# ======================================================================================================================
# name        : convnet.py
# type        : ConvNet class
# purpose     : define main functionality of a general purpose convolutional neural network
# author      : Olga Dergachyova
# last update : 10/2020
# ======================================================================================================================

import os
import shutil
import numpy as np
import logging
import tensorflow as tf
from skimage.measure import compare_ssim as ssim

from utils import util
import convnet.tensorboard as tb
from convnet.cost_functions import (MSE_cost, MAE_cost, MSEw2_cost, MSEw_cost, MAEw2_cost, MAEw_cost, SSIM_cost)
from networks.unet import create_unet
from networks.DnCNN import create_DnCNN
from networks.c2net import create_c2net
from networks.offresnet import (create_offresnet2D, create_offresnet3D)
from networks.Knet import create_Knet2D


class ConvNet(object):

    def __init__(self, cfg, data_provider):
        """
         A convnet implementation

        :param n_class: number of output labels
        :param network: type of network
        :param conv: type of convolution
        :param cost_name: name of the cost function
        :param cost_reg: power of the L2 regularizers added to the loss function
        """

        # set variables
        self.n_class = data_provider.n_class
        self.network = cfg.network.name
        self.conv = cfg.network.conv
        self.cost_name = cfg.train.cost
        self.cost_reg = cfg.train.cost_reg

        # reset graph
        tf.reset_default_graph()
        with tf.Graph().as_default():
            tf.set_random_seed(cfg.general.seed)

        # assign placeholders for data and hyper-parameters
        self.x = tf.placeholder("float", shape=[None, None, None, data_provider.channels], name="x")
        self.y = tf.placeholder("float", shape=[None, None, None, self.n_class], name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_probability")
        self.training = tf.placeholder(tf.bool, name='training')

        # construct network graph
        if self.network == 'unet':
            if self.conv == '2D' or self.conv == '3D':
                logits, self.variables, self.layers = create_unet(cfg, data_provider, self.x, self.training, self.keep_prob, self.conv)
            else:
                raise ValueError('Undefined convolution type')

        elif self.network == 'DnCNN':
            if self.conv == '2D' or self.conv == '3D':
                if self.conv == '3D':
                    logging.info('3D DnCNN is not available yet. 2D version will be created instead')
                logits, self.variables, self.layers = create_DnCNN(cfg, data_provider, self.x, self.training, '2D')
            else:
                raise ValueError('Undefined convolution type')

        elif self.network == 'c2net':
            if self.conv == '2D' or self.conv == '3D':
                logits, self.variables, self.layers = create_c2net(cfg, data_provider, self.x, self.training, self.keep_prob, self.conv)
            else:
                raise ValueError('Undefined convolution type')

        elif self.network == 'offresnet':
            if self.conv == '2D':
                logits, self.variables, self.layers = create_offresnet2D(cfg, data_provider, self.x, self.training, self.keep_prob)
            elif self.conv == '3D':
                logits, self.variables, self.layers = create_offresnet3D(cfg, data_provider, self.x, self.training, self.keep_prob)
            else:
                raise ValueError('Undefined convolution type')

        else:
            raise ValueError('Wrong network name')

        logging.info('Created ' + self.network + ' using ' + self.conv + ' convolutions')

        # define cost function
        self.cost = self._get_cost(logits, self.cost_name, self.cost_reg)

        # define predicter operation
        with tf.name_scope("results"):
            self.predicter = logits

        # compute MSE for summary
        with tf.name_scope("MSE"):
            self.MSE = MSE_cost(self.y, logits)

        # compute MAE for summary
        with tf.name_scope("MAE"):
            self.MAE = MAE_cost(self.y, logits)

        # compute SSIM for summary
        with tf.name_scope("SSIM"):
            self.SSIM = SSIM_cost(self.y, logits)

        # count and display number of parameters in the network
        n_params = util.get_n_params(tf.trainable_variables())
        logging.info('----- Number of trainable parameters: ' + str(n_params))

    def _get_cost(self, logits, cost_name, cost_reg):
        """
        Constructs the cost function.

        :param logits: predicted values
        :param cost_name: name of cost function to use for training
        :param cost_reg: power of the L2 regularizers added to the loss function
        """

        with tf.name_scope("cost"):

            if cost_name == "MSE":
                loss = MSE_cost(self.y, logits)

            elif cost_name == "MAE":
                loss = MAE_cost(self.y, logits)

            elif cost_name == "MSEw2":
                loss = MSEw2_cost(self.x, self.y, logits)

            elif cost_name == "MSEw":
                loss = MSEw_cost(self.x, self.y, logits)

            elif cost_name == "MAEw2":
                loss = MAEw2_cost(self.x, self.y, logits)

            elif cost_name == "MAEw":
                loss = MAEw2_cost(self.x, self.y, logits)

            elif cost_name == "SSIM":
                loss = SSIM_cost(self.y, logits)

            else:
                raise ValueError('Unknown cost function')

            if cost_reg is not None:
                regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
                loss += (cost_reg * regularizers)

            return loss

    def predict(self, model_path, x, close_sess=True):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x: data to predict on
        :param close_sess: flag indicating if session should be closed at the end of the function or left open
        :returns prediction: network prediction
        """

        # open new session
        init = tf.global_variables_initializer()
        sess = tf.Session()

        # initialize variables
        sess.run(init)

        # restore model weights from previously saved model
        self.restore(sess, model_path)

        # make prediction
        y_dummy = np.empty((x.shape[0], x.shape[1], x.shape[2], self.n_class))
        prediction = sess.run(self.predicter, feed_dict={self.x: x, self.y: y_dummy, self.keep_prob: 1., self.training: 0})

        # close session
        if close_sess:
            sess.close()

        return prediction, sess

    def predict_batch(self, sess, x):
        """
        Uses the model from the current session to create a prediction for the given data

        :param sess: session with the model restored from checkpoint using self.predict beforehand
        :param x: Data to predict on
        :returns prediction: network predictiom
        """
        # make prediction
        y_dummy = np.empty((x.shape[0], x.shape[1], x.shape[2], self.n_class))
        prediction = sess.run(self.predicter, feed_dict={self.x: x, self.y: y_dummy, self.keep_prob: 1., self.training: 0})

        return prediction

    def evaluate_prediction(self, predictions, y):
        """
        Evaluates and prints model performance metrics comparing actual labels and model's predictions

        :param predictions: predictions made by the current model
        :param y: actual labels for the current data
        """

        logging.info("Final prediction: R square = {:.4f},  MSE = {:.6f}, MAE = {:.6f}, SSIM = {:.6f}".
                     format(R_square_score(predictions, y),  mean_squared_error(predictions, y),
                            mean_absolute_error(predictions, y), structural_similarity(predictions, y)))

    def save(self, sess, model_path, epoch=None, step=None):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        :param epoch: training epoch number
        :param step: training step number (batch-wise)
        """

        model_path = os.path.join(model_path, "model", "model.ckpt")
        saver = tf.train.Saver()
        if epoch is None and step is None:
            save_path = saver.save(sess, model_path)
        else:
            save_path = saver.save(sess, model_path, global_step=step, write_meta_graph=False)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(model_path, "model", "model.ckpt"))
        logging.info("Model restored from file: %s" % model_path)


class Trainer(object):

    def __init__(self, net, cfg, optimizer='adam', optimizer_kwargs={}, train_batch_size=1, valid_batch_size=1):
        """
        Trains a convnet instance

        :param net: the convnet instance to train
        :param optimizer: (optional) name of the optimizer to use (momentum or adam)
        :param optimizer_kwargs: (optional) additional parameters for optimizer. May contain learning_rate, decay_rate, and momentum.
        :param train_batch_size: size of training batch
        :param valid_batch_size: size of validation batch
        """

        # set variables
        self.net = net
        self.optimizer = cfg.optimizer if cfg.optimizer else optimizer
        self.optimizer_kwargs = cfg.optimizer_kwargs if cfg.optimizer_kwargs else optimizer_kwargs
        self.train_batch_size = cfg.train_batch_size if cfg.train_batch_size else train_batch_size
        self.valid_batch_size = cfg.valid_batch_size if cfg.valid_batch_size else valid_batch_size

    def _get_optimizer(self, training_iters, global_step):
        """
        Returns network optimizer

        :param training_iters: number of training iterations (do not confuse with epochs)
        :param global_step: absolute number of interations
        :returns optimizer: optimizer object
        """

        # Momentum Optimizer
        if self.optimizer == 'momentum':
            learning_rate = self.optimizer_kwargs.get('learning_rate', 0.1)  # 0.1
            decay_rate = self.optimizer_kwargs.get('decay_rate', 0.95)  # learning_rate / epochs
            momentum = self.optimizer_kwargs.get('momentum', 0.2)  # from 0.5 to 0.9

            with tf.name_scope("lr"):
                self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step, decay_steps=training_iters,
                                                                 decay_rate=decay_rate, staircase=True)

            with tf.name_scope("optimizer"):
                opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum)
                optimizer = opt.minimize(self.net.cost, global_step=global_step)

        # Adam Optimizer
        elif self.optimizer == 'adam':
            learning_rate = self.optimizer_kwargs.get('learning_rate', 0.001)
            self.learning_rate = tf.Variable(learning_rate, name='learning_rate')

            with tf.name_scope("optimizer"):
                opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                optimizer = opt.minimize(self.net.cost, global_step=global_step)

            with tf.name_scope("lr"):
                self.learning_rate_node = (opt._lr * tf.sqrt(1 - opt._beta2) / (1 - opt._beta1))

        return optimizer

    def _initialize(self, training_iters, restore, model_path, prediction_path, summaries):
        """
        Initializes training process and variables

        :param training_iters: number of training iterations (do not confuse with epochs)
        :param restore: flag if previous model should be restored
        :param model_path: path where to store checkpoints
        :param prediction_path: path to save predictions made during train and test time
        :param summaries: flag to write tensor board summaries
        :returns init: object with global variables
        """

        # optimizer + link BN update ops
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step = tf.Variable(0, name="global_step")
            self.optimizer = self._get_optimizer(training_iters, global_step)

        # summary
        if summaries:
            # metrics
            metrics = [tf.summary.scalar('loss', self.net.cost)]
            if self.net.cost_name != 'MSE':
                metrics.append(tf.summary.scalar('MSE_metric', self.net.MSE))
            if self.net.cost_name != 'MAE':
                metrics.append(tf.summary.scalar('MAE_metric', self.net.MAE))
            if self.net.cost_name != 'SSIM':
                metrics.append(tf.summary.scalar('SSIM_metric', self.net.SSIM))
            metrics_smr = tf.summary.merge(metrics)

            # learning rate
            lr_smr = tf.summary.scalar('learning_rate', self.learning_rate_node)

            # histograms
            variables_smr = tb.variables_summary(self.net.variables)
            gradients_smr = tb.gradients_summary(self.net.cost, self.net.variables)
            activations_smr = tb.activations_summary(self.net.layers['activs'], self.net.network)
            features_smr = tb.features_summary(self.net.layers['convs'], self.net.layers['deconvs'], self.net.network)

            # images
            variables_img_smr = tb.variables_image_summary(self.net.variables)
            activations_img_smr = tb.activations_image_summary(self.net.layers['activs'])
            features_img_smr = tb.features_image_summary(self.net.layers['convs'], self.net.layers['deconvs'])

            # separate train, valid and visual summaries
            self.train_summary = tf.summary.merge([metrics_smr, lr_smr, variables_smr, gradients_smr, activations_smr, features_smr, variables_img_smr, activations_img_smr, features_img_smr])
            self.valid_summary = tf.summary.merge([metrics_smr, gradients_smr, activations_smr, features_smr])
            self.visual_summary = tf.summary.merge([metrics_smr, activations_img_smr, features_img_smr])

        # no summary (only loss)
        else:
            self.train_summary = self.valid_summary = self.visual_summary = tf.summary.scalar('loss', self.net.cost)

        # init global variables
        init = tf.global_variables_initializer()

        # create new directories for output model and predictions or restore from existing ones
        self.prediction_path = prediction_path
        prepare_output_dir(restore, prediction_path, model_path)

        return init

    def train(self, data_provider, cfg, training_iters=None, epochs=100, keep_prob=0.75, display_step=1, patience=None,
              model_path='./convnet_trained', prediction_path='prediction', restore=False, summaries=False):
        """
        Launches the training process

        :param data_provider: callable returning training and verification data
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param keep_prob: dropout probability of keeping neurons
        :param display_step: number of steps till outputting stats
        :param patience: number of epochs to wait untill stop if validation loss increases
        :param model_path: path where to store checkpoints
        :param prediction_path: path to save predictions made during train and test time
        :param restore: flag if previous model should be restored
        :param summaries: flag to write tensor board summaries
        """

        # set variables
        training_iters = cfg.train.training_iters if cfg.train.training_iters else training_iters
        epochs = cfg.train.epochs if cfg.train.epochs is not None else epochs
        keep_prob = cfg.network.keep_prob if cfg.network.keep_prob else keep_prob
        display_step = cfg.train.display_step if cfg.train.display_step else display_step
        patience = cfg.train.patience if cfg.train.patience is not None else patience
        model_path = cfg.results.model_path if cfg.results.model_path else model_path
        prediction_path = cfg.results.prediction_path if cfg.results.prediction_path else prediction_path
        restore = cfg.train.restore if cfg.train.restore is not None else restore
        summaries = cfg.results.summaries if cfg.results.summaries is not None else summaries

        # count number of training iterations
        ratio = data_provider.get_train_data_n() / self.train_batch_size
        if (training_iters is None) or (training_iters > ratio):
            training_iters = int(ratio) if ratio.is_integer() else int(ratio) + 1

        # initialize network
        init = self._initialize(training_iters, restore, model_path, prediction_path, summaries)

        # trick to enable GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # restore existing session or start with a new one
            if restore:
                self.net.restore(sess, model_path)
            else:
                sess.run(init)
                # save initial graph
                self.net.save(sess, model_path)

            if epochs > 0:
                # get visualization samples and save initial prediction
                visual_x, visual_y = data_provider.get_visual_data()
                if data_provider.visualize_during_training:
                    train_vis_x, train_vis_y = data_provider.get_train_vis_data()
                    valid_vis_x, valid_vis_y = data_provider.get_valid_vis_data()
                    self.store_prediction(sess, train_vis_x, train_vis_y, "train_init")
                    self.store_prediction(sess, valid_vis_x, valid_vis_y, "valid_init")

                # initialize summary writers
                train_summary_writer = tf.summary.FileWriter(model_path + '/train', graph=sess.graph)
                valid_summary_writer = tf.summary.FileWriter(model_path + '/valid')
                if visual_x is not None and visual_y is not None:
                    visual_summary_writer = tf.summary.FileWriter(model_path + '/visual')

                logging.info("Start optimization\r\n")

                for epoch in range(epochs):
                    total_loss = 0
                    for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):

                        # get new batch data
                        train_batch_x, train_batch_y = data_provider.get_train_batch(self.train_batch_size)

                        # Run optimization op (backprop)
                        _, loss, learning_rate = sess.run(
                            (self.optimizer, self.net.cost, self.learning_rate_node),
                            feed_dict={self.net.x: train_batch_x,
                                       self.net.y: train_batch_y,
                                       self.net.keep_prob: keep_prob,
                                       self.net.training: 1})

                        total_loss += loss

                        # display minibatch statistics
                        self.output_minibatch_stats(sess, train_summary_writer, step, display_step, train_batch_x, train_batch_y)

                        # save model current state
                        self.net.save(sess, model_path, epoch, step)

                    # display epoch statistics
                    valid_batch_x, valid_batch_y = data_provider.get_valid_batch(self.valid_batch_size)
                    valid_loss = self.output_epoch_stats(sess, epoch, step, learning_rate, valid_batch_x, valid_batch_y,
                                                         visual_x, visual_y, valid_summary_writer, visual_summary_writer)

                    # save predictions for visualization samples
                    if data_provider.visualize_during_training:
                        self.store_prediction(sess, train_vis_x, train_vis_y, "train_epoch_%s" % epoch)
                        self.store_prediction(sess, valid_vis_x, valid_vis_y, "valid_epoch_%s" % epoch)

                    # early stopping
                    if self.stop_early(valid_loss, patience, epoch):
                        break

                    # save model current state
                    self.net.save(sess, model_path)

                logging.info("Optimization Finished!")

    def stop_early(self, valid_loss, patience, epoch):
        """
        Stops training when the performance on validation data starts to decrease (i.e. loss increases)

        :param valid_loss: loss on validation data
        :param patience: number of epochs to wait before stopping the training
        :param epoch: current epoch number
        :returns flag to either stop or continue training
        """

        if epoch == 0:
            self.valid_loss = np.Inf
            self.patience_cnt = 0

        if patience is not None:
            if valid_loss > self.valid_loss:
                self.patience_cnt += 1
            elif valid_loss < self.valid_loss:
                self.patience_cnt = 0
            if self.patience_cnt > patience:
                logging.info("Early stopping after epoch {:} after patience for {:} epochs".format(epoch, patience))
                return True
            else:
                self.valid_loss = valid_loss
                return False
        else:
            return False

    def store_prediction(self, sess, batch_x, batch_y, name):
        """
        Makes a prediction on current batch and stores it

        :param sess: current session instance
        :param batch_x: current batch data
        :param batch_y: current batch label
        :param name: name of saving file
        """

        prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x, self.net.y: batch_y,
                                                             self.net.keep_prob: 1., self.net.training: 0})

        img = util.combine_img_prediction(batch_x, batch_y, prediction)
        util.save_image(img, "%s/%s" % (self.prediction_path, name))

    def output_minibatch_stats(self, sess, summary_writer, step, display_step, batch_x, batch_y):
        """
        Displays performances on current batch and adds them to summary

        :param sess: current session instance
        :param summary_writer: object saving model's summary
        :param step: current training iteration
        :param display_step: number indicating how often statistics to be displayed
        :param batch_x: current batch data
        :param batch_y: current batch label

        """
        if step % display_step == 0:
            summary_str, loss, predictions = sess.run([self.train_summary, self.net.cost, self.net.predicter],
                                                      feed_dict={self.net.x: batch_x, self.net.y: batch_y, self.net.keep_prob: 1., self.net.training: 0})

            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
            logging.info("Iter {:}, Minibatch Loss = {:.6f}, MSE = {:.6f}, MAE = {:.6f}".
                         format(step, loss, mean_squared_error(predictions, batch_y), mean_absolute_error(predictions, batch_y)))

    def output_epoch_stats(self, sess, epoch, step, learning_rate,
                           valid_x, valid_y, visual_x, visual_y, valid_summary_writer, visual_summary_writer):

        """
        Displays performances on current batch and adds them to summary

        :param sess: current session instance
        :param epoch: current training epoch number
        :param step: current training iteration
        :param learning_rate: current learning rate
        :param valid_x: validation input data
        :param valid_y: validation label
        :param visual_x: visualization input data
        :param visual_y: visualization label
        :param visual_x: visualization input data
        :param visual_y: visualization label
        :param valid_summary_writer: summary writer object for validation data
        :param visual_summary_writer: summary writer object for visualization data
        :returns loss: loss function value for validation data

        """

        logging.info("Epoch {:} ------------------ learning rate: {:.6f}".format(epoch, learning_rate))

        # for validation
        summary_str, loss, predictions = sess.run([self.valid_summary, self.net.cost, self.net.predicter],
                                                  feed_dict={self.net.x: valid_x, self.net.y: valid_y,
                                                             self.net.keep_prob: 1., self.net.training: 0})
        valid_summary_writer.add_summary(summary_str, step)
        valid_summary_writer.flush()
        logging.info("Validation loss = {:.6f}, MSE = {:.6f}, MAE = {:.6f}\r\n".
                     format(loss, mean_squared_error(predictions, valid_y), mean_absolute_error(predictions, valid_y)))

        # for visualization
        if visual_x is not None and visual_y is not None:
            summary_str = sess.run(self.visual_summary, feed_dict={self.net.x: visual_x, self.net.y: visual_y,
                                                                   self.net.keep_prob: 1., self.net.training: 0})
            visual_summary_writer.add_summary(summary_str, step)
            visual_summary_writer.flush()

        return loss


def prepare_output_dir(restore, prediction_path, model_path):
    """
    Prepares directories where the model and predictions will be saved

    :param restore: flag if previous model should be restored
    :param prediction_path: path to save predictions made during train and test time
    :param model_path: path to save model's checkpoints

    """

    abs_prediction_path = os.path.abspath(prediction_path)
    model_path = os.path.abspath(model_path)

    if not restore:
        logging.info("Removing '{:}'".format(abs_prediction_path))
        shutil.rmtree(abs_prediction_path, ignore_errors=True)
        logging.info("Removing '{:}'".format(model_path))
        shutil.rmtree(model_path, ignore_errors=True)

    if not os.path.exists(abs_prediction_path):
        logging.info("Allocating '{:}'".format(abs_prediction_path))
        os.makedirs(abs_prediction_path)

    if not os.path.exists(model_path):
        logging.info("Allocating '{:}'".format(model_path))
        os.makedirs(model_path)
        os.makedirs(os.path.join(model_path, "model"))


# -------- Performance scores
def mean_squared_error(predictions, labels):
    return ((predictions - labels) ** 2).mean()


def mean_absolute_error(predictions, labels):
    return (np.absolute(predictions - labels)).mean()


def R_square_score(predictions, labels):
    total_error = np.sum((labels - labels.mean()) ** 2)
    unexplained_error = np.sum((labels - predictions) ** 2)
    R_squared = 1.0 - unexplained_error / total_error
    return R_squared


def structural_similarity(predictions, labels):
    predictions = predictions.astype(np.float32)
    labels = labels.astype(np.float32)
    total_ssim = 0
    for s in range(0, labels.shape[0]):
        total_ssim += ssim(predictions[s,:,:,:], labels[s, :, :, :], multichannel=True)
    return total_ssim / labels.shape[0]
