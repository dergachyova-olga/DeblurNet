# ======================================================================================================================
# name        : main.py
# type        : main script
# purpose     : performs deblurring and/or denoising of MRI using deep learning;
#               constructs, trains, validates and saves a convolutional neural network according to chosen configuration
# parameters  : change settings in 'configs/config.ini'
# author      : Olga Dergachyova
# last update : 10/2020
# ======================================================================================================================

from __future__ import division, print_function
import sys
import logging
import config
import numpy as np
from convnet import convnet, loader
from utils import util


# ---------------------------- Load configuration ---------------------------- #
# load a config file provided as input argument; if no config provided, load the default configuration
cfg = config.load_config(sys.argv[1] if len(sys.argv) > 1 else None)


# ---------------------------- Setup data provider ---------------------------- #
# create and configure an object of a class that loads, holds and manipulates input and output data
data_provider = loader.MatDataProvider(cfg.data)

# load training data
data_provider.load_train_data()
channels = data_provider.channels


# ------------------------- Setup and train a network ------------------------- #
# construct a convolutional network model according to the configuration
net = convnet.ConvNet(cfg, data_provider)

# train the model on provided input data
trainer = convnet.Trainer(net, cfg.train)
trainer.train(data_provider, cfg)


# --------- Make image predictions for train and valid visualization data --------- #
# this part is necessary to have a visual representation of model's performance;

# ----- for train visualization data
logging.info('Predict from train visualization data')

# load data set apart from training data for visualization purposes
train_vis_x, train_vis_y = data_provider.get_train_vis_data()
# process the data using trained model
train_vis_prediction, _ = net.predict(cfg.results.model_path, train_vis_x)
# evaluate model's performance and save the processed data
net.evaluate_prediction(train_vis_prediction, train_vis_y)
path = cfg.results.prediction_path + '/train_vis_results.png'
util.plot_result_image(data_provider.n_vis, channels, train_vis_x, train_vis_y, train_vis_prediction, path)

# ----- for validation visualization data
logging.info('Predict from valid visualization data')

# load data set apart from validation data for visualization purposes
valid_vis_x, valid_vis_y = data_provider.get_valid_vis_data()
# process the data using trained model
valid_vis_prediction, _ = net.predict(cfg.results.model_path, valid_vis_x)
# evaluate model's performance and save processed data as image
net.evaluate_prediction(valid_vis_prediction, valid_vis_y)
path = cfg.results.prediction_path + '/valid_vis_results.png'
util.plot_result_image(data_provider.n_vis, channels, valid_vis_x, valid_vis_y, valid_vis_prediction, path)


# -------- Make mat predictions for entire valid and visual data sets -------- #
# this part is necessary to perform quantitative evaluation of model's performances

# ----- for train data
logging.info('Predict from train data')

# compute the number of iterations needed to process all training data
ratio = data_provider.get_train_data_n() / cfg.train.train_batch_size
iters = int(ratio) if ratio.is_integer() else int(ratio) + 1
sess = train_y = train_prediction = None

# process all training data in batches (last batch may be incomplete)
for i in range(iters):
    if sess is None:
        # get the first training data batch
        train_batch_x, train_batch_y = data_provider.get_train_batch(cfg.train.train_batch_size, idx=0)
        # process the batch using trained model
        train_batch_prediction, sess = net.predict(cfg.results.model_path, train_batch_x, close_sess=False)
        # get actual results and processing results
        train_y = train_batch_y
        train_prediction = train_batch_prediction
    else:
        # get all the remaining training data batches one-by-one
        train_batch_x, train_batch_y = data_provider.get_train_batch(cfg.train.train_batch_size)
        # process the batch using trained model
        train_batch_prediction = net.predict_batch(sess, train_batch_x)
        # get actual results and processing results
        train_y = np.concatenate((train_y, train_batch_y))
        train_prediction = np.concatenate((train_prediction, train_batch_prediction))
sess.close()

# evaluate model's performance and save processed data as (MATLAB) matrix
net.evaluate_prediction(train_prediction, train_y)
util.save_mat(train_prediction, cfg.results.prediction_path + '/train_results.mat')
util.save_mat(train_y, cfg.results.prediction_path + '/train_gt.mat')

# ----- for valid data
logging.info('Predict from valid data')

# compute the number of iterations needed to process all validation data
ratio = data_provider.get_valid_data_n() / cfg.train.valid_batch_size
iters = int(ratio) if ratio.is_integer() else int(ratio) + 1
sess = valid_y = valid_prediction = None

# process all validation data in batches (last batch may be incomplete)
for i in range(iters):
    if sess is None:
        # get the first validation data batch
        valid_batch_x, valid_batch_y = data_provider.get_valid_batch(cfg.train.valid_batch_size, idx=0)
        # process the batch using trained model
        valid_batch_prediction, sess = net.predict(cfg.results.model_path, valid_batch_x, close_sess=False)
        # get actual results and processing results
        valid_y = valid_batch_y
        valid_prediction = valid_batch_prediction
    else:
        # get all the remaining validation data batches one-by-one
        valid_batch_x, valid_batch_y = data_provider.get_valid_batch(cfg.train.valid_batch_size)
        # process the batch using trained model
        valid_batch_prediction = net.predict_batch(sess, valid_batch_x)
        # get actual results and processing results
        valid_y = np.concatenate((valid_y, valid_batch_y))
        valid_prediction = np.concatenate((valid_prediction, valid_batch_prediction))

# evaluate model's performance and save processed data as (MATLAB) matrix
sess.close()
net.evaluate_prediction(valid_prediction, valid_y)
util.save_mat(valid_prediction, cfg.results.prediction_path + '/valid_results.mat')
util.save_mat(valid_y, cfg.results.prediction_path + '/valid_gt.mat')

# ----- for visual data
if cfg.data.visual_data:
    logging.info('Predict from visual data')
    # get visualization data
    visual_x, visual_y = data_provider.get_visual_data()
    # process data
    visual_prediction, _ = net.predict(cfg.results.model_path, visual_x)
    # evaluate model's performance and save processed data as (MATLAB) matrix
    net.evaluate_prediction(visual_prediction, visual_y)
    util.save_mat(visual_prediction, cfg.results.prediction_path + '/visual_results.mat')
    util.save_mat(visual_y, cfg.results.prediction_path + '/visual_gt.mat')

