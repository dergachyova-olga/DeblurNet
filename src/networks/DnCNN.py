# ======================================================================================================================
# name        : DnCNN.py
# type        : network architecture
# purpose     : define DnCNN network architecture
# author      : Olga Dergachyova
# last update : 10/2020
# ======================================================================================================================

import tensorflow as tf
from collections import OrderedDict
import logging
from convnet.layers import (conv, activation, batch_norm)


def create_DnCNN(cfg, data_provider, x, training, type='2D', AF='relu', kernel_size=3,
                  kernel_init='ortho', middle_layers=15, features=64, residual=True, use_BN=True, BN_axis=3, BN_momentum=0.0, BN_epsilon=0.0001, seed=None):
    """
    Creates a new DnCNN for the given parametrization..

    :param cfg: configuration object containing data parameters
    :param data_provider: object containing and manipulation data
    :param x: input tensor, shape [?,nx,ny,channels]
    :param training: defines if the run is for training or tests
    :param AF: activation function on intermediate conv layers
    :param kernel_size: (optional) size of the convolution kernel
    :param kernel_init: (optional) initializer for weights
    :param middle_layers: (optional) number of layers between first and last Conv layers
    :param features: (optional) number of features in the first layer
    :param residual: (optional) defines if residual formulation is used
    :param use_BN: (optional) flag if batch normalization layers should be used
    :param BN_axis: (optional) axis for batch normalization
    :param BN_momentum: (optional) momentum for batch normalization
    :param BN_epsilon: (optional) epsilon for batch normalization
    :param seed: (optional) seed for kernel initialization and dropout
    """

    # set variables
    nx = data_provider.nx
    ny = data_provider.ny
    channels = data_provider.channels
    n_class = data_provider.n_class

    AF = cfg.network.af if cfg.network.af else AF
    kernel_size = cfg.network.kernel_size if cfg.network.kernel_size else kernel_size
    kernel_init = cfg.network.kernel_init if cfg.network.kernel_init else kernel_init
    middle_layers = cfg.DnCNN.middle_layers if cfg.DnCNN.middle_layers else middle_layers
    features = cfg.DnCNN.features if cfg.DnCNN.features else features
    residual = cfg.DnCNN.residual if cfg.DnCNN.residual is not None else residual
    use_BN = cfg.network.use_bn if cfg.network.use_bn is not None else use_BN
    BN_axis = cfg.DnCNN.bn_kwargs.get('axis', BN_axis)
    BN_momentum = cfg.DnCNN.bn_kwargs.get('momentum', BN_momentum)
    BN_epsilon = cfg.DnCNN.bn_kwargs.get('epsilon', BN_epsilon)
    seed = cfg.network.seed if cfg.network.seed is not None else seed

    kernel_params = {'init': kernel_init, 'size': kernel_size, 'seed': seed}

    logging.info("----- Network has {layers} layers, {features} features, and {kernel_size}x{kernel_size} kernel size".
                 format(layers=middle_layers+2, features=features, kernel_size=kernel_params['size']))

    # Manually enforce image size
    with tf.name_scope("preprocessing"):
        input = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = input

    convs = OrderedDict()
    activs = OrderedDict()

    # -------- Network architecture

    # first layer
    with tf.name_scope("first_layer"):
        first_conv = conv(type, in_node, features, kernel_params, name="conv")
        first_activ = activation(first_conv, AF, name="activ")

        in_node = first_activ
        convs["first_layer"] = first_conv
        activs["first_layer"] = first_activ

    # middle layers
    for layer in range(0, middle_layers):
        name_scope = "middle_layer{0}".format(str(layer))
        with tf.name_scope(name_scope):
            middle_conv = conv(type, in_node, features, kernel_params, use_bias=False, name="conv")
            middle_bn = batch_norm(middle_conv, use_BN, training, BN_axis, BN_momentum, BN_epsilon)
            middle_activ = activation(middle_bn, AF, name="activ")

            in_node = middle_activ
            convs[name_scope] = middle_conv
            activs[name_scope] = middle_activ

    # TODO: 3D conv

    # last layer
    with tf.name_scope("last_layer"):
        last_conv = conv(type, in_node, n_class, kernel_params, use_bias=False, name="conv")
        convs["last_layer"] = last_conv

    # input - noise
    if residual:
        with tf.name_scope("residual"):
            if n_class > 1:
                # TODO: make n-channel
                if data_provider.order == 'stacked':
                    input_ = [input[:, :, :, i] for i in range(0, channels) if i % 2 == 0]
                elif data_provider.order == 'grouped':
                    input_ = [input[:, :, :, i] for i in range(0, int(channels/2))]
                input_ = tf.stack(input_, axis=3)
                output = tf.keras.layers.Subtract(name="subtract")([input_, last_conv])
            else:
                input_ = tf.reshape(input[:, :, :, 0], [-1, nx, ny, 1])
                last_conv_ = tf.reshape(last_conv[:, :, :, 0], [-1, nx, ny, 1])
                output = tf.keras.layers.Subtract(name="subtract")([input_, last_conv_])
    else:
        output = last_conv

    # -------- Network architecture

    variables = tf.trainable_variables()
    layers = {'convs': convs, 'activs': activs}

    return output, variables, layers


