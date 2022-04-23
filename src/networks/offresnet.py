# ======================================================================================================================
# name        : offresnet.py
# type        : network architecture
# purpose     : define offresnet network architecture
# author      : Olga Dergachyova
# last update : 10/2020
# ======================================================================================================================

import tensorflow as tf
from collections import OrderedDict
import logging
from convnet.layers import (conv2D, conv3D, activation, dropout, batch_norm)


def create_offresnet2D(cfg, data_provider, x, training, keep_prob, AF='relu', output_AF='linear', kernel_size=3, kernel_init='he',
                       blocks=0, layers=2, features=128, use_skip=True, use_dropout=False, use_BN=False, seed=None):
    """
    Creates a new 2D Off-Resnet for the given parametrization.

    :param cfg: configuration object containing data parameters
    :param data_provider: object containing and manipulation data
    :param x: input tensor, shape [?,nx,ny,channels]
    :param training: defines if the run is for training or tests
    :param keep_prob: dropout probability tensor indicating the part of neurons to keep
    :param nx: width of input image
    :param ny: height of input image
    :param n_class: number of output labels
    :param AF: activation function on intermediate conv layers
    :param output_AF: activation function on last conv layers
    :param kernel_size: (optional) size of the convolution kernel
    :param kernel_init: (optional) initializer for weights
    :param blocks: number of residual blocks in the net
    :param layers: number of convolutional layers in one residual block
    :param features: number of features in the first layer
    :param use_skip: flag if feature concatenation skip connections should be used
    :param use_dropout: flag if dropout layers should be used
    :param use_BN: flag if batch normalization layers should be used
    :param seed: (optional) seed for kernel initialization and dropout
    """

    # set variables
    nx = data_provider.nx
    ny = data_provider.ny
    channels = data_provider.channels
    n_class = data_provider.n_class

    AF = cfg.network.af if cfg.network.af else AF
    output_AF = cfg.network.output_af if cfg.network.output_af else output_AF
    kernel_size = cfg.network.kernel_size if cfg.network.kernel_size else kernel_size
    kernel_init = cfg.network.kernel_init if cfg.network.kernel_init else kernel_init
    blocks = cfg.offresnet.blocks if cfg.offresnet.blocks else blocks
    layers = cfg.offresnet.layers if cfg.offresnet.layers else layers
    features = cfg.offresnet.features if cfg.offresnet.features else features
    use_skip = cfg.offresnet.use_skip if cfg.offresnet.use_skip is not None else use_skip
    use_dropout = cfg.network.use_dropout if cfg.network.use_dropout is not None else use_dropout
    use_BN = cfg.network.use_bn if cfg.network.use_bn is not None else use_BN
    seed = cfg.network.seed if cfg.network.seed is not None else seed

    kernel_params = {'init': kernel_init, 'size': kernel_size, 'seed': seed}

    logging.info("----- Layers {layers}, features {features}, kernel size {kernel_size}x{kernel_size}".
                 format(layers=layers, features=features, kernel_size=kernel_params['size']))

    convs = OrderedDict()
    activs = OrderedDict()
    concats = OrderedDict()
    skip_cons = OrderedDict()

    # Placeholder for the input image
    with tf.name_scope("preprocessing"):
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image

    # -------- Network architecture

    # first convolutional layer
    with tf.name_scope("input"):
        conv = conv2D(in_node, features, kernel_params, name="conv")
        drop = dropout(conv, keep_prob, use_dropout, seed, name="drop")
        bn = batch_norm(drop, use_BN, training, name="bn-in")
        activ = activation(bn, AF, name="activ")

        in_node = activ
        skip_cons[0] = in_node
        convs["input"] = conv
        activs["input"] = activ

    # residual layers
    for block in range(1, blocks+1):

        block_convs = list()
        block_activs = list()

        name_scope = "res_block_{}".format(str(block))
        with tf.name_scope(name_scope):
            for layer in range(1, layers+1):
                conv = conv2D(in_node, features, kernel_params, name="conv{}".format(str(layer)))
                drop = dropout(conv, keep_prob, use_dropout, seed, name="drop{}".format(str(layer)))
                bn = batch_norm(drop, use_BN, training, name="bn{}".format(str(layer)))
                activ = activation(bn, AF, name="activ{}".format(str(layer)))

                in_node = activ
                block_convs.append(conv)
                block_activs.append(activ)

            if use_skip:
                in_node += skip_cons[block-1]
            skip_cons[block] = in_node
            concats[name_scope] = in_node
            convs[name_scope] = tuple(block_convs)
            activs[name_scope] = tuple(block_activs)

    # last convolutional layer
    kernel_params['size'] = 1
    with tf.name_scope("output"):
        conv = conv2D(in_node, n_class, kernel_params, name="conv")
        drop = dropout(conv, keep_prob, use_dropout, seed, name="drop")
        bn = batch_norm(drop, use_BN, training, name="bn-out")
        activ = activation(bn, output_AF, name="activ")

        convs["output"] = conv
        activs["output"] = activ
        output_map = activ

    # -------- Network architecture

    variables = tf.trainable_variables()
    layers = {'convs': convs, 'activs': activs, 'concats': concats, 'deconvs': {}}

    return output_map, variables, layers


def create_offresnet3D(cfg, data_provider, x, training, keep_prob, AF='relu', output_AF='linear', kernel_size=3, kernel_init='he',
                       blocks=0, layers=2, features=128, use_skip=True, use_dropout=False, use_BN=False, seed=None):
    """
    Creates a new 3D Off-Resnet for the given parametrization.

    :param cfg: configuration object containing data parameters
    :param data_provider: object containing and manipulation data
    :param x: input tensor, shape [?,nx,ny,channels]
    :param training: defines if the run is for training or tests
    :param keep_prob: dropout probability tensor indicating the part of neurons to keep
    :param nx: width of input image
    :param ny: height of input image
    :param n_class: number of output labels
    :param AF: activation function on intermediate conv layers
    :param output_AF: activation function on last conv layers
    :param kernel_size: (optional) size of the convolution kernel
    :param kernel_init: (optional) initializer for weights
    :param blocks: number of residual blocks in the net
    :param layers: number of convolutional layers in one residual block
    :param features: number of features in the first layer
    :param use_skip: flag if feature concatenation skip connections should be used
    :param use_dropout: flag if dropout layers should be used
    :param use_BN: flag if batch normalization layers should be used
    :param seed: (optional) seed for kernel initialization and dropout
    """

    # set variables
    nx = data_provider.nx
    ny = data_provider.ny
    channels = data_provider.channels
    n_class = data_provider.n_class

    AF = cfg.network.af if cfg.network.af else AF
    output_AF = cfg.network.output_af if cfg.network.output_af else output_AF
    kernel_size = cfg.network.kernel_size if cfg.network.kernel_size else kernel_size
    kernel_init = cfg.network.kernel_init if cfg.network.kernel_init else kernel_init
    blocks = cfg.offresnet.blocks if cfg.offresnet.blocks else blocks
    layers = cfg.offresnet.layers if cfg.offresnet.layers else layers
    features = cfg.offresnet.features if cfg.offresnet.features else features
    use_skip = cfg.offresnet.use_skip if cfg.offresnet.use_skip is not None else use_skip
    use_dropout = cfg.network.use_dropout if cfg.network.use_dropout is not None else use_dropout
    use_BN = cfg.network.use_bn if cfg.network.use_bn is not None else use_BN
    seed = cfg.network.seed if cfg.network.seed is not None else seed

    kernel_params = {'init': kernel_init, 'size': kernel_size, 'seed': seed}

    logging.info("----- Layers {layers}, features {features}, kernel size {kernel_size}x{kernel_size}".
                 format(layers=layers, features=features, kernel_size=kernel_params['size']))

    convs = OrderedDict()
    activs = OrderedDict()
    concats = OrderedDict()
    skip_cons = OrderedDict()

    with tf.name_scope("preprocessing"):
        in_node = tf.reshape(x, tf.stack([-1, nx, ny, channels, 1]))

    # -------- Network architecture

    # first convolutional layer
    with tf.name_scope("input"):
        conv = conv3D(in_node, features, kernel_params, name="conv-in")
        drop = dropout(conv, keep_prob, use_dropout, seed, name="drop-in")
        bn = batch_norm(drop, use_BN, training, name="bn-in")
        activ = activation(bn, AF, name="activ-in")

        in_node = activ
        skip_cons[0] = in_node
        convs["input"] = conv
        activs["input"] = activ

    # residual layers
    for block in range(1, blocks+1):

        block_convs = list()
        block_activs = list()

        name_scope = "res_block_{}".format(str(block))
        with tf.name_scope(name_scope):
            for layer in range(1, layers+1):
                conv = conv3D(in_node, features, kernel_params, name="conv{}".format(str(layer)))
                drop = dropout(conv, keep_prob, use_dropout, seed, name="drop{}".format(str(layer)))
                bn = batch_norm(drop, use_BN, training, name="bn{}".format(str(layer)))
                activ = activation(bn, AF, name="activ{}".format(str(layer)))

                in_node = activ
                block_convs.append(conv)
                block_activs.append(activ)

            if use_skip:
                in_node += skip_cons[block-1]
            skip_cons[block] = in_node
            concats[name_scope] = in_node
            convs[name_scope] = tuple(block_convs)
            activs[name_scope] = tuple(block_activs)


    # flatten data to go from 3D to 2D output
    kernel_params['size'] = 1
    with tf.name_scope("flat"):
        conv = conv3D(in_node, n_class, kernel_params, name="conv-flat")
        drop = dropout(conv, keep_prob, use_dropout, seed, name="drop-flat")
        bn = batch_norm(drop, use_BN, training, name="bn-flat")
        activ = activation(bn, AF, name="activ-flat")
        in_node = tf.reshape(activ, tf.stack([-1, nx, ny, channels]))

        activs["flat"] = activ
        convs["flat"] = conv

    # last convolutional layer
    with tf.name_scope("output"):
        conv = conv2D(in_node, n_class, kernel_params, name="conv-out")
        drop = dropout(conv, keep_prob, use_dropout, seed, name="drop-out")
        bn = batch_norm(drop, use_BN, training, name="bn-out")
        activ = activation(bn, output_AF, name="activ-out")

        convs["output"] = conv
        activs["output"] = activ
        output_map = activ

    # -------- Network architecture

    variables = tf.trainable_variables()
    layers = {'convs': convs, 'activs': activs, 'concats': concats, 'deconvs': {}}

    return output_map, variables, layers
