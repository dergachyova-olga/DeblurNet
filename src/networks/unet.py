# ======================================================================================================================
# name        : unet.py
# type        : network architecture
# purpose     : define Unet network architecture
# author      : Olga Dergachyova
# last update : 10/2020
# ======================================================================================================================

import tensorflow as tf
from collections import OrderedDict
import logging
from convnet.layers import (conv, deconv, max_pool, concat, activation, dropout, batch_norm)


def create_unet(cfg, data_provider, x, training, keep_prob, type='2D', AF='relu', output_AF='linear', kernel_size=3, kernel_init='he',
                  layers=3, features_root=16, pool_size=2, upscale_method='transpose', use_skip=True, use_dropout=False, use_BN=False, seed=None):
    """
    Creates a new Unet for the given parametrization.

    :param cfg: configuration object containing data parameters
    :param data_provider: object containing and manipulation data
    :param x: input tensor, shape [?,nx,ny,channels]
    :param training: defines if the run is for training or tests
    :param keep_prob: dropout probability tensor indicating the part of neurons to keep
    :param type: convolution dimensionality ('2D' or '3D')
    :param AF: activation function on intermediate conv layers
    :param output_AF: activation function on last conv layers
    :param kernel_size: (optional) size of the convolution kernel
    :param kernel_init: (optional) initializer for weights
    :param layers: (optional) number of layers in the net
    :param features_root: (optional) number of features in the first layer
    :param pool_size: (optional) size of the max pooling operation
    :param upscale_method: (optional) method used in up side of the net to increase input size.
           Options: transpose, nn, biliniar, bicubic. If not transpose, interpolation + conv are used.
    :param use_skip: (optional) flag if feature concatenation skip connections should be used
    :param use_dropout: (optional) flag if dropout layers should be used
    :param use_BN: (optional) flag if batch normalization layers should be used
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
    layers = cfg.unet.layers if cfg.unet.layers else layers
    features_root = cfg.unet.features_root if cfg.unet.features_root else features_root
    pool_size = cfg.unet.pool_size if cfg.unet.pool_size else pool_size
    upscale_method = cfg.unet.upscale_method if cfg.unet.upscale_method else upscale_method
    use_skip = cfg.unet.use_skip if cfg.unet.use_skip is not None else use_skip
    use_dropout = cfg.network.use_dropout if cfg.network.use_dropout is not None else use_dropout
    use_BN = cfg.network.use_bn if cfg.network.use_bn is not None else use_BN
    seed = cfg.network.seed if cfg.network.seed is not None else seed

    kernel_params = {'init': kernel_init, 'size': kernel_size, 'seed': seed}

    logging.info("----- Network has {layers} layers, {features} features, "
                 "{kernel_size}x{kernel_size} kernel size, and {pool_size}x{pool_size} pool size".
                 format(layers=layers, features=features_root, kernel_size=kernel_params['size'], pool_size=pool_size))

    # Placeholder for the input image
    with tf.name_scope("preprocessing"):
        # nx = tf.shape(x)[1]
        # ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image

    convs = OrderedDict()
    deconvs = OrderedDict()
    pools = OrderedDict()
    activs = OrderedDict()
    concats = OrderedDict()
    skip_cons = OrderedDict()
    lid = 1  # layer id

    # -------- Network architecture
    # down layers
    for layer in range(0, layers):
        name_scope = "{0:d}_down{1}".format(lid, str(layer))
        lid += 1
        with tf.name_scope(name_scope):
            features = 2 ** layer * features_root

            conv1 = conv(type, in_node, features, kernel_params, name="conv1")
            drop1 = dropout(conv1, keep_prob, use_dropout, seed, name="drop1")
            bn1 = batch_norm(drop1, use_BN, training, name="bn1")
            activ1 = activation(bn1, AF, name="activ1")

            conv2 = conv(type, activ1, features, kernel_params, name="conv2")
            drop2 = dropout(conv2, keep_prob, use_dropout, seed, name="drop2")
            bn2 = batch_norm(drop2, use_BN, training, name="bn2")
            activ2 = activation(bn2, AF, name="activ2")

            in_node = activ2  # in_node - layer passed as input for next iteration
            skip_cons[layer] = activ2

            if layer < layers - 1:
                pools[name_scope] = max_pool('2D', in_node, pool_size)
                in_node = pools[name_scope]

            convs[name_scope] = (conv1, conv2)
            activs[name_scope] = (activ1, activ2)

    # up layers
    for layer in range(layers - 2, -1, -1):
        name_scope = "{0:d}_up{1}".format(lid, str(layer))
        lid += 1
        with tf.name_scope(name_scope):
            features = (2 ** (layer + 1) * features_root) // 2

            deconv0 = deconv(type, in_node, features, kernel_params, pool_size, upscale_method, name="deconv")
            activ0 = activation(deconv0, AF, name="activ-deconv")
            concat0 = concat(type, skip_cons[layer], activ0) if use_skip else activ0

            conv1 = conv(type, concat0, features, kernel_params, name="conv1")
            drop1 = dropout(conv1, keep_prob, use_dropout, seed, name="drop1")
            bn1 = batch_norm(drop1, use_BN, training, name="bn1")
            activ1 = activation(bn1, AF, name="activ1")

            conv2 = conv(type, activ1, features, kernel_params, name="conv2")
            drop2 = dropout(conv2, keep_prob, use_dropout, seed, name="drop2")
            bn2 = batch_norm(drop2, use_BN, training, name="bn2")
            activ2 = activation(bn2, AF, name="activ2")

            in_node = activ2
            deconvs[name_scope] = deconv
            concats[name_scope] = concat
            convs[name_scope] = (conv1, conv2)
            activs[name_scope] = (activ0, activ1, activ2)

    if type == '3D':
        # flatten data to go from 3D to 2D output
        kernel_params['size'] = 1
        name_scope = "{}_flat".format(str(lid))
        with tf.name_scope(name_scope):
            conv_flat = conv(type, in_node, n_class, kernel_params, name="conv")
            bn_flat = batch_norm(conv_flat, use_BN, training, name="bn")
            activ_flat = activation(bn_flat, AF, name="activ")
            in_node = tf.reshape(activ_flat, tf.stack([-1, nx, ny, channels]))

            convs[name_scope] = conv_flat
            activs[name_scope] = activ_flat

    # output map
    kernel_params['size'] = 1
    name_scope = "{}_output".format(str(lid))
    with tf.name_scope(name_scope):
        conv_output = conv('2D', in_node, n_class, kernel_params, name="conv")
        bn_output = batch_norm(conv_output, use_BN, training, name="bn")
        activ_output = activation(bn_output, output_AF, name="activ")

        convs[name_scope] = conv_output
        activs[name_scope] = activ_output

    # -------- Network architecture

    variables = tf.trainable_variables()
    layers = {'convs': convs, 'deconvs': deconvs, 'pools': pools, 'activs': activs, 'concats': concats}

    return activ_output, variables, layers