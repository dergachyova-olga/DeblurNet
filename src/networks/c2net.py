# ======================================================================================================================
# name        : c2net.py
# type        : network architecture
# purpose     : define c2net network architecture
# author      : Olga Dergachyova
# last update : 10/2020
# ======================================================================================================================

import tensorflow as tf
from collections import OrderedDict
import logging
from convnet.layers import (conv, deconv, max_pool, concat, activation, dropout, batch_norm)


def create_c2net(cfg, data_provider, x, training, keep_prob, type='2D', AF='relu', output_AF='linear', kernel_size=3, kernel_init='he',
                n_layers=3, n_features_root=16, pool_size=2, upscale_method='transpose', use_dropout=False, use_BN=False, seed=None):
    """
    Creates a new convolutional c2net for the given parametrization.

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
    :param n_layers: (optional) number of layers in the net
    :param n_features_root: (optional) number of features in the first layer
    :param pool_size: (optional) size of the max pooling operation
    :param upscale_method: (optional) method used in up side of the net to increase input size.
           Options: transpose, nn, biliniar, bicubic. If not transpose, interpolation + conv are used.
    :param use_dropout: (optional) flag if dropout layers should be used
    :param use_BN: (optional) flag if batch normalization layers should be used
    :param seed: (optional) seed for kernel initialization and dropout
    """

    # Set variables
    nx = data_provider.nx
    ny = data_provider.ny
    n_channels = int(data_provider.n_channels / 2)
    n_class = int(data_provider.n_class / 2)

    type = cfg.network.conv if cfg.network.conv else type
    AF = cfg.network.af if cfg.network.af else AF
    output_AF = cfg.network.output_af if cfg.network.output_af else output_AF
    kernel_size = cfg.network.kernel_size if cfg.network.kernel_size else kernel_size
    kernel_init = cfg.network.kernel_init if cfg.network.kernel_init else kernel_init
    n_layers = cfg.unet.layers if cfg.unet.layers else n_layers
    n_features_root = cfg.unet.features_root if cfg.unet.features_root else n_features_root
    pool_size = cfg.unet.pool_size if cfg.unet.pool_size else pool_size
    upscale_method = cfg.unet.upscale_method if cfg.unet.upscale_method else upscale_method
    use_dropout = cfg.network.use_dropout if cfg.network.use_dropout is not None else use_dropout
    use_BN = cfg.network.use_bn if cfg.network.use_bn is not None else use_BN
    seed = cfg.network.seed if cfg.network.seed is not None else seed

    kernel_params = {'init': kernel_init, 'size': kernel_size, 'seed': seed}

    # Create dicts to store references to different layers
    layers_ref = {'convs': OrderedDict(), 'deconvs': OrderedDict(), 'activs': OrderedDict(),
                  'pools': OrderedDict(), 'concats': OrderedDict(), 'skip_cons': OrderedDict()}

    # Separate c1 from c2
    with tf.name_scope("preprocessing"):
        c1_x = tf.reshape(x[:, :, :, 0:n_channels], tf.stack([-1, nx, ny, n_channels]))
        c2_x = tf.reshape(x[:, :, :, n_channels:2*n_channels], tf.stack([-1, nx, ny, n_channels]))

    # Create separate networks for c1 and c2
    c1_output, c1_containers = create_branch(type, 'c1', c1_x, nx, ny, n_channels, n_class, n_layers, n_features_root, kernel_params, pool_size, AF, output_AF,
                                             upscale_method, training, keep_prob, use_dropout, use_BN, seed, layers_ref)

    c2_output, c2_containers = create_branch(type, 'c2', c2_x, nx, ny, n_channels, n_class, n_layers, n_features_root, kernel_params, pool_size, AF, output_AF,
                                             upscale_method, training, keep_prob, use_dropout, use_BN, seed, layers_ref)

    # Concatenate c1 and c2 networks
    output = tf.concat([c1_output, c2_output], 3)

    # Concatenate layers content in dicts
    layers_ref = OrderedDict(c1_containers.items() + c2_containers.items())

    # Get all trainable variables
    variables = tf.trainable_variables()

    return output, variables, layers_ref


# -------- Network architecture
def create_branch(type, cmp, in_node, nx, ny, n_channels, n_class, n_layers, n_features_root, kernel_params, pool_size, AF, output_AF,
                  upscale_method, training, keep_prob, use_dropout, use_BN, seed, layers_ref):

    lid = 1  # layer id

    # down layers
    for layer in range(0, n_layers):
        name_scope = "{0}_{0:d}_down{1}".format(cmp, lid, str(layer))
        lid += 1
        with tf.name_scope(name_scope):
            features = 2 ** layer * n_features_root

            conv1 = conv(type, in_node, features, kernel_params, name=cmp+"_conv1")
            drop1 = dropout(conv1, keep_prob, use_dropout, seed, name=cmp+"_drop1")
            bn1 = batch_norm(drop1, use_BN, training, name=cmp+"bn1")
            activ1 = activation(bn1, AF, name=cmp+"activ1")

            conv2 = conv(type, activ1, features, kernel_params, name=cmp+"conv2")
            drop2 = dropout(conv2, keep_prob, use_dropout, seed, name=cmp+"drop2")
            bn2 = batch_norm(drop2, use_BN, training, name=cmp+"bn2")
            activ2 = activation(bn2, AF, name=cmp+"activ2")

            in_node = activ2  # in_node - layer passed as input for next iteration
            layers_ref['skip_cons'][layer] = activ2

            if layer < n_layers - 1:
                layers_ref['pools'][name_scope] = max_pool(type, in_node, pool_size)
                in_node = layers_ref['pools'][name_scope]

            layers_ref['convs'][name_scope] = (conv1, conv2)
            layers_ref['activs'][name_scope] = (activ1, activ2)

    # up layers
    for layer in range(n_layers - 2, -1, -1):
        name_scope = "{0}_{0:d}_up{1}".format(cmp, lid, str(layer))
        lid += 1
        with tf.name_scope(name_scope):
            features = (2 ** (layer + 1) * n_features_root) // 2

            deconv0 = deconv(type, in_node, features, kernel_params, pool_size, upscale_method, name=cmp+"deconv")
            activ0 = activation(deconv0, AF, name=cmp+"activ-deconv")
            concat0 = concat(type, layers_ref['skip_cons'][layer], activ0)

            conv1 = conv(type, concat0, features, kernel_params, name=cmp+"conv1")
            drop1 = dropout(conv1, keep_prob, use_dropout, seed, name=cmp+"drop1")
            bn1 = batch_norm(drop1, use_BN, training, name=cmp+"bn1")
            activ1 = activation(bn1, AF, name=cmp+"activ1")

            conv2 = conv(type, activ1, features, kernel_params, name=cmp+"conv2")
            drop2 = dropout(conv2, keep_prob, use_dropout, seed, name=cmp+"drop2")
            bn2 = batch_norm(drop2, use_BN, training, name=cmp+"bn2")
            activ2 = activation(bn2, AF, name=cmp+"activ2")

            in_node = activ2
            layers_ref['deconvs'][name_scope] = deconv
            layers_ref['concats'][name_scope] = concat
            layers_ref['convs'][name_scope] = (conv1, conv2)
            layers_ref['activs'][name_scope] = (activ0, activ1, activ2)

    if type == '3D':
        # flatten data to go from 3D to 2D output
        kernel_params['size'] = 1
        name_scope = "{}_{}_flat".format(cmp, str(lid))
        with tf.name_scope(name_scope):
            conv_flat = conv(in_node, n_class, kernel_params, name=cmp+"conv")
            bn_flat = batch_norm(conv_flat , use_BN, training, name=cmp+"bn")
            activ_flat = activation(bn_flat, AF, name=cmp+"activ")
            in_node = tf.reshape(activ_flat, tf.stack([-1, nx, ny, n_channels]))

            layers_ref['convs'][name_scope] = conv_flat
            layers_ref['activs'][name_scope] = activ_flat

    # output map
    kernel_params['size'] = 1
    name_scope = "{}_{}_output".format(cmp, lid)
    with tf.name_scope(name_scope):
        conv_output = conv(in_node, n_class, kernel_params, name=cmp+"conv")
        bn_output = batch_norm(conv_output, use_BN, training, name=cmp+"bn")
        activ_output = activation(bn_output, output_AF, name=cmp+"activ")

        layers_ref['convs'][name_scope] = conv_output
        layers_ref['activs'][name_scope] = activ_output

        output_map = activ_output

    return output_map, layers_ref