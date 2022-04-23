# ======================================================================================================================
# name        : layers.py
# type        : functions
# purpose     : define possible layers and building bricks of a convolutional network
# author      : Olga Dergachyova
# last update : 10/2020
# ======================================================================================================================

import tensorflow as tf
import numpy as np
import logging


# -------------- GENERIC ------------- #
def conv(type, x, features, kernel_params, strides=1, use_bias=True, name=None):
    if type == '2D':
        return conv2D(x, features, kernel_params, strides, use_bias, name if name else 'conv2D')
    elif type == '3D':
        return conv3D(x, features, kernel_params, strides, use_bias, name if name else 'conv3D')


def deconv(type, x, features, kernel_params, stride, upscale_method, name=None):
    if type == '2D':
        return deconv2D(x, features, kernel_params, stride, upscale_method, name if name else 'deconv2D')
    elif type == '3D':
        return deconv3D(x, features, kernel_params, stride, upscale_method, name if name else 'deconv3D')


def max_pool(type, x, pool_size, name=None):
    if type == '2D':
        return max_pool2D(x, pool_size, name=name if name else 'maxpool2D')
    elif type == '3D':
        return max_pool3D(x, pool_size, name=name if name else 'maxpool3D')


def concat(type, x1, x2):
    if type == '2D':
        return concat2D(x1, x2)
    elif type == '3D':
        return concat3D(x1, x2)


def kernel_init(kernel_params, features):
    if kernel_params['init'] == 'same':
        return tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=kernel_params['seed'])
    elif kernel_params['init'] == 'original':
        stddev = np.sqrt(2 / (kernel_params['size'] ** 2 * features))
        return tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=stddev, seed=kernel_params['seed'])
    elif kernel_params['init'] == 'glorot':
        return tf.keras.initializers.glorot_normal(seed=kernel_params['seed'])
    elif kernel_params['init'] == 'he':
        return tf.keras.initializers.he_normal(seed=kernel_params['seed'])
    elif kernel_params['init'] == 'ortho':
        return tf.keras.initializers.Orthogonal(gain=1.0, seed=kernel_params['seed'])


def bias_init():
    return tf.keras.initializers.Constant(value=0.1)


def activation(x, type, name="activ"):
    if type == 'relu':
        return tf.nn.relu(x, name=name)
    elif type == 'leaky_relu':
        return tf.nn.leaky_relu(x, alpha=0.2, name=name)
    elif type == 'linear':
        return tf.keras.activations.linear(x)
    elif type == 'tanh':
        return tf.nn.tanh(x, name=name)
    elif type == 'sigmoid':
        return tf.keras.activations.sigmoid(x)
    elif type == 'norm':
        [_, h, w, ch] = x.get_shape().as_list()
        s = tf.reduce_sum(x, axis=[1, 2, 3])
        s = tf.reshape(s, [-1, 1, 1, 1])
        s = tf.tile(s, [1, h, w, ch])
        return x / s


def dropout(x, keep_prob, apply=True, seed=None, name="dropout"):
    if apply:
        return tf.nn.dropout(x, rate=1 - keep_prob, seed=seed, name=name)
    else:
        return x


def batch_norm(x, apply=True, training=False, axis=-1, momentum=0.99, epsilon=0.001, name="bn"):
    if apply:
        return tf.layers.batch_normalization(x, training=training, axis=axis, momentum=momentum, epsilon=epsilon)
    else:
        return x


# -------------- 2D ------------- #
def conv2D(x, features, kernel_params, strides, use_bias, name):
    return tf.keras.layers.Conv2D(features, kernel_size=kernel_params['size'],  strides=strides, padding='SAME', use_bias=use_bias,
                                  kernel_initializer=kernel_init(kernel_params, features), bias_initializer=bias_init(), name=name)(x)


def deconv2D(x, features, kernel_params, strides, upscale_method, name='deconv2D'):
    if upscale_method == 'transpose':
        return tf.keras.layers.Conv2DTranspose(features, kernel_size=kernel_params['size'], strides=strides, padding='SAME',
                                               kernel_initializer=kernel_init(kernel_params, features), bias_initializer=bias_init(), name=name)(x)
    else:
        new_size = x.get_shape().as_list()[1]*2
        if upscale_method == 'nn':
            x = tf.image.resize_images(x, [new_size, new_size], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        elif upscale_method == 'bilinear':
            x = tf.image.resize_images(x, [new_size, new_size], tf.image.ResizeMethod.BILINEAR)
        elif upscale_method == 'bicubic':
            x = tf.image.resize_images(x, [new_size, new_size], tf.image.ResizeMethod.BICUBIC)
        return tf.keras.layers.Conv2D(features, kernel_size=kernel_params['size'], strides=1, padding='SAME',
                                      kernel_initializer=kernel_init(kernel_params, features), bias_initializer=bias_init(), name=name)(x)


def max_pool2D(x, pool_size, name='maxpool2D'):
    return tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_size, padding='VALID', name=name)(x)


def concat2D(x1, x2):
    return tf.concat([x1, x2], 3)


# -------------- 3D ------------- #
def conv3D(x, features, kernel_params, strides, use_bias, name):
    return tf.keras.layers.Conv3D(features, kernel_size=kernel_params['size'], strides=strides, padding='SAME', use_bias=use_bias,
                                  kernel_initializer=kernel_init(kernel_params, features), bias_initializer=bias_init(), name=name)(x)


def deconv3D(x, features, kernel_params, strides, upscale_method, name):
    if upscale_method != 'transpose':
        logging.info('Warning: interpolation upscaling is not implemented for 3D. Transposed convolution used.')

    return tf.keras.layers.Conv3DTranspose(features, kernel_size=kernel_params['size'], strides=strides, padding='SAME',
                                           kernel_initializer=kernel_init(kernel_params, features), bias_initializer=bias_init(), name=name)(x)


def max_pool3D(x, pool_size, name='maxpool3D'):
    return tf.keras.layers.MaxPool3D(pool_size=pool_size, strides=pool_size, padding='VALID', name=name)(x)


def concat3D(x1, x2):
    return tf.concat([x1, x2], 4)
