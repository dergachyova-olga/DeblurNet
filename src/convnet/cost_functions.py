# ======================================================================================================================
# name        : cost_functions.py
# type        : functions
# purpose     : define cost (loss) functions that can be minimized during training
# author      : Olga Dergachyova
# last update : 10/2020
# ======================================================================================================================

import tensorflow as tf
import numpy as np


def MSE_cost(labels, predictions):
    return tf.losses.mean_squared_error(labels, predictions)


def MAE_cost(labels, predictions):
    return tf.losses.absolute_difference(labels, predictions)


def MSEw2_cost(x, labels, predictions):
    input = tf.stack([x[:, :, :, -1]], axis=3)  # take last input
    intensity_change = tf.subtract(1.0, tf.squared_difference(input, labels))
    error = tf.squared_difference(labels, predictions)
    loss = tf.reduce_mean(tf.multiply(error, intensity_change))
    return loss


def MSEw_cost(x, labels, predictions):
    input = tf.stack([x[:, :, :, -1]], axis=3)  # take last input
    intensity_change = tf.subtract(1.0, tf.math.abs(input - labels))
    error = tf.squared_difference(labels, predictions)
    loss = tf.reduce_mean(tf.multiply(error, intensity_change))
    return loss


def MAEw2_cost(x, labels, predictions):
    input = tf.stack([x[:, :, :, -1]], axis=3)  # take last input
    intensity_change = tf.subtract(1.0, tf.squared_difference(input, labels))
    error = tf.losses.absolute_difference(labels, predictions)
    loss = tf.reduce_mean(tf.multiply(error, intensity_change))
    return loss


def MAEw_cost(x, labels, predictions):
    input = tf.stack([x[:, :, :, -1]], axis=3)  # take last input
    intensity_change = tf.subtract(1.0, tf.losses.absolute_difference(input, labels))
    error = tf.losses.absolute_difference(labels, predictions)
    loss = tf.reduce_mean(tf.multiply(error, intensity_change))
    return loss


def MSE_Energy_cost(labels, predictions, lambda_value=0.0):
    MSE_loss = tf.losses.mean_squared_error(labels, predictions)

    if lambda_value is not None and lambda_value != 0.0:
        # ------  Energy loss
        norm_labels = tf.sqrt(tf.reduce_sum(tf.square(labels), axis=[1, 2, 3]))
        norm_predictions = tf.sqrt(tf.reduce_sum(tf.square(predictions), axis=[1, 2, 3]))
        Energy_loss = tf.losses.mean_squared_error(norm_labels, norm_predictions)

        # ------  Total cost
        cost = (1 - lambda_value) * MSE_loss + lambda_value * Energy_loss
        return cost

    else:
        return MSE_loss


def MSE_Hermitian_cost(labels, predictions, lambda_value=0.0):
    MSE_loss = tf.losses.mean_squared_error(labels, predictions)

    if lambda_value is not None and lambda_value != 0.0:
        # ------  Hermitian symmetry loss
        [_, h, w, _] = predictions.get_shape().as_list()
        h = int(h / 2)
        w = int(w / 2)

        # extract center halfs and border lines
        center_left = predictions[:, 1:, 1:w + 1, :]
        center_right = predictions[:, 1:, w:, :]

        vert_top = predictions[:, 1:h + 1, 0, :]
        vert_bottom = predictions[:, h:, 0, :]

        horiz_left = predictions[:, 0, 1:w + 1, :]
        horiz_right = predictions[:, 0, w:, :]

        # mirror and conjugate
        center_right = tf.image.flip_up_down(tf.image.flip_left_right(center_right)) * [1, -1]
        vert_bottom = tf.reverse(vert_bottom, [1]) * [1, -1]
        horiz_right = tf.reverse(horiz_right, [1]) * [1, -1]

        # compute sum for every region
        center_tf = tf.reduce_sum(tf.abs(tf.subtract(center_left, center_right)), axis=(1, 2))
        vert_tf = tf.reduce_sum(tf.abs(tf.subtract(vert_top, vert_bottom)), axis=1)
        horiz_tf = tf.reduce_sum(tf.abs(tf.subtract(horiz_left, horiz_right)), axis=1)

        # compute mean of Hermitian symmetry score across all the regions
        Hermitian_score = (center_tf + vert_tf + horiz_tf) / ((2 * h - 1) * w + w + h)

        # compute loss for all samples
        Hermitian_loss = tf.reduce_mean(tf.reduce_sum(Hermitian_score, axis=1))

        # ------  Total cost
        cost = (1 - lambda_value) * MSE_loss + lambda_value * Hermitian_loss
        return cost

    else:
        return MSE_loss


def SSIM_cost(labels, predictions):
    ssim = tf.image.ssim(predictions, labels, max_val=1.0)
    return (1.0 - tf.reduce_mean(ssim)) / 2.0


def cross_entropy_cost(flat_labels, flat_logits, class_weights):
    if class_weights is not None:
        class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

        weight_map = tf.multiply(flat_labels, class_weights)
        weight_map = tf.reduce_sum(weight_map, axis=1)

        loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                              labels=flat_labels)
        weighted_loss = tf.multiply(loss_map, weight_map)

        loss = tf.reduce_mean(weighted_loss)

    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                         labels=flat_labels))
    return loss