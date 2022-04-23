# ======================================================================================================================
# name        : utils.py
# type        : utility functions
# purpose     : provide a set of useful functions to process and transform data
# author      : Olga Dergachyova
# last update : 10/2020
# ======================================================================================================================

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import numpy as np
from PIL import Image
from scipy.io import savemat
import matplotlib.pyplot as plt


def plot_prediction(x_test, y_test, prediction, save=False):
    """
    Plots three images side by side: input, ground truth, and predicted output.

    :param x_test: input test data
    :param y_test: ground truth test data (label)
    :param prediction: prediction provided by the model
    :param save: flag to save the image
    """

    test_size = x_test.shape[0]
    fig, ax = plt.subplots(test_size, 3, figsize=(12,12), sharey=True, sharex=True)

    ax = np.atleast_2d(ax)
    for i in range(test_size):
        cax = ax[i, 0].imshow(x_test[i])
        plt.colorbar(cax, ax=ax[i,0])
        cax = ax[i, 1].imshow(y_test[i, ..., 1])
        plt.colorbar(cax, ax=ax[i,1])
        pred = prediction[i, ..., 1]
        pred -= np.amin(pred)
        pred /= np.amax(pred)
        cax = ax[i, 2].imshow(pred)
        plt.colorbar(cax, ax=ax[i,2])
        if i==0:
            ax[i, 0].set_title("x")
            ax[i, 1].set_title("y")
            ax[i, 2].set_title("pred")
    fig.tight_layout()

    if save:
        fig.savefig(save)
    else:
        fig.show()
        plt.show()


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    :param img: the array to convert [nx, ny, channels]
    :returns img: the rgb image [nx, ny, 3]
    """

    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    if np.amax(img) != 0:
        img /= np.amax(img)

    img *= 255
    return img


def combine_img_prediction(data, gt, pred):
    """
    Combines the data, grouth thruth and the prediction into one rgb image

    :param data: the data tensor
    :param gt: the ground thruth tensor
    :param pred: the prediction tensor
    :returns img: the concatenated rgb image
    """

    ny = pred.shape[2]
    ch = data.shape[3]
    cl = pred.shape[3]

    data_ch = np.split(data, ch, axis=3)
    images = [to_rgb(c.reshape(-1, ny, 1)) for c in data_ch]

    if cl > 1:
        # In case multiple output classes
        pred_cl = np.split(pred, cl, axis=3)
        gt_cl = np.split(gt, cl, axis=3)

        images.extend([to_rgb(c.reshape(-1, ny, 1)) for c in gt_cl])
        images.extend([to_rgb(c.reshape(-1, ny, 1)) for c in pred_cl])
    else:
        images.extend((to_rgb((gt[..., 0]).reshape(-1, ny, 1)), to_rgb(pred[..., 0].reshape(-1, ny, 1))))

    img = np.concatenate(images, axis=1)
    return img


def save_image(img, path):
    """
    Writes the image to disk

    :param img: the rgb image to save
    :param path: the target path
    """

    im = Image.fromarray(img.astype(np.uint8))
    im.save(path + '.png', 'PNG', dpi=[600, 600], quality=100)


def save_mat(mat, path):
    """
    Writes the mat to disk

    :param mat: mat to save
    :param path: the target path
    """
    savemat(path, {"data": mat})


def create_training_path(output_path, prefix="run_"):
    """
    Enumerates a new path using the prefix under the given output_path

    :param output_path: the root path
    :param prefix: (optional) defaults to `run_`
    :return: the generated path as string in form `output_path`/`prefix_` + `<number>`
    """

    idx = 0
    path = os.path.join(output_path, "{:}{:03d}".format(prefix, idx))
    while os.path.exists(path):
        idx += 1
        path = os.path.join(output_path, "{:}{:03d}".format(prefix, idx))
    return path


def plot_result_image(rows, cols, x, y, prediction, path):
    """
    Plots and saves series of input, ground truth, and predicted output results in one image

    :param rows: number of rows in image (number of data samples)
    :param cols: number of columns in image (depends on input and output dimensions)
    :param x: input data
    :param y: ground truth data (label)
    :param prediction: prediction provided by the model
    :param path: path to save created image
    """

    plt.rcParams.update({'font.size': 10})
    if rows > 1:
        fig, ax = plt.subplots(rows, cols + 2, sharex=True, sharey=True, figsize=(17, 2.5*rows))
        for i in range(rows):
            for j in range(cols):
                ax[i, j].imshow(x[i, ..., j])
                ax[0, j].set_title("Input " + str(j + 1))
            ax[i, cols].imshow(y[i, ..., 0])
            ax[0, cols].set_title("Ground truth")
            ax[i, cols + 1].imshow(prediction[i, ..., 0])
            ax[0, cols + 1].set_title("Prediction")
            for j in range(cols+2):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)
    else:
        fig, ax = plt.subplots(1, cols + 2, sharex=True, sharey=True, figsize=(17, 2.5))
        for i in range(cols):
            ax[i].imshow(x[0, ..., i])
            ax[i].set_title("Input " + str(i + 1))
        ax[cols].imshow(y[0, ..., 0])
        ax[cols].set_title("Ground truth")
        ax[cols + 1].imshow(prediction[0, ..., 0])
        ax[cols + 1].set_title("Prediction")
        for i in range(cols + 2):
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


def get_n_params(variables):
    """
    Returns number of trainable parameters in a network

    :param variables: entity holding network variables
    :returns total_parameters: number of trainable parameters in a network
    """
    total_parameters = 0
    for variable in variables:
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

