# ======================================================================================================================
# name        : loader.py
# type        : data manipulation classes
# purpose     : load, prepare and manipulate data
# author      : Olga Dergachyova
# last update : 10/2020
# ======================================================================================================================

from __future__ import print_function, division, absolute_import, unicode_literals

import glob
import numpy as np
import os
from scipy.io import loadmat
import logging


class BaseDataProvider(object):
    """
    Abstract base class for MatDataProvider implementation. Subclasses have to
    overwrite the `_next_data`, `_find_data_files` and `_load_file` methods
    that load the input and output data arrays.
    """

    def __init__(self, cfg, n_vis=1, valid=0.2, a_min=-np.inf, a_max=np.inf):
        """
        Initializes abstract BaseDataProvider class.

        :param cfg: configuration object containing data parameters
        :param train_data_path: a glob search pattern for finding all training input and output image matrices
        :param valid_data_path: a glob search pattern for finding all validation input and output image matrices
        :param visual_data_path: (optional) a glob search pattern for finding input and output image matrices to use for visual evaluation
        :param test_data_path: (optional) a glob search pattern for finding all test input and output image matrices
        :param n_vis: (optional) number of data samples to use for result visualization after each epoch
        :param visualize_during_training: flag for storing results on visualization data set apart
        :param valid: (optional) portion of train data to be used as validation set
        :param a_min: (optional) min value used for clipping
        :param a_max: (optional) max value used for clipping

        """

        # set variables
        self.train_data_path = cfg.train_data
        self.valid_data_path = cfg.valid_data
        self.visual_data_path = cfg.visual_data
        self.test_data_path = cfg.test_data
        self.n_vis = cfg.n_vis if cfg.n_vis else n_vis
        self.visualize_during_training = True if cfg.n_vis else False
        self.valid = cfg.valid if cfg.valid else valid
        self.a_min = cfg.a_min if cfg.a_min else a_min
        self.a_max = cfg.a_max if cfg.a_max else a_max
        self.train_file_idx = 0
        self.valid_file_idx = 0
        self.current_file_idx = -1

    def load_train_data(self):
        """
        Loads training and validation data (more exact their file names) according to configuration file
        """

        logging.info('Loading train data...')

        # load training data
        self.train_data_files = self._find_data_files(self.train_data_path)
        np.random.shuffle(self.train_data_files)
        assert len(self.train_data_files) > 0, "No training files"

        # check if separate path to validation data
        if self.valid_data_path:
            self.valid_data_files = self._find_data_files(self.valid_data_path)
            np.random.shuffle(self.valid_data_files)
            assert len(self.valid_data_files) > 0, "No validation files"

        else:
            # separate to train and validation sets
            valid_size = round(self.valid * len(self.train_data_files))
            self.valid_data_files = self.train_data_files[0:valid_size]
            self.train_data_files = self.train_data_files[valid_size:]

        logging.info("----- Number of files used for training: %s" % len(self.train_data_files))
        logging.info("----- Number of files used for validation: %s" % len(self.valid_data_files))

        # load visualisation data
        if self.visual_data_path:
            self.visual_data_files = self._find_data_files(self.visual_data_path, display=True)
            logging.info("----- Number of files used for visualization: %s" % len(self.visual_data_files))

        # select n_vis training samples for intermediate visualization during training
        self.train_vis_files = self.train_data_files[0:self.n_vis]
        self.valid_vis_files = self.valid_data_files[0:self.n_vis]

        # try to load data to check input shape and number of output classes
        data_path = self.train_data_files[0]
        data_input = self._load_file(data_path[0])
        data_output = self._load_file(data_path[1])

        # store input data shape
        self.nx = data_input.shape[0]
        self.ny = data_input.shape[1]
        self.channels = 1 if len(data_input.shape) == 2 else data_input.shape[-1]
        self.n_class = 1 if len(data_output.shape) == 2 else data_output.shape[-1]

        logging.info("----- Number of channels: %s"%self.channels)
        logging.info("----- Number of classes: %s"%self.n_class)

    def load_test_data(self):
        """
        Loads testing data (more exact file names) according to configuration file
        """

        self.test_data_files = self._find_data_files(self.test_data_path)

        assert len(self.test_data_files) > 0, "No test files"
        logging.info("----- Number of files used for tests: %s" % len(self.test_data_files))

    def get_train_batch(self, batch_size, idx=None):
        """
        Returns a batch of training data

        :param batch_size: batch size (number of input samples in a batch)
        :param idx: index of a particular input file to start from
        :returns X: input train data of current training batch
        :returns Y: labels of current training batch
        """

        # start from a particular file to be able to run through all data without repetitions
        if idx is not None:
            self.train_file_idx = idx

        # compute current batch size in case it's the last one and is smaller than others
        if (self.train_file_idx + batch_size) > len(self.train_data_files):
            batch_size = len(self.train_data_files) - self.train_file_idx

        # get input data and labels
        self.current_file_list = self.train_data_files[self.train_file_idx: self.train_file_idx+batch_size]
        X, Y = self._get_data()

        # update index
        self.train_file_idx += batch_size
        if self.train_file_idx == len(self.train_data_files):
            np.random.shuffle(self.train_data_files)
            self.train_file_idx = 0

        return X, Y

    def get_train_batch_names(self):
        """
        Returns names of current training batch files
        """
        return self.current_file_list

    def get_valid_batch(self, batch_size, idx=None):
        """
        Returns a batch of validation data

        :param batch_size: batch size (number of input samples in a batch)
        :param idx: index of a particular input file to start from
        :returns X: input train data of current validation batch
        :returns Y: labels of current validation batch
        """

        assert len(self.valid_data_files) >= batch_size, "Not enough samples to create validation batch"

        # start from a particular file to be able to run through all data without repetitions
        if idx is not None:
            self.valid_file_idx = idx

        # compute current batch size in case it's the last one and is smaller than others
        if (self.valid_file_idx + batch_size) > len(self.valid_data_files):
            batch_size = len(self.valid_data_files) - self.valid_file_idx

        # get input data and labels
        self.current_file_list = self.valid_data_files[self.valid_file_idx: self.valid_file_idx+batch_size]
        X, Y = self._get_data()

        # update index
        self.valid_file_idx += batch_size
        if self.valid_file_idx == len(self.valid_data_files):
            np.random.shuffle(self.valid_data_files)
            self.valid_file_idx = 0

        return X, Y

    def get_valid_batch_names(self):
        """
        Returns names of current validation batch files
        """
        return self.current_file_list

    def get_train_vis_data(self):
        """
        Returns training visualization data
        """
        self.current_file_list = self.train_vis_files
        return self._get_data()

    def get_valid_vis_data(self):
        """
        Returns validation visualization data
        """
        self.current_file_list = self.valid_vis_files
        return self._get_data()

    def get_valid_data(self):
        """
        Returns all validation data
        """
        self.current_file_list = self.valid_data_files
        return self._get_data()

    def get_visual_data(self):
        """
        Returns all visualization data
        """
        if self.visual_data_path:
            self.current_file_list = self.visual_data_files
            return self._get_data()
        else:
            return None, None

    def get_test_data(self):
        """
        Returns all testing data
        """
        self.current_file_list = self.test_data_files
        return self._get_data()

    def get_train_data_n(self):
        """
        Returns number of training data files
        """
        return len(self.train_data_files)

    def get_valid_data_n(self):
        """
        Returns number of validation data files
        """
        return len(self.valid_data_files)

    def _get_data(self):
        """
        Loads data from `current_file_list` and returns data content (X) and label (Y)
        """

        # load sample input data (X) and labels (Y) to extract data shape
        input, output = self._load_input_and_output()

        # get shape
        n = len(self.current_file_list)
        nx = input.shape[1]
        ny = input.shape[2]

        # create containers
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))

        # actually load data
        X[0] = input
        Y[0] = output
        for i in range(1, n):
            input, output = self._load_input_and_output()
            X[i] = input
            Y[i] = output

        self.current_file_idx = -1
        return X, Y

    def _load_input_and_output(self):
        """
        Loads data from `current_file_list`
        """

        # load
        input, output = self._next_data()

        # get shape
        nx = input.shape[0]
        ny = input.shape[1]

        # process if needed
        input = self._process_data(input)
        output = self._process_data(output)

        return np.reshape(input, (1, nx, ny, self.channels)), np.reshape(output, (1, nx, ny, self.n_class))

    def _process_data(self, data):
        """
        Pre-process input data and labels
        """

        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)

        if np.amax(data) != 0:
            data /= np.amax(data)

        return data

class MatDataProvider(BaseDataProvider):
    """
    Generic data provider for .mat files
    """

    def __init__(self, cfg, input_suffix='input.mat', output_suffix='output.mat', multi_input=False, multi_output=False, order='stacked'):
        """
        Initializes data provider for .mat files.

        :param cfg: configuration object containing data parameters
        :param input_suffix: suffix pattern for the input images.
        :param output_suffix: suffix pattern for the output images (labels).
        :param multi_input: indicates if all *input.mat should be combined in one data sample
        :param multi_output: indicates if all *output.mat should be combined in one data sample
        :param order: indicates if multidimensional data should be stacked (RIRIRI...) or grouped by dimension (RRR...III...). Options: stacked, grouped
        """

        super(MatDataProvider, self).__init__(cfg)

        # set variables
        self.input_suffix = cfg.input_suffix if cfg.input_suffix else input_suffix
        self.output_suffix = cfg.output_suffix if cfg.output_suffix else output_suffix
        self.multi_input = cfg.multi_input if cfg.multi_input is not None else multi_input
        self.multi_output = cfg.multi_output if cfg.multi_output is not None else multi_output
        self.order = cfg.order if cfg.order else order

    def _find_data_files(self, search_path, display=False):
        """
         Finds appropriate files corresponding to search criteria

         :param search_path: a glob search pattern for finding all input and output image matrices
         :param display: flag to display pre-fixes of found files
         :returns grouped_files: input-label data pairs grouped by name
        """

        logging.info('Searching data files in ' + search_path)

        # extract all files
        files = glob.glob(search_path + '.mat')

        # find file prefixes (the part before the last '_') and sort them
        prefixes = ['_'.join(os.path.basename(file).split('_')[:-1]) + '_' for file in files]
        prefixes = sort_prefixes(list(set(prefixes)))

        # group files
        grouped_files = []
        if self.multi_input and self.multi_output:
            for prefix in prefixes:
                prefix_files = glob.glob(search_path[:-1] + prefix + '*')
                prefix_input_files = sorted([file for file in prefix_files if self.input_suffix in file])
                prefix_output_files = sorted([file for file in prefix_files if self.output_suffix in file])
                grouped_files.append((prefix_input_files, prefix_output_files))
        elif self.multi_input and not self.multi_output:
            for prefix in prefixes:
                prefix_files = glob.glob(search_path[:-1] + prefix + '*')
                prefix_input_files = sorted([file for file in prefix_files if self.input_suffix in file])
                prefix_output_file = search_path[:-1] + prefix + self.output_suffix
                grouped_files.append((prefix_input_files, prefix_output_file))
        elif not self.multi_input and not self.multi_output:
            input_files = sorted([file for file in files if self.input_suffix in file])
            grouped_files = [(file, file.replace(self.input_suffix, self.output_suffix)) for file in input_files]

        if display:
            logging.info('Prefixes:')
            for p in prefixes:
                logging.info(p)

        return grouped_files

    def _load_file(self, path, dtype=np.float32):
        """
         Loads files content form .mat file as data array

         :param path: path to input file
         :param dtype: data type
         :returns data array
        """

        if type(path) is list:
            data = [np.array(loadmat(p)['data'], dtype) for p in path]
            data = np.dstack(data)
        else:
            data = loadmat(path)['data']

        # TODO: check if executed correctly and make n-channel
        if self.order == 'grouped':
            n = data.shape[-1]
            data = [data[:, :, 0:2:n]] + [data[:, :, 1:2:n]]
            data = np.dstack(data)

        return np.array(data, dtype)

    def _next_data(self):
        """
         Switches to current file and loads it
        """

        # get current file name
        self.current_file_idx += 1
        input_files = self.current_file_list[self.current_file_idx][0]
        output_files = self.current_file_list[self.current_file_idx][1]

        # load
        input = self._load_file(input_files, np.float32)
        output = self._load_file(output_files, np.float32)

        return input, output

    def _process_data(self, data):
        """
        Place here any eventual pre-processing for this specific data type
        """
        # no processing needed
        return data


def sort_prefixes(prefixes):
    """
    Sorts prefixes

     :param prefixes: list of prefixes
     :returns sorted_prefixes: list of sorted prefixes
    """

    # find max number of '_'
    n = max([prefix.count('_') for prefix in prefixes])

    # sort prefixes by each sub prefix
    sorted_prefixes = list()
    for i in range(n):
        try:
            prefixes = sorted(prefixes, key=lambda x: int(x.split('_')[i]))
        except:
            prefixes = sorted(prefixes, key=lambda x: x.split('_')[i])
        sorted_prefixes += [prefix for prefix in prefixes if prefix.count('_') == i+1]
        prefixes = [prefix for prefix in prefixes if prefix.count('_') > i+1]

    return sorted_prefixes

