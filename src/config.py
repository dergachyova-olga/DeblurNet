# ======================================================================================================================
# name        : config.py
# type        : configuration script
# purpose     : read configuration file and create an object containing all configuration parameters
# author      : Olga Dergachyova
# last update : 10/2020
# ======================================================================================================================

import configparser
from collections import namedtuple
from collections import OrderedDict
import ast
import logging
import os
import random
import numpy as np
import tensorflow as tf

# location of the default configuration file
default_config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'configs', 'config.ini')
# default location for a log
default_log_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'log.txt')


# main function called by the main script to read provided config file and create a config object
def load_config(config_path):

    # load configuration file
    config = configparser.ConfigParser()
    if config_path:
        config.read(config_path)
    else:
        config.read(default_config_path)

    # convert config sections and options into objects
    undefined = list()
    config_sections = OrderedDict()
    for section in config.sections():

        # create dict of section options
        section_options = OrderedDict()
        for option in config.options(section):
            if config.get(section, option) == '':
                undefined.append(section + '.' + option)
                section_options[option] = None
            else:
                section_options[option] = typed(config.get(section, option))

        # put section dict to config dict
        section_struct = namedtuple(section, ' '.join([option for option in config.options(section)]))
        config_sections[section] = section_struct(**section_options)

    # create config object
    config_struct = namedtuple('config', ' '.join([key for key in config_sections.keys()]))
    config = config_struct(**config_sections)

    # set up seed
    set_global_seed(config)

    # set up logging
    set_logger(config)

    # print undefined options in config file
    print_undefined(undefined)

    return config


# function to transform a string value from configuration file to an actual appropriate type
def typed(value):
    try:
        value = ast.literal_eval(value)
        return value
    except:
        return value


# function to setup logging and mute internal tensorflow messages
def set_logger(cfg):

    # mute internal tensorflow messages
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # setup logging
    if cfg.general.log_to_file:
        log_file = cfg.general.log_path if cfg.general.log_path else default_log_path
        if os.path.exists(log_file):
            os.remove(log_file)
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# function to indicate if some parameters are missing
def print_undefined(undefined):
    for i in undefined:
        logging.info('Configuration file: undefined value for ' + i)


# function to setup randomness seed
def set_global_seed(cfg):
    # set python and numpy seed
    os.environ['PYTHONHASHSEED'] = str(cfg.general.seed)
    random.seed(cfg.general.seed)
    np.random.seed(cfg.general.seed)
