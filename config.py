#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : config.py
# Purpose :
# Creation Date : 09-04-2018
# Last Modified : 2018年04月09日 星期一 22时09分30秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

from env import Env
from utils import *


class Config(object):
    def __init__(self):
        # task properties
        self.number_of_tasks = 4            # n_c
        self.trajectory_dimension = 2       # n_dim
        self.image_size = (100, 100)        # sz_im
        self.image_x_range = (-1., 1.)
        self.image_y_range = (0., 1.)
        self.image_channels = 3             # ch_im
        self.number_of_hidden = 4           # n_z
        self.number_of_MP_kernels = 10      # n_k
        self.number_time_samples = 100      # n_t
        self.number_of_oversample = 10      # n_oversample
        # data loader
        self.generator_train = batch_train  # function pointer
        self.generator_test = batch_test    # function pointer
        self.env = Env                      # class pointer
        # training properties
        self.batch_size_train = 20          # n_batch
        self.batch_size_test = 6
        self.batches_train = 100
        self.epochs = 100
        self.continue_training = True
        self.save_interval = 10             # -1 for saving best model
        self.display_interval = 5
        # program properties
        self.use_gpu = True
        self.multi_threads = 4
        self.log_path = "./logs"
        self.check_point_path = "./logs/checkpoints"
        self.experiment_name = "Four_Point_Reacher"


if __name__ == '__main__':
    pass
