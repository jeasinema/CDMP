#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : config.py
# Purpose :
# Creation Date : 09-04-2018
# Last Modified : Fri 20 Apr 2018 05:40:12 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

from env import *
from utils import *


class Config(object):
    def __init__(self):
        # task properties
        self.number_of_tasks = 10          # n_c
        self.trajectory_dimension = 2       # n_dim
        self.image_size = (224, 224)        # sz_im
        self.image_x_range = (-.45, .45)
        self.image_y_range = (-.4, .4)
        self.image_channels = 3             # ch_im
        self.number_of_hidden = 16          # n_z
        self.number_of_MP_kernels = 10      # n_k
        self.number_of_oversample = 30      # n_oversample
        self.trajectory_variance = 0.05
        # environment
        self.env = YCBEnv                      # class pointer
        self.image_path = "./data/cdmp_images"
        self.number_time_samples = 100      # n_t
        self.trajectory_variance = 0.05
        self.totally_random = False          # if True, target can be anywhere
        # training properties
        self.batch_size_train = 1         # n_batch
        self.batch_size_test = 6
        self.batches_train = 2
        self.epochs = 1500
        self.continue_training = True
        self.save_interval = 10             # -1 for saving best model
        self.display_interval = 1
        # program properties
        self.use_gpu = True
        self.multi_threads = 32
        self.log_path = "./assets/log"
        self.check_point_path = "./assets/learned_model"
        self.experiment_name = "YCB_Ten_Point_Reacher_256_16_spatialsoftmax_224_fast"
        self.gpu=3


if __name__ == '__main__':
    pass
