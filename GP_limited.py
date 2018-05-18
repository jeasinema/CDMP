#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : GP_limited.py
# Purpose :
# Creation Date : 14-05-2018
# Last Modified : Mon 14 May 2018 01:12:10 AM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import os
import argparse

from config import Config

parser = argparse.ArgumentParser(description='cnn_dmp')
parser.add_argument('--train-data-path', type=str, nargs='?', default='', help='load train data')
parser.add_argument('--test-data-path', type=str, nargs='?', default='', help='load test data')

args = parser.parse_args()

cfg = Config() 

def main():
    train_data = pickle.load(open(args.train_data_path, 'wb'))
    test_data = pickle.load(open(args.test_data_path, 'wb'))
    print('Load data done.')
    gp = GaussianProcessRegressor()
    # only support t as one-hot
     



if __name__ == '__main__':
    main()	
