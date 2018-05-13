#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : prepare_data.py
# Purpose :
# Creation Date : 13-05-2018
# Last Modified : Sun 13 May 2018 08:44:15 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import torch
import numpy as np
import argparse
import pickle
import os
import time

from config import Config
from utils import bar, build_loader


parser = argparse.ArgumentParser(description='prepare data')
parser.add_argument('--amount', type=int, nargs='?', default=1000, help='amount of data')
parser.add_argument('--tag', type=str, nargs='?', default='default', help='tag of data')
args = parser.parse_args()


def main():
    cfg = Config() 
    cfg.batches_train = args.amount
    cfg.batch_size_train = 1

    generator_train = build_loader(cfg, True)  # function pointer
    t0 = time.time()
    batches = []
    for i, batch in enumerate(generator_train):
        batches.append(batch[0])
        print(i)
    
    pickle.dump(batches, open(os.path.join('./assets/data/{}_{}.pkl'.format(args.tag, args.amount)), 'wb'), 
            protocol=pickle.HIGHEST_PROTOCOL)
    print('done, time cost: {}s'.format(time.time()-t0))

if __name__ == '__main__':
    main()	
