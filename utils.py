#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : utils.py
# Purpose :
# Creation Date : 09-04-2018
# Last Modified : Fri 20 Apr 2018 10:48:40 AM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np


def bar(current, total, prefix="", suffix="", bar_sz=25, end_string=None):
    sp = ""
    print("\x1b[2K\r", end='')
    for i in range(bar_sz):
        if current * bar_sz // total > i:
            sp += '='
        elif current * bar_sz // total == i:
            sp += '>'
        else:
            sp += ' '
    if current == total:
        if end_string is None:
            print("\r%s[%s]%s" % (prefix, sp, suffix))
        else:
            if end_string != "":
                print("\r%s" % end_string)
            else:
                print("\r", end='')
    else:
        print("\r%s[%s]%s" % (prefix, sp, suffix), end='')


class EnvDataset(data.Dataset):
    def __init__(self, config, train=True):
        self.cfg = config
        self.train = train
        self.env = self.cfg.env(self.cfg)

    def __getitem__(self, index):
        return self.env.sample()

    def __len__(self):
        return self.cfg.epochs*self.cfg.batches_train*self.cfg.batch_size_train*10


def collate_fn_env(batch):
    ret = []
    for sample in batch:
        ret.append(tuple(sample))
    return ret


# generator: (traj, task, image) x batch_size
def build_loader(config, train=True):
    return torch.utils.data.DataLoader(
        EnvDataset(
            config=config,
            train=train
        ),
        batch_size=config.batch_size_train if train else config.batch_size_test,
        num_workers=config.multi_threads,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn_env
    )


if __name__ == '__main__':
    pass
