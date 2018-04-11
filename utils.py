#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : utils.py
# Purpose :
# Creation Date : 09-04-2018
# Last Modified : Wed 11 Apr 2018 05:25:35 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]


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


# generator: (traj, task, image) x batch_size
def batch_train(config):
    env = config.env(config)
    while True:
        yield tuple(env.sample() for _ in range(config.batch_size_train))


# generator: (traj, task, image) x batch_size
def batch_test(config):
    env = config.env(config)
    while True:
        yield tuple(env.sample() for _ in range(config.batch_size_test))


if __name__ == '__main__':
    pass
