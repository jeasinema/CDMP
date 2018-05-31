#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : rbf.py
# Purpose :
# Creation Date : 09-04-2018
# Last Modified : 2018年04月09日 星期一 22时11分52秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import numpy as np


class RBF(object):
    # from (n_batch, t_samples, dim_in) or (t_samples, dim_in) to average kernel (n_k, dim_k)
    @staticmethod
    def calculate(traj, n_k):
        if len(traj.shape) != 3:
            if len(traj.shape) == 2:
                traj = np.expand_dims(traj, axis=0)
            else:
                print("What ?!")
                exit(-1)

        dim_out = traj.shape[-1]
        N = traj.shape[0]
        nt = traj.shape[1]
        c = np.linspace(0, 1, n_k, dtype=np.float32)
        w = np.zeros((n_k, dim_out), dtype=np.float32)

        t = np.linspace(0, 1, nt, dtype=np.float32)
        # wc: N x K; w: K x D;
        wc = np.exp(-((np.tile(t[:, np.newaxis], (1, n_k)
                               ) - np.tile(c, (nt, 1))) * 10.) ** 2)
        wc /= wc.sum(1, keepdims=True)

        for _traj in traj:
            w += np.linalg.solve(wc.T @ wc, wc.T @ _traj) / N

        return w

    # from (n_k, dim_out) to (t_samples, dim_out)
    @staticmethod
    def generate(w, t_samples):
        n_k = w.shape[0]
        c = np.linspace(0, 1, n_k, dtype=np.float32)

        t = np.linspace(0, 1, t_samples, dtype=np.float32)
        # wc: N x K; w: K x D;
        wc = np.exp(-((np.tile(t[:, np.newaxis], (1, n_k)) -
                       np.tile(c, (t_samples, 1))) * 10.) ** 2)
        wc /= wc.sum(1, keepdims=True)

        traj = wc @ w
        return traj


if __name__ == '__main__':
    pass
