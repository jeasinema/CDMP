#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : env.py
# Purpose :
# Creation Date : 09-04-2018
# Last Modified : 2018年04月09日 星期一 22时14分27秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import *


class Env(object):
    def __init__(self, config):
        self.cfg = config

        self.t = np.linspace(
            0, 1, self.cfg.number_time_samples, dtype=np.float32)
        self.center = ((-0.75, 0.75), (-0.3, 0.75), (0.3, 0.75), (0.75, 0.75))
        self.color = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (0., 0., 0.))
        self.traj_mean = (np.vstack([self.center[0][0] * self.t, self.center[0][1] * self.t ** .5]).T,
                          np.vstack([self.center[1][0] * self.t,
                                     self.center[1][1] * self.t ** .5]).T,
                          np.vstack([self.center[2][0] * self.t,
                                     self.center[2][1] * self.t ** .5]).T,
                          np.vstack([self.center[3][0] * self.t, self.center[3][1] * self.t ** .5]).T)

    # remap x[-1, 1], y[0, 1] to image coordinate
    def __remap_data_to_image(self, x, y):
        im_sz = self.cfg.image_size
        im_xr = self.cfg.image_x_range
        im_yr = self.cfg.image_y_range
        return (x - im_xr[0]) / (im_xr[1] - im_xr[0]) * im_sz[0], (im_yr[1] - y) / (im_yr[1] - im_yr[0]) * im_sz[1]

    # task: a 0~n_task-1 value, or None for random one; im: tuple of 4 color index(0~3), or None for random
    # return tau, task_id, im
    def sample(self, task_id=None, im_id=None):
        if task_id is None:
            task_id = np.random.randint(0, self.cfg.number_of_tasks)
        if im_id is None:
            im_id = list(range(4))
            np.random.shuffle(im_id)
        traj_id = 0
        for i in range(self.cfg.number_of_tasks):
            if task_id == im_id[i]:
                traj_id = i
                break

        tau = self.traj_mean[traj_id]
        tau += np.random.normal(0., 0.025, tau.shape) * \
            np.expand_dims(np.sin(self.t * np.pi), 1)
        im = np.ones(self.cfg.image_size +
                     (self.cfg.image_channels,), np.float32)
        for i in range(self.cfg.number_of_tasks):
            x, y = self.__remap_data_to_image(*self.center[i])
            cv2.rectangle(im, (int(x - 2), int(y - 2)), (int(x + 2),
                                                         int(y + 2)), self.color[im_id[i]], cv2.FILLED)
        return tau, task_id, im

    def display(self, tau, im, c=None, interactive=False):
        if interactive:
            plt.close()
            plt.ion()

        if (isinstance(im, np.ndarray) and len(im.shape) == 3) or len(im) == 1:
            if len(im) == 1:
                im = im[0]
                tau = tau[0]
                c = c[0]

            plt.imshow(im)
            plt.plot(*self.__remap_data_to_image(tau[:, 0], tau[:, 1]))
            plt.xticks([])
            plt.yticks([])
            if c is not None:
                plt.title("Task_%d" % c)

        else:
            if len(im) > 8:
                im = im[:8]
                tau = tau[:8]
                c = c[:8]
                print(
                    "Warning: more then 8 samples are provided, only first 8 will be displayed")

            n_batch = len(im)
            if n_batch <= 3:
                fig, axarr = plt.subplots(n_batch)
            elif n_batch == 4:
                fig, axarr = plt.subplots(2, 2)
            elif n_batch <= 6:
                fig, axarr = plt.subplots(2, 3)
            else:
                fig, axarr = plt.subplots(2, 4)
            for w, i, t, f in zip(tau, im, c, range(n_batch)):
                if n_batch <= 3:
                    axarr[f].imshow(i)
                    axarr[f].plot(
                        *self.__remap_data_to_image(w[:, 0], w[:, 1]))
                    if t is not None:
                        axarr[f].set_title("Task_%d" % t)
                elif n_batch == 4:
                    axarr[f // 2, f % 2].imshow(i)
                    axarr[f // 2, f %
                          2].plot(*self.__remap_data_to_image(w[:, 0], w[:, 1]))
                    if t is not None:
                        axarr[f // 2, f % 2].set_title("Task_%d" % t)
                elif n_batch <= 6:
                    axarr[f // 3, f % 3].set_yticklabels([])
                    axarr[f // 3, f % 3].set_xticklabels([])
                    axarr[f // 3, f % 3].imshow(i)
                    axarr[f // 3, f %
                          3].plot(*self.__remap_data_to_image(w[:, 0], w[:, 1]))
                    if t is not None:
                        axarr[f // 3, f % 3].set_title("Task_%d" % t)
                else:
                    axarr[f // 4, f % 4].set_yticklabels([])
                    axarr[f // 4, f % 4].set_xticklabels([])
                    axarr[f // 4, f % 4].imshow(i)
                    axarr[f // 4, f %
                          4].plot(*self.__remap_data_to_image(w[:, 0], w[:, 1]))
                    if t is not None:
                        axarr[f // 4, f % 4].set_title("Task_%d" % t)
        if interactive:
            plt.pause(0.01)
        else:
            plt.show()


if __name__ == '__main__':
    pass
