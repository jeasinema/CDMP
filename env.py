#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : env.py
# Purpose :
# Creation Date : 09-04-2018
# Last Modified : Sat 14 Apr 2018 12:13:09 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from rbf import RBF


class Env(object):
    def __init__(self, config):
        self.cfg = config

        t = np.linspace(0, 1, self.cfg.number_time_samples, dtype=np.float32)
        self.center = ((-.6, .75),  (-.2, .75),  (.2, .75),  (.6, .75),
                       (-.6, 0.),                            (.6, 0.),
                       (-.6, -.75), (-.2, -.75), (.2, -.75), (.6, -.75))
        self.color = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (0., 0., 0.), (1., 1., 0.),
                      (0., 1., 1.), (1., 0., 1.), (1., .5, 0.), (.5, 0., 1.), (0., 1., .5))
        self.traj_mean = tuple(np.vstack([self.center[i][0] * t, self.center[i][1] * t ** .5]).T
                               for i in range(self.cfg.number_of_tasks))

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
            im_id = list(range(self.cfg.number_of_tasks))
            np.random.shuffle(im_id)
        traj_id = 0
        for i in range(self.cfg.number_of_tasks):
            if task_id == im_id[i]:
                traj_id = i
                break

        tau_mean = self.traj_mean[traj_id]
        noise = np.random.normal(0., self.cfg.trajectory_variance) * np.sin(np.linspace(0, 1, tau_mean.shape[0]) * np.pi)
        noise_dir = np.asarray((-(tau_mean[-1] - tau_mean[0])[1], (tau_mean[-1] - tau_mean[0])[0]), dtype=np.float32)
        noise_dir /= np.linalg.norm(noise_dir)
        tau = tau_mean + noise_dir.reshape(1, 2) * noise.reshape(tau_mean.shape[0], 1)
        im = np.ones(self.cfg.image_size+(self.cfg.image_channels,), np.float32)
        for i in range(self.cfg.number_of_tasks):
            x, y = self.__remap_data_to_image(*self.center[i])
            cv2.rectangle(im, (int(x - 5), int(y - 5)), (int(x + 5), int(y + 5)), self.color[im_id[i]], cv2.FILLED)
            txsz, baseline = cv2.getTextSize(str(im_id[i]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, 1)
            pos = int(x - txsz[0] // 2), int(y + txsz[1] // 2)
            cv2.putText(im, str(im_id[i]), pos, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (.6, .6, .6), 1)
        return tau, task_id, im

    def display(self, tau, im, c=None, interactive=False):
        if interactive:
            plt.close()
            plt.ion()

        if (isinstance(tau, np.ndarray) and len(tau.shape) == 3) or len(tau) == 1:
            if len(tau) == 1:
                im = im[0]
                tau = tau[0]
                c = c[0]

            fig = plt.figure()
            plt.imshow(im)
            plt.plot(*self.__remap_data_to_image(tau[:, 0], tau[:, 1]))
            plt.xticks([])
            plt.yticks([])
            if c is not None:
                plt.title("Task_%d" % c)

        else:
            if len(tau) > 8:
                im = im[:8]
                tau = tau[:8]
                c = c[:8]
                print(
                    "Warning: more then 8 samples are provided, only first 8 will be displayed")

            n_batch = len(tau)
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
                    axarr[f].plot(*self.__remap_data_to_image(w[:, 0], w[:, 1]))
                    if t is not None:
                        axarr[f].set_title("Task_%d" % t)
                elif n_batch == 4:
                    axarr[f // 2, f % 2].imshow(i)
                    axarr[f // 2, f % 2].plot(*self.__remap_data_to_image(w[:, 0], w[:, 1]))
                    if t is not None:
                        axarr[f // 2, f % 2].set_title("Task_%d" % t)
                elif n_batch <= 6:
                    axarr[f // 3, f % 3].set_yticklabels([])
                    axarr[f // 3, f % 3].set_xticklabels([])
                    axarr[f // 3, f % 3].imshow(i)
                    axarr[f // 3, f % 3].plot(*self.__remap_data_to_image(w[:, 0], w[:, 1]))
                    if t is not None:
                        axarr[f // 3, f % 3].set_title("Task_%d" % t)
                else:
                    axarr[f // 4, f % 4].set_yticklabels([])
                    axarr[f // 4, f % 4].set_xticklabels([])
                    axarr[f // 4, f % 4].imshow(i)
                    axarr[f // 4, f % 4].plot(*self.__remap_data_to_image(w[:, 0], w[:, 1]))
                    if t is not None:
                        axarr[f // 4, f % 4].set_title("Task_%d" % t)
        if interactive:
            plt.pause(0.01)
        else:
            plt.show()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return img

if __name__ == '__main__':
    pass
