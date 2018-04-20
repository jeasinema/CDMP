#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : env.py
# Purpose :
# Creation Date : 09-04-2018
# Last Modified : 2018年04月20日 星期五 16时05分15秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from utils import *
from rbf import RBF


class Env(object):
    def __init__(self, config):
        self.cfg = config

    def sample(self):
        raise NotImplementedError

    def display(self):
        raise NotImplementedError


class ToyEnv(Env):
    def __init__(self, config):
        self.cfg = config

        self.center = ((-.6, .75),  (-.2, .75),  (.2, .75),  (.6, .75),
                       (-.6, 0.),                            (.6, 0.),
                       (-.6, -.75), (-.2, -.75), (.2, -.75), (.6, -.75))
        self.color = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (0., 0., 0.), (1., 1., 0.),
                      (0., 1., 1.), (1., 0., 1.), (1., .5, 0.), (.5, 0., 1.), (0., 1., .5))

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
        if self.cfg.totally_random:
            center = []
            p_i = list(
                range((self.cfg.image_size[0] // 10) * (self.cfg.image_size[1] // 10)))
            np.random.shuffle(p_i)
            p_i = p_i[3: 3 + self.cfg.number_of_tasks]
            for p in p_i:
                x = (p // (self.cfg.image_size[1] // 10)) * 10. + 5.
                y = (p % (self.cfg.image_size[1] // 10)) * 10. + 5.
                x = (x / self.cfg.image_size[0]) * (self.cfg.image_x_range[1] -
                                                    self.cfg.image_x_range[0]) + self.cfg.image_x_range[0]
                y = (y / self.cfg.image_size[1]) * (self.cfg.image_y_range[1] -
                                                    self.cfg.image_y_range[0]) + self.cfg.image_y_range[0]
                center.append(np.asarray((x, y), dtype=np.float32))
        else:
            center = self.center

        t = np.linspace(0, 1, self.cfg.number_time_samples, dtype=np.float32)
        tau_mean = np.vstack(
            [center[traj_id][0] * t, center[traj_id][1] * t ** .5]).T
        noise = np.random.normal(
            0., self.cfg.trajectory_variance) * np.sin(t * np.pi)
        noise_dir = np.asarray(
            (-(tau_mean[-1] - tau_mean[0])[1], (tau_mean[-1] - tau_mean[0])[0]), dtype=np.float32)
        noise_dir /= np.linalg.norm(noise_dir)
        tau = tau_mean + \
            noise_dir.reshape(1, 2) * noise.reshape(tau_mean.shape[0], 1)
        im = np.ones(self.cfg.image_size +
                     (self.cfg.image_channels,), np.float32)
        for i in range(self.cfg.number_of_tasks):
            x, y = self.__remap_data_to_image(*center[i])
            cv2.rectangle(im, (int(x - 5), int(y - 5)), (int(x + 5),
                                                         int(y + 5)), self.color[im_id[i]], cv2.FILLED)
            txsz, baseline = cv2.getTextSize(
                str(im_id[i]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, 1)
            pos = int(x - txsz[0] // 2), int(y + txsz[1] // 2)
            cv2.putText(
                im, str(im_id[i]), pos, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (.6, .6, .6), 1)
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
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return img


class YCBEnv(Env):
    def __init__(self, config):
        self.cfg = config

        self.t = np.linspace(0, 1, self.cfg.number_time_samples, dtype=np.float32)
        # self.backgrounds = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        #                     for p in sorted(glob(self.cfg.image_path+"/background/*.png", recursive=False))]
        self.backgrounds = [p for p in sorted(glob(self.cfg.image_path+"/background/*.png", recursive=False))]
        self.objects = {}
        for pth in sorted(glob(self.cfg.image_path+"/clipped_obj/*", recursive=False)):
            # img = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in sorted(glob(pth+"/*.png"))]
            # self.objects[pth.split('/')[-1]] = [cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.
            #                                     for im in img]
            img = [p for p in sorted(glob(pth+"/*.png"))]
            self.objects[pth.split('/')[-1]] = img

        self.center = ((-.27, .2),  (-.09, .2),  (.09, .2),  (.27, .2),
                       (-.27, 0.),                           (.27, 0.),
                       (-.27, -.2), (-.09, -.2), (.09, -.2), (.27, -.2))


    # draw obj(RGBA) into scene(RGB) and scene_mask(u8) at pos(x, y)
    # no return and ops directly on inputs
    def __mask_add_image(self, scene, obj, pos, scene_mask=None):
        im_sz = scene.shape[:2]
        im_xr = self.cfg.image_x_range
        im_yr = self.cfg.image_y_range
        x, y = (pos[0] - im_xr[0]) / (im_xr[1] - im_xr[0]) * im_sz[1],\
               (im_yr[1] - pos[1]) / (im_yr[1] - im_yr[0]) * im_sz[0]
        obj_sz = obj.shape[:2]
        rect = [0, 0, 0, 0]
        rect[0] = int(y - obj_sz[0] // 2)
        rect[1] = int(x - obj_sz[1] // 2)
        rect[2] = rect[0] + obj_sz[0]
        rect[3] = rect[1] + obj_sz[1]
        
        scene[rect[0]:rect[2], rect[1]: rect[3], :] *= (1. - obj[:, :, 3:])
        scene[rect[0]:rect[2], rect[1]: rect[3], :] += obj[:, :, :3]

        if scene_mask is not None:
            scene_mask[rect[0]:rect[2], rect[1]: rect[3], :] += obj[:, :, 3:]

    # task: a 0~n_task-1 value, or None for random one;
    # return tau, task_id, im
    def sample(self, objects=None, task_id=None):
        if task_id is not None and isinstance(task_id, str):
            for i, t in enumerate(self.objects.keys()):
                if task_id == t:
                    task_id = i
                    break
        else:
            task_id = np.random.randint(0, self.cfg.number_of_tasks)

        if objects is None:
            objects = tuple(range(len(self.objects.keys())))

        if self.cfg.totally_random:
            while True:
                ret = np.array(poisson_dics_samples(2700, 2700, 500))/2700 # critic
                if len(ret) >= len(self.center):
                    break
                ret = ret[np.random.shuffle(np.arange(len(ret)))[:len(ret)]]
                factor = 0.8
                ret[:, 0] *= (self.cfg.image_x_range[1]-self.cfg.image_x_range[0])*0.8  # critic
                ret[:, 1] *= (self.cfg.image_y_range[1]-self.cfg.image_y_range[0])*0.8  # critic
                offset_x = (self.cfg.image_x_range[1]-self.cfg.image_x_range[0])*0.8
                ret[:, 0] += self.cfg.image_x_range[0]
        else:
            centers = list(self.center).copy()
        
        np.random.shuffle(centers)
        back_id = np.random.randint(0, len(self.backgrounds))
        im = cv2.cvtColor(cv2.imread(self.backgrounds[back_id]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.

        for i in objects:
            obj_list = self.objects[list(self.objects.keys())[i]]
            object_id = np.random.randint(0, len(obj_list))
            object_im = cv2.cvtColor(cv2.imread(obj_list[object_id], cv2.IMREAD_UNCHANGED), 
                    cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.
            self.__mask_add_image(im, object_im, centers[i])
        im = cv2.resize(im, self.cfg.image_size)

        tau_mean = np.vstack([centers[task_id][0] * self.t,
                              (centers[task_id][1] - self.cfg.image_y_range[0])
                              * self.t ** .5 + self.cfg.image_y_range[0]]).T
        noise = np.random.normal(0., self.cfg.trajectory_variance) * np.sin(self.t * np.pi)
        noise_dir = np.asarray((-(tau_mean[-1] - tau_mean[0])[1], (tau_mean[-1] - tau_mean[0])[0]), dtype=np.float32)
        noise_dir /= np.linalg.norm(noise_dir)
        tau = tau_mean + noise_dir.reshape(1, 2) * noise.reshape(tau_mean.shape[0], 1)
        return tau, task_id, im


    # remap x[-1, 1], y[0, 1] to image coordinate
    def __remap_data_to_image(self, x, y):
        im_sz = self.cfg.image_size
        im_xr = self.cfg.image_x_range
        im_yr = self.cfg.image_y_range
        return (x - im_xr[0]) / (im_xr[1] - im_xr[0]) * im_sz[0], (im_yr[1] - y) / (im_yr[1] - im_yr[0]) * im_sz[1]


    def display(self, tau, im, c=None, name="", interactive=False):
        if interactive:
            plt.ion()

        plt.figure(num=name)
        plt.clf()

        if (isinstance(tau, np.ndarray) and len(tau.shape) == 3) or len(tau) == 1:
            if len(tau) == 1:
                im = im[0]
                tau = tau[0]
                c = c[0]

            fig = plt.figure()
            plt.imshow(im)
            plt.plot(*self.__remap_data_to_image(tau[:, 0], tau[:, 1]), "#00FF00")
            plt.xticks([])
            plt.yticks([])
            if c is not None:
                plt.title("Task_%d"%c)

        else:
            if len(tau) > 8:
                im = im[:8]
                tau = tau[:8]
                c = c[:8]
                print("Warning: more then 8 samples are provided, only first 8 will be displayed")

            n_batch = len(tau)
            if n_batch <= 3:
                fig, axarr = plt.subplots(n_batch, num=name)
            elif n_batch == 4:
                fig, axarr = plt.subplots(2, 2, num=name)
            elif n_batch <= 6:
                fig, axarr = plt.subplots(2, 3, num=name)
            else:
                fig, axarr = plt.subplots(2, 4, num=name)
            for w, i, t, f in zip(tau, im, c, range(n_batch)):
                if n_batch <= 3:
                    axarr[f].imshow(i)
                    axarr[f].plot(*self.__remap_data_to_image(w[:, 0], w[:, 1]), "#00FF00")
                    if t is not None:
                        axarr[f].set_title("Task_%d"%t)
                elif n_batch == 4:
                    axarr[f // 2, f % 2].imshow(i)
                    axarr[f // 2, f % 2].plot(*self.__remap_data_to_image(w[:, 0], w[:, 1]), "#00FF00")
                    if t is not None:
                        axarr[f // 2, f % 2].set_title("Task_%d"%t)
                elif n_batch <= 6:
                    axarr[f // 3, f % 3].set_yticklabels([])
                    axarr[f // 3, f % 3].set_xticklabels([])
                    axarr[f // 3, f % 3].imshow(i)
                    axarr[f // 3, f % 3].plot(*self.__remap_data_to_image(w[:, 0], w[:, 1]), "#00FF00")
                    if t is not None:
                        axarr[f // 3, f % 3].set_title("Task_%d"%t)
                else:
                    axarr[f // 4, f % 4].set_yticklabels([])
                    axarr[f // 4, f % 4].set_xticklabels([])
                    axarr[f // 4, f % 4].imshow(i)
                    axarr[f // 4, f % 4].plot(*self.__remap_data_to_image(w[:, 0], w[:, 1]), "#00FF00")
                    if t is not None:
                        axarr[f // 4, f % 4].set_title("Task_%d"%t)
        if interactive:
            plt.pause(0.01)
        else:
            plt.show()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return img



if __name__ == '__main__':
    pass
