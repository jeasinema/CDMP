#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : model.py
# Purpose :
# Creation Date : 09-04-2018
# Last Modified : 2018年04月10日 星期二 02时09分03秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import torch
import numpy as np


class NN_pw_zimc(torch.nn.Module):
    def __init__(self, sz_image, ch_image, n_z, tasks, dim_w, n_k):
        super(NN_pw_zimc, self).__init__()
        self.sz_image = sz_image
        self.ch = ch_image
        self.tasks = tasks
        self.n_z = n_z
        self.dim_w = dim_w
        self.n_k = n_k

        # for image input
        self.conv1 = torch.nn.Conv2d(self.ch, 16, kernel_size=3, padding=1)
        self.pool1 = torch.nn.MaxPool2d(3)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool3 = torch.nn.MaxPool2d(2)
        self.fc_img1 = torch.nn.Linear(
            32 * (sz_image[0] // 3 // 2 // 2) * (sz_image[1] // 3 // 2 // 2), 64)

        # for z input
        self.fc_z1 = torch.nn.Linear(self.n_z, 64)
        self.fc_z2 = torch.nn.Linear(64, 64)

        # for c input
        self.fc_c1 = torch.nn.Linear(self.tasks, 64)
        self.fc_c2 = torch.nn.Linear(64, 64)

        # merge
        self.fc1 = torch.nn.Linear(3 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.mean = torch.nn.Linear(64, self.dim_w * self.n_k)
        self.logvar = torch.nn.Linear(64, self.dim_w * self.n_k)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    # input z(n_batch, n_z), im(n_batch, ch, h, w), c(n_batch, tasks);
    # output mean(n_batch, k_w, dim_w), logvar(n_batch, k_w, dim_w)
    def forward(self, z, im, c):
        n_batch = z.size(0)

        # image
        im_x = self.relu(self.conv1(im))
        im_x = self.pool1(im_x)
        im_x = self.relu(self.conv2(im_x))
        im_x = self.pool2(im_x)
        im_x = self.relu(self.conv3(im_x))
        im_x = self.pool3(im_x).view(n_batch, -1)
        im_x = self.fc_img1(im_x)

        # tasks
        c_x = self.relu(self.fc_c1(c))
        c_x = self.fc_c2(c_x)

        # conditions
        z_x = self.relu(self.fc_z1(z))
        z_x = self.fc_z2(z_x)

        # merge
        x = self.relu(self.fc1(torch.cat((im_x, c_x, z_x), 1)))
        x = self.relu(self.fc2(x))
        mean = self.mean(x).view(n_batch, self.n_k, self.dim_w)
        logvar = self.logvar(x).view(n_batch, self.n_k, self.dim_w)
        return mean, logvar

    # input z(n_batch, n_z), im(n_batch, ch, h, w), c(n_batch, tasks);
    # output mean(n_samples, n_batch, n_k, dim_w), logvar(n_samples, n_batch, n_k, dim_w)
    def sample(self, z, im, c, samples=None, fixed=True):
        mean, logvar = self.forward(z, im, c)
        dist = torch.distributions.Normal(mean, torch.exp(logvar))
        if samples is None:
            if fixed:
                return mean
            else:
                return dist.sample()
        else:
            return dist.sample_n(samples)

    # input w(n_batch, k_w, dim_w) z(n_batch, n_z), im(n_batch, ch, h, w), c(n_batch, tasks);
    # output (n_batch,)
    # or if z is batch of samples, output E(n_batch)
    def log_prob(self, w, z, im, c):
        if len(z.size()) == len(c.size()) + 1:
            m = z.size(0)
            mean, logvar = self.forward(
                z.view(-1, z.size(-1)), im.repeat(m, 1, 1, 1), c.repeat(m, 1))
            dist = torch.distributions.Normal(mean, torch.exp(logvar))
            return dist.log_prob(w.repeat(m, 1, 1)).view(m, *w.size()).sum(0).sum(-1).sum(-1) / m
        else:
            mean, logvar = self.forward(z, im, c)
            dist = torch.distributions.Normal(mean, torch.exp(logvar))
            return dist.log_prob(w).sum(-1).sum(-1)

    # input w(n_batch, k_w, dim_w) z(n_batch, n_z), im(n_batch, ch, h, w), c(n_batch, tasks);
    # output (n_batch,)
    # or if z is batch of samples, output E(n_batch)
    def mse_error(self, w, z, im, c):
        if len(z.size()) == len(c.size()) + 1:
            m = z.size(0)
            mean, _ = self.forward(
                z.view(-1, z.size(-1)), im.repeat(m, 1, 1, 1), c.repeat(m, 1))
            return ((w.repeat(m, 1, 1) - mean)**2).view(m, *w.size()).sum(0).sum(-1).sum(-1) / m
        else:
            mean, logvar = self.forward(z, im, c)
            dist = torch.distributions.Normal(mean, torch.exp(logvar))
            return ((w - mean)**2).sum(-1).sum(-1)


class NN_qz_w(torch.nn.Module):
    def __init__(self, sz_image, ch_image, n_z, tasks, dim_w, n_k):
        super(NN_qz_w, self).__init__()
        self.sz_image = sz_image
        self.ch = ch_image
        self.tasks = tasks
        self.n_z = n_z
        self.dim_w = dim_w
        self.n_k = n_k

        # for image input
        self.conv1 = torch.nn.Conv2d(self.ch, 16, kernel_size=3, padding=1)
        self.pool1 = torch.nn.MaxPool2d(3)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool3 = torch.nn.MaxPool2d(2)
        self.fc_img1 = torch.nn.Linear(
            32 * (sz_image[0] // 3 // 2 // 2) * (sz_image[1] // 3 // 2 // 2), 64)

        # for c input
        self.fc_c1 = torch.nn.Linear(self.tasks, 64)
        self.fc_c2 = torch.nn.Linear(64, 64)

        # for w input
        self.fc_w1 = torch.nn.Linear(self.dim_w * self.n_k, 64)
        self.fc_w2 = torch.nn.Linear(64, 64)

        # merge
        self.fc1 = torch.nn.Linear(3 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.mean = torch.nn.Linear(64, self.n_z)
        self.logvar = torch.nn.Linear(64, self.n_z)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    # input w(n_batch, k_w, dim_w), im(n_batch, ch, h, w), c(n_batch, tasks);
    # output mean(n_batch, n_z), logvar(n_batch, n_z)
    def forward(self, w, im, c):
        n_batch = w.size(0)

        # image
        im_x = self.relu(self.conv1(im))
        im_x = self.pool1(im_x)
        im_x = self.relu(self.conv2(im_x))
        im_x = self.pool2(im_x)
        im_x = self.relu(self.conv3(im_x))
        im_x = self.pool3(im_x).view(n_batch, -1)
        im_x = self.fc_img1(im_x)

        # tasks
        c_x = self.relu(self.fc_c1(c))
        c_x = self.fc_c2(c_x)

        # w
        w_x = self.relu(self.fc_w1(w.view(n_batch, -1)))
        w_x = self.relu(self.fc_w2(w_x))

        # merge
        x = self.relu(self.fc1(torch.cat((im_x, c_x, w_x), 1)))
        x = self.relu(self.fc2(x))
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar

    # if samples is None, then return shape (n_batch, n_z)
    # else return shape (n_samples, n_batch, n_z)
    def sample(self, w, im, c, samples=None, reparameterization=False):
        mean, logvar = self.forward(w, im, c)
        if reparameterization:
            dist = torch.distributions.Normal(
                torch.zeros_like(mean), torch.ones_like(logvar))
            if samples is None:
                s = dist.sample().detach()
                return mean + torch.exp(logvar) * s
            else:
                s = dist.sample_n(samples).detach()
                return mean.unsqueeze(0) + torch.exp(logvar).unsqueeze(0) * s
        else:
            dist = torch.distributions.Normal(mean, torch.exp(logvar))
            if samples is None:
                sample = dist.sample().detach()
                return sample
            else:
                sample = dist.sample_n(samples).detach()
                return sample

    # return (n_batch,)
    def Dkl(self, w, im, c):
        def norm_Dkl(mean1, logvar1, mean2, logvar2):
            # p(x)~N(u1, v1), q(x)~N(u2, v2)
            # Dkl(p||q) = 0.5 * (log(|v2|/|v1|) - d + tr(v2^-1 * v1) + (u1 - u2)' * v2^-1 * (u1 - u2))
            # for diagonal v, Dkl(p||q) = 0.5*(sum(log(v2[i])-log(v1[i])+v1[i]/v2[i]+(u1[i]-u2[i])**2/v2[i]-1))
            return 0.5 * ((logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2)**2.) / torch.exp(logvar2) - 1)\
                .sum(-1).sum(-1)

        mean, logvar = self.forward(w, im, c)
        if next(self.fc1.parameters()).is_cuda:
            mean_t, logvar_t = torch.zeros_like(mean).cuda(
            ).detach(), torch.zeros_like(logvar).cuda().detach()
        else:
            mean_t, logvar_t = torch.zeros_like(
                mean).detach(), torch.zeros_like(logvar).detach()

        return norm_Dkl(mean, logvar, mean_t, logvar_t)


if __name__ == '__main__':
    pass
