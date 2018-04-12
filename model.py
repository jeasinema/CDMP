#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : model.py
# Purpose :
# Creation Date : 09-04-2018
# Last Modified : Thu 12 Apr 2018 10:26:04 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import torch
import numpy as np


class NN_img_c(object):
    class Model(torch.nn.Module):
        def __init__(self, sz_image, ch_image, tasks):
            super(NN_img_c.Model, self).__init__()
            self.sz_iamge = sz_image
            self.ch = ch_image
            self.tasks = tasks

            # for image input
            self.conv1 = torch.nn.Conv2d(self.ch, 64, kernel_size=3, padding=1)
            self.bn1 = torch.nn.BatchNorm2d(64)
            self.pool1 = torch.nn.MaxPool2d(3)
            self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn2 = torch.nn.BatchNorm2d(128)
            self.pool2 = torch.nn.MaxPool2d(2)
            self.conv3 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn3 = torch.nn.BatchNorm2d(128)
            self.pool3 = torch.nn.MaxPool2d(2)
            self.fc_img1 = torch.nn.Linear(
                128 * (sz_image[0] // 3 // 2 // 2) * (sz_image[1] // 3 // 2 // 2), 512)

            # for c input
            self.fc_c1 = torch.nn.Linear(self.tasks, 128)
            self.drop_c = torch.nn.Dropout()
            self.fc_c2 = torch.nn.Linear(128, 512)

            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, im, c):
            n_batch = c.size(0)

            # image
            im_x = self.relu(self.bn1(self.conv1(im)))
            im_x = self.pool1(im_x)
            im_x = self.relu(self.bn2(self.conv2(im_x)))
            im_x = self.pool2(im_x)
            im_x = self.relu(self.bn3(self.conv3(im_x)))
            im_x = self.pool3(im_x)
            feature_map = im_x
            im_x = self.fc_img1(im_x.view(n_batch, -1))

            # tasks
            c_x = self.relu(self.fc_c1(c))
            c_x = self.drop_c(c_x)
            c_x = self.fc_c2(c_x)

            return torch.cat((im_x, c_x), 1), feature_map

    def __init__(self, *args, **kwargs):
        self.model = self.Model(*args, **kwargs)

    def feature_map(self, im, c):
        _, feature_map = self.model(im, c) 
        return feature_map


class NN_pw_zimc(object):
    class Model(torch.nn.Module):
        def __init__(self, sz_image, ch_image, n_z, tasks, dim_w, n_k):
            super(NN_pw_zimc.Model, self).__init__()
            self.sz_image = sz_image
            self.ch = ch_image
            self.tasks = tasks
            self.n_z = n_z
            self.dim_w = dim_w
            self.n_k = n_k

            # for z input
            self.fc_z1 = torch.nn.Linear(self.n_z, 128)
            self.drop_z = torch.nn.Dropout()
            self.fc_z2 = torch.nn.Linear(128, 512)

            # merge
            self.fc1 = torch.nn.Linear(3 * 512, 512)
            self.drop1 = torch.nn.Dropout()
            self.fc2 = torch.nn.Linear(512, 512)
            self.drop2 = torch.nn.Dropout()
            self.mean = torch.nn.Linear(512, self.dim_w * self.n_k)
            self.logvar = torch.nn.Linear(512, self.dim_w * self.n_k)

            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()

        # input z(n_batch, n_z), im_c(n_batch, channel)
        # output mean(n_batch, k_w, dim_w), logvar(n_batch, k_w, dim_w)
        def forward(self, z, im_c):
            n_batch = z.size(0)

            # conditions
            z_x = self.relu(self.fc_z1(z))
            z_x = self.drop_z(z_x)
            z_x = self.fc_z2(z_x)

            # merge
            x = self.relu(self.fc1(torch.cat((im_c, z_x), 1)))
            x = self.drop1(x)
            x = self.relu(self.fc2(x))
            x = self.drop2(x)
            mean = self.mean(x).view(n_batch, self.n_k, self.dim_w)
            logvar = self.logvar(x).view(n_batch, self.n_k, self.dim_w)
            return mean, logvar

    def __init__(self, *args, **kwargs):
        self.model = self.Model(*args, **kwargs)

    # input z(n_batch, n_z), im_c(n_batch, channel)
    # output mean(n_samples, n_batch, n_k, dim_w), logvar(n_samples, n_batch, n_k, dim_w)
    def sample(self, z, im_c, samples=None, fixed=True):
        mean, logvar = self.model.forward(z, im_c)
        dist = torch.distributions.Normal(mean, torch.exp(logvar))
        if samples is None:
            if fixed:
                return mean
            else:
                return dist.sample()
        else:
            return dist.sample_n(samples)

    # input w(n_batch, k_w, dim_w) z(n_batch, n_z), im_c(n_batch, channel)
    # output (n_batch,)
    # or if z is batch of samples, output E(n_batch)
    def log_prob(self, w, z, im_c):
        if len(z.size()) == len(c.size()) + 1:
            m = z.size(0)
            mean, logvar = self.model.forward(
                z.view(-1, z.size(-1)), im_c.repeat(m, 1))
            dist = torch.distributions.Normal(mean, torch.exp(logvar))
            return dist.log_prob(w.repeat(m, 1, 1)).view(m, *w.size()).sum(-1).sum(-1).mean()
        else:
            mean, logvar = self.model.forward(z, im_c)
            dist = torch.distributions.Normal(mean, torch.exp(logvar))
            return dist.log_prob(w).sum(-1).sum(-1)

    # input w(n_batch, k_w, dim_w) z(n_batch, n_z), im_c(n_batch, channel)
    # output (n_batch,)
    # or if z is batch of samples, output E(n_batch)
    def mse_error(self, w, z, im_c):
        if len(z.size()) == len(im_c.size()) + 1:
            m = z.size(0)
            mean, _ = self.model.forward(
                z.view(-1, z.size(-1)), im_c.repeat(m, 1))
            return ((w.repeat(m, 1, 1) - mean)**2).view(m, *w.size()).sum(-1).sum(-1).mean()
        else:
            mean, logvar = self.model.forward(z, im_c)
            dist = torch.distributions.Normal(mean, torch.exp(logvar))
            return ((w - mean)**2).sum(-1).sum(-1)


class NN_qz_w(object):
    class Model(torch.nn.Module):
        def __init__(self, sz_image, ch_image, n_z, tasks, dim_w, n_k):
            super(NN_qz_w.Model, self).__init__()
            self.sz_image = sz_image
            self.ch = ch_image
            self.tasks = tasks
            self.n_z = n_z
            self.dim_w = dim_w
            self.n_k = n_k

            # for w input
            self.fc_w1 = torch.nn.Linear(self.dim_w * self.n_k, 128)
            self.drop_w = torch.nn.Dropout()
            self.fc_w2 = torch.nn.Linear(128, 512)

            # merge
            self.fc1 = torch.nn.Linear(3 * 512, 512)
            self.drop1 = torch.nn.Dropout()
            self.fc2 = torch.nn.Linear(512, 512)
            self.drop2 = torch.nn.Dropout()
            self.mean = torch.nn.Linear(512, self.n_z)
            self.logvar = torch.nn.Linear(512, self.n_z)

            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()

        # input w(n_batch, k_w, dim_w), im_c(n_batch, channel)
        # output mean(n_batch, n_z), logvar(n_batch, n_z)
        def forward(self, w, im_c):
            n_batch = w.size(0)

            # w
            w_x = self.relu(self.fc_w1(w.view(n_batch, -1)))
            w_x = self.drop_w(w_x)
            w_x = self.relu(self.fc_w2(w_x))

            # merge
            x = self.relu(self.fc1(torch.cat((im_c, w_x), 1)))
            x = self.drop1(x)
            x = self.relu(self.fc2(x))
            x = self.drop2(x)
            mean = self.mean(x)
            logvar = self.logvar(x)
            return mean, logvar

    def __init__(self, *args, **kwargs):
        self.model = self.Model(*args, **kwargs)

    # if samples is None, then return shape (n_batch, n_z)
    # else return shape (n_samples, n_batch, n_z)
    def sample(self, w, im_c, samples=None, reparameterization=False):
        mean, logvar = self.model.forward(w, im_c)

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
    def Dkl(self, w, im_c):
        def norm_Dkl(mean1, logvar1, mean2, logvar2):
            # p(x)~N(u1, v1), q(x)~N(u2, v2)
            # Dkl(p||q) = 0.5 * (log(|v2|/|v1|) - d + tr(v2^-1 * v1) + (u1 - u2)' * v2^-1 * (u1 - u2))
            # for diagonal v, Dkl(p||q) = 0.5*(sum(log(v2[i])-log(v1[i])+v1[i]/v2[i]+(u1[i]-u2[i])**2/v2[i]-1))
            return (logvar2 - logvar1 + (torch.exp(logvar1)**2 + (mean1 - mean2)**2) / (2 * torch.exp(logvar2)**2) - 1 / 2).sum(-1)

        mean, logvar = self.model.forward(w, im_c)
        if mean.is_cuda:
            mean_t, logvar_t = torch.zeros_like(mean).cuda(
            ).detach(), torch.zeros_like(logvar).cuda().detach()
        else:
            mean_t, logvar_t = torch.zeros_like(
                mean).detach(), torch.zeros_like(logvar).detach()

        return norm_Dkl(mean, logvar, mean_t, logvar_t)


if __name__ == '__main__':
    pass
