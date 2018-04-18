#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : model.py
# Purpose :
# Creation Date : 09-04-2018
# Last Modified : Tue 17 Apr 2018 11:31:34 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        if self.data_format == 'NHWC':
            feature = feature.permute(0,2,3,1).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)
        
        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(Variable(self.pos_x)*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(Variable(self.pos_y)*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints


class NN_img_c(torch.nn.Module):
    def __init__(self, sz_image, ch_image, tasks):
        super(NN_img_c, self).__init__()
        self.sz_iamge = sz_image
        self.ch = ch_image
        self.tasks = tasks

        # for image input
        self.conv1 = torch.nn.Conv2d(self.ch, 64, kernel_size=4, padding=1, stride=2)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=4, padding=1, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.fc_img1 = torch.nn.Linear(
        #     32 * (sz_image[0] // 3 // 2 // 2) * (sz_image[1] // 3 // 2 // 2), 64)
        self.spatial_softmax = SpatialSoftmax(sz_image[0] // 2 // 2, sz_image[1] // 2 // 2, 64) # (N, 32*2)

        # for merge 
        self.fc1 = torch.nn.Linear(self.tasks + 64*2, 128)
        self.fc2 = torch.nn.Linear(128, 128)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, im, c):
        n_batch = c.size(0)

        # image
        im_x = self.relu(self.conv1(im))
        im_x = self.relu(self.conv2(im_x))
        im_x = self.relu(self.conv3(im_x))
        im_x = self.relu(self.conv4(im_x))
        im_x = self.relu(self.conv5(im_x))
        im_x = self.spatial_softmax(im_x)
        
        # merge 
        im_c = torch.cat((im_x, c), 1)
        im_c = self.relu(self.fc1(im_c))
        im_c = self.fc2(im_c)

        return im_c

    def feature_map(self, im):
        im_x = self.relu(self.conv1(im))
        im_x = self.relu(self.conv2(im_x))
        im_x = self.relu(self.conv3(im_x))
        im_x = self.relu(self.conv4(im_x))
        im_x = self.relu(self.conv5(im_x))
       
        return im_x


class NN_pw_zimc(torch.nn.Module):
    def __init__(self, sz_image, ch_image, n_z, tasks, dim_w, n_k):
        super(NN_pw_zimc, self).__init__()
        self.sz_image = sz_image
        self.ch = ch_image
        self.tasks = tasks
        self.n_z = n_z
        self.dim_w = dim_w
        self.n_k = n_k

        # for z input
        self.fc_z1 = torch.nn.Linear(self.n_z, 64)
        self.fc_z2 = torch.nn.Linear(64, 64)

        # merge
        self.fc1 = torch.nn.Linear(3 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.mean = torch.nn.Linear(64, self.dim_w * self.n_k)
        self.logvar = torch.nn.Linear(64, self.dim_w * self.n_k)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    # input z(n_batch, n_z), im_c(n_batch, channel)
    # output mean(n_batch, k_w, dim_w), logvar(n_batch, k_w, dim_w)
    def forward(self, z, im_c):
        n_batch = z.size(0)

        z_x = self.relu(self.fc_z1(z))
        z_x = self.fc_z2(z_x)

        # merge
        x = self.relu(self.fc1(torch.cat((im_c, z_x), 1)))
        x = self.relu(self.fc2(x))
        mean = self.mean(x).view(n_batch, self.n_k, self.dim_w)
        logvar = self.logvar(x).view(n_batch, self.n_k, self.dim_w)
        return mean, logvar

    # input z(n_batch, n_z), im_c(n_batch, channel)
    # output mean(n_samples, n_batch, n_k, dim_w), logvar(n_samples, n_batch, n_k, dim_w)
    def sample(self, z, im_c, samples=None, fixed=True):
        mean, logvar = self.forward(z, im_c)
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
            mean, logvar = self.forward(
                z.view(-1, z.size(-1)), im_c.repeat(m, 1))
            dist = torch.distributions.Normal(mean, torch.exp(logvar))
            return dist.log_prob(w.repeat(m, 1, 1)).view(m, *w.size()).sum(-1).sum(-1).mean()
        else:
            mean, logvar = self.forward(z, im_c)
            dist = torch.distributions.Normal(mean, torch.exp(logvar))
            return dist.log_prob(w).sum(-1).sum(-1)

    # input w(n_batch, k_w, dim_w) z(n_batch, n_z), im_c(n_batch, channel)
    # output (n_batch,)
    # or if z is batch of samples, output E(n_batch)
    def mse_error(self, w, z, im_c):
        if len(z.size()) == len(im_c.size()) + 1:
            m = z.size(0)
            mean, _ = self.forward(
                z.view(-1, z.size(-1)), im_c.repeat(m, 1))
            return ((w.repeat(m, 1, 1) - mean)**2).view(m, *w.size()).sum(-1).sum(-1).mean()
        else:
            mean, logvar = self.forward(z, im_c)
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

    # input w(n_batch, k_w, dim_w), im_c(n_batch, channel)
    # output mean(n_batch, n_z), logvar(n_batch, n_z)
    def forward(self, w, im_c):
        n_batch = w.size(0)

        # w
        w_x = self.relu(self.fc_w1(w.view(n_batch, -1)))
        w_x = self.relu(self.fc_w2(w_x))

        # merge
        x = self.relu(self.fc1(torch.cat((im_c, w_x), 1)))
        x = self.relu(self.fc2(x))
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar

    # if samples is None, then return shape (n_batch, n_z)
    # else return shape (n_samples, n_batch, n_z)
    def sample(self, w, im_c, samples=None, reparameterization=False):
        mean, logvar = self.forward(w, im_c)
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

        mean, logvar = self.forward(w, im_c)
        if next(self.fc1.parameters()).is_cuda:
            mean_t, logvar_t = torch.zeros_like(mean).cuda(
            ).detach(), torch.zeros_like(logvar).cuda().detach()
        else:
            mean_t, logvar_t = torch.zeros_like(
                mean).detach(), torch.zeros_like(logvar).detach()

        return norm_Dkl(mean, logvar, mean_t, logvar_t)


if __name__ == '__main__':
    from torch.autograd import Variable
    data = Variable(torch.zeros([1,3,3,3]))
    data[0,0,0,1] = 10
    data[0,1,1,1] = 10
    data[0,2,1,2] = 10
    layer = SpatialSoftmax(3,3,3, temperature=1)
    print(layer(data))


