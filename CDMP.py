import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from datetime import datetime as dt

from config import Config
from utils import bar
from rbf import RBF
from model import *


class CMP(object):
    def __init__(self, config):
        self.cfg = config
        self.encoder = NN_qz_w(n_z=self.cfg.number_of_hidden,
                               dim_w=self.cfg.trajectory_dimension,
                               n_k=self.cfg.number_of_MP_kernels)
        self.decoder = NN_pw_zimc(sz_image=self.cfg.image_size,
                                  ch_image=self.cfg.image_channels,
                                  n_z=self.cfg.number_of_hidden,
                                  tasks=self.cfg.number_of_tasks,
                                  dim_w=self.cfg.trajectory_dimension,
                                  n_k=self.cfg.number_of_MP_kernels)
        self.use_gpu = (self.cfg.use_gpu and torch.cuda.is_available())
        if self.use_gpu:
            print("Use GPU for training, all parameters will move to GPU 0")
            self.encoder.cuda(0)
            self.decoder.cuda(0)

        # TODO: loading from check points

    # generator: (traj, task_id, img) x n_batch
    def train(self):
        def batchToVariable(traj_batch):
            batch_im = torch.zeros(self.cfg.batch_size_train, self.cfg.image_channels,
                                   self.cfg.image_size[0], self.cfg.image_size[1])
            batch_w = torch.zeros(
                self.cfg.batch_size_train, self.cfg.number_of_MP_kernels, self.cfg.trajectory_dimension)
            batch_c = torch.zeros(self.cfg.batch_size_train,
                                  self.cfg.number_of_tasks)
            for i, b in enumerate(traj_batch):
                batch_w[i] = torch.from_numpy(RBF.calculate(
                    b[0], self.cfg.number_of_MP_kernels))
                batch_c[i, b[1]] = 1.
                batch_im[i] = torch.from_numpy(b[2].transpose(2, 0, 1))

            if self.use_gpu:
                return torch.autograd.Variable(batch_w.cuda()),\
                    torch.autograd.Variable(batch_c.cuda()),\
                    torch.autograd.Variable(batch_im.cuda())
            else:
                return torch.autograd.Variable(batch_w),\
                    torch.autograd.Variable(batch_c),\
                    torch.autograd.Variable(batch_im)

        optim = torch.optim.Adam(
            list(self.decoder.parameters()) + list(self.encoder.parameters()))
        loss = []
        for epoch in range(self.cfg.epochs):
            avg_loss = []
            for i, batch in enumerate(self.cfg.generator_train(self.cfg)):
                w, c, im = batchToVariable(batch)
                optim.zero_grad()
                z = self.encoder.sample(
                    w, samples=self.cfg.number_of_oversample, reparameterization=True)
                de = self.decoder.mse_error(w, z, im, c).mean()
                ee = self.encoder.Dkl(w).mean()
                l = de + ee
                l.backward()
                optim.step()

                avg_loss.append(l.data[0])

                bar(i + 1, self.cfg.batches_train, "Epoch %d/%d: " % (epoch + 1, self.cfg.epochs),
                    " | D-Err=%f; E-Err=%f" % (de.data[0], ee.data[0]), end_string='')

                # update training counter and make check points
                if i + 1 >= self.cfg.batches_train:
                    loss.append(sum(avg_loss) / len(avg_loss))
                    print("Epoch=%d, Average Loss=%f" % (epoch + 1, loss[-1]))
                    break
            if (epoch % self.cfg.save_interval == 0 and epoch != 0) or\
                    (self.cfg.save_interval <= 0 and loss[-1] == min(loss)):
                net_param = {
                    "epoch": epoch,
                    "config": self.cfg,
                    "loss": loss,
                    "encoder": self.encoder.state_dict(),
                    "decoder": self.decoder.state_dict()
                }
                os.makedirs(self.cfg.check_point_path, exist_ok=True)
                check_point_file = os.path.join(self.cfg.check_point_path,
                                                "%s:%s.check" % (self.cfg.experiment_name, str(dt.now())))
                torch.save(net_param, check_point_file)
                print("Check point saved @ %s" % check_point_file)
            if epoch != 0 and epoch % self.cfg.display_interval == 0:
                self.test()

    # generator: (task_id, img) x n_batch
    def test(self):
        def batchToVariable(traj_batch):
            batch_im = torch.zeros(self.cfg.batch_size_test, self.cfg.image_channels,
                                   self.cfg.image_size[0], self.cfg.image_size[1])
            batch_z = torch.normal(torch.zeros(self.cfg.batch_size_test, self.cfg.number_of_hidden),
                                   torch.ones(self.cfg.batch_size_test, self.cfg.number_of_hidden))
            batch_c = torch.zeros(self.cfg.batch_size_test,
                                  self.cfg.number_of_tasks)
            for i, b in enumerate(traj_batch):
                batch_c[i, b[1]] = 1.
                batch_im[i] = torch.from_numpy(b[2].transpose(2, 0, 1))

            if self.use_gpu:
                return torch.autograd.Variable(batch_z.cuda(), volatile=True),\
                    torch.autograd.Variable(batch_c.cuda(), volatile=True),\
                    torch.autograd.Variable(batch_im.cuda(), volatile=True)
            else:
                return torch.autograd.Variable(batch_z, volatile=True),\
                    torch.autograd.Variable(batch_c, volatile=True),\
                    torch.autograd.Variable(batch_im, volatile=True)

        batch = next(self.cfg.generator_test(self.cfg))
        z, c, im = batchToVariable(batch)
        tauo = (RBF.generate(wo, self.cfg.number_time_samples)
                for wo in self.decoder.sample(z, im, c).cpu().data.numpy())
        tau, cls, imo = tuple(zip(*batch))
        env = self.cfg.env(self.cfg)
        env.display(tauo, imo, cls, interactive=True)
        # env.display(tau, imo, cls, interactive=True)


def main():
    cfg = Config()
    alg = CMP(config=cfg)
    alg.train()
    alg.test()


if __name__ == "__main__":
    main()
    # env = Env(Config())
    # for i in range(10):
    #     batch = (env.sample() for j in range(6))
    #     batch = tuple(zip(*batch))
    #     env.display(batch[0], batch[2], batch[1], interactive=True)
    #     plt.pause(3)
