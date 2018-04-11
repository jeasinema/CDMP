import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from datetime import datetime as dt

from config import Config
from utils import bar
from rbf import RBF
from model import *
from colorize import *
from tensorboardX import SummaryWriter

cfg = Config()
logger = SummaryWriter(os.path.join(cfg.log_path, cfg.experiment_name))

class CMP(object):
    def __init__(self, config):
        self.cfg = config
        self.condition_net = NN_img_c(sz_image=self.cfg.image_size,
                                      ch_image=self.cfg.image_channels,
                                      tasks=self.cfg.number_of_tasks)
        self.encoder = NN_qz_w(n_z=self.cfg.number_of_hidden,
                               ch_image=self.cfg.image_channels,
                               sz_image=self.cfg.image_size,
                               tasks=self.cfg.number_of_tasks,
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
            print("Use GPU for training, all parameters will move to GPU {}".format(self.cfg.device_id))
            self.encoder.model = nn.DataParallel(self.encoder.model, device_ids=cfg.device_id).cuda()
            self.decoder.model = nn.DataParallel(self.decoder.model, device_ids=cfg.device_id).cuda()
            self.condition_net.model = nn.DataParallel(self.condition_net.model, device_ids=cfg.device_id).cuda()


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
            list(self.decoder.model.parameters()) + list(self.encoder.model.parameters()) + list(self.condition_net.model.parameters()))
        loss = []
        for epoch in range(self.cfg.epochs):
            self.encoder.model.train()
            self.decoder.model.train()
            self.condition_net.model.train()
            avg_loss = []
            avg_loss_de = []
            avg_loss_ee = []
            for i, batch in enumerate(self.cfg.generator_train(self.cfg)):
                w, c, im = batchToVariable(batch)
                optim.zero_grad()
                im_c = self.condition_net.model(im, c)[0]
                z = self.encoder.sample(
                    w, im_c, samples=self.cfg.number_of_oversample, reparameterization=True)
                de = self.decoder.mse_error(w, z, im_c).mean()
                ee = self.encoder.Dkl(w, im_c).mean()
                l = de + ee
                l.backward()
                optim.step()

                avg_loss.append(l.data[0])
                avg_loss_de.append(de.data[0])
                avg_loss_ee.append(ee.data[0])

                bar(i + 1, self.cfg.batches_train, "Epoch %d/%d: " % (epoch + 1, self.cfg.epochs),
                    " | D-Err=%f; E-Err=%f" % (de.data[0], ee.data[0]), end_string='')

                # update training counter and make check points
                if i + 1 >= self.cfg.batches_train:
                    loss.append(sum(avg_loss) / len(avg_loss))
                    print("Epoch=%d, Average Loss=%f" % (epoch + 1, loss[-1]))
                    logger.add_scalar('loss', sum(avg_loss)/len(avg_loss), epoch)
                    logger.add_scalar('loss_de', sum(avg_loss_de)/len(avg_loss_de), epoch)
                    logger.add_scalar('loss_ee', sum(avg_loss_ee)/len(avg_loss_ee), epoch)
                    break
            if (epoch % self.cfg.save_interval == 0 and epoch != 0) or\
                    (self.cfg.save_interval <= 0 and loss[-1] == min(loss)):
                net_param = {
                    "epoch": epoch,
                    "config": self.cfg,
                    "loss": loss,
                    "encoder": self.encoder.model.state_dict(),
                    "decoder": self.decoder.model.state_dict(),
                    "condition_net": self.condition_net.model.state_dict(),
                }
                os.makedirs(self.cfg.check_point_path, exist_ok=True)
                check_point_file = os.path.join(self.cfg.check_point_path,
                                                "%s:%s.check" % (self.cfg.experiment_name, str(dt.now())))
                torch.save(net_param, check_point_file)
                print("Check point saved @ %s" % check_point_file)
            if epoch != 0 and epoch % self.cfg.display_interval == 0:
                img, img_gt, feature = self.test()
                feature = feature.transpose([0,2,3,1]).sum(axis=-1, keepdims=True)
                h = feature.shape[1]*12 # CNN factor
                heatmap = np.zeros((h*2 + 20*3, h*3 + 20*4, 3),  # output 2*3
                        dtype=np.uint8)
                for ind in range(feature.shape[0]):
                    heatmap[(ind//3)*(h+20)+20:(ind//3)*(h+20)+20+h, 
                            (ind%3)*(h+20)+20:(ind%3)*(h+20)+20+h, :] = colorize(feature[ind, ...], 12)
                logger.add_image('test_img', img, epoch)
                logger.add_image('heatmap', heatmap, epoch)
                logger.add_image('test_img_gt', img_gt, epoch)


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


        self.encoder.model.eval()
        self.decoder.model.eval()
        self.condition_net.model.eval()
        batch = next(self.cfg.generator_test(self.cfg))
        z, c, im = batchToVariable(batch)
        tauo = tuple(RBF.generate(wo, self.cfg.number_time_samples)
                for wo in self.decoder.sample(z, self.condition_net.model(im, c)[0]).cpu().data.numpy())
        tau, cls, imo = tuple(zip(*batch))
        env = self.cfg.env(self.cfg)
        img = env.display(tauo, imo, cls, interactive=True)
        img_gt = env.display(tau, imo, cls, interactive=True)
        feature = self.condition_net.feature_map(im, c).data.cpu().numpy()
        return img, img_gt, feature  


def main():
    alg = CMP(config=cfg)
    alg.train()
    alg.test()


if __name__ == "__main__":
    main()
    # from env import Env
    # env = Env(Config())
    # for i in range(10):
    #     batch = (env.sample(task_id=0, im_id=list(range(10))) for j in range(6))
    #     batch = tuple(zip(*batch))
    #     env.display(batch[0], batch[2], batch[1], interactive=True)
    #     plt.pause(3)
