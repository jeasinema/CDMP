import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from datetime import datetime as dt
from torchvision import datasets, transforms 

from config_mnist import Config
from utils import bar
from rbf import RBF
from model_mnist import *
from tensorboard_logging import Logger

cfg = Config()
logger = Logger(os.path.join(cfg.log_path, cfg.experiment_name))

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=cfg.batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=cfg.batch_size_test, shuffle=True)


class CMP(object):
    def __init__(self, config):
        self.cfg = config
        self.encoder = NN_qz_w(n_z=self.cfg.number_of_hidden,
                               ch_image=self.cfg.image_channels,
                               sz_image=self.cfg.image_size,
                               tasks=self.cfg.number_of_tasks)
        self.decoder = NN_pw_zimc(sz_image=self.cfg.image_size,
                                  ch_image=self.cfg.image_channels,
                                  n_z=self.cfg.number_of_hidden,
                                  tasks=self.cfg.number_of_tasks)
        self.use_gpu = (self.cfg.use_gpu and torch.cuda.is_available())
        if self.use_gpu:
            print("Use GPU for training, all parameters will move to GPU 0")
            self.encoder.cuda(0)
            self.decoder.cuda(0)

        # TODO: loading from check points

    # generator: (traj, task_id, img) x n_batch
    def train(self):
        optim = torch.optim.Adam(
            list(self.decoder.parameters()) + list(self.encoder.parameters()))
        loss = []
        for epoch in range(self.cfg.epochs):
            avg_loss = []
            avg_loss_de = []
            avg_loss_ee = []
            for i, (data, target) in enumerate(train_loader):
                batch_n = data.shape[0]
                c = torch.zeros(batch_n, 10).float()
                for ind in range(batch_n):
                    c[ind, target[ind]] = 1
                if self.use_gpu:
                    data, c = data.cuda(), c.cuda()
                im = Variable(data)
                c = Variable(c)
                
                optim.zero_grad()
                z = self.encoder.sample(
                    im, c, samples=1, reparameterization=True)
                res, _ = self.decoder(z.view(-1, z.size(-1)), c.repeat(z.size(0), 1))
                de = F.mse_loss(res, im).mean()
                ee = self.encoder.Dkl(im, c).mean()
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
                    logger.log_scalar('loss', sum(avg_loss)/len(avg_loss), epoch)
                    logger.log_scalar('loss_de', sum(avg_loss_de)/len(avg_loss_de), epoch)
                    logger.log_scalar('loss_ee', sum(avg_loss_ee)/len(avg_loss_ee), epoch)
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
                img = self.test()
                logger.log_images('test_img', img, epoch)

    # generator: (task_id, img) x n_batch
    def test(self):
        batch_z = torch.normal(torch.zeros(self.cfg.batch_size_test, self.cfg.number_of_hidden),
                               torch.ones(self.cfg.batch_size_test, self.cfg.number_of_hidden))
        batch_c = torch.zeros(self.cfg.batch_size_test,
                              self.cfg.number_of_tasks)
        for ind in range(self.cfg.batch_size_test):
            batch_c[ind, ind % self.cfg.number_of_tasks] = 1
        if self.use_gpu:
            batch_c, batch_z = batch_c.cuda(), batch_z.cuda()
        z, c = Variable(batch_z), Variable(batch_c)
        res = self.decoder.sample(z, c).cpu().data.numpy() # N * 1 * 28 * 28
        res = res.transpose((0,2,3,1)) # N * 28 * 28 * 1
        res = np.tile(res, [1,1,1,3]) # N * 28 * 28 * 3
        img = np.zeros((28*5, 28*10, 3))
        for ind in range(self.cfg.batch_size_test):
            img[(ind//10)*28:((ind//10)+1)*28, (ind%10)*28:((ind%10)+1)*28, :] = res[ind, ...]*255
        
        return img.astype(np.uint8)


def main():
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
