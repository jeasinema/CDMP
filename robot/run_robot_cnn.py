#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : run_robot.py
# Purpose :
# Creation Date : 26-04-2018
# Last Modified : Fri 27 Apr 2018 11:47:14 AM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import torch
import sys
import glob
import argparse
import cv2
import os
import numpy as np
import multiprocessing as mp
from tensorboardX import SummaryWriter

from robot import *
from utils import *
from config import Config
from model import *
from rbf import RBF
from dmp import DMP
from colorize import *

from env import display

parser = argparse.ArgumentParser(description='CDMP_run')
parser.add_argument('--model-path', type=str, default='')
parser.add_argument('--ip', type=str, default='166.111.138.136')
parser.add_argument('--port', type=int, default=6776)
parser.add_argument('--camera', type=int, default=0)
parser.add_argument('--robot', action='store_true')
args = parser.parse_args()


g_net_param = torch.load(args.model_path, map_location='cpu') if args.model_path else None

if g_net_param:
    cfg = g_net_param['config']
else:
    cfg = Config() 
logger = SummaryWriter(os.path.join(cfg.log_path, cfg.experiment_name))
torch.cuda.set_device(0)

if cfg.use_DMP:
    dmp = DMP(cfg)

#loader
generator_train = build_loader(cfg, True)  # function pointer
generator_test = build_loader(cfg, False)    # function pointer


class CNN_DMP(object):
    def __init__(self, config):
        self.cfg = config
        self.condition_net = NN_img_c(sz_image=self.cfg.image_size,
                                      ch_image=self.cfg.image_channels,
                                      tasks=self.cfg.number_of_tasks,
                                      task_img_sz=(self.cfg.object_size[0] if self.cfg.img_as_task else -1))
        self.cnn_dmp = NN_cnn_dmp(dim_w=self.cfg.trajectory_dimension,
                                  n_k=self.cfg.number_of_MP_kernels)
        if g_net_param:
            self.cnn_dmp.load_state_dict(g_net_param['cnn_dmp'])
            self.condition_net.load_state_dict(g_net_param['condition_net'])
        self.use_gpu = (self.cfg.use_gpu and torch.cuda.is_available())
        if self.use_gpu:
            print("Use GPU for training, all parameters will move to GPU {}".format(self.cfg.gpu))
            self.cnn_dmp.cuda()
            self.condition_net.cuda()

        # TODO: loading from check points

    # generator: (traj, task_id, img) x n_batch
    def train(self):
        def batchToVariable(traj_batch):
            batch_im = torch.zeros(self.cfg.batch_size_train, self.cfg.image_channels,
                                   self.cfg.image_size[0], self.cfg.image_size[1])
            batch_w = torch.zeros(
                self.cfg.batch_size_train, self.cfg.number_of_MP_kernels, self.cfg.trajectory_dimension)
            if self.cfg.img_as_task:
                batch_c = torch.zeros(self.cfg.batch_size_train, self.cfg.image_channels,
                                       self.cfg.object_size[0], self.cfg.object_size[1])
            else:
                batch_c = torch.zeros(self.cfg.batch_size_train, self.cfg.number_of_tasks)
            for i, b in enumerate(traj_batch):
                batch_w[i] = torch.from_numpy(b[0])
                if self.cfg.img_as_task:
                    batch_c[i] = torch.from_numpy(b[2].transpose(2, 0, 1))
                    batch_im[i] = torch.from_numpy(b[3].transpose(2, 0, 1))
                else:
                    batch_c[i,b[1]] = 1.
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
            list(self.cnn_dmp.parameters()) + list(self.condition_net.parameters()))
        loss = []
        if g_net_param:
            base = g_net_param['epoch'] 
        else:
            base = 0
        for epoch in range(base, self.cfg.epochs+base):
            avg_loss = []
            for i, batch in enumerate(generator_train):
                w, c, im = batchToVariable(batch)
                optim.zero_grad()
                im_c = self.condition_net(im, c)
                w_pred = self.cnn_dmp(im_c)
                l = F.mse_loss(w, w_pred).mean()
                l.backward()
                optim.step()

                avg_loss.append(l.item())

                bar(i + 1, self.cfg.batches_train, "Epoch %d/%d: " % (epoch + 1, self.cfg.epochs),
                    " | Err=%f" % (l.item()), end_string='')

                # update training counter and make check points
                if i + 1 >= self.cfg.batches_train:
                    loss.append(sum(avg_loss) / len(avg_loss))
                    print("Epoch=%d, Average Loss=%f" % (epoch + 1, loss[-1]))
                    logger.add_scalar('loss', sum(avg_loss)/len(avg_loss), epoch)
                    break
            if (epoch % self.cfg.save_interval == 0 and epoch != 0) or\
                    (self.cfg.save_interval <= 0 and loss[-1] == min(loss)):
                net_param = {
                    "epoch": epoch,
                    "config": self.cfg,
                    "loss": loss,
                    "cnn_dmp": self.cnn_dmp.state_dict(),
                    "condition_net": self.condition_net.state_dict()
                }
                os.makedirs(self.cfg.check_point_path, exist_ok=True)
                check_point_file = os.path.join(self.cfg.check_point_path,
                                                "%s:%s.check" % (self.cfg.experiment_name, str(dt.now())))
                torch.save(net_param, check_point_file)
                print("Check point saved @ %s" % check_point_file)
            if epoch != 0 and epoch % self.cfg.display_interval == 0:
                if self.cfg.img_as_task:
                    img, img_gt, feature, c = self.test()
                else:
                    img, img_gt, feature = self.test()
                feature = feature.transpose([0,2,3,1]).sum(axis=-1, keepdims=True)
                h = feature.shape[1]*4 # CNN factor
                heatmap = np.zeros((h*2 + 20*3, h*3 + 20*4, 3),  # output 2*3
                        dtype=np.uint8)
                for ind in range(feature.shape[0]):
                    heatmap[(ind//3)*(h+20)+20:(ind//3)*(h+20)+20+h, 
                            (ind%3)*(h+20)+20:(ind%3)*(h+20)+20+h, :] = colorize(feature[ind, ...], 4)
                if self.cfg.img_as_task:
                    # output 2*3
                    h, w = self.cfg.object_size
                    task_map = np.zeros((h*2+20*3, w*3+20*4, 3)).astype(np.uint8)
                    for ind, task_img in enumerate(c.cpu().data.numpy()):
                        task_map[(ind//3)*(h+20)+20:(ind//3)*(h+20)+20+h,
                                (ind%3)*(w+20)+20:(ind%3)*(w+20)+20+w, :] = task_img.transpose([1,2,0])*255
                    logger.add_image('test_task_img', task_map, epoch)

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
            batch_w = torch.zeros(
                self.cfg.batch_size_test, self.cfg.number_of_MP_kernels, self.cfg.trajectory_dimension)

            batch_target = torch.zeros(
                self.cfg.batch_size_test, 2)

            if self.cfg.img_as_task:
                batch_c = torch.zeros(self.cfg.batch_size_test, self.cfg.image_channels,
                                       self.cfg.object_size[0], self.cfg.object_size[1])
            else:
                batch_c = torch.zeros(self.cfg.batch_size_test, self.cfg.number_of_tasks)

            for i, b in enumerate(traj_batch):
                batch_w[i] = torch.from_numpy(b[0])
                batch_target[i] = torch.from_numpy(b[-1])
                if self.cfg.img_as_task:
                    batch_c[i] = torch.from_numpy(b[2].transpose(2, 0, 1))
                    batch_im[i] = torch.from_numpy(b[3].transpose(2, 0, 1))
                else:
                    batch_c[i,b[1]] = 1.
                    batch_im[i] = torch.from_numpy(b[2].transpose(2, 0, 1))
            

            if self.use_gpu:
                return torch.autograd.Variable(batch_z.cuda(), volatile=True),\
                    torch.autograd.Variable(batch_c.cuda(), volatile=True),\
                    torch.autograd.Variable(batch_im.cuda(), volatile=True),\
                    batch_target,\
                    batch_w
            else:
                return torch.autograd.Variable(batch_z, volatile=True),\
                    torch.autograd.Variable(batch_c, volatile=True),\
                    torch.autograd.Variable(batch_im, volatile=True),\
                    batch_target,\
                    batch_w

        for batch in generator_test:
            break
        z, c, im, target, wgt = batchToVariable(batch)
        if self.cfg.use_DMP:
            p0 = np.tile(np.asarray((0., self.cfg.image_y_range[0]), dtype=np.float32), (self.cfg.batch_size_test, 1)) 
            w = self.cnn_dmp(self.condition_net(im, c)).cpu().data.numpy()
            tauo = tuple(dmp.generate(w, target.cpu().numpy(), self.cfg.number_time_samples, p0=p0, init=True))
            tau = tuple(dmp.generate(wgt.cpu().numpy(), target.cpu().numpy(), self.cfg.number_time_samples, p0=p0, init=True))
        else:
            tauo = tuple(RBF.generate(wo, self.cfg.number_time_samples)
                    for wo in self.cnn_dmp(self.condition_net(im, c)).cpu().data.numpy())
            tau = tuple(RBF.generate(wo, self.cfg.number_of_MP_kernels)
                    for wo in wgt)
        if self.cfg.img_as_task:
            _, cls, _, imo, _ = tuple(zip(*batch))
        else:
            _, cls, imo, _ = tuple(zip(*batch))
        env = self.cfg.env(self.cfg)
        img = display(self.cfg, tauo, imo, cls, interactive=True)
        img_gt = display(self.cfg, tau, imo, cls, interactive=True)
        feature = self.condition_net.feature_map(im).data.cpu().numpy()
        if self.cfg.img_as_task:
            return img, img_gt, feature, c
        else:
            return img, img_gt, feature  

    def eval(self, im, c):
        im = Variable(torch.from_numpy(im.transpose([2,0,1])).unsqueeze(0).float())
        tmp = Variable(torch.zeros(1, self.cfg.number_of_tasks).float())
        tmp[:, int(c)] = 1.

        if self.use_gpu:
            im = im.cuda()
            tmp = tmp.cuda()
        im_c = self.condition_net(im, tmp)
        w = self.cnn_dmp(im_c).cpu().data.numpy()
        return w



cfg.number_time_samples = 10

cam_exit = mp.Value('i', 0)
cam_retrieve = mp.Value('i', 0)
cam_queue = mp.Queue(1)
# TODO
init_position = np.array([0.30, -0.40, 0.0])
init_orintation = np.array([-0.373, 0.928, 0.0135, 0.015])
# imgList = glob.glob('./data/cdmp_images/scene/*.png')
imgList = ['./data/scene.png']

# Tasks:
tasks = {
    0 : "apple",
    1 : "ball",
    2 : "banana",
    3 : "can", 
    4 : "coffee can", 
    5 : "cup", 
    6 : "pear",
    7 : "screwdriver",
    8 : "suger", 
    9 : "yello bottle",
}

def main():
    alg = CNN_DMP(cfg)
    print('1. Model [{}]'.format('READY'))
    try:
        if args.robot:
            baxter = RemoteBaxter(args.ip, args.port)
            res = baxter.connect()
            if not res:
                print('2. Robot [{}]'.format('FAILED'))
                raise RuntimeError
            print('2. Robot [{}]'.format('READY'))
            print('Back to init pose..')
            baxter.gotoPose(init_position, init_orintation)
        else:
            print('2. Bypass Robot [{}]'.format('READY'))
        # cam_p = mp.Process(target=camera_process_opencv, args=(args.camera,))
        cam_p = mp.Process(target=camera_process_imglist, args=(imgList,))
        cam_p.start()
        img = cam_queue.get()
        if not img:
            print('3. Camera [{}]'.format('FAILED'))
            raise RuntimeError
        else:
            print('3. Camera [{}]'.format('READY'))
        print('4. Commander [{}]'.format('READY'))
        print('=======Command Mode=======')
        i = -1
        while True:
            i += 1
            print('------ Round {} ------'.format(i))
            cmd = int(input('Cmd:'))
            if cmd == -1:
                print('User specified exit..')
                print('------ Round end ------\n')
                break
            elif 0 <= cmd < cfg.number_of_tasks:
                print('Start to compute traj for task [{}:{}]'.format(cmd, tasks[cmd]))
                cam_retrieve.value = 1
                img = cam_queue.get()
                if img is False:
                    print('Camera shut down, stop')
                    print('------ Round end ------\n')
                    break
                # TODO: pre-process image
                img = cv2.resize(img, cfg.image_size)/255.
                img = img[..., [2, 1, 0]]
                position_traj = RBF.generate(alg.eval(img, cmd)[0], cfg.number_time_samples)
                position_traj = np.hstack([position_traj, init_position[-1]*np.ones((len(position_traj), 1))])
                orintation_traj = init_orintation[np.newaxis, ...].repeat(
                    len(position_traj), 0)
                print('Finish traj computation for task [{}]'.format(cmd))
                # plot the traj and img
                display(cfg, [position_traj], [img], [0])
                choice = input('Confirmed?(y/n):')
                if choice != 'y':
                    print('Traj canceled')
                    print('------ Round end ------\n')
                    continue
                print('Traj confirmed')
                # confirmed the result
                if args.robot:
                    start_t = time.time()
                    res = baxter.followTraj(position_traj, orintation_traj, continuous=True)
                    if res:
                        print('Task [{}] exec success!'.format(cmd))
                    else:
                        print('Task [{}] exec failed!'.format(cmd))
                    print('Time cost:{:.4f}s'.format(time.time() - start_t))
                    print('Back to init pose..')
                    baxter.gotoPose(init_position, init_orintation)
                else:
                    print('Bypass exec..')
                print('------ Round end ------\n')
            else:
                print('Please input an valid task number!')
                print('------ Round end ------\n')
        if args.robot:
            print('5. Disconnecting with robot..')
            baxter.close()
        else:
            print('5. Bypass disconnecting with robot..')
    except:
        print('Something going wrong, exit.')
        cam_exit.value = 1
        sys.exit(1)

    print('6. Finish and clean-up [{}]'.format('DONE'))
    cam_exit.value = 1
    sys.exit(0)


def camera_process_opencv(cam_id):
    cam = cv2.VideoCapture(cam_id)
    if cam.isOpened():
        success, img = camera.read()
        cam_queue.put(True)  # notice main thread
    else:
        cam_queue.put(False)
        return 0
    buf = []
    while not cam_exit.value:
        success, img = camera.read()
        if success:
            buf = img
        if cam_retrieve.value:
            cam_queue.put(buf)
            cam_retrieve.value = 0
    camera.close()


def camera_process_imglist(imglist):
    if imglist:
        cam_queue.put(True)  # notice main thread
    else:
        cam_queue.put(False)
        return 0
    img_list = iter(imglist)
    while not cam_exit.value:
        try:
            if cam_retrieve.value:
                img = next(img_list)
                img = cv2.imread(img)
                cam_queue.put_nowait(img)
                cam_retrieve.value = 0
        except:
            img_list = iter(imglist)
            img = next(img_list)
            img = cv2.imread(img)
            cam_queue.put(img)
            cam_retrieve.value = 0



if __name__ == '__main__':
    main()
