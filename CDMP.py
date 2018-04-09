import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from datetime import datetime as dt


class Config(object):
    def __init__(self):
        # task properties
        self.number_of_tasks = 4            # n_c
        self.trajectory_dimension = 2       # n_dim
        self.image_size = (100, 100)        # sz_im
        self.image_x_range = (-1., 1.)
        self.image_y_range = (0., 1.)
        self.image_channels = 3             # ch_im
        self.number_of_hidden = 4           # n_z
        self.number_of_MP_kernels = 10      # n_k
        self.number_time_samples = 100      # n_t
        self.number_of_oversample = 10      # n_oversample
        # data loader
        self.generator_train = batch_train  # function pointer
        self.generator_test = batch_test    # function pointer
        self.env = Env                      # class pointer
        # training properties
        self.batch_size_train = 20          # n_batch
        self.batch_size_test = 6
        self.batches_train = 100
        self.epochs = 100
        self.continue_training = True
        self.save_interval = 10             # -1 for saving best model
        self.display_interval = 5
        # program properties
        self.use_gpu = True
        self.multi_threads = 4
        self.log_path = "./logs"
        self.check_point_path = "./logs/checkpoints"
        self.experiment_name = "Four_Point_Reacher"


class Env(object):
    def __init__(self, config):
        self.cfg = config

        self.t = np.linspace(0, 1, self.cfg.number_time_samples, dtype=np.float32)
        self.center = ((-0.75, 0.75), (-0.3, 0.75), (0.3, 0.75), (0.75, 0.75))
        self.color = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (0., 0., 0.))
        self.traj_mean = (np.vstack([self.center[0][0] * self.t, self.center[0][1] * self.t ** .5]).T,
                          np.vstack([self.center[1][0] * self.t, self.center[1][1] * self.t ** .5]).T,
                          np.vstack([self.center[2][0] * self.t, self.center[2][1] * self.t ** .5]).T,
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
        tau += np.random.normal(0., 0.025, tau.shape) * np.expand_dims(np.sin(self.t * np.pi), 1)
        im = np.ones(self.cfg.image_size+(self.cfg.image_channels,), np.float32)
        for i in range(self.cfg.number_of_tasks):
            x, y = self.__remap_data_to_image(*self.center[i])
            cv2.rectangle(im, (int(x - 2), int(y - 2)), (int(x + 2), int(y + 2)), self.color[im_id[i]], cv2.FILLED)
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
                print("Warning: more then 8 samples are provided, only first 8 will be displayed")

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


# generator: (traj, task, image) x batch_size
def batch_train(config):
    env = config.env(config)
    while True:
        yield tuple(env.sample() for _ in range(config.batch_size_train))


# generator: (traj, task, image) x batch_size
def batch_test(config):
    env = config.env(config)
    while True:
        yield tuple(env.sample() for _ in range(config.batch_size_test))


def bar(current, total, prefix="", suffix="", bar_sz=25, end_string=None):
    sp = ""
    for i in range(bar_sz):
        if current * bar_sz // total > i:
            sp += '='
        elif current * bar_sz // total == i:
            sp += '>'
        else:
            sp += ' '
    if current == total:
        if end_string is None:
            print("\r                                                                                              "
                  "\r%s[%s]%s" % (prefix, sp, suffix))
        else:
            if end_string != "":
                print("\r                                                                                          "
                      "\r%s" % end_string)
            else:
                print("\r                                                                                          "
                      "\r", end='')
    else:
        print("\r                                                                                                  "
              "\r%s[%s]%s" % (prefix, sp, suffix), end='')


class CMP(object):
    def __init__(self, config):
        self.cfg = config
        self.encoder = self.NN_qz_w(n_z=self.cfg.number_of_hidden,
                                    dim_w=self.cfg.trajectory_dimension,
                                    n_k=self.cfg.number_of_MP_kernels)
        self.decoder = self.NN_pw_zimc(sz_image=self.cfg.image_size,
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

    class NN_pw_zimc(torch.nn.Module):
        def __init__(self, sz_image, ch_image, n_z, tasks, dim_w, n_k):
            super(CMP.NN_pw_zimc, self).__init__()
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
            self.fc_img1 = torch.nn.Linear(32*(sz_image[0] // 3 // 2 // 2)*(sz_image[1] // 3 // 2 // 2), 64)

            # for z input
            self.fc_z1 = torch.nn.Linear(self.n_z, 64)
            self.fc_z2 = torch.nn.Linear(64, 64)

            # for c input
            self.fc_c1 = torch.nn.Linear(self.tasks, 64)
            self.fc_c2 = torch.nn.Linear(64, 64)

            # merge
            self.fc1 = torch.nn.Linear(3*64, 64)
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
                mean, logvar = self.forward(z.view(-1, z.size(-1)), im.repeat(m, 1, 1, 1), c.repeat(m, 1))
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
                mean, _ = self.forward(z.view(-1, z.size(-1)), im.repeat(m, 1, 1, 1), c.repeat(m, 1))
                return ((w.repeat(m, 1, 1) - mean)**2).view(m, *w.size()).sum(0).sum(-1).sum(-1) / m
            else:
                mean, logvar = self.forward(z, im, c)
                dist = torch.distributions.Normal(mean, torch.exp(logvar))
                return ((w - mean)**2).sum(-1).sum(-1)

    class NN_qz_w(torch.nn.Module):
        def __init__(self, n_z, dim_w, n_k):
            super(CMP.NN_qz_w, self).__init__()
            self.n_k = n_k
            self.dim_w = dim_w
            self.n_z = n_z

            self.fc1 = torch.nn.Linear(self.dim_w * self.n_k, 64)
            self.fc2 = torch.nn.Linear(64, 64)
            self.mean = torch.nn.Linear(64, self.n_z)
            self.logvar = torch.nn.Linear(64, self.n_z)

            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()

        # input w(n_batch, k_w, dim_w);
        # output mean(n_batch, n_z), logvar(n_batch, n_z)
        def forward(self, w):
            n_batch = w.size(0)
            x = self.relu(self.fc1(w.view(n_batch, -1)))
            x = self.relu(self.fc2(x))
            mean = self.mean(x)
            logvar = self.logvar(x)
            return mean, logvar

        # if samples is None, then return shape (n_batch, n_k, dim_w)
        # else return shape (n_samples, n_batch, n_k, dim_w)
        def sample(self, w, samples=None, reparameterization=False):
            mean, logvar = self.forward(w)
            if reparameterization:
                dist = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(logvar))
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
        def Dkl(self, w):
            def norm_Dkl(mean1, logvar1, mean2, logvar2):
                # p(x)~N(u1, v1), q(x)~N(u2, v2)
                # Dkl(p||q) = 0.5 * (log(|v2|/|v1|) - d + tr(v2^-1 * v1) + (u1 - u2)' * v2^-1 * (u1 - u2))
                # for diagonal v, Dkl(p||q) = 0.5*(sum(log(v2[i])-log(v1[i])+v1[i]/v2[i]+(u1[i]-u2[i])**2/v2[i]-1))
                return 0.5*((logvar2-logvar1+torch.exp(logvar1-logvar2)+(mean1-mean2)**2.)/torch.exp(logvar2)-1)\
                            .sum(-1).sum(-1)

            mean, logvar = self.forward(w)
            if next(self.fc1.parameters()).is_cuda:
                mean_t, logvar_t = torch.zeros_like(mean).cuda().detach(), torch.zeros_like(logvar).cuda().detach()
            else:
                mean_t, logvar_t = torch.zeros_like(mean).detach(), torch.zeros_like(logvar).detach()

            return norm_Dkl(mean, logvar, mean_t, logvar_t)

    class RBF(object):
        # from (n_batch, t_samples, dim_in) or (t_samples, dim_in) to average kernel (n_k, dim_k)
        @staticmethod
        def calculate(traj, n_k):
            if len(traj.shape) != 3:
                if len(traj.shape) == 2:
                    traj = np.expand_dims(traj, axis=0)
                else:
                    print("What ?!")
                    exit(-1)

            dim_out = traj.shape[-1]
            N = traj.shape[0]
            nt = traj.shape[1]
            c = np.linspace(0, 1, n_k, dtype=np.float32)
            w = np.zeros((n_k, dim_out), dtype=np.float32)

            t = np.linspace(0, 1, nt, dtype=np.float32)
            # wc: N x K; w: K x D;
            wc = np.exp(-((np.tile(t[:, np.newaxis], (1, n_k)) - np.tile(c, (nt, 1))) * 10.) ** 2)
            wc /= wc.sum(1, keepdims=True)

            for _traj in traj:
                w += np.linalg.solve(wc.T @ wc, wc.T @ _traj) / N

            return w

        # from (n_k, dim_out) to (t_samples, dim_out)
        @staticmethod
        def generate(w, t_samples):
            n_k = w.shape[0]
            c = np.linspace(0, 1, n_k, dtype=np.float32)

            t = np.linspace(0, 1, t_samples, dtype=np.float32)
            # wc: N x K; w: K x D;
            wc = np.exp(-((np.tile(t[:, np.newaxis], (1, n_k)) - np.tile(c, (t_samples, 1))) * 10.) ** 2)
            wc /= wc.sum(1, keepdims=True)

            traj = wc @ w
            return traj

    # generator: (traj, task_id, img) x n_batch
    def train(self):
        def batchToVariable(traj_batch):
            batch_im = torch.zeros(self.cfg.batch_size_train, self.cfg.image_channels, self.cfg.image_size[0], self.cfg.image_size[1])
            batch_w = torch.zeros(self.cfg.batch_size_train, self.cfg.number_of_MP_kernels, self.cfg.trajectory_dimension)
            batch_c = torch.zeros(self.cfg.batch_size_train, self.cfg.number_of_tasks)
            for i, b in enumerate(traj_batch):
                batch_w[i] = torch.from_numpy(self.RBF.calculate(b[0], self.cfg.number_of_MP_kernels))
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

        optim = torch.optim.Adam(list(self.decoder.parameters())+list(self.encoder.parameters()))
        loss = []
        for epoch in range(self.cfg.epochs):
            avg_loss = []
            for i, batch in enumerate(self.cfg.generator_train(self.cfg)):
                w, c, im = batchToVariable(batch)
                optim.zero_grad()
                z = self.encoder.sample(w, samples=self.cfg.number_of_oversample, reparameterization=True)
                de = self.decoder.mse_error(w, z, im, c).sum()
                ee = self.encoder.Dkl(w).sum()
                l = de + ee
                l.backward()
                optim.step()

                avg_loss.append(l.data[0])

                bar(i+1, self.cfg.batches_train, "Epoch %d/%d: " % (epoch+1, self.cfg.epochs),
                    " | D-Err=%f; E-Err=%f" % (de.data[0], ee.data[0]), end_string='')

                # update training counter and make check points
                if i+1 >= self.cfg.batches_train:
                    loss.append(sum(avg_loss) / len(avg_loss))
                    print("Epoch=%d, Average Loss=%f" % (epoch+1, loss[-1]))
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
            batch_im = torch.zeros(self.cfg.batch_size_test, self.cfg.image_channels, self.cfg.image_size[0], self.cfg.image_size[1])
            batch_z = torch.normal(torch.zeros(self.cfg.batch_size_test, self.cfg.number_of_hidden),
                                   torch.ones(self.cfg.batch_size_test, self.cfg.number_of_hidden))
            batch_c = torch.zeros(self.cfg.batch_size_test, self.cfg.number_of_tasks)
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
        tauo = (self.RBF.generate(wo, self.cfg.number_time_samples) for wo in self.decoder.sample(z, im, c).cpu().data.numpy())
        tau, cls, imo = tuple(zip(*batch))
        env = self.cfg.env(self.cfg)
        env.display(tauo, imo, cls, interactive=True)


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
