import numpy as np
import matplotlib.pyplot as plt
import torch


class RBF(object):
    def __init__(self, dim_out, Tasks, K=10):
        self.dim = dim_out
        self.K = K
        self.Tasks = Tasks

        self.center = np.linspace(0, 1, K, dtype=np.float32)
        self.weight = np.zeros((self.Tasks, self.K, self.dim), dtype=np.float32)
        self.weight_var = np.ones((self.Tasks, self.K, self.dim), dtype=np.float32)
        self.pi = np.ones((self.Tasks,), dtype=np.float32) / self.Tasks

    @staticmethod
    def kernel_func(t, c):
        return np.exp(-((t - c) * 10.) ** 2)

    def train(self, traj, epoch=5000):
        """
        :param traj: [n_batch, n_t, n_dim]
        :return: None
        """

        ## first calculate kernel weights ##

        n_t = traj.shape[1]
        t = np.linspace(0, 1, n_t, dtype=np.float32)
        # wc: N x K; w: K x D;
        wc = self.kernel_func(np.tile(t[:, np.newaxis], (1, self.K)), np.tile(self.center, (n_t, 1)))
        wc /= wc.sum(1, keepdims=True)
        w = []
        for _traj in traj:
            wi = np.linalg.solve(wc.T @ wc, wc.T @ _traj)
            w.append(wi)
        # w: NxKxD
        w = np.asarray(w, dtype=np.float32)

        ## then calculate clustering ##

        # ag_mean & ag_var: CxKxD; ag_pi: C
        ag_logpi = torch.autograd.Variable(torch.ones(self.Tasks), requires_grad=True)
        ag_mean = torch.autograd.Variable(torch.rand(self.Tasks, self.K, self.dim), requires_grad=True)
        ag_logvar2 = torch.autograd.Variable(torch.ones(self.Tasks, self.K, self.dim) * -1)

        ag_w = torch.autograd.Variable(torch.from_numpy(w))

        def log_pw():
            # calculate norm pdf
            def pw_normal(x, mean, var2):
                log_pdf = -(x-mean)**2 / (2. * var2) - 0.5*torch.log(var2 * 2. * np.pi)
                return torch.exp(log_pdf.sum(-1).sum(-1))

            # CxN
            from ipdb import set_trace; set_trace()
            pw_c_mul_pc = pw_normal(ag_w.unsqueeze(0), ag_mean.unsqueeze(1), torch.exp(ag_logvar2).unsqueeze(1)) # *\
                          # torch.multinomial(self.ag_logpi, , True)
                          # (torch.exp(ag_logpi)/torch.exp(ag_logpi).sum()).unsqueeze(1)
            pw = torch.log(pw_c_mul_pc.sum(dim=0)).sum(dim=0)
            return pw

        plt.ion()
        plt.figure()
        optim = torch.optim.Adam([ag_logpi, ag_mean])
        for e in range(epoch):
            optim.zero_grad()
            pw = -log_pw()
            pw.backward()
            optim.step()
            if e % 100 == 0:
                print("Step %d: log(pw): %f" % (e, -pw[0]))

                self.weight = ag_mean.data.numpy()
                self.weight_var = np.exp(ag_logvar2.data.numpy() / 2.)
                self.pi = np.exp(ag_logpi.data.numpy())
                self.pi /= self.pi.sum()

                plt.clf()
                for i in range(self.Tasks):
                    taug = self.generate(task=i, fix=True)
                    plt.plot(taug[:, 0], taug[:, 1])
                plt.pause(0.01)

        plt.ioff()
        self.weight = ag_mean.data.numpy()
        self.weight_var = np.exp(ag_logvar2.data.numpy() / 2.)
        self.pi = np.exp(ag_logpi.data.numpy())
        self.pi /= self.pi.sum()

    def generate(self, nt=100, task=0, fix=True):
        t = np.linspace(0, 1, nt, dtype=np.float32)
        wc = self.kernel_func(np.tile(t[:, np.newaxis], (1, self.K)), np.tile(self.center, (nt, 1)))
        wc /= wc.sum(1, keepdims=True)

        if fix:
            w = self.weight[task]
        else:
            w = np.random.normal(self.weight[task], np.sqrt(self.weight_var[task]))
        return wc @ w


class Net_qc_w(torch.nn.Module):
    def __init__(self, K, C):
        super(Net_qc_w, self).__init__()
        self.K = K
        self.fc1 = torch.nn.Linear(K, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, C)
        self.active = torch.nn.Tanh()

    def forward(self, x):
        y = self.active(self.fc1(x))
        y = self.active(self.fc2(y))
        y = self.fc3(y)
        return y


if __name__ == "__main__":
    t = np.linspace(0, 1, 50, dtype=np.float32)
    tau1 = np.vstack([-1.*t, (1*t)**.25]).T
    tau2 = np.vstack([-0.3*t, (1*t)**.25]).T
    tau3 = np.vstack([0.3*t, (1*t)**.25]).T
    tau4 = np.vstack([1.*t, (1*t)**.25]).T
    tau = np.vstack([np.tile(tau1, (10, 1, 1)), np.tile(tau2, (40, 1, 1)),
                     np.tile(tau3, (300, 1, 1)), np.tile(tau4, (60, 1, 1))])

    tau += np.random.normal(0., 0.05, tau.shape) * np.sin(t*np.pi).reshape(1, 50, 1)

    # from ipdb import set_trace; set_trace()
    rbf = RBF(dim_out=2, Tasks=4)
    rbf.train(tau, epoch=5000)

    plt.figure()
    for taui in tau:
        plt.plot(taui[:, 0], taui[:, 1], '-.')
    plt.figure()
    for i in range(4):
        taug = rbf.generate(task=i, fix=True)
        plt.plot(taug[:, 0], taug[:, 1])

    plt.show()
