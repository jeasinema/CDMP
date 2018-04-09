import numpy as np
import matplotlib.pyplot as plt
import torch


class RBF(object):
    def __init__(self, dim_out, K=10):
        self.dim = dim_out
        self.K = K

        self.center = np.linspace(0, 1, K, dtype=np.float32)
        self.weight = np.zeros((self.K, self.dim), dtype=np.float32)
        self.weight_var = np.ones((self.K, self.dim), dtype=np.float32)

    @staticmethod
    def kernel_func(t, c):
        return np.exp(-((t - c) * 10.) ** 2)

    def train(self, traj):
        """
        :param traj: [n_batch, n_t, n_dim]
        :return: None
        """
        n_t = traj.shape[1]
        t = np.linspace(0, 1, n_t, dtype=np.float32)
        # wc:N x K; w:K x D;
        wc = self.kernel_func(
            np.tile(t[:, np.newaxis], (1, self.K)), np.tile(self.center, (n_t, 1)))
        wc /= wc.sum(1, keepdims=True)
        w = []
        self.weight *= 0
        self.weight_var *= 0
        for _traj in traj:
            wi = np.linalg.solve(wc.T @ wc, wc.T @ _traj)
            w.append(wi)
            self.weight += wi
        self.weight /= float(traj.shape[0])
        for wi in w:
            self.weight_var += (wi - self.weight)**2
        self.weight_var /= float(traj.shape[0])

    def generate(self, nt=100, fix=False):
        t = np.linspace(0, 1, nt, dtype=np.float32)
        wc = self.kernel_func(
            np.tile(t[:, np.newaxis], (1, self.K)), np.tile(self.center, (nt, 1)))
        wc /= wc.sum(1, keepdims=True)

        if fix:
            w = self.weight
        else:
            w = np.random.normal(self.weight, np.sqrt(self.weight_var))

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


def main():
    pi = np.ones((4,), dtype=np.float32) * 0.25
    mean = np.ones((5, 10, 4), dtype=np.float32)
    var = np.ones((5, 10, 4), dtype=np.float32)

    def log_pwc(w, c):
        return log(w - mean@ c) + np.log(pi[np.argmax(c, axis=1)])


if __name__ == "__main__":
    t = np.linspace(0, 1, 50, dtype=np.float32)
    tau1 = np.vstack([-.25 * t, (1 * t)**.25]).T
    tau2 = np.vstack([-0.1 * t, (1 * t)**.25]).T
    tau3 = np.vstack([0.1 * t, (1 * t)**.25]).T
    tau4 = np.vstack([.25 * t, (1 * t)**.25]).T
    tau = np.vstack([np.tile(tau1, (100, 1, 1)), np.tile(tau2, (100, 1, 1)),
                     np.tile(tau3, (100, 1, 1)), np.tile(tau4, (100, 1, 1))])

    tau += np.random.normal(0., 0.1, tau.shape) * \
        np.sin(t * np.pi).reshape(1, 50, 1)

    rbf = RBF(2)
    rbf.train(tau)
    taug = rbf.generate()

    plt.figure()
    for taui in tau:
        plt.plot(taui[:, 0], taui[:, 1], '-.')
    plt.plot(taug[:, 0], taug[:, 1])

    plt.show()
