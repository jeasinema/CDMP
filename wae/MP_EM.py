import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import sklearn.mixture as mx 


class RBF(object):
    def __init__(self, dim_out, Tasks, K=10):
        self.dim = dim_out
        self.K = K
        self.Tasks = Tasks

        self.center = np.linspace(0, 1, K, dtype=np.float32)
        self.weight = np.zeros((self.Tasks, self.K, self.dim), dtype=np.float32)
        self.mix_weight = np.zeros((self.K), dtype=np.float32)
        self.weight_var = np.ones((self.Tasks, self.K, self.dim), dtype=np.float32)
        self.c_cond_x = Net_qc_w(dim_out*K, self.Tasks)
        self.gmm = mx.GaussianMixture(n_components=Tasks, init_params='random')

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
        self.gmm.fit(w.reshape(w.shape[0], -1))

        ## then calculate clustering ##

        # ag_mean & ag_var: CxKxD; ag_pi: C
        ag_mean = torch.autograd.Variable(torch.rand(self.Tasks, self.K, self.dim), requires_grad=True)
        ag_logvar2 = torch.autograd.Variable(torch.ones(self.Tasks, self.K, self.dim) * -1, requires_grad=True)
        ag_weight = torch.autograd.Variable(torch.ones(self.Tasks), requires_grad=True) 
        # ag_weight = torch.autograd.Variable(torch.FloatTensor([400,200,100,100])/800, requires_grad=True) 

        ag_w = torch.autograd.Variable(torch.from_numpy(w)) # N * k * dim

        def pw_normal(x, mean, var2):
            # N*K*dim K*dim K*dim 
            log_pdf = -(x-mean)**2 / (2. * var2) - 0.5*torch.log(var2 * 2. * np.pi) # N*K*dim
            return torch.exp(log_pdf.sum(-1).sum(-1)) # N
        
        def pw_normal_mixture(x, mean, var2, unnormed_pi):
            # N*K*dim T*K*dim T*K*dim T
            x = x.unsqueeze(1)
            log_pdf = -(x-mean)**2 / (2. * var2) - 0.5*torch.log(var2 * 2. * np.pi) # N*T*K*dim
            pi = F.softmax(unnormed_pi) # T
            pdf = torch.exp(log_pdf.sum(-1).sum(-1)) * pi 
            return pdf.sum(-1) # N

        def log_pw():
            # GMM EM
            # cal p(z|x) with bayesian rules
            mle_obj = []
            for i in range(self.Tasks):
                qz = F.softmax(ag_weight) # learnable q(z)
                # qz = F.softmax(torch.autograd.Variable(torch.FloatTensor([1,1,1,1]))) # fixed q(z)

                Q = torch.div(pw_normal(ag_w, ag_mean[i], torch.exp(ag_logvar2[i]))*qz[i], 
                    pw_normal_mixture(ag_w, ag_mean, torch.exp(ag_logvar2), ag_weight))
                log_prob = torch.log(pw_normal(ag_w, ag_mean[i], torch.exp(ag_logvar2[i]))*qz[i])
                mle_obj.append(Q*log_prob) # T*N
            
            pw = torch.stack(mle_obj).sum(0).sum(0)

            # Variational EM 
            # mle_obj = []
            # Q = self.c_cond_x(ag_w.view((ag_w.shape[0],-1))) # N*T
            # for i in range(self.Tasks):
            #     qz = F.softmax(ag_weight) # learnable q(z)
            #     # qz = F.softmax(torch.autograd.Variable(torch.FloatTensor([1,1,1,1]))) # fixed q(z)

            #     log_prob = torch.log(pw_normal(ag_w, ag_mean[i], torch.exp(ag_logvar2[i]))*qz[i])
            #     mle_obj.append(Q[:, i]*log_prob) # T*N
            # 
            # pw = torch.stack(mle_obj).sum(0).sum(0)

            # totally MLE
            # pw = torch.log(pw_normal_mixture(ag_w, ag_mean, torch.exp(ag_logvar2), ag_weight)).sum(0)

            return pw

        plt.ion()
        plt.figure()
        optim = torch.optim.Adam([ag_mean, ag_weight] + [x for x in self.c_cond_x.parameters()])
        for e in range(epoch):
            optim.zero_grad()
            pw = -log_pw()
            pw.backward()
            optim.step()
            # pw = -PRML_EM()

            if e % 100 == 0:
                print("Step %d: log(pw): %f" % (e, -pw[0]))
                print(F.softmax(ag_weight))

                self.weight = ag_mean.data.numpy()
                self.weight_var = np.exp(ag_logvar2.data.numpy() / 2.)
                # self.weight_var = ag_logvar2.data.numpy()
                self.mix_weight = F.softmax(ag_weight).data.cpu().numpy()

                plt.clf()
                # for i in range(1000):
                #     tau = self.generate(task=-1, fix=False)
                #     plt.plot(tau[:, 0], tau[:, 1], '.')

                for i in range(self.Tasks):
                    taug = self.generate(task=i, fix=True)
                    plt.plot(taug[:, 0], taug[:, 1])
                plt.pause(0.01)

        plt.ioff()
        self.weight = ag_mean.data.numpy()
        self.weight_var = np.exp(ag_logvar2.data.numpy() / 2.)
        self.mix_weight = F.softmax(ag_weight).data.cpu().numpy()

    def generate(self, nt=100, task=0, fix=True):
        t = np.linspace(0, 1, nt, dtype=np.float32)
        wc = self.kernel_func(np.tile(t[:, np.newaxis], (1, self.K)), np.tile(self.center, (nt, 1)))
        wc /= wc.sum(1, keepdims=True)

        if task == -1:
            ind = np.where(np.random.multinomial(1, self.mix_weight) != 0)[0][0]
            if fix:
                w = self.weight[ind]
            else:
                w = np.random.normal(self.weight[ind], self.weight_var[ind])
        elif task == -2:
            w = self.gmm.sample()[0].reshape((10, 2))
        else:
            if fix:
                w = self.weight[task]
            else:
                w = np.random.normal(self.weight[task], self.weight_var[task])
        return wc @ w


class Net_qc_w(torch.nn.Module):
    def __init__(self, K, C):
        super(Net_qc_w, self).__init__()
        self.K = K
        self.fc1 = torch.nn.Linear(K, 32)
        # self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, C)
        self.active = torch.nn.Tanh()

    def forward(self, x):
        y = self.active(self.fc1(x))
        # y = self.active(self.fc2(y))
        y = self.fc3(y)
        return F.softmax(y)


if __name__ == "__main__":
    t = np.linspace(0, 1, 50, dtype=np.float32)
    tau1 = np.vstack([-1.*t, (1*t)**.25]).T
    tau2 = np.vstack([-0.3*t, (1*t)**.25]).T
    tau3 = np.vstack([0.3*t, (1*t)**.25]).T
    tau4 = np.vstack([1.*t, (1*t)**.25]).T
    tau = np.vstack([np.tile(tau1, (400, 1, 1)), np.tile(tau2, (100, 1, 1)),
                     np.tile(tau3, (200, 1, 1)), np.tile(tau4, (50, 1, 1))])

    tau += np.random.normal(0., 0.05, tau.shape) * np.sin(t*np.pi).reshape(1, 50, 1)

    # from ipdb import set_trace; set_trace()
    rbf = RBF(dim_out=2, Tasks=4)
    rbf.train(tau, epoch=50000)

    plt.figure()
    for taui in tau:
        plt.plot(taui[:, 0], taui[:, 1], '-.')
    plt.figure()
    for i in range(4):
        taug = rbf.generate(task=i, fix=True)
        plt.plot(taug[:, 0], taug[:, 1])

    plt.show()
