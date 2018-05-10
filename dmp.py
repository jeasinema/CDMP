import torch
import numpy as np
import scipy.integrate as ode


class DMP(object):
    def __init__(self, config):
        self.cfg = config
        self.k_func = self._k_func_exp
        self.nk = self.cfg.number_of_MP_kernels
        self.c = np.linspace(0, 1, self.nk, dtype=np.float32, endpoint=False)
        self.r = 1. / self.nk
        self.K = self.cfg.DMP_wn * self.cfg.DMP_wn
        self.B = 2. * self.cfg.DMP_wn * self.cfg.DMP_xi
        self.p0 = None
        self.v0 = None
        self.t0 = None

    @staticmethod
    def _k_func_exp(t, c, r):
        return np.exp(-((t - c) * 10.) ** 2)

    @staticmethod
    def _k_func_tri(t, c, r):
        return np.maximum(0., 1. - np.abs((t - c) / r))

    # xt: (t_samples, n_dim), t: (t_samples,)
    @staticmethod
    def _num_diff2(xt, t, d4mode=True):
        # D-2 mode:
        # f0 = [ 1, (t0-t1), (t0-t1)^2/2 ]   f1
        # f1   [ 1,       0,           0 ] x f1'
        # f2   [ 1, (t2-t1), (t2-t1)^2/2 ]   f1''
        # D-4 mode:
        # f0   [ 1, (t0-t2), (t0-t2)^2/2, (t0-t2)^3/6, (t0-t2)^4/24 ]   f2
        # f1   [ 1, (t1-t2), (t1-t2)^2/2, (t1-t2)^3/6, (t1-t2)^4/24 ]   f2'
        # f2 = [ 1,       0,           0,           0,            0 ] x f2''
        # f3   [ 1, (t3-t2), (t3-t2)^2/2, (t3-t2)^3/6, (t3-t2)^4/24 ]   f2(3)
        # f4   [ 1, (t4-t2), (t4-t2)^2/2, (t4-t2)^3/6, (t4-t2)^4/24 ]   f2(4)
        if len(xt) < 5 and d4mode:
            print("Error, 2nd diff cannot be calculate by less than 5 points")
            d4mode = False

        if len(xt) < 3 and not d4mode:
            print("Error, 2nd diff cannot be calculate by less than 3 points")
            return np.zeros_like(xt, dtype=np.float32)

        diff1 = np.zeros_like(xt, dtype=np.float32)
        diff2 = np.zeros_like(xt, dtype=np.float32)

        if d4mode:
            t10, t20, t30, t40 = t[1] - t[0], t[2] - t[0], t[3] - t[0], t[4] - t[0]
            A = np.asarray(((1., 0., 0., 0., 0.),
                            (1., t10, t10 ** 2 / 2., t10 ** 3 / 6., t10 ** 4 / 24.),
                            (1., t20, t20 ** 2 / 2., t20 ** 3 / 6., t20 ** 4 / 24.),
                            (1., t30, t30 ** 2 / 2., t30 ** 3 / 6., t30 ** 4 / 24.),
                            (1., t40, t40 ** 2 / 2., t40 ** 3 / 6., t40 ** 4 / 24.)), dtype=np.float32)
            _, diff1[0, :], diff2[0, :], _, _ = np.linalg.solve(A, xt[:5])
            ##
            t01, t21, t31, t41 = t[0] - t[1], t[2] - t[1], t[3] - t[1], t[4] - t[1]
            A = np.asarray(((1., t01, t01 ** 2 / 2., t01 ** 3 / 6., t01 ** 4 / 24.),
                            (1., 0., 0., 0., 0.),
                            (1., t21, t21 ** 2 / 2., t21 ** 3 / 6., t21 ** 4 / 24.),
                            (1., t31, t31 ** 2 / 2., t31 ** 3 / 6., t31 ** 4 / 24.),
                            (1., t41, t41 ** 2 / 2., t41 ** 3 / 6., t41 ** 4 / 24.)), dtype=np.float32)
            _, diff1[1, :], diff2[1, :], _, _ = np.linalg.solve(A, xt[:5])

            for i in range(2, len(xt) - 2):
                t02, t12, t32, t42 = t[i - 2] - t[i], t[i - 1] - t[i], t[i + 1] - t[i], t[i + 2] - t[i]
                A = np.asarray(((1., t02, t02 ** 2 / 2., t02 ** 3 / 6., t02 ** 4 / 24.),
                                (1., t12, t12 ** 2 / 2., t12 ** 3 / 6., t12 ** 4 / 24.),
                                (1., 0., 0., 0., 0.),
                                (1., t32, t32 ** 2 / 2., t32 ** 3 / 6., t32 ** 4 / 24.),
                                (1., t42, t42 ** 2 / 2., t42 ** 3 / 6., t42 ** 4 / 24.)), dtype=np.float32)
                _, diff1[i, :], diff2[i, :], _, _ = np.linalg.solve(A, xt[i - 2:i + 3])

            t41, t31, t21, t01 = t[-5] - t[-2], t[-4] - t[-2], t[-3] - t[-2], t[-1] - t[-2]
            A = np.asarray(((1., t41, t41 ** 2 / 2., t41 ** 3 / 6., t41 ** 4 / 24.),
                            (1., t31, t31 ** 2 / 2., t31 ** 3 / 6., t31 ** 4 / 24.),
                            (1., t21, t21 ** 2 / 2., t21 ** 3 / 6., t21 ** 4 / 24.),
                            (1., 0., 0., 0., 0.),
                            (1., t01, t01 ** 2 / 2., t01 ** 3 / 6., t01 ** 4 / 24.)), dtype=np.float32)
            _, diff1[-2, :], diff2[-2, :], _, _ = np.linalg.solve(A, xt[-5:])
            ##
            t40, t30, t20, t10 = t[-5] - t[-1], t[-4] - t[-1], t[-3] - t[-1], t[-2] - t[-1]
            A = np.asarray(((1., t40, t40 ** 2 / 2., t40 ** 3 / 6., t40 ** 4 / 24.),
                            (1., t30, t30 ** 2 / 2., t30 ** 3 / 6., t30 ** 4 / 24.),
                            (1., t20, t20 ** 2 / 2., t20 ** 3 / 6., t20 ** 4 / 24.),
                            (1., t10, t10 ** 2 / 2., t10 ** 3 / 6., t10 ** 4 / 24.),
                            (1., 0., 0., 0., 0.)), dtype=np.float32)
            _, diff1[-1, :], diff2[-1, :], _, _ = np.linalg.solve(A, xt[-5:])

        else:
            t10, t20 = t[1] - t[0], t[2] - t[0]
            A = np.asarray(((1., 0., 0.),
                            (1., t10, t10 ** 2 / 2.),
                            (1., t20, t20 ** 2 / 2.)), dtype=np.float32)
            _, diff1[0, :], diff2[0, :] = np.linalg.solve(A, xt[:3])

            for i in range(1, len(xt) - 1):
                t01, t21 = t[i - 1] - t[i], t[i + 1] - t[i]
                A = np.asarray(((1., t01, t01 ** 2 / 2.),
                                (1., 0., 0.),
                                (1., t21, t21 ** 2 / 2.)), dtype=np.float32)
                _, diff1[i, :], diff2[i, :] = np.linalg.solve(A, xt[i - 1:i + 2])

            t20, t10 = t[-3] - t[-1], t[-2] - t[-1]
            A = np.asarray(((1., t20, t20 ** 2 / 2.),
                            (1., t10, t10 ** 2 / 2.),
                            (1., 0., 0.)), dtype=np.float32)
            _, diff1[-1, :], diff2[-1, :] = np.linalg.solve(A, xt[-3:])

        return diff1, diff2

    @staticmethod
    def _ode_func(x, t, kfunc, c, r, w, ep, K, B):
        # return dx(t) = [dx(t); d2x(t)]
        # d2x(t) = -k x(t) - b dx(t) + f_ctl(t)
        dimo = x.shape[0] // 2
        x, dx = x[:dimo], x[dimo:]
        f = kfunc(t, c, r)[:, np.newaxis]
        # now w * f=(nk, n_dim)
        f = (w * f / (f.sum() + 1e-5)).sum(0)
        d2x = -K * (x - ep) - B * dx + f
        return np.concatenate((dx, d2x), axis=0)

    # from (n_batch, t_samples, dim_in) to kernel (n_batch, n_k, dim_k)
    # target can be None for using the last point, or (n_batch, dim_in)
    # t can be None for using equally sliced from 0~1, or (t_samples,) or (n_batch, t_samples) !! incremental only !!
    def calculate(self, traj, target=None, t=None):
        if len(traj.shape) != 3:
            print("What ?!")
            exit(-1)

        if target is None:
            target = traj[:, -1, :]
        elif len(target.shape) != 2:
            print("What ?!")
            exit(-1)

        N = traj.shape[0]
        nt = traj.shape[1]
        dimo = traj.shape[-1]

        if t is None:
            t = np.tile(np.linspace(0, 1, nt, dtype=np.float32), (N, 1))
        elif len(t.shape) == 1:
            t = np.tile(t, (N, 1))

        # Now, t=(n_batch, t_samples), traj=(n_batch, t_samples, dim_in), target=(n_batch, dim_in)

        ret_w = np.zeros((N, self.nk, dimo), dtype=np.float32)
        ret_t = np.zeros((N, dimo), dtype=np.float32)

        for i, (_traj, _ep, _t) in enumerate(zip(traj, target, t)):
            diff1, diff2 = self._num_diff2(_traj, _t)
            ft = self.K * (_traj - _ep) + self.B * diff1 + diff2

            # wc: N x K; w: K x D;
            wc = self.k_func(np.tile(_t[:, np.newaxis], (1, self.nk)), np.tile(self.c, (nt, 1)), self.r)

            ret_w[i] = np.linalg.solve(wc.T @ wc, wc.T @ (ft * wc.sum(1, keepdims=True)))
            ret_t[i] = _ep

        return ret_w, ret_t

    # from (n_batch, n_k, dim_out) to (n_batch, t_samples, dim_out)
    # target: (n_batch, dim_out), p0, v0: (n_batch, dim_out) or None for use last state, t0: (n_batch)
    # t_samples can be None for using cfg, int n for n steps, list of query points in 0.~1.
    def generate(self, w, target, t_samples=None, p0=None, v0=None, t0=None, init=False):
        if t_samples is None:
            t_samples = self.cfg.number_time_samples

        if hasattr(t_samples, '__iter__'):
            t = np.asarray(t_samples, dtype=np.float32)
        else:
            t = np.linspace(0, 1, t_samples, dtype=np.float32)

        N = w.shape[0]
        dimo = w.shape[-1]

        if len(t.shape) == 1:
            t = np.tile(t, (N, 1))

        if init:
            v0 = np.zeros_like(p0, dtype=np.float32)
            t0 = np.zeros((N, 1), dtype=np.float32)

        if p0 is None:
            p0 = self.p0
            if p0 is None:
                print("DMP Error: p0 == None but system has not been queried yet")
        if v0 is None:
            v0 = self.v0
        if t0 is None:
            t0 = self.t0

        ret_traj = np.zeros((N, t.shape[1], dimo), dtype=np.float32)

        for i, (_w, _ep, _t, _p0, _v0, _t0) in enumerate(zip(w, target, t, p0, v0, t0)):
            x = np.concatenate((_p0, _v0))
            x_query = ode.odeint(self._ode_func, x, np.insert(_t, 0, _t0),
                                 (self.k_func, self.c, self.r, _w, _ep, self.K, self.B))
            ret_traj[i] = x_query[1:, :dimo]

        self.t0 = t[-1]
        self.p0, self.v0 = ret_traj[:, -1, :dimo], ret_traj[:, -1, dimo:]

        return ret_traj
