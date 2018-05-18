#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : robot.py
# Purpose :
# Creation Date : 26-04-2018
# Last Modified : Fri 27 Apr 2018 01:24:35 AM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import socket
import time

import numpy as np


class RemoteRobot(object):
    def __init__(self, ip='127.0.0.1', port='6776', buffer_size=65536*256):
        self.host_ip = ip
        self.host_port = port
        self.buf_size = buffer_size

    def connect(self, timeout=100000):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if timeout:
            self.socket.settimeout(timeout)
        while timeout:
            try:
                self.socket.connect((self.host_ip, self.host_port))
                print('Sucessfuly connect to {}:{}'.format(
                    self.host_ip, self.host_port))
                return True 
            except:
                print('Failed to connect, retrying...')
                time.sleep(1)
                timeout -= 1
        return False

    def close(self):
        try:
            self.socket.close()
            print('Bye')
        except:
            print('Failed to close, force to do')

    def send(self, s):
        try:
            self.socket.send('{}'.format(s).encode())
        except:
            print('Failed to send specified sequences')

    def __exit__(self):
        self.socket.close()

    def wait(self):
        try:
            s = self.socket.recv(self.buf_size).decode()
            if s == 'ok' or s == 'OK':
                return True
            else:
                return False
        except:
            return False

    def formatCmd(self, *args):
        raise NotImplementedError

    def cartesianTransform(self, x, y, z):
        raise NotImplementedError

    def quaternionTransform(self, x, y, z, w):
        raise NotImplementedError

    def gotoPos(self, pos):
        raise NotImplementedError

    def followTraj(self, traj):
        raise NotImplementedError


class RemoteBaxter(RemoteRobot):
    def __init__(self, *args, **kwargs):
        super(RemoteBaxter, self).__init__(*args, **kwargs)

    def formatCmd(self, *args):
        return '#'.join([str(i) for i in args])

    def _cartesianTransform(self, x, y, z):
        nz = z
        A = np.array([
            [-2.01711039e-02,  9.15804999e-01,  7.20279308e-01],
            [-8.11264031e-01, -4.92506785e-04, -2.51673001e-01],
        ], dtype=np.float32)
        nx, ny = A @ np.array([x, y, 1.]).T
        return nx, ny, nz

    def _quaternionTransform(self, x, y, z, w):
        return x, y, z, w

    def cartesianTransform(self, traj):
        traj = traj.copy()
        for ind, p in enumerate(traj):
            traj[ind] = self._cartesianTransform(*p)
        return traj 

    def quaternionTransform(self, traj):
        traj = traj.copy()
        for ind, p in enumerate(traj):
            traj[ind] = self._quaternionTransform(*p)
        return traj 

    def gotoPose(self, position, orintation):
        x, y, z = position
        qx, qy, qz, qw = orintation
        x, y, z = self._cartesianTransform(x, y, z)
        qx, qy, qz, qw = self._quaternionTransform(qx, qy, qz, qw)
        self.send(self.formatCmd('arm', 'right', x, y, z, qx, qy, qz, qw))
        return self.wait(), (x, y, z)

    def followTraj(self, position_traj, orintation_traj, continuous=True):
        if continuous:
            position_traj = self.cartesianTransform(position_traj) 
            orintation_traj = self.quaternionTransform(orintation_traj)
            N = len(position_traj)
            traj = list(np.hstack([position_traj, orintation_traj]).reshape(-1))
            print(traj)
            self.send(self.formatCmd('arm_cont', 'right',N, *traj))
            return self.wait(), position_traj[-1]  
        else:
            for ind, (position, orintation) in enumerate(zip(position_traj, orintation_traj)):
                position = list(position)
                orintation = list(orintation)
                res = self.gotoPose(position, orintation)
                if not res:
                    print('Exec {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} [FAILED] {}/{}'.format(
                        *(position + orintation),
                        ind, len(position_traj)
                    ))
                    break
                else:
                    print('Exec {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} [SUCCESS] {}/{}'.format(
                        *(position + orintation),
                        ind, len(position_traj)
                    ))
            else:
                return True 
            return False


if __name__ == '__main__':
    pass
