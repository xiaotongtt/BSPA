import os
import math
import time
import datetime
from functools import reduce

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc as misc

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from collections.abc import Iterable
from math import log, cos, pi, floor
from torch.optim.lr_scheduler import _LRScheduler

from multiprocessing import Process
from multiprocessing import Queue
import imageio
import pickle
import cv2


class CyclicCosineDecayLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 init_interval,
                 min_lr,
                 restart_multiplier=None,
                 restart_interval=None,
                 restart_lr=None,
                 last_epoch=-1):
        """
        Initialize new CyclicCosineDecayLR object
        :param optimizer: (Optimizer) - Wrapped optimizer.
        :param init_interval: (int) - Initial decay cycle interval.
        :param min_lr: (float or iterable of floats) - Minimal learning rate.
        :param restart_multiplier: (float) - Multiplication coefficient for increasing cycle intervals,
            if this parameter is set, restart_interval must be None.
        :param restart_interval: (int) - Restart interval for fixed cycle intervals,
            if this parameter is set, restart_multiplier must be None.
        :param restart_lr: (float or iterable of floats) - Optional, the learning rate at cycle restarts,
            if not provided, initial learning rate will be used.
        :param last_epoch: (int) - Last epoch.
        """

        if restart_interval is not None and restart_multiplier is not None:
            raise ValueError("You can either set restart_interval or restart_multiplier but not both")

        if isinstance(min_lr, Iterable) and len(min_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(min_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(min_lr), len(optimizer.param_groups)))

        if isinstance(restart_lr, Iterable) and len(restart_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(restart_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(restart_lr), len(optimizer.param_groups)))

        if init_interval <= 0:
            raise ValueError("init_interval must be a positive number, got {} instead".format(init_interval))

        group_num = len(optimizer.param_groups)
        self._init_interval = init_interval
        self._min_lr = [min_lr] * group_num if isinstance(min_lr, float) else min_lr
        self._restart_lr = [restart_lr] * group_num if isinstance(restart_lr, float) else restart_lr
        self._restart_interval = restart_interval
        self._restart_multiplier = restart_multiplier
        super(CyclicCosineDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self._init_interval:
            return self._calc(self.last_epoch,
                              self._init_interval,
                              self.base_lrs)

        elif self._restart_interval is not None:
            cycle_epoch = (self.last_epoch - self._init_interval) % self._restart_interval
            lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
            return self._calc(cycle_epoch,
                              self._restart_interval,
                              lrs)

        elif self._restart_multiplier is not None:
            n = self._get_n(self.last_epoch)
            sn_prev = self._partial_sum(n)
            cycle_epoch = self.last_epoch - sn_prev
            interval = self._init_interval * self._restart_multiplier ** n
            lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
            return self._calc(cycle_epoch,
                              interval,
                              lrs)
        else:
            return self._min_lr

    def _calc(self, t, T, lrs):
        return [min_lr + (lr - min_lr) * (1 + cos(pi * t / T)) / 2
                for lr, min_lr in zip(lrs, self._min_lr)]

    def _get_n(self, epoch):
        a = self._init_interval
        r = self._restart_multiplier
        _t = 1 - (1 - r) * epoch / a
        return floor(log(_t, r))

    def _partial_sum(self, n):
        a = self._init_interval
        r = self._restart_multiplier
        return a * (1 - r ** n) / (1 - r)

class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = '../experiment/' + args.save
        else:
            self.dir = '../experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            # misc.imsave('{}{}.png'.format(filename, p), ndarr)
            imageio.imsave('{}{}.png'.format(filename, p), ndarr)


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)
    '''
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    '''
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse), mse


def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    elif args.decay_type == 'cosine':
        scheduler = lrs.CosineAnnealingLR(
            my_optimizer,
            T_max=1000,
            eta_min=0.00001,
            last_epoch=-1
        )
    elif args.decay_type == 'cosine2':
        # T_period = [250000, 250000, 250000, 250000]
        # restarts = [250000, 500000, 750000]
        T_period = [1000]
        restarts = [1000]
        restart_weights = [1]
        scheduler = CosineAnnealingLR_Restart(
            my_optimizer,
            T_period,
            eta_min=1e-7,
            restarts=restarts,
            weights=restart_weights
        )

    elif args.decay_type == 'cycle':

        scheduler = CyclicCosineDecayLR(
            my_optimizer,
            init_interval=20,
            min_lr=1e-9,
            restart_interval=20,
            restart_lr=2e-4)

    return scheduler


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

def crop(img, crop_sz, step):
    b, c, h, w = img.shape
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list=[]
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            crop_img = img[:, :, x:x + crop_sz, y:y + crop_sz]
            lr_list.append(crop_img)
    new_h=x + crop_sz # new height after crop
    new_w=y + crop_sz # new width  after crop
    return lr_list, num_h, num_w, new_h, new_w

def combine(sr_list, num_h, num_w, h, w, patch_size, step):
    index=0
    sr_img = torch.zeros((1, 3, h, w)).to(sr_list[0].device)
    for i in range(num_h):
        for j in range(num_w):
            sr_img[:, :, i*step:i*step+patch_size, j*step:j*step+patch_size] += sr_list[index]
            index+=1

    # mean the overlap region
    for j in range(1,num_w):
        sr_img[:, :, :, j*step:j*step+(patch_size-step)]/=2
    for i in range(1,num_h):
        sr_img[:, :, i*step:i*step+(patch_size-step), :]/=2

    return sr_img

def seamless_combine(sr_list, num_h, num_w, h, w, patch_size, step):
    index=0
    sr_img = torch.zeros((1, 3, h, w)).to(sr_list[0].device)
    border = [1,1,1,1]
    for i in range(num_h):
        if i == 0:  # top side
            border[1] = 0
            border[3] = 1
        elif i < num_h-1: # middle
            border[1] = 1
            border[3] = 1
        else: # bottom side
            border[1] = 1
            border[3] = 0
        for j in range(num_w):
            if j == 0:  # left side
                border[0] = 0
                border[2] = 1
            elif j < num_w-1: # middle
                border[0] = 1
                border[2] = 1
            else: # right side
                border[0] = 1
                border[2] = 0
            sr_img[:, :, i*step:i*step+patch_size, j*step:j*step+patch_size] += fade_border(sr_list[index], patch_size-step, border)
            index+=1

    return sr_img

def fade_border(img, border_size, border=[1,1,1,1]):
    '''
    gradually fade the border while maintain the center,
    "border" indicates fading at [left, top, right, bottom]
    '''
    if border_size > 0: # overlap
        if border[0] != 0: # left border
            img[:, :, :border_size] *= torch.linspace(0, 1, border_size).unsqueeze(0).unsqueeze(0)
        if border[1] != 0: # top border
            img[:, :border_size, :] *= torch.linspace(0, 1, border_size).unsqueeze(0).transpose(1,0).unsqueeze(0)
        if border[2] != 0: # right border
            img[:, :, -border_size:] *= torch.linspace(1, 0, border_size).unsqueeze(0).unsqueeze(0)
        if border[3] != 0: # bottom border
            img[:, -border_size:,:] *= torch.linspace(1, 0, border_size).unsqueeze(0).transpose(1,0).unsqueeze(0)
        return img
    else: # non-overlap
        return img


def crop_parallel(img, crop_sz, step):
    b, c, h, w = img.shape
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list=torch.Tensor().to(img.device)
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            crop_img = img[:, :, x:x + crop_sz, y:y + crop_sz]
            lr_list = torch.cat([lr_list, crop_img])
    new_h=x + crop_sz # new height after crop
    new_w=y + crop_sz # new width  after crop
    return lr_list, num_h, num_w, new_h, new_w

def combine_parallel(sr_list, num_h, num_w, h, w, patch_size, step):
    index=0
    sr_img = torch.zeros((1, 3, h, w)).to(sr_list.device)
    for i in range(num_h):
        for j in range(num_w):
            sr_img[:, :, i*step:i*step+patch_size, j*step:j*step+patch_size] += sr_list[index]
            index+=1

    # mean the overlap region
    for j in range(1,num_w):
        sr_img[:, :, :, j*step:j*step+(patch_size-step)]/=2
    for i in range(1,num_h):
        sr_img[:, :, i*step:i*step+(patch_size-step), :]/=2

    return sr_img


def add_mask(sr_img, scale, num_h, num_w, h, w, patch_size, step, exit_index, show_number=True):
    # white and 7-rainbow
    # color_list = [(255,255,255),(255,0,0),(255,165,0),(255,255,0),(0,255,0),(0,127,255),(0,0,255),(139,0,255)]
    color_list = [(255,255,255),(255,225,0),(255,165,0),(240,0,0),(0,255,0),(0,127,255),(0,0,255),(139,0,255)]

    idx = 0
    sr_img = sr_img.squeeze().permute(1,2,0).numpy() # (H,W,C)
    mask = np.zeros((sr_img.shape), 'float32')
    for i in range(num_h):
        for j in range(num_w):
            bbox = [j * step + 2*scale,
                     i * step + 2*scale,
                     j * step + patch_size - (2*scale+1),
                     i * step + patch_size - (2*scale+1)]  # xl,yl,xr,yr

            color = color_list[int(exit_index[idx])]
            cv2.rectangle(mask, (bbox[0]+1, bbox[1]+1), (bbox[2]-1, bbox[3]-1), color=color, thickness=-1)
            cv2.putText(mask, '{}'.format(int(exit_index[idx]+1)),
                        (bbox[0]+4*scale, bbox[3]-4*scale), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
            idx += 1

    # add_mask
    alpha = 0.7
    beta = 1 - alpha
    gamma = 0
    sr_mask = cv2.addWeighted(sr_img, alpha, mask, beta, gamma)
    sr_mask = torch.from_numpy(sr_mask).permute(2,0,1).unsqueeze(0)

    return sr_mask
