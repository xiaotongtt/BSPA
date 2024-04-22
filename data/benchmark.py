import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data


class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _set_filesystem(self, dir_data):
        dir_data_path = '/data/xxx/DIV2K/'
        # self.apath = os.path.join(dir_data_path, 'benchmark', self.args.data_test)
        # self.dir_hr = os.path.join(self.apath, 'HR')  # /x4
        # self.dir_lr = os.path.join(self.apath, 'LR_bicubic/')
        self.dir_hr = os.path.join(dir_data_path, 'DIV2K_train_HR_crop/X4')  # /x4
        self.dir_lr = os.path.join(dir_data_path, 'DIV2K_train_LR_bicubic_crop')
        self.ext = '.png'
        #print(self.dir_hr, self.dir_lr)

    # def _set_filesystem_inverse(self, dir_data):
    #     self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
    #     self.dir_hr = os.path.join(self.apath, 'HR')
    #     self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
    #     self.ext = '.png'

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        #print(self.dir_hr, self.dir_lr)
        hr = sorted(os.listdir(self.dir_hr))  # [:10]
        for i in range(len(hr)):
            filename = hr[i]  # 0001_s001.png
            list_hr.append(os.path.join(self.dir_hr, filename))
            #print(filename)
            #lr_filename = filename.split('.')[0]  # 0001x4_s001.png
            lr_filename = filename.split('_')[0] + 'x4_' + filename.split('_')[1]
            #print(lr_filename)
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{}/{}'.format(s, lr_filename)
                    #'X{}/{}x{}{}'.format(s, lr_filename, s, self.ext)
                    # 'x{}/{}{}'.format(s, lr_filename, self.ext)
                ))
                #list_lr[si].append(os.path.join(
                #    self.dir_lr,
                #    'X{}/{}x{}{}'.format(s, lr_filename, s, self.ext)
                #    # 'x{}/{}{}'.format(s, lr_filename, self.ext)
                #))
                #list_lr[si].append(os.path.join(
                #    self.dir_lr,
                #    'x{}/{}'.format(s, filename)
                #))

        return list_hr, list_lr

    # def _scan_inverse(self):
    #     list_hr = []
    #     list_lr = [[] for _ in self.scale]
    #
    #     hr = sorted(os.listdir(self.dir_hr))  # [:10]
    #     for i in range(len(hr)):
    #         filename = hr[i]
    #         list_hr.append(os.path.join(self.dir_hr, filename))
    #         lr_filename = filename.split('.')[0]
    #         for si, s in enumerate(self.scale):
    #             # list_lr[si].append(os.path.join(
    #             #     self.dir_lr,
    #             #     'X{}/{}x{}{}'.format(s, lr_filename, s, self.ext)
    #             #     # 'x{}/{}{}'.format(s, lr_filename, self.ext)
    #             # ))
    #             list_lr[si].append(os.path.join(
    #                 self.dir_lr,
    #                 'x{}/{}'.format(s, filename)
    #             ))
    #
    #     return list_hr, list_lr


    def _get_patch_bench(self, lr, hr):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1

        ih, iw = lr.shape[0:2]
        hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)  # , "normal"
        lr, hr = self._get_patch_bench(lr, hr)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        return lr_tensor, hr_tensor, filename, self.idx_scale