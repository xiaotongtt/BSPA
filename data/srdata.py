import os

from data import common

import numpy as np
import scipy.misc as misc
import imageio

import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, name = '', train=True, benchmark=False):
        self.args = args
        self.train = train
        # self.split = 'train' if train else 'test'
        self.name = name
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        self.skip_num=5
        self._set_filesystem(args.dir_data)
        # self._set_filesystem_normal(args.dir_data)
        self.class_dict = dict()
        self.class_weight = []
        self.sum_weight = 0.0

        def _load_bin():
            self.images_hr = np.load(self._name_hrbin())
            self.images_lr = [
                np.load(self._name_lrbin(s)) for s in self.scale
            ]

        if args.ext == 'img' or benchmark:
            self.images_hr, self.images_lr = self._scan()   #get list of filename hr lr
            # self.images_hr_normal, self.images_lr_normal = self._scan_normal()
            # self.images_hr_inverse, self.images_lr_inverse = self._scan_inverse()
        elif args.ext.find('sep') >= 0:
            # modify by xt
            self.images_hr, self.images_lr = self._scan()
            # self.images_hr_normal, self.images_lr_normal = self._scan_normal()
            # self.images_hr_inverse, self.images_lr_inverse = self._scan_inverse()
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_hr:
                    # hr = misc.imread(v)
                    hr = imageio.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
                for si, s in enumerate(self.scale):
                    for v in self.images_lr[si]:
                        # lr = misc.imread(v)
                        lr = imageio.imread(v)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr)

            self.images_hr = [
                v.replace(self.ext, '.npy') for v in self.images_hr
            ]
            self.images_lr = [
                [v.replace(self.ext, '.npy') for v in self.images_lr[i]]
                for i in range(len(self.scale))
            ]

        elif args.ext.find('bin') >= 0:
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a binary file')
                _load_bin()
            except:
                print('Preparing a binary file')
                bin_path = os.path.join(self.apath, 'bin')
                if not os.path.isdir(bin_path):
                    os.mkdir(bin_path)

                list_hr, list_lr = self._scan()
                hr = [misc.imread(f) for f in list_hr]
                np.save(self._name_hrbin(), hr)
                del hr
                for si, s in enumerate(self.scale):
                    lr_scale = [misc.imread(f) for f in list_lr[si]]
                    np.save(self._name_lrbin(s), lr_scale)
                    del lr_scale
                _load_bin()
        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def _name_lrbin(self, scale):
        raise NotImplementedError

    # def __getitem__(self, idx):
    #     lr, hr, filename = self._load_file(idx)
    #     lr, hr = self._get_patch(lr, hr)
    #     lr, hr = common.set_channel([lr, hr], self.args.n_colors)
    #     lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
    #     return lr_tensor, hr_tensor, filename, self.idx_scale

    def __len__(self):
        return len(self.images_hr)


    def _get_index(self, idx):
        # print(idx)
        return idx

    # def _get_index_normal(self, idx):
    #     # print(idx)
    #     return idx
    #
    # def _get_index_inverse(self, idx):
    #     # print(idx)
    #     return idx

    # def _load_file(self, idx, sampling):
    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]
        # e = time.time()
        #print("oooo", e - s)
        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            #lr = misc.imread(lr)
            #hr = misc.imread(hr)
            lr = imageio.imread(lr)
            hr = imageio.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            # s = time.time()
            filename = hr
            lr = np.load(lr)
            hr = np.load(hr)
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return lr, hr, filename

    def _get_patch(self, lr, hr, sampling):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            if sampling == "normal":
                lr, hr = common.get_patch(
                    lr, hr, patch_size, scale, multi_scale=multi_scale
                )
            lr, hr = common.augment([lr, hr])
            lr = common.add_noise(lr, self.args.noise)
        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

