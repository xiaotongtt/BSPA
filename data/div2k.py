import os

from data import common
from data import srdata
import random
import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class DIV2K(srdata.SRData):
    def __init__(self, args, train=True):
        super(DIV2K, self).__init__(args, train)
        self.repeat = 1 # args.test_every // (args.n_train // args.batch_size)#20
        self.args = args

    def _scan_normal(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train  # 800
        else:
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val

        for i in range(idx_begin + 1, idx_end + 1):
            filename = '{:0>4}'.format(i)
            list_hr.append(os.path.join(self.dir_hr_normal, filename + self.ext))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr_normal,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))

        return list_hr, list_lr


    def _scan_inverse(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train
        else:
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val

        # for i in range(idx_begin + 1, idx_end + 1):
        #     filename = '{:0>4}'.format(i)
        #     list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
        #     for si, s in enumerate(self.scale):
        #         list_lr[si].append(os.path.join(
        #             self.dir_lr,
        #             'X{}/{}x{}{}'.format(s, filename, s, self.ext)
        #         ))
        i = 0
        # print(self.class_dict)
        #for id in range(1, 6): 
        for id in range(1, 11):
        #for id in range(1, 11): #sorted(os.listdir(self.class_path)):
            class_id = str(id)
            print(class_id)
            for data_name in os.listdir(os.path.join(self.class_path, class_id)  + '/LR'):  # + '/old'
                # if data_name.endswith('npy.npy') or (not data_name.endswith('npy')):
                #     print(class_id, data_name)
                #     continue
                if not class_id in self.class_dict:
                    self.class_dict[class_id] = []
                self.class_dict[class_id].append(i)

                lr_filepath = os.path.join(os.path.join(self.class_path, class_id)  + '/LR', data_name)  # + '/old'
                hr_filepath = os.path.join(os.path.join(self.class_path, class_id)  + '/HR', data_name.replace('x4', ''))  # + '/old'
                i += 1
            #print(lr_filepath)
                list_lr[0].append(lr_filepath)
                list_hr.append(hr_filepath)
        # print(len(self.class_dict['0']), len(self.class_dict['1']), len(self.class_dict['2']))
        print(len(list_lr[0]), len(list_hr))
        # rand_list = np.random.randint(0, len(list_hr), size=16000)
        # list_new_hr= []
        # list_new_lr = [[] for _ in self.scale]
        # for i in rand_list:
        #     list_new_hr.append(list_hr[i])
        #     list_new_lr[0].append(list_lr[0][i])
        # #
        # print(list_new_lr[0][0])
        # print(list_new_hr[0])
        # print(os.path.isfile(list_new_hr[0]))
        num_list = [len(self.class_dict[k]) for k in self.class_dict.keys()]
        print(num_list)
        self.class_weight = [(max(num_list) / i) for i in num_list]
        self.sum_weight = sum(self.class_weight)
        print(self.class_weight, self.sum_weight)  # [9.665930773476306, 41.15240013427324, 2.883320005644668, 1.0, 2.6426030911168112, 1.401014822348948, 1.5995537694736568, 30648.25, 1.1348576718352232, 228.292364990689] 30938.022045258855
        return list_hr, list_lr   # 499875

    def _set_filesystem_inverse(self, dir_data):
        self.apath = dir_data + '/DIV2K'
        self.dir_hr = os.path.join(self.apath, 'HR_train_sub')#'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'LR_train_sub')#'DIV2K_train_LR_bicubic')
        self.ext = '.png'
        self.class_path = self.apath + '/subimages_classes_mse_' + self.args.un_factor +'_group10'


    def _set_filesystem_normal(self, dir_data):
        self.apath = dir_data + '/DIV2K'
        self.dir_hr_normal = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr_normal = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        self.ext = '.png'


    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(len(self.class_dict)):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i+1

    def __len__(self):
        if self.train:
            # return len(self.images_hr_normal) * self.repeat
            return len(self.images_hr_inverse)  # [0:1600]
        else:
            # return len(self.images_hr_normal)
            return len(self.images_hr_inverse)

    def _get_index_normal(self, idx):
        if self.train:
            #rint(idx)
            return idx % len(self.images_hr_normal)  # images_hr_normal
        else:
            return idx

    def _get_index_inverse(self, idx):
        if self.train:
            #rint(idx)
            return idx % len(self.images_hr_inverse)  # images_hr_normal
        else:
            return idx

    def __getitem__(self, idx):
        # import time
        # s = time.time()
        # print(self.sample_class_index_by_weight())
        sample_class = self.sample_class_index_by_weight()  # 随机取类别
        sample_indexes = self.class_dict[str(sample_class)]  # 获得类别的样本索引集合
        # print(sample_indexes)
        sample_index = random.choice(sample_indexes)#随机采样反转样本的索引
        # print(sample_index)
        ###正常样本, random sampling
        lr, hr, filename = self._load_file(idx, "normal")
        lr, hr = self._get_patch(lr, hr, "normal")
        # e = time.time()
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        ###反转采样
        meta = dict()
        lr_sample, hr_sample, sample_name = self._load_file(sample_index, "inverse")
        lr_sample, hr_sample = self._get_patch(lr_sample, hr_sample, "inverse")
        lr_sample, hr_sample = common.set_channel([lr_sample, hr_sample], self.args.n_colors)
        lr_sample_tensor, hr_sample_tensor = common.np2Tensor([lr_sample, hr_sample], self.args.rgb_range)
        meta['lr_sample'], meta['hr_sample'] = lr_sample_tensor, hr_sample_tensor
        #print(idx, end='\t')
        # print("hhhhh", e - s)
        return lr_tensor, hr_tensor, filename, self.idx_scale, meta