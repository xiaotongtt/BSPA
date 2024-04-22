import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.misc as misc
import os
###rgb统计
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import torch

file_path = r'/data/xxx/dataset/DIV2K/'
lr_path = os.path.join(file_path, 'DIV2K_train_LR_bicubic_crop/X4')#低分子文件目录
hr_path = os.path.join(file_path, 'DIV2K_train_HR_crop')#高分子文件目录

lr_files = tqdm(os.listdir(lr_path))


i = 0
simple, hard_all, hard = 0, 0, 0
source = r'subimages_classes/'

division = {}

for i in range(11):
    division[str(i)] = 0
std = []
i = 0

for lr_name in lr_files:
    # if not path.startswith('0018x4_s049'):
    #     continue
    i += 1
    # if i > 10000:
    #     break

    if lr_name.endswith('npy'):
        print(lr_name)
        continue

    hr_name = lr_name.replace('x4', '')
    lr_data = misc.imread(os.path.join(lr_path, lr_name))
    hr_data = misc.imread(os.path.join(hr_path, hr_name))
    cal_data = cv2.imread(os.path.join(lr_path, lr_name))

    cal_data = torch.from_numpy(cal_data / 255.0)
    cal_data = torch.std(cal_data)
    std.append(cal_data.item())


    for key in division.keys():
        if cal_data < (float(key) + 1) * 0.05:
            division[key] += 1
            save_file = file_path + source + key
            if not os.path.isdir(save_file):
                os.mkdir(save_file)
            np.save(os.path.join(save_file + '/LR', lr_name.replace('png', 'npy')), lr_data)
            np.save(os.path.join(save_file + '/HR', hr_name.replace('png', 'npy')), hr_data)
            break



def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def save_data(data_list, name):
    with open(name, 'w') as f:
        for data in data_list:
            f.write(str(data) + '\n')

def cal_hist(data, style):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(data, bins=10, color='blue', alpha=0.7)
    plt.savefig('./' + style + '.jpg')
    plt.show()






