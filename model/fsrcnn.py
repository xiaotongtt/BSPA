# https://github.com/cvlab-yonsei/PISR/blob/master/models/fsrcnn.py

import torch.nn as nn
from collections import OrderedDict
from model import common

#import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def make_model(args, parent=False):
    return FSRCNN(args)


class FSRCNN(nn.Module):
    def __init__(self, args):
        super(FSRCNN, self).__init__()
        scale = args.scale[0]
        n_colors = 3
        d = 56
        s = 12
        m = 4
        fsrcnn_weight_init = False
        self.scale = scale
        self.feature_extraction = []
        self.feature_extraction.append(nn.Sequential(
            nn.Conv2d(in_channels=n_colors,
                out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU()))
        self.shrinking = []
        self.shrinking.append(nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=s,
                kernel_size=1, stride=1, padding=0),
            nn.PReLU()))
        self.mapping = []
        for _ in range(m):
            self.mapping.append(nn.Sequential(
                nn.Conv2d(in_channels=s, out_channels=s,
                    kernel_size=3, stride=1, padding=1),
                nn.PReLU()))
        self.expanding = []
        self.expanding.append(nn.Sequential(
            nn.Conv2d(in_channels=s, out_channels=d,
                    kernel_size=1, stride=1, padding=0),
            nn.PReLU()))
        self.last_layer = []
        self.last_layer.append(nn.Sequential(
            nn.ConvTranspose2d(d, n_colors, kernel_size=9, stride=scale, padding=9//2,
                              output_padding=scale-1))
        )
        self.network = nn.Sequential(
            OrderedDict([
                ('feature_extraction', nn.Sequential(*self.feature_extraction)),
                ('shrinking', nn.Sequential(*self.shrinking)),
                ('mapping', nn.Sequential(*self.mapping)),
                ('expanding', nn.Sequential(*self.expanding)),
                ('last_layer', nn.Sequential(*self.last_layer)),
            ]))

        if fsrcnn_weight_init:
            self.fsrcnn_weight_init()

    def fsrcnn_weight_init(self, mean=0.0, std=0.001):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.network(x)

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
               if isinstance(param, nn.Parameter):
                   param = param.data
               try:
                   own_state[name].copy_(param)
               except Exception:
                   if name.find('tail') >= 0:
                       print('Replace pre-trained upsampler to new one...')
                   else:
                       raise RuntimeError('While copying the parameter named {}, '
                                          'whose dimensions in the model are {} and '
                                          'whose dimensions in the checkpoint are {}.'
                                          .format(name, own_state[name].size(), param.size()))
            elif strict:
               if name.find('tail') == -1:
                   raise KeyError('unexpected key "{}" in state_dict'
                                  .format(name))

        if strict:
           missing = set(own_state.keys()) - set(state_dict.keys())
           if len(missing) > 0:
               raise KeyError('missing keys in state_dict: "{}"'.format(missing))