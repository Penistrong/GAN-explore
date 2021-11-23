#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File  : dcdiscriminator.py
@Author: Penistrong
@Email : chen18296276027@gmail.com
@Date  : 2021-11-23 周二 21:07:09
@Desc  : Deep Convolutional Discriminator
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log2

class DCDiscriminator(nn.Module):
    '''
    深度卷积判别器

    Params
    ------
    in_dim -> int : 输入维度
    num_feat -> int : 最后隐含层的特征维数
    img_size -> int : 输入图片尺寸
    '''
    def __init__(self, in_dim=3, num_feat=512, img_size=64):
        super(DCDiscriminator, self).__init__()

        self.in_dim = in_dim
        num_layers = int(log2(img_size) - 2)
        self.layers = nn.ModuleList(
            [nn.Conv2d(in_channels=in_dim,
                       out_channels=int(num_feat / (2 ** (num_layers - 1))),
                       kernel_size=4,
                       stride=2,
                       padding=1,
                       bias=False
                       )]
            +
            [nn.Conv2d(in_channels=int(num_feat / (2 ** (num_layers - i))),
                       out_channels=int(num_feat / (2 ** (num_layers - i - 1))),
                       kernel_size=4,
                       stride=2,
                       padding=1,
                       bias=False
                       )for i in range(1, num_layers)]
        )

        self.conv_out = nn.Conv2d(num_feat, 1, 4, 1, 0, bias=False)
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        if x.shape[1] != self.in_dim:
            x = x[:, :self.in_dim]
        for layer in self.layers:
            x = self.actvn(layer(x))

        out = self.conv_out(x)
        out = out.reshape(batch_size, 1)

        return out