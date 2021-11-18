#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File  : layers.py
@Author: Penistrong
@Email : chen18296276027@gmail.com
@Date  : 2021-11-17 周三 16:42:57
@Desc  : 常用的网络层结构，抽出封装
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import filter2d

class Blur(nn.Module):
    """
    使用空间滤波核进行图片平滑

    Kernel:
    torch.Tensor([[[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]]
    """
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)
