#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File  : neural_renderer.py
@Author: Penistrong
@Email : chen18296276027@gmail.com
@Date  : 2021-11-17 Wed 17:39:26
@Desc  : 2D神经渲染器,用于将前一步3D体积渲染生成的16x16的特征图上采样为更高分辨率的RGB图片
'''
from math import log2

import torch
import torch.nn as nn

from .layers import Blur


class NeuralRenderer(nn.Module):
    '''
    2D神经渲染器

    Params:
        num_feat -> int : 特征数量
        min_feat -> int : 最小特征数
        input_dim -> int : 输入维数，如果与特征数量不同会使用1x1的卷积投影到特征数量上
        out_dim -> int : 输出维数(神经渲染器是用来生成RGB图片的，显然这里默认输出维数是3)
        img_size -> int : 输出图像尺寸，方形图片边长
        use_final_actvn -> bool : 是否使用最后一层的激活函数(论文中是sigmoid)
        use_rgb_skip -> bool : 是否使用RGB跳连,即在每一个空间分辨率上将特征图映射到RGB图片，并通过双线性上采样将前一个输出叠加到下一个输出
        use_norm -> bool : 是否归一化
        upsample_feat -> str : 特征上采样类型，论文中使用最近邻插值上采样(nn=Nearest Nearby)
        upsample_rgb -> str: RGB上采样类型，论文中使用双线性插值上采样(bilinear)
    '''

    def __init__(self,
                 num_feat=128,
                 min_feat=32,
                 input_dim=128,
                 out_dim=3,
                 img_size=64,
                 use_final_actvn=True,
                 use_rgb_skip=True,
                 use_norm=False,
                 upsample_feat="nn",
                 upsample_rgb="bilinear",
                 **kwargs):
        super(NeuralRenderer, self).__init__()
        self.input_dim = input_dim
        self.use_final_actvn = use_final_actvn
        self.use_rgb_skip = use_rgb_skip
        self.use_norm = use_norm

        # 使用log(img_size) - 4计算网络层数
        num_layers=int(log2(img_size) - 4)
        
        assert(upsample_feat in ("nn", "bilinear"))
        if upsample_feat == "nn":
            self.upsample_feat = nn.Upsample(scale_factor=2.)
        elif upsample_feat == "bilinear":
            self.upsample_feat = nn.Sequential(
                nn.Upsample(scale_factor=2,
                            mode='bilinear',
                            align_corners=False),
                Blur()
            )

        assert(upsample_rgb in ("nn", "bilinear"))
        if upsample_rgb == "nn":
            self.upsample_rgb = nn.Upsample(scale_factor=2.)
        elif upsample_rgb == "bilinear":
            self.upsample_rgb = nn.Sequential(
                nn.Upsample(scale_factor=2,
                            mode='bilinear',
                            align_corners=False),
                Blur()
            )

        # 定义卷积层前的输入层
        # 如果给定特征数与输入维数不相同，则使用1x1卷积核将输入维数映射到给定特征数的维数
        if num_feat == input_dim:
            self.conv_in = lambda x: x
        else:
            self.conv_in = nn.Conv2d(in_channels=input_dim,
                                     out_channels=num_feat,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)

        # 特征图进行上采样后使用的3x3-卷积层
        # I_v: H_v x W_v x M_f -> 2^{i}H_v x 2^{i}W_v x 2^{-i}M_f, i=1,2,...,n
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(in_channels=num_feat,
                       out_channels=num_feat // 2,
                       kernel_size=3,
                       stride=1,
                       padding=1)]
            +
            [nn.Conv2d(in_channels=max(num_feat // (2 ** (i + 1)), min_feat),
                       out_channels=max(num_feat // (2 ** (i + 2)), min_feat),
                       kernel_size=3,
                       stride=1,
                       padding=1)
                for i in range(0, num_layers - 1)]
        )

        # 如果使用RGB图片跳连(Style GAN提出的方法)
        # ModuleList的第一个卷积层用于初始特征场直接映射到初始RGB图片
        # 后续卷积层用于特征层不断上采样+卷积后，将每一层的输出进行卷积以便叠加到对应RGB层双线性上采样后的结果上
        if use_rgb_skip:
            self.conv_rgb = nn.ModuleList(
                [nn.Conv2d(in_channels=input_dim,
                           out_channels=out_dim,
                           kernel_size=3,
                           stride=1,
                           padding=1)]
                +
                [nn.Conv2d(in_channels=max(num_feat // (2 ** (i + 1)), min_feat),
                           out_channels=out_dim,
                           kernel_size=3,
                           stride=1,
                           padding=1)
                    for i in range(0, num_layers)]
            )
        else:
            self.conv_rgb = nn.Conv2d(in_channels=max(num_feat // (2 ** num_layers), min_feat),
                                      out_channels=out_dim,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)

        # 如果采用归一化
        if use_norm:
            self.norms = nn.ModuleList([
                nn.InstanceNorm2d(max(num_feat // (2 ** (i + 1)), min_feat))
                for i in range(num_layers)
            ])

        # 2D神经渲染器的激活函数为LeakyReLU
        self.actvn = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # 参考原论文Figure 4的模型结构
    def forward(self, x):
        net = self.conv_in(x)

        # 使用RGB跳连的情况下，初始特征图使用3x3卷积得到RGB图片
        if self.use_rgb_skip:
            # 使用从input_dim -> out_dim的首个卷积层
            rgb = self.upsample_rgb(self.conv_rgb[0](x))

        for idx, layer in enumerate(self.conv_layers):
            hidden = layer(self.upsample_feat(net))
            # 如果指明需要归一化，则对每一个隐藏层都进行一次归一化
            if self.use_norm:
                hidden = self.norms[idx](hidden)
            net = self.actvn(hidden)

            # 使用RGB跳连，将已计算完毕的当前特征图上采样后的卷积结果叠加到对应的rgb上采样层上
            if self.use_rgb_skip:
                rgb = rgb + self.conv_rgb[idx+1](net)
                # 最后一层不需要上采样
                if idx < len(self.conv_layers) - 1:
                    rgb = self.upsample_rgb(rgb)

        # 不使用RGB跳连的话，特征层不断进行上采样+卷积+LeakyReLU后，直接通过卷积层映射到RGB图片
        if not self.use_rgb_skip:
            rgb = self.conv_rgb(net)

        # 最后使用sigmoid作为激活函数
        if self.use_final_actvn:
            rgb = torch.sigmoid(rgb)

        return rgb
