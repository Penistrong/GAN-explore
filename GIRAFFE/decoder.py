#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File  : decoder.py
@Author: Penistrong
@Email : chen18296276027@gmail.com
@Date  : 2021-11-18 周四 13:06:45
@Desc  : 
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import pi


class Decoder(nn.Module):
    '''
    解码器网络

    利用基于神经辐射场(Neural Radiance Fields, NeRFs)的GRAF生成模型(Generative RAdiance Fields)
    辐射场(Radiance Fields)是一个连续函数f
    它将 3D点x 和 视角方向d 的位置编码(Positional Encoding)映射到体积密度σ和3维RGB值c， 这个映射使用MLP实现
    GRAF为了学习NeRFs的隐空间，将物体形状(shape)和外观(appearance)特点的隐向量z_s与z_a添加到MLP层的输入里
    GIRAFFE作者改进了这个方法，用更通用的M_f维特征f替代输出的3维RGB值c

    Params
    ------
    hidden_size -> int : 解码器网络的隐藏层大小
    num_layers -> int : 网络层数
    num_layers_view -> int : 景深(view-depth)网络层数
    skips -> list : 网络中需要添加跳连的网络层编号，用列表装载
    use_view_direction -> bool : 是否需要使用视角方向
    max_freq_pos_enc -> int :  3D坐标位置编码的最大频率
    max_freq_pos_enc_vd -> int : 视角方向位置编码的最大频率
    input_dim -> int : 输入维数
    z_dim -> int : 隐向量z的维数
    out_dim_feat -> int : 模型预测 特征/RGB 的输出维数
    use_final_actvn -> bool : 是否在 特征/RGB 输出层上应用激活函数(论文中使用sigmoid)
    downscale_by -> float : 在位置编码前输入点的缩小因子
    type_pos_enc -> str : 位置编码的类型
    gauss_dim_pos_enc -> int : 对3D点进行高斯位置编码的维数
    gauss_dim_pos_enc_vd -> int : 对视角方向进行高斯位置编码的维数
    gauss_std_pos_enc -> int : 高斯位置编码的标准差
    '''
    def __init__(self,
                 hidden_size=128,
                 num_layers=8,
                 num_layers_view=1,
                 skips=[4],
                 use_view_direction=True,
                 max_freq_pos_enc=10,
                 max_freq_pos_enc_vd=4,
                 z_dim=64,
                 out_dim_feat=128,
                 use_final_actvn=False,
                 downscale_by=2.,
                 type_pos_enc="normal",
                 gauss_dim_pos_enc=10,
                 gauss_dim_pos_enc_vd=4,
                 gauss_std_pos_enc=4,
                 **kwargs):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.num_layers_view = num_layers_view
        self.use_view_direction = use_view_direction
        self.max_freq_pos_enc = max_freq_pos_enc
        self.max_freq_pos_enc_vd = max_freq_pos_enc_vd
        self.skips = skips
        self.downscale_by = downscale_by
        self.z_dim = z_dim
        self.use_final_actvn = use_final_actvn

        # 位置编码分为 普通 和 高斯 两种方式
        assert(type_pos_enc in ("normal", "gauss"))
        self.type_pos_enc = type_pos_enc

        if(type_pos_enc == 'gauss'):
            np.random.seed(42)
            self.B_pos = gauss_std_pos_enc * \
                torch.from_numpy(np.random.randn(1, gauss_dim_pos_enc * 3, 3)) \
                    .float().cuda()
            self.B_view = gauss_std_pos_enc * \
                torch.from_numpy(np.random.randn(1, gauss_dim_pos_enc * 3, 3)) \
                    .float().cuda()
            # 3D点的嵌入向量维数
            emb_dim = 3 * gauss_dim_pos_enc * 2
            # 视角方向的嵌入向量维数
            emb_dim_view = 3 * gauss_dim_pos_enc_vd * 2
        else:
            emb_dim = 3 * self.max_freq_pos_enc * 2
            emb_dim_view = 3 * self.max_freq_pos_enc_vd * 2

        # 预测体积密度(Volume Density)的MLP层
        self.fc_in = nn.Linear(emb_dim, hidden_size)
        if z_dim > 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for i in range(num_layers - 1)
        ])

        # 计算需要添加跳连的总层数，注意skips里存储的是需要添加跳连的层的编号
        num_skips = sum([i in skips for i in range(num_layers - 1)])
        if num_skips > 0:
            # 处理隐向量z时，增加跳连需要的全连接层
            self.fc_z_skips = nn.ModuleList(
                [nn.Linear(z_dim, hidden_size) for i in range(num_skips)]
            )
            # 处理3D点p的位置编码时，增加跳连需要的全连接层
            self.fc_p_skips = nn.ModuleList(
                [nn.Linear(emb_dim, hidden_size) for i in range*num_skips]
            )
        # sigma(σ)即体积密度(Volume Density)，输出预测的体积密度
        self.out_sigma = nn.Linear(hidden_size, 1)

        # 使用视角方向(viewing direction)时需要的MLP层
        if use_view_direction and num_layers_view > 1:
            self.layers_view = nn.ModuleList([
                nn.Linear(emb_dim_view + hidden_size, hidden_size)
                for i in range(num_layers_view - 1)
            ])

        # 预测M_f维特征(M_f-dimensional feature f)的MLP层
        self.fc_z_view = nn.Linear(z_dim, hidden_size)
        self.feat_view = nn.Linear(hidden_size, hidden_size)
        # 视角方向(View Direction)的嵌入向量映射到隐含层
        self.fc_view = nn.Linear(emb_dim_view, hidden_size)
        # 隐含层映射到输出的M_f维特征
        self.feat_out = nn.Linear(hidden_size, out_dim_feat)

    def pos_encode(self, p, is_vd : bool):
        '''
        位置编码(Positional Encoding)函数
        公式: \gamma(t,L) = (sin(2^{0}t*pi), cos(2^{0}t*pi), ..., sin(2^{L}t*pi), cos(2^{L}t*pi))
        t为输入标量，L为频率倍频程
        
        在论文中为了计算 3D点x 和 视角方向d 的位置编码\gamma(x)和\gamma(d)
        要对x和d的每一维分量都应用位置编码
        '''

        # 将给定的p归一化到[-1,1]
        p = p / self.downscale_by

        # 假定点都分布在[-1,1]区间内，不需要再进行放缩
        # 采用高斯位置编码
        if self.type_pos_enc == 'gauss':
            B = self.B_view if is_vd else self.B_pos
            encoded_p = (B @ (pi * p.permute(0, 2, 1))).permute(0, 2, 1)
            encoded_p = torch.cat(
                [torch.sin(encoded_p),
                 torch.cos(encoded_p)],
                dim=-1
            )
        # 采用普通位置编码(论文中默认采用的方法)
        else:
            # 计算倍频程L
            L = self.max_freq_pos_enc_vd if is_vd else self.max_freq_pos_enc
            encoded_p = torch.cat(
                [
                    torch.cat(
                        [torch.sin((2 ** i) * pi * p),
                         torch.cos((2 ** i) * pi * p)],
                        dim=-1
                    )
                    for i in range(L)
                ],
                dim=-1
            )

        return encoded_p

    def forward(self, p_in, ray_d, z_s=None, z_a=None, **kwargs):
        '''
        Params:
            z_s -> Tensor : 形状的隐向量
            z_a -> Tensor : 外观的隐向量

        Return:
            out_feat -> Tensor : 输出的M_f维特征f
            out_sigma -> Tensor : 输出的体积密度σ
        '''
        if self.z_dim > 0:
            batch_size = p_in.shape[0]
            if z_s is None:
                z_s = torch.randn(batch_size, self.z_dim).to(p_in.device)
            if z_a is None:
                z_a = torch.randn(batch_size, self.z_dim).to(p_in.device)
        
        # 对输入的点进行位置编码
        p = self.pos_encode(p_in, is_vd=False)

        decoder = self.fc_in(p)
        if z_s is not None:
            decoder = decoder + self.fc_z(z_s).unsqueeze(1)

        # ReLU作为激活函数
        decoder = F.relu(decoder)

        # 处理跳连
        skip_idx = 0    # 指示需要跳连的层在self.fc_xx_skips的ModuleList中的索引
        for idx, layer in enumerate(self.fc_layers):
            # 每经过一个全连接层都使用ReLU作为激活函数
            decoder = F.relu(layer(decoder))
            # 如果当前层编号匹配了需要跳连的层编号
            if (idx + 1) in self.skips and (idx < len(self.fc_layers) - 1):
                decoder = decoder + self.fc_z_skips[skip_idx](z_s).unsqueeze(1)
                decoder = decoder + self.fc_p_skips[skip_idx][p]
                skip_idx += 1
        
        out_sigma = self.out_sigma(decoder).squeeze(-1)

        # 处理特征到视角方向的映射
        decoder = self.feat_view(decoder)
        # 外观隐向量映射到MLP的隐含层
        decoder = decoder + self.fc_z_view(z_a).unsqueeze(1)
        if self.use_view_direction and ray_d is not None:
            ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            # 对视角方向进行位置编码
            ray_d = self.pos_encode(ray_d, is_vd=True)
            decoder = decoder + self.fc_view(ray_d)
            decoder = F.relu(decoder)
            # 每经过一个全连接层都使用ReLU作为激活函数
            if self.num_layers_view > 1:
                for layer in self.layers_view:
                    decoder = F.relu(layer(decoder))

        out_feat = self.feat_out(decoder)

        if self.use_final_actvn:
            out_feat = torch.sigmoid(out_feat)

        return out_feat, out_sigma
