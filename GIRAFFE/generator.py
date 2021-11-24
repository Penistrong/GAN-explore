#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File  : generator.py
@Author: Penistrong
@Email : chen18296276027@gmail.com
@Date  : 2021-11-20 周六 20:54:37
@Desc  : GIRAFFE的生成器网络
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot

from GIRAFFE.bounding_box_generator import BoundingBoxGenerator
from GIRAFFE.camera import get_camera_matrix, get_camera_pose, get_random_pose
from GIRAFFE.decoder import Decoder
from GIRAFFE.neural_renderer import NeuralRenderer
from GIRAFFE.utils import arrange_pixels, img_points_to_world, origin_to_world


class Generator(nn.Module):
    """
    GIRAFFE生成网络

    Params
    ------
    device -> device : PyTorch设备(cpu or gpu)
    z_dim -> int : 隐向量z的维度
    z_dim_bg -> int : 背景(background)作为物体的隐向量z的维度
    range_u -> Tuple : 旋转范围rotation range(0~1)
    range_v -> Tuple : 仰角范围elevation range(0~1)
    range_r -> Tuple : 半径范围radius range
    range_d -> Tuple : 在图片深度方向上的近、远之分
    range_bg_rot -> Tuple : 背景旋转范围background rotation range(0~1)
    num_ray_samples -> int : 体积渲染中每次光线投射的采样数
    resolution_vol -> int : 体积渲染生成图片的分辨率，默认为16x16，这里只需设为16即可
    fov -> float : 视野范围(field of view)

    Parts of Generator
    -----------------
    decoder -> nn.Module : Decoder解码器
    background_generator -> nn.Module : 背景生成器
    bounding_box_generator -> nn.Module : 碰撞箱生成器
    neural_renderer -> nn.Module : 2D神经渲染器
    """

    def  __init__(self,
                  device,
                  z_dim=256,
                  z_dim_bg=128,
                  range_u=(0, 0),
                  range_v=(0.25, 0.25),
                  range_r=(2.732, 2.732),
                  range_d=(0.5, 6.),
                  range_bg_rot=(0., 0.),
                  num_ray_samples=64,
                  resolution_vol=16,
                  fov=49.13,
                  decoder : Decoder = None,
                  background_generator=None,
                  bounding_box_generator : BoundingBoxGenerator = None,
                  neural_renderer : NeuralRenderer = None,
                  **kwargs):
        super(Generator, self).__init__()

        self.device = device
        self.z_dim = z_dim
        self.z_dim_bg = z_dim_bg
        self.range_u = range_u
        self.range_v = range_v
        self.range_r = range_r
        self.range_d = range_d
        self.range_bg_rot = range_bg_rot
        self.num_ray_samples = num_ray_samples
        self.resolution_vol = resolution_vol
        self.fov = fov

        self.decoder = decoder
        self.background_generator = background_generator
        self.bounding_box_generator = bounding_box_generator
        self.neural_renderer = neural_renderer

        # 将相机矩阵添加到device上
        self.camera_matrix = get_camera_matrix(fov=fov).to(device)

        # 将Generator的各组件添加到device上
        if decoder is not None:
            self.decoder = decoder.to(device)
        else:
            self.decoder = None
        if background_generator is not None:
            self.background_generator = background_generator.to(device)
        else:
            self.background_generator = None
        if bounding_box_generator is not None:
            self.bounding_box_generator = bounding_box_generator.to(device)
        else:
            self.bounding_box_generator = bounding_box_generator
        if neural_renderer is not None:
            self.neural_renderer = neural_renderer.to(device)
        else:
            self.neural_renderer = None

    def forward(self,
                batch_size=32,
                latent_codes=None,
                camera_matrices=None,
                transformations=None,
                bg_rotation=None,
                mode='training',
                it=0):
        if latent_codes is None:
            latent_codes = self.get_latent_codes(batch_size)
        if camera_matrices is None:
            camera_matrices = self.get_random_camera(batch_size)
        if transformations is None:
            transformations = self.get_random_transformations(batch_size)
        if bg_rotation is None:
            bg_rotation = self.get_random_bg_rotation(batch_size)

        rgb_v = self.volume_render_image(latent_codes, camera_matrices, transformations,
                                         bg_rotation, mode=mode, it=it)
        
        # 利用2D神经渲染器生成目标图片
        if self.neural_renderer is not None:
            rgb = self.neural_renderer(rgb_v)
        else:
            rgb = rgb_v

        return rgb

    def get_num_boxes(self):
        return self.bounding_box_generator.num_boxes if self.bounding_box_generator is not None else 1

    def get_latent_codes(self, batch_size=32, tmp=1.):
        '''
        获取各物体(包含普通物体与背景)形状和外观的隐向量
        '''
        z_dim, z_dim_bg = self.z_dim, self.z_dim_bg
        num_boxes = self.get_num_boxes()

        z_s_obj = self.sample_z((batch_size, num_boxes, z_dim))
        z_a_obj = self.sample_z((batch_size, num_boxes, z_dim))
        z_s_bg = self.sample_z((batch_size, z_dim_bg))
        z_a_bg = self.sample_z((batch_size, z_dim_bg))

        return z_s_obj, z_a_obj, z_s_bg, z_a_bg

    def sample_z(self, size, tmp=1.):
        z = torch.randn(*size) * tmp
        z = z.to(self.device)

        return z

    def get_vis_dict(self, batch_size=32):
        '''
        生成可视化过程中使用到的字典
        '''
        vis_dict = {
            'batch_size': batch_size,
            'latent_codes': self.get_latent_codes(batch_size),
            'camera_matrices': self.get_random_camera(batch_size),
            'transformations': self.get_random_transformations(batch_size),
            'bg_rotation': self.get_random_bg_rotation(batch_size)
        }
        
        return vis_dict

    def get_random_camera(self, batch_size=32):
        '''
        获取随机相机姿态

        Returns
        -------
        camera_matrix -> Tensor : 相机张量
        world_matrix -> Tensor : 姿态张量
        '''
        camera_matrix = self.camera_matrix.repeat(batch_size, 1, 1)
        world_matrix = get_random_pose(self.range_u, self.range_v, self.range_r, batch_size)
        world_matrix = world_matrix.to(self.device)

        return camera_matrix, world_matrix

    def get_camera(self, val_u=0.5, val_v=0.5, val_r=0.5, batch_size=32):
        '''
        获取固定相机姿态

        Returns
        -------
        camera_matrix -> Tensor : 相机张量
        world_matrix -> Tensor : 姿态张量
        '''
        camera_matrix = self.camera_matrix.repeat(batch_size, 1, 1)
        world_matrix = get_camera_pose(self.range_u, self.range_v, self.range_r, val_u, val_v, val_r, batch_size=batch_size)
        world_matrix = world_matrix.to(self.device)

        return camera_matrix, world_matrix

    def get_random_bg_rotation(self, batch_size):
        '''
        获取随机值背景旋转
        '''
        if self.range_bg_rot != [0., 0.]:
            bg_r = self.range_bg_rot
            r_random = bg_r[0] + np.random.rand() * (bg_r[1] - bg_r[0])
            R_bg = [torch.from_numpy(Rot.from_euler('z', r_random * 2 * np.pi).as_dcm())
                    for i in range(batch_size)]
            R_bg = torch.stack(R_bg, dim=0).reshape(
                batch_size, 3, 3).float()
        else:
            R_bg = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()

        R_bg = R_bg.to(self.device)
        
        return R_bg

    def get_bg_rotation(self, val, batch_size=32):
        '''
        获取固定值背景旋转
        '''
        if self.range_bg_rot != [0., 0.]:
            bg_r = self.range_bg_rot
            r_val = bg_r[0] + val * (bg_r[1] - bg_r[0])
            r = torch.from_numpy(
                Rot.from_euler('z', r_val * 2 * np.pi).as_dcm()
            ).reshape(1, 3, 3).repeat(batch_size, 1, 1).float()
        else:
            r = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()

        r = r.to(self.device)

        return r

    def get_random_transformations(self, batch_size=32):
        '''
        获取随机仿射变换
        '''
        s, t, R = self.bounding_box_generator(batch_size)
        s, t, R = s.to(self.device), t.to(self.device), R.to(self.device)

        return s, t, R

    def get_transformations(self, 
                            val_s=[[0.5, 0.5, 0.5]],
                            val_t=[[0.5, 0.5, 0.5]],
                            val_r=[0.5],
                            batch_size=32):
        '''
        获取固定仿射变换
        '''
        s = self.bounding_box_generator.get_scale(batch_size=batch_size, val=val_s)
        t = self.bounding_box_generator.get_translation(batch_size=batch_size, val=val_t)
        R = self.bounding_box_generator.get_rotation(batch_size=batch_size, val=val_r)

        s, t, R = s.to(self.device), t.to(self.device), R.to(self.device)

        return s, t, R

    def get_transformations_in_range(self,
                                     range_s=[0., 1.],
                                     range_t=[0., 1.],
                                     range_r=[0., 1.],
                                     num_boxes=1,
                                     batch_size=32):
        '''
        获取给定范围内的随机仿射变换
        '''
        s, t, R = [], [], []

        def rand_s(): return range_s[0] + \
            np.random.rand() * (range_s[1] - range_s[0])
        def rand_t(): return range_t[0] + \
            np.random.rand() * (range_t[1] - range_t[0])
        def rand_r(): return range_r[0] + \
            np.random.rand() * (range_r[1] - range_r[0])

        for i in range(batch_size):
            val_s = [[rand_s(), rand_s(), rand_s()] for j in range(num_boxes)]
            val_t = [[rand_t(), rand_t(), rand_t()] for j in range(num_boxes)]
            val_r = [rand_r() for j in range(num_boxes)]
            si, ti, Ri = self.get_transformations(
                val_s, val_t, val_r, batch_size=1)
            s.append(si)
            t.append(ti)
            R.append(Ri)

        s, t, R = torch.cat(s), torch.cat(t), torch.cat(R)
        s, t, R = s.to(self.device), t.to(self.device), R.to(self.device)

        return s, t, R

    def get_rotation(self, val_r, batch_size=32):
        R = self.bounding_box_generator.get_rotation(batch_size=batch_size, val=val_r)
        R = R.to(self.device)

        return R

    def add_noise_to_interval(self, di):
        di_mid = .5 * (di[..., 1:] + di[..., :-1])
        di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
        di_low = torch.cat([di[..., :1], di_mid], dim=-1)
        noise = torch.rand_like(di_low)
        ti = di_low + (di_high - di_low) * noise
        return ti

    def transform_points_to_box(self, p, transformations, box_idx=0, scale_factor=1.):
        # T = {s, t, R}
        bb_s, bb_t, bb_R = transformations
        p_box = (bb_R[:, box_idx] @ (p - bb_t[:, box_idx].unsqueeze(1)).permute(0, 2, 1)) \
            .permute(0, 2, 1) / bb_s[:, box_idx].unsqueeze(1) * scale_factor

        return p_box

    def get_evaluation_points(self, pixels_world, camera_world, di, transformations, i,
                              mode='object', bg_rotation=None):
        '''
        获取物体的测量点，物体分为普通物体和背景两种情况

        Params
        ------
        pixels_world -> Tensor : 像素点在真实世界中的坐标
        camera_world -> Tensor : 相机位置在真实世界中的坐标
        di -> Tensor : 物体在3维空间中深度的可能值，按照每次光线投射的采样数构建的深度步进张量(range_d[0]~range_d[1])
        transformations -> Tensor : 在该物体上应用的仿射变换
        i -> int : 物体上待变换的该点对应的索引i
        mode -> str : 默认为'object', 指明为'background'时，对背景进行处理
        bg_rotation -> Tensor : 对背景应用的旋转矩阵，mode=‘background’时要给定该参数

        Returns
        -------
        p_i -> Tensor : 采样得到的3D点坐标
        ray_i -> Tensor : 该点对应的光线向量
        '''
        batch_size = pixels_world.shape[0]
        num_steps = di.shape[-1]

        assert(mode in ('object', 'background'))

        if mode == 'object':
            pixels_world_i = self.transform_points_to_box(pixels_world, transformations, box_idx=i)
            camera_world_i = self.transform_points_to_box(camera_world, transformations, box_idx=i)
        else:
            pixels_world_i = (bg_rotation @ pixels_world.permute(0, 2, 1)).permute(0, 2, 1)
            camera_world_i = (bg_rotation @ camera_world.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 计算光线向量
        ray_i = pixels_world_i - camera_world_i
        
        p_i = camera_world_i.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * ray_i.unsqueeze(-2).contiguous()
        ray_i = ray_i.unsqueeze(-2).repeat(1, 1, num_steps, 1)
        
        p_i = p_i.reshape(batch_size, -1, 3)
        ray_i = ray_i.reshape(batch_size, -1, 3)

        return p_i, ray_i

    def composite_operator(self, sigma, feat):
        '''
        将(f_i, σ_i)对合成为加权特征f和体积密度σ

        C(x, d)=(σ, f), σ=\sum_{i=1}^{N}σ_i; f=\\frac{1}{σ}\sum_{i=1}^{N}σ_i*f_i

        Params
        ------
        sigma -> Tensor : 各个采样点的体积密度
        feat -> Tensor : 各个采样点的对应特征

        Returns
        -------
        sigma_sum -> Tensor : 累加的体积密度σ
        feat_weighted -> Tensor : 加权特征f
        '''
        num_boxes = sigma.shape[0]
        if num_boxes > 1:
            sigma_sum = torch.sum(sigma, dim=0)

            denom_sigma = torch.sum(sigma, dim=0, keepdim=True)
            denom_sigma[denom_sigma == 0] = 1e-4
            portion_sigma = sigma / denom_sigma
            feat_weighted = (feat * portion_sigma.unsqueeze(-1)).sum(0)
        else:
            sigma_sum = sigma.squeeze(0)
            feat_weighted = feat.squeeze(0)
        
        return sigma_sum, feat_weighted

    def calc_volume_weights(self, di, ray_vector, sigma, last_dist=1e10):
        '''
        3D体积渲染中计算T_j * α_j * f_j的特征图权重分量

        基于前面Composition Operator处理得到的一个给定像素点时生成的体积密度σ_j和特征场中的特征向量f_j
        生成最终的16x16特征图

        使用论文中的3.3节的公式(10): f = \sum_{j=1}^{N_s} T_j * α_j * f_j
        其中:
        T_j = \prod_{k=1}^{j-1}(1 - α_k)
        α_j = 1 - exp{-σ_j*δ_j}
        δ_j = ||x_{j+1} - x_{j}||
        '''
        # 计算相邻采样点间的距离δ_j，用2-范数计算||x_{j+1} - x{j}||
        dist = di[..., 1:] - di[..., :-1]
        dist = torch.cat([dist, torch.ones_like(di[..., :1]) * last_dist], dim=-1)
        dist = dist * torch.norm(ray_vector, dim=-1, keepdim=True)
        alpha = 1. - torch.exp(-F.relu(sigma) * dist)
        weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[:, :, :1]),(1. - alpha + 1e-10)],
                                                  dim=-1),
                                        dim=-1)[..., :-1]

        return weights

    def volume_render_image(self, latent_codes,
                            camera_matrices, transformations,
                            bg_rotation, mode='training',
                            it=0):
        '''
        生成器的主要部分，生成体积渲染图片(分辨率默认为16x16)

        Params
        ------
        camera_matrices -> Tuple(Tensor, Tensor) : 即camera_matrix, world_matrix形成的元组
        '''
        res = self.resolution_vol
        num_points = res ** 2
        device = self.device
        num_steps = self.num_ray_samples
        range_d = self.range_d
        z_s_obj, z_a_obj, z_s_bg, z_a_bg = latent_codes
        batch_size, num_boxes = z_s_obj.shape[:2]

        # 排列像素点 Arrange Pixels
        _, pixels = arrange_pixels((res, res), batch_size, invert_y_axis=False)
        pixels = pixels.to(device)
        pixels[..., -1] *= -1.
    
        # 投影至3维世界 Project to 3D world
        camera_matrix, world_matrix = camera_matrices[:2]
        pixels_world = img_points_to_world(pixels, camera_matrix, world_matrix)
        camera_world = origin_to_world(num_points, camera_matrix, world_matrix)
        # 获取3维世界中，从原点到目标像素点的光线向量
        ray_vector = pixels_world - camera_world

        # 注意size=batch_size x num_points x num_steps
        di = range_d[0] + torch.linspace(0., 1., steps=num_steps).reshape(1, 1, -1) * (range_d[1] - range_d[0])
        di = di.repeat(batch_size, num_points, 1).to(device)
        if mode == 'training':  # 如果是训练模式，添加噪声
            di = self.add_noise_to_interval(di)

        # M_f维特征和体积密度σ
        feat, sigma = [], []
        # 迭代次数还要算上将背景作为物体的情况
        num_iters = num_boxes + 1

        for i in range(num_iters):
            if i < num_boxes:   # 普通对象情况
                p_i, r_i = self.get_evaluation_points(pixels_world, camera_world, di, transformations, i)
                z_s_i, z_a_i = z_s_obj[:, i], z_a_obj[:, i]

                feat_i, sigma_i = self.decoder(p_i, r_i, z_s_i, z_a_i)

                # NeRF中，训练时要给预测的体积密度σ添加噪声
                if mode == 'training':
                    sigma_i == torch.randn_like(sigma_i)

                # Mask out values outside
                padd = 0.1
                mask_box = torch.all(p_i <= 1. + padd, dim=-1) & torch.all(p_i >= -1. - padd, dim=-1)
                sigma_i[mask_box == 0] = 0.

                # Reshape
                sigma_i = sigma_i.reshape(batch_size, num_points, num_steps)
                feat_i = feat_i.reshape(batch_size, num_points, num_steps, -1)
            else: # 对于背景
                p_bg, r_bg = self.get_evaluation_points(pixels_world, camera_world, di, transformations,
                                                        i, mode='background', bg_rotation=bg_rotation)
                feat_i, sigma_i = self.background_generator(p_bg, r_bg, z_s_bg, z_a_bg)

                sigma_i = sigma_i.reshape(batch_size, num_points, num_steps)
                feat_i = feat_i.reshape(batch_size, num_points, num_steps, -1)

                # NeRF中，训练时要给预测的体积密度σ添加噪声
                if mode == 'training':
                    sigma_i == torch.randn_like(sigma_i)

            feat.append(feat_i)
            sigma.append(sigma_i)

        sigma = F.relu(torch.stack(sigma, dim=0))
        feat = torch.stack(feat, dim=0)

        # 合成 利用Composition Operator: sigma_sum = \sum_{i=1}^{N}\sigma_i
        sigma_sum, feat_weighted = self.composite_operator(sigma, feat)

        # 3D体积渲染 Volume Rendering
        # 获取体积权重
        weights = self.calc_volume_weights(di, ray_vector, sigma_sum)
        # 得到特征图
        feat_map = torch.sum(weights.unsqueeze(-1) * feat_weighted, dim=-2)

        # 格式化输出
        # size=B x feat x h x w
        feat_map = feat_map.permute(0, 2, 1).reshape(batch_size, -1, res, res)
        # 特征图的x轴与y轴互相翻转(后两维)
        feat_map = feat_map.permute(0, 1, 3, 2)

        return feat_map
