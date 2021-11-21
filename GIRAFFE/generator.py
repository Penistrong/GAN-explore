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
from torch.autograd import backward

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

    def forward(self):
        pass

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

    def get_evaluation_points(self, pixels_world, camera_world, di, transformations, i):
        batch_size = pixels_world.shape[0]
        num_steps = di.shape[-1]

        pixels_world_i = self.transform_points_to_box(pixels_world, transformations, box_idx=i)
        camera_world_i = self.transform_points_to_box(camera_world, transformations, box_idx=i)
        ray_i = pixels_world_i - camera_world_i # 计算光线向量
        
        p_i = camera_world_i.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * ray_i.unsqueeze(-2).contiguous()
        ray_i = ray_i.unsqueeze(-2).repeat(1, 1, num_steps, 1)
        
        p_i = p_i.reshape(batch_size, -1, 3)
        ray_i = ray_i.reshape(batch_size, -1, 3)

        return p_i, ray_i

    def volume_render_image(self, latent_codes,
                            camera_matrices, transformations,
                            bg_rotation, mode='training',
                            it=0, return_alpha_map=False,
                            not_render_bg=False,
                            only_render_bg=False):
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
                p_bg, r_bg = self.get_evaluation_points()

        # 合成 利用Composition Operator


        # 获取体积权重


        # 格式化输出
