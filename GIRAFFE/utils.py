#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File  : utils.py
@Author: Penistrong
@Email : chen18296276027@gmail.com
@Date  : 2021-11-20 周六 19:04:41
@Desc  : 工具类，包含各种细节处理
'''
import torch
import numpy as np


def to_tensor(tensor):
    '''
    将numpy的ndarray转换为pytorch的tensor类型

    Param
    ------
    tensor -> Tensor/ndarray : Numpy数组或者Torch张量

    Return
    ------
    tensor -> Tensor
    '''
    if type(tensor) == np.ndarray:
        tensor = torch.from_numpy(tensor)
    # 防止torch.from_numpy造成共享内存的情况
    tensor = tensor.clone()

    return tensor


def arrange_pixels(resolution=(128, 128), batch_size=1, img_range=(-1., 1.),
                   subsample_to=None, invert_y_axis=False):
    '''
    按照给定图片分辨率排列像素点

    Params
    ------
    resolution -> Tuple : 图片分辨率
    batch_size -> int : 批处理大小
    img_range -> Tuple : 输出点的范围，默认为[-1, 1]
    subsample_to -> int : 如果该值是正整数，则生成的像素点会随机地子采样到该值上
    invert_y_axis -> bool : 是否反转y轴

    Returns
    -------
    pixel_locations -> Tensor : 整数型未放缩的像素位置
    pixel_scaled -> Tensor : 浮点型已放缩的像素位置
    '''
    h, w = resolution
    num_points = h * w

    # 在放缩后的分辨率下排列像素点
    pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
    pixel_locations = torch.stack([pixel_locations[0], pixel_locations[1]], dim=-1) \
                        .long().view(1, -1, 2).repeat(batch_size, 1, 1)

    pixel_scaled = pixel_locations.clone().float()
    
    # 移动并缩放像素点以匹配给定范围img_range
    scale = img_range[1] - img_range[0]
    loc = scale / 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    # 给定upsample_to为正整数时
    if subsample_to is not None and subsample_to > 0 and subsample_to < num_points:
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,), replace=False)
        pixel_scaled = pixel_scaled[:, idx]
        pixel_locations = pixel_locations[:, idx]

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.
        pixel_locations[..., -1] = (h - 1) - pixel_locations[..., -1]
    
    return pixel_locations, pixel_scaled


def transform_to_world(pixels, depth, camera_matrix, world_matrix,
                       scale_matrix=None, invert=True, use_abs_depth=True):
    '''
    将像素点位置p根据给定的深度值变换到真实世界中的坐标

    Params
    ------
    pixels -> Tensor : Pixel张量, size = Batch_size x N x 2
    depth -> Tensor : Depth张量, size = Batch_size x N x 1
    camera_matrix -> Tensor : 相机矩阵
    world_matrix -> Tensor : 姿态矩阵
    scale_matrix -> Tensor : 放缩矩阵
    invert -> bool : 是否要对矩阵求逆，默认为True
    use_abs_depth -> bool : 使用绝对深度

    Return
    ------
    p_world -> Tensor : 像素点在真实世界的3维空间坐标表示，size = Batch_size x N x 3
    '''
    if scale_matrix is None:
        scale_matrix = torch.eye(4).unsqueeze(0).repeat(camera_matrix.shape[0], 1, 1).to(camera_matrix.device)
    
    # 将给定的张量参数都转换为torch.Tensor
    pixels = to_tensor(pixels)
    depth = to_tensor(depth)
    camera_matrix = to_tensor(camera_matrix)
    world_matrix = to_tensor(world_matrix)
    scale_matrix = to_tensor(scale_matrix)

    # 对各矩阵求逆
    if invert:
        camera_matrix = torch.inverse(camera_matrix)
        world_matrix = torch.inverse(world_matrix)
        scale_matrix = torch.inverse(scale_matrix)

    # 将像素点转换为齐次坐标表示
    pixels = pixels.permute(0, 2, 1)    # 维度换位，后两维调换
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

    # 将像素点投影到相机空间
    if use_abs_depth:
        pixels[:, :2] = pixels[:, :2] * depth.permute(0, 2, 1).abs()
        pixels[:, 2:3] = pixels[:, 2:3] * depth.permute(0, 2, 1)
    else:
        pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # 将像素点转换到真实世界,利用公式
    p_world = scale_matrix @ world_matrix @ camera_matrix @ pixels

    # 将P_world转换为3维空间坐标
    p_world = p_world[:, :3].permute(0, 2, 1)

    return p_world


def transform_to_camera_space(p_world, camera_matrix, world_matrix, scale_matrix):
    '''
    将真实世界坐标转换到相机空间

    Params
    ------
    p_world -> Tensor : 点坐标张量 size = Batch_size x N x 3
    camera_matrix -> Tensor : 相机矩阵
    world_matrix -> Tensor : 姿态矩阵
    scale_matrix -> Tensor : 放缩矩阵

    Return
    ------
    p_cam -> Tensor : 相机空间中的坐标表示
    '''
    batch_size, N= p_world.shape[:2]
    device = p_world.device

    # 将真实世界坐标转换为齐次坐标表示
    p_world = torch.cat([p_world, torch.ones(batch_size, N, 1).to(device)], dim=-1).permute(0, 2, 1)

    # 应用公式
    p_cam = camera_matrix @ world_matrix @ scale_matrix @ p_world

    # 从齐次坐标转换到3维空间坐标
    p_cam = p_cam[:, :3].permute(0, 2, 1)

    return p_cam


def origin_to_world(n_points, camera_matrix, world_matrix, scale_matrix=None,
                    invert=False):
    ''' 
    将原点位置(即相机位置)转换为真实世界坐标
    
    Params
    ------
    num_points -> int: 转换后的原点坐标以(batch_size, n_points, 3)的形式重复的频率
    camera_matrix -> Tensor : 相机矩阵
    world_matrix -> Tensor : 姿态矩阵
    scale_matrix -> Tensor : 放缩矩阵
    invert -> bool : 是否对以上矩阵求逆，默认为False

    Return
    ------
    p_world -> Tensor : 原点的真实世界坐标
    '''
    batch_size = camera_matrix.shape[0]
    device = camera_matrix.device

    # 在齐次坐标系中创建原点, 每个batch中所有原点的最后一维为1
    p = torch.zeros(batch_size, 4, n_points).to(device)
    p[:, -1] = 1.

    if scale_matrix is None:
        scale_matrix = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)

    # Invert matrices
    if invert:
        camera_matrix = torch.inverse(camera_matrix)
        world_matrix = torch.inverse(world_matrix)
        scale_matrix = torch.inverse(scale_matrix)

    # 应用变换公式
    p_world = scale_matrix @ world_matrix @ camera_matrix @ p

    # 从齐次坐标转换到3维空间坐标
    p_world = p_world[:, :3].permute(0, 2, 1)

    return p_world


def img_points_to_world(img_points, camera_matrix, world_matrix,
                        scale_matrix=None, invert=False, negative_depth=True):
    '''
    将图片平面上的点转换到真实世界坐标

    与函数transform_to_world()相比，由于是图像平面，不需给定深度值(固定为1)

    Params
    ------
    img_points -> int: 图片上点的坐标张量 size = Batch_size x N x 2
    camera_matrix -> Tensor : 相机矩阵
    world_matrix -> Tensor : 姿态矩阵
    scale_matrix -> Tensor : 放缩矩阵
    invert -> bool : 是否对以上矩阵求逆，默认为False
    negative_depth -> bool : 是否采用反向深度

    Return
    ------
    p_world -> Tensor : 真实世界坐标
    '''
    batch_size , num_points, dim = img_points.shape
    device = img_points.device
    depth_img = torch.ones(batch_size, num_points, 1).to(device)
    if negative_depth:
        depth_img *= -1
    
    # 利用写好的转换函数
    return transform_to_world(img_points, depth_img, camera_matrix, world_matrix, scale_matrix, invert=invert)


def interpolate_sphere(z1, z2, t):
    p = (z1 * z2).sum(dim=-1, keepdim=True)
    p = p / z1.pow(2).sum(dim=-1, keepdim=True).sqrt()
    p = p / z2.pow(2).sum(dim=-1, keepdim=True).sqrt()
    omega = torch.acos(p)
    s1 = torch.sin((1-t)*omega)/torch.sin(omega)
    s2 = torch.sin(t*omega)/torch.sin(omega)
    z = s1 * z1 + s2 * z2
    return z