#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File  : camera.py
@Author: Penistrong
@Email : chen18296276027@gmail.com
@Date  : 2021-11-21 周日 14:55:40
@Desc  : GIRAFFE中使用Camera获取样本姿态
'''
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot


def to_sphere(u, v):
    '''
    根据各点的旋转值和仰角值，获取其在3D球坐标系上的坐标(直角坐标表示)

    Params
    ------
    u -> ndarray(float) : 旋转值
    v -> ndarray(float) : 仰角值

    Return
    ------
    coordinate_on_sphere -> ndarray : 各3D点在球坐标系上的直角坐标表示
    '''
    theta = 2 * np.pi * u
    phi = np.arccos(1 - 2 * v)
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)

    return np.stack([cx, cy, cz], axis=-1)


def sample_on_sphere(range_u=(0, 1), range_v=(0, 1), size=(1,)):
    '''
    在球坐标系中随机采样

    Return
    ------
    sample -> Tensor : 随机采样点在空间直角坐标系中的坐标表示的一维张量
    size -> Tuple(int,) :  批次生成的采样点个数
    '''
    u = np.random.uniform(*range_u, size=size)
    v = np.random.uniform(*range_v, size=size)

    sample = to_sphere(u, v)
    sample = torch.tensor(sample).float()

    return sample


def get_camera_matrix(fov=49.13, invert=True):
    '''
    获取相机矩阵

    Params
    ------
    fov -> float : 视野范围
    invert -> bool : 是否对矩阵求逆

    Return
    ------
    matrix -> Tensor : 相机矩阵
    '''
    focal = 1. / np.tan(0.5 * fov * np.pi / 180.)
    focal = focal.astype(np.float32)
    matrix = torch.tensor([
        [focal, 0., 0., 0.],
        [0., focal, 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ]).reshape(1, 4, 4)

    if invert:
        matrix = torch.inverse(matrix)

    return matrix


def get_random_pose(range_u, range_v, range_r, batch_size=32, invert=False):
    '''
    获取随机姿态矩阵

    Params
    ------
    range_u -> Tuple : 旋转范围rotation range(0~1)
    range_v -> Tuple : 仰角范围elevation range(0~1)
    range_r -> Tuple : 半径范围radius range
    batch_size -> int : 批处理大小
    invert -> bool : 是否对获取的姿态矩阵求逆
    '''
    coord = sample_on_sphere(range_u, range_v, size=(batch_size))
    radius = range_r[0] + torch.rand(batch_size) * (range_r[1] - range_r[0])
    coord = coord * radius.unsqueeze(-1)
    R = look_at(coord)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = coord

    if invert:
        RT = torch.inverse(RT)

    return RT


def get_camera_pose(range_u, range_v, range_r, val_u=0.5, val_v=0.5, val_r=0.5, batch_size=32, invert=False):
    '''
    获取相机姿态
    '''
    u0, ur = range_u[0], range_u[1] - range_u[0]
    v0, vr = range_v[0], range_v[1] - range_v[0]
    r0, rr = range_r[0], range_r[1] - range_r[0]
    u = u0 + val_u * ur
    v = v0 + val_v * vr
    r = r0 + val_r * rr

    # 固定的相机姿态，采用固定采样，随机范围限制到固定点
    coord = sample_on_sphere((u, u), (v, v), size=(batch_size))
    radius = torch.ones(batch_size) * r
    coord = coord * radius.unsqueeze(-1)
    R = look_at(coord)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = coord

    if invert:
        RT = torch.inverse(RT)
    return RT


# 获取在3维欧式空间进行旋转的旋转矩阵
def get_rotation_matrix(axis='z', value=0, batch_size=32):
    matrix = Rot.from_euler(seq=axis, angles=value * 2 * np.pi).as_dcm()
    matrix = torch.from_numpy(matrix).reshape(1, 3, 3).repeat(batch_size, 1, 1)
    return matrix


def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5):
    '''
    
    Params
    ------
    eye -> ndrray : 观测点

    '''
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)
    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)

    z_axis = eye - at
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis,
                                              axis=1, keepdims=True), eps]))

    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis,
                                              axis=1, keepdims=True), eps]))

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis,
                                              axis=1, keepdims=True), eps]))

    r_mat = np.concatenate(
        (x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(
            -1, 3, 1)), axis=2)

    r_mat = torch.tensor(r_mat).float()

    return r_mat