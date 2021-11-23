#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File  : bounding_box_generator.py
@Author: Penistrong
@Email : chen18296276027@gmail.com
@Date  : 2021-11-19 周五 11:35:05
@Desc  : 碰撞箱生成器
'''
import numpy as np
import torch
import torch.nn as nn

from GIRAFFE.camera import get_rotation_matrix
from scipy.spatial.transform import Rotation as Rot


class BoundingBoxGenerator(nn.Module):
    '''
    在论文章节3.1中的Object Representation里
    作者在NeRF和GRAF的基础上进行改进, 将每个物体用分离的特征场进行表示
    添加了一个仿射变换T = {s, t, R}, s,t \in R^3 代表比例和平移的参数, R为旋转矩阵
    这样就以 k(x) = R·diag(s_1, s_2, s_3)·x + t 的形式将物体的点变换到场景空间

    碰撞箱生成器，用于生成物体(Object)的碰撞箱
    GIRAFFE作者认为背景(Background)也是一种Object，但是要作特殊处理

    Params
    ------
    num_boxes -> int : 除了背景之外，其他物体的碰撞箱总数
    is_fix_scale_ratio -> bool : 是否固定x/y/z的放缩率
    scale_range_min -> list : x, y, z的最小值
    scale_range_max -> list : x, y, z的最大值
    trans_range_min -> list : 翻译后的x,y,z的最小值
    trans_range_max -> list : 翻译后的x,y,z的最大值
    rotation_range -> list : 以列表存放的最小/最大旋转值
    is_check_collision -> bool : 是否启用碰撞检测
    collision_padding -> float : 碰撞检测的填充内边距
    is_object_on_plane -> bool : 是否将物体放在以z_level_plane定义的景深平面上
    z_level_plane -> float : 景深平面z的值
    '''

    def __init__(self,
                 num_boxes=1,
                 is_fix_scale_ratio=True,
                 scale_range_min=[0.5, 0.5, 0.5],
                 scale_range_max=[0.5, 0.5, 0.5],
                 trans_range_min=[-0.75, -0.75, 0.],
                 trans_range_max=[0.75, 0.75, 0.],
                 rotation_range=[0., 1.],
                 is_check_collision=False,
                 collision_padding=0.1,
                 is_object_on_plane=False,
                 z_level_plane=0.,
                 **kwargs):
        super(BoundingBoxGenerator, self).__init__()

        self.num_boxes = num_boxes
        self.is_fix_scale_ratio = is_fix_scale_ratio
        # x, y, z的范围最小值形成一个1行1列深度为3的张量
        self.scale_min = torch.tensor(scale_range_min).reshape(1, 1, 3)
        self.scale_range = (torch.tensor(scale_range_max) - torch.tensor(scale_range_min)).reshape(1, 1, 3)

        self.trans_min = torch.tensor(trans_range_min).reshape(1, 1, 3)
        self.trans_range = (torch.tensor(trans_range_max) - torch.tensor(trans_range_min)).reshape(1, 1, 3)

        self.rotation_range = rotation_range

        self.is_check_collision = is_check_collision
        self.collision_padding = collision_padding

        self.is_object_on_plane = is_object_on_plane
        self.z_level_plane = z_level_plane
 
    def check_collision(self, s, t):
        """
        检测物体的碰撞情况(只支持1~3个碰撞箱)

        Params
        ------
        s -> Tensor : 放缩比例scale, s.size=[batch_size, num_boxes, point's dimensions]
        t -> Tensor : 平移量translation

        Return
        ------
        is_free -> Tensor : 指示各物体是否有碰撞产生, 
        """
        num_boxes = s.shape[1]
        if num_boxes == 1:      # 只有一个碰撞箱，不用检测了
            is_free = torch.ones_like(s[..., 0]).bool().squeeze(1)
        elif num_boxes == 2:    # 2个碰撞箱情况
            d_t = (t[:, :1] - t[:, 1:2]).abs()
            d_s = (s[:, :1] - s[:, 1:2]).abs() + self.collision_padding
            is_free = (d_t >= d_s).any(-1).squeeze(1)
        elif num_boxes == 3:    # 3个碰撞箱情况, 递归调用本函数
            is_free_1 = self.check_collision(s[:, [0, 1]], t[:, [0, 1]])    # 检测第1个与第2个
            is_free_2 = self.check_collision(s[:, [0, 2]], t[:, [0, 2]])    # 检测第1个与第3个
            is_free_3 = self.check_collision(s[:, [1, 2]], t[:, [1, 2]])    # 检测第2个与第3个
            is_free = is_free_1 & is_free_2 & is_free_3
        else:
            print("ERROR: Collision Check for num_boxes > 3 not implemented")
        
        return is_free
    
    # 计算放缩比例
    def get_scale(self, batch_size=32, val=[[0.5, 0.5, 0.5]]):
        num_boxes = len(val)
        if self.is_fix_scale_ratio:
            s = self.scale_min + torch.tensor(val).reshape(1, num_boxes, -1)[..., -1] * self.scale_range
        else:
            s = self.scale_min + torch.tensor(val).reshape(1, num_boxes, 3) * self.scale_range
        s = s.repeat(batch_size, 1, 1)
        return  s

    # 计算平移量
    def get_translation(self, batch_size=32, val=[[0.5, 0.5, 0.5]]):
        num_boxes = len(val)
        t = self.trans_min + torch.tensor(val).reshape(1, num_boxes, 3) * self.trans_range
        t = t.repeat(batch_size, 1, 1)
        # 如果指定了对象在图片深度方向上的平面level，则将所有平移的z方向平移量从初始0改为给定z_level_plane
        if self.is_object_on_plane:
            t[..., -1] = self.z_level_plane
        
        return t

    # 计算旋转量
    def get_rotation(self, batch_size=32, val=[0.]):
        range = self.rotation_range
        values = [range[0] + v * (range[1] - range[0]) for v in val]
        R = torch.cat([get_rotation_matrix(value=v, batch_size=batch_size).unsqueeze(1) for v in values],
                      dim=1)
        R = R.float()

        return R

    def get_random_offset(self, batch_size=32):
        num_boxes = self.num_boxes
        if self.is_fix_scale_ratio:
            s_rand = torch.rand(batch_size, num_boxes, 1)
        else:
            s_rand = torch.rand(batch_size, num_boxes, 3)
        s = self.scale_min + s_rand * self.scale_range

        # 平移取样
        t = self.trans_min + torch.rand(batch_size, num_boxes, 3) * self.trans_range
        if self.is_check_collision:
            is_free = self.check_collision(s, t)
            # 当物体应用了平移量后各物体的碰撞箱存在碰撞情况
            # 持续给定随机平移量直到所有物体经过变换后都未发生碰撞
            while not torch.all(is_free):
                t_new = self.trans_min + torch.rand(batch_size, num_boxes, 3) * self.trans_range
                t[is_free == 0] = t_new[is_free == 0]
                is_free = self.check_collision(s, t)
        
        if self.is_object_on_plane:
            t[..., -1] = self.z_level_plane

        def r_val(): 
            return self.rotation_range[0] + np.random.rand() * (self.rotation_range[1] - self.rotation_range[0])
        
        R = [torch.from_numpy(Rot.from_euler('z', r_val() * 2 * np.pi).as_dcm())
             for i in range(batch_size * self.num_boxes)]
        R = torch.stack(R, dim=0).reshape(batch_size, self.num_boxes, -1).cuda().float()

        return s, t, R

    def forward(self, batch_size=32):
        s, t, R = self.get_random_offset(batch_size=batch_size)
        R = R.reshape(batch_size, self.num_boxes, 3, 3)
        
        return s, t, R


