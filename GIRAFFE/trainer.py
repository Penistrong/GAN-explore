#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File  : train.py
@Author: Penistrong
@Email : chen18296276027@gmail.com
@Date  : 2021-11-22 周一 17:21:18
@Desc  : GIRAFFE模型训练器
'''
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Evaluation.fid_score import (calculate_activation_statistics,
                                  calculate_frechet_distance)
from numpy.random import multivariate_normal
from torch import autograd
from torch.functional import Tensor
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


class BaseTrainer(object):
    '''
    基础训练器
    '''

    def evaluate(self, *args, **kwargs):
        '''
        执行评估
        '''
        eval_list = defaultdict(list)
        eval_step_dict = self.eval_step()
        for k, v in eval_step_dict.items():
            eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def train_step(self, *args, **kwargs):
        '''
        执行一次训练
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        '''
        执行一次评估
        '''
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        '''
        执行可视化
        '''
        raise NotImplementedError

    def toggle_grad(self, model : nn.Module, requires_grad):
        '''
        由于GIRAFFE模型由各组件组合而成，切换训练/评估时需要批次更改梯度模式
        '''
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        reg = grad_dout2.reshape(batch_size, -1).sum(1)
        return reg

    def update_average(self, dst_model : nn.Module, src_model : nn.Module, beta):
        self.toggle_grad(src_model, requires_grad=False)
        self.toggle_grad(dst_model, requires_grad=False)

        src_params_dict = dict(src_model.named_parameters())

        for p_name, p_dst in dst_model.named_parameters():
            p_src = src_params_dict[p_name]
            assert(p_src is not p_dst)
            p_dst.copy_(beta * p_dst + (1. - beta) * p_src)

    def compute_bce(self, d_out : Tensor, target):
        '''
        计算二值交叉熵损失
        '''
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = F.binary_cross_entropy_with_logits(d_out, targets)

        return loss


class Trainer(BaseTrainer):
    '''
    GIRAFFE训练器

    Params
    ------
    model -> nn.Module : GIRAFFE模型
    optimizer_g -> optimizer : 生成模型的优化器
    optimizer_d -> optimizer : 判别模型的优化器
    device -> torch.device : PyTorch设备
    vis_dir -> str : 可视化文件夹
    multi_gpu -> bool : 训练时是否使用多gpu
    fid_dict -> dict : FID的GT数据字典
    num_eval_iters -> int : 进行模型评估的迭代次数间隔
    overwrite_visualization -> bool : 是否覆盖可视化文件
    '''
    def __init__(self, model, optimizer_g, optimizer_d,
                 device=None, vis_dir=None, multi_gpu=False,
                 fid_dict={}, num_eval_iters=10, overwrite_visualization=True,
                **kwargs):
        self.model = model
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.device = device
        self.vis_dir = vis_dir
        self.multi_gpu = multi_gpu
        self.overwrite_visualization = overwrite_visualization

        self.fid_dict = fid_dict
        self.vis_dict = model.generator.get_vis_dict(16)
        self.num_eval_iters = num_eval_iters

        self.generator = self.model.generator
        self.discriminator = self.model.discriminator
        self.generator_test = self.model.generator_test

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, it=None):
        loss_g = self.train_step_G(data, it)
        loss_d, reg_d, fake_d, real_d = self.train_step_D(data, it)

        return {
            'generator': loss_g,
            'discriminator': loss_d,
            'regularizer': reg_d
        }

    def eval_step(self):
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()

        x_fake = []
        num_iters = self.num_eval_iters

        for i in tqdm(range(num_iters)):
            with torch.no_grad():
                x_fake.append(gen().cpu()[:, :3])

        x_fake = torch.cat(x_fake, dim=0)
        x_fake.clamp_(0., 1.)
        
        mu, sigma = calculate_activation_statistics(x_fake)
        fid = calculate_frechet_distance(mu, sigma, self.fid_dict['m'], self.fid_dict['s'], eps=1e-4)

        eval_dict = {
            'fid_score': fid
        }

        return eval_dict

    def train_step_generator(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator

        self.toggle_grad(generator, True)
        self.toggle_grad(discriminator, False)

        generator.train()
        discriminator.train()

        self.optimizer_g.zero_grad()

        x_fake = generator()

        d_fake = discriminator(x_fake)
        g_loss = self.compute_bce(d_fake, 1)

        g_loss.backward()
        self.optimizer_g.step()

        if self.generator_test is not None:
            self.update_average(self.generator_test, generator, beta=0.999)

        return g_loss.item()
        
    def train_step_discriminator(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator

        self.toggle_grad(generator, False)
        self.toggle_grad(discriminator, True)

        generator.train()
        discriminator.train()

        self.optimizer_d.zero_grad()

        x_real = data.get('image').to(self.device)
        loss_d_full = 0

        x_real.requires_grad_()
        d_real = discriminator(x_real)

        d_loss_real = self.compute_bce(d_real, 1)
        loss_d_full += d_loss_real

        reg = 10. * self.compute_grad2(d_real, x_real).mean()
        loss_d_full += reg

        with torch.no_grad():
            x_fake = generator()
        
        x_fake.requires_grad_()
        d_fake = discriminator(x_fake)

        d_loss_fake = self.compute_bce(d_fake, 0)
        loss_d_full += d_loss_fake

        loss_d_full.backward()
        self.optimizer_d.step()

        d_loss = (d_loss_fake + d_loss_real)

        return (d_loss.item(), reg.item(), d_loss_fake.item(), d_loss_real.item())


    def visualize(self, it=0):
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()
        with torch.no_grad():
            img_fake = self.generator(**self.vis_dict, mode='val').cpu()
        
        if self.overwrite_visualization:
            out_file_name = 'visualization.png'
        else:
            out_file_name = 'visualization_%010d.png' % it
        
        img_grid = make_grid(img_fake.clamp_(0., 1.), nrow=4)
        save_image(img_grid, os.path.join(self.vis_dir, out_file_name))

        return img_grid