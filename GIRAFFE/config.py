#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File  : config.py
@Author: Penistrong
@Email : chen18296276027@gmail.com
@Date  : 2021-11-23 周二 18:29:41
@Desc  : GIRAFFE模型配置相关，由外层config.py传入模型配置文件并调用
'''
import os
from copy import deepcopy

import numpy as np
from discriminator import discriminator_dict

import GIRAFFE
from GIRAFFE.trainer import Trainer
from GIRAFFE.renderer import Renderer


def get_model(cfg, device=None, len_dataset=0, **kwargs):
    '''
    返回GIRAFFE模型

    Params
    ------
    cfg -> dict : 配置字典,从yaml中解析得到
    device -> device : PyTorch设备
    len_dataset -> int : 模型使用的数据集长度
    '''
    decoder = cfg['model']['decoder']
    decoder_kwargs = cfg['model']['decoder_kwargs']

    discriminator = cfg['model']['discriminator']
    discriminator_kwargs = cfg['model']['discriminator_kwargs']
    
    generator = cfg['model']['generator']
    generator_kwargs = cfg['model']['generator_kwargs']

    background_generator = cfg['model']['background_generator']
    background_generator_kwargs = cfg['model']['background_generator_kwargs']

    bounding_box_generator = cfg['model']['bounding_box_generator']
    bounding_box_generator_kwargs = cfg['model']['bounding_box_generator_kwargs']

    neural_renderer = cfg['model']['neural_renderer']
    neural_renderer_kwargs = cfg['model']['neural_renderer_kwargs']

    z_dim = cfg['model']['z_dim']
    z_dim_bg = cfg['model']['z_dim_bg']
    img_size = cfg['data']['img_size']


    decoder = GIRAFFE.decoder_dict[decoder](z_dim=z_dim, **decoder_kwargs)

    if discriminator is not None:
        discriminator = \
            discriminator_dict[discriminator](
                img_size=img_size, **discriminator_kwargs)
    
    if background_generator is not None:
        background_generator = \
            GIRAFFE.background_generator_dict[background_generator](
                z_dim=z_dim_bg, **background_generator_kwargs)

    if bounding_box_generator is not None:
        bounding_box_generator = \
            GIRAFFE.bounding_box_generator_dict[bounding_box_generator](
                z_dim=z_dim, **bounding_box_generator_kwargs)

    if neural_renderer is not None:
        neural_renderer = \
            GIRAFFE.neural_renderer_dict[neural_renderer](
                z_dim=z_dim, img_size=img_size, **neural_renderer_kwargs)

    if generator is not None:
        generator = GIRAFFE.generator_dict[generator](
            device, z_dim=z_dim, z_dim_bg=z_dim_bg, decoder=decoder,
            background_generator=background_generator,
            bounding_box_generator=bounding_box_generator,
            neural_renderer=neural_renderer, **generator_kwargs)

    if cfg['test']['take_generator_average']:
        generator_test = deepcopy(generator)
    else:
        generator_test = None
    
    model = GIRAFFE.GIRAFFE(
        device=device,
        discriminator=discriminator,
        generator=generator,
        generator_test=generator_test
    )

    return model

def get_trainer(model, optimizer_g, optimizer_d, cfg, device, **kwargs):
    '''
    返回GIRAFFE训练器

    Params
    ------
    model -> nn.Module : GIRAFFE模型
    optimizer_g -> optimizer: Generator optimizer object
    optimizer_d -> optimizer: Discriminator optimizer object
    cfg -> dict : 配置字典,从yaml中解析得到
    device -> device: pytorch device
    '''
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    overwrite_visualization = cfg['training']['overwrite_visualization']
    multi_gpu = cfg['training']['multi_gpu']
    num_eval_iters = (cfg['training']['n_eval_images'] // cfg['training']['batch_size'])

    fid_file = cfg['data']['fid_file']
    assert(fid_file is not None)
    fid_dict = np.load(fid_file)

    trainer = Trainer(
        model, optimizer_g, optimizer_d, device=device, vis_dir=vis_dir,
        multi_gpu=multi_gpu,fid_dict=fid_dict,
        num_eval_iters=num_eval_iters,
        overwrite_visualization=overwrite_visualization
    )

    return trainer


def get_renderer(model, cfg, device, **kwargs):
    '''
    Returns the renderer object.
    
    Params
    ------
    model -> nn.Module : GIRAFFE模型
    cfg -> dict : 配置字典,从yaml中解析得到
    device -> device: pytorch device
    '''

    renderer = Renderer(
        model,
        device=device
    )

    return renderer