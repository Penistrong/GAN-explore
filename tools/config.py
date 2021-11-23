#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File  : config.py
@Author: Penistrong
@Email : chen18296276027@gmail.com
@Date  : 2021-11-23 周二 17:02:41
@Desc  : 模型配置相关，可以调用位于不同模型模块下的config类加载对应模型的配置
'''
import logging
import os

import GIRAFFE
import yaml

method_dict = {
    'giraffe': GIRAFFE
}

def load_config(path, default_path=None):
    '''
    加载配置文件

    Params
    ------
    path -> str : 配置文件路径
    default_path -> bool : 是否使用默认路径
    '''
    # 指定配置文件
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # 如果配置文件中有inherit_from字段，说明该配置文件属于扩展配置文件
    inherit_from = cfg.get('inherit_from')

    if inherit_from is not None:
        parent_cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            parent_cfg = yaml.load(f, Loader=yaml.Loader)
    else:
        parent_cfg = dict()

    # 存在父配置文件的情况下，当前配置文件只是对其中某些字段的更新
    update_recursively(parent_cfg, cfg)

    return parent_cfg


def update_recursively(dict_src : dict, dict_update : dict):
    '''
    递归更新配置文件，部分键值对中 值仍是个字典，所以递归更新

    Params
    ------
    dict_src -> dict : 需要被升级的配置文件
    dict_update -> dict : 包含升级字段的配置文件
    '''
    for k, v in dict_update.items():
        if k not in dict_src:
            dict_src[k] = dict()
        if isinstance(v, dict):
            update_recursively(dict_src[k], v)
        else:
            dict_src[k] = v


def get_model(cfg, device=None, len_dataset=0):
    '''
    返回模型实例

    Params
    ------
    cfg -> dict : 配置字典,从yaml中解析得到
    device -> device : PyTorch设备
    len_dataset -> int : 模型使用的数据集长度
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model(cfg, device=device, len_dataset=len_dataset)

    return model


def get_trainer(model, optimizer_g, optimizer_d, cfg, device):
    '''
    返回模型训练器

    Params
    ------
    model -> nn.Module : 使用的模型
    optimizer -> optimizer : PyTorch Optimizer
    cfg -> dict : 配置字典
    device -> device : PyTorch Device
    '''
    method = cfg['method']
    set_logger(cfg)
    trainer = method_dict[method].config.get_trainer(model, optimizer_g, optimizer_d, cfg, device)

    return trainer


def get_renderer(model, cfg, device):
    '''
    返回模型渲染器

    Params
    ------
    model -> nn.Module : 使用的模型
    cfg -> dict : 配置字典
    device -> device : PyTorch Device
    '''
    method = cfg['method']
    renderer = method_dict[method].config.get_renderer(model, cfg, device)

    return renderer


def get_dataset(cfg, **kwargs):
    '''
    调用对应模型的dataset.py获取其使用的图片数据集
    '''
    method = cfg['method']
    # 获取配置中的字段
    dataset_name = cfg['data']['dataset_name']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']
    img_size = cfg['data']['img_size']

    dataset = method_dict[method].dataset.ImagesDataset(
        dataset_folder, size=img_size,
        use_tanh_range=cfg['data']['use_tanh_range'],
        celebA_center_crop=cfg['data']['celebA_center_crop'],
        random_crop=cfg['data']['random_crop'],
    )
    
    return dataset


def set_logger(cfg):
    logfile = os.path.join(cfg['training']['out_dir'],
                           cfg['training']['logfile'])
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s %(name)s: %(message)s',
        datefmt='%m-%d %H:%M',
        filename=logfile,
        filemode='a',
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console_handler)
