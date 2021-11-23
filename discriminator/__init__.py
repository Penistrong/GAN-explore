#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File  : __init__.py
@Author: Penistrong
@Email : chen18296276027@gmail.com
@Date  : 2021-11-23 周二 21:16:14
@Desc  : 通用判别器类，包含各种判别器的模型结构
'''
from discriminator import dcdiscriminator

discriminator_dict = {
    'dc': dcdiscriminator.DCDiscriminator
}