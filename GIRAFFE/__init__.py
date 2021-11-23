#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File  : __init__.py
@Author: Penistrong
@Email : chen18296276027@gmail.com
@Date  : 2021-11-17 Wed 17:22:18
@Desc  : GIRAFFE总体网络结构
'''
import torch.nn as nn

from GIRAFFE import (bounding_box_generator, camera, config, dataset, decoder,
                     generator, neural_renderer, renderer)

__all__ = [
    camera, config, dataset, renderer
]

decoder_dict = {
    'default': decoder.Decoder
}

generator_dict = {
    'default': generator.Generator
}

# 背景生成器仍采用Decoder
background_generator_dict = {
    'default': decoder.Decoder
}

bounding_box_generator_dict = {
    'default': bounding_box_generator.BoundingBoxGenerator
}

neural_renderer_dict = {
    'default': neural_renderer.NeuralRenderer
}


class GIRAFFE(nn.Module):
    '''
    GIRAFFE模型类

    Params:
        device -> device : torch.device
        discriminator -> nn.Module : discriminator network
        generator -> nn.Module : generator network
        generator_test -> nn.Module : generator network for test
    '''

    def __init__(self, 
                 device=None, 
                 discriminator=None,
                 generator=None,
                 generator_test=None,
                 **kwargs):
        super().__init__()

        # 初始化网络结构
        if discriminator is not None:
            self.discriminator = discriminator.to(device)
        else:
            self.discriminator = None
        if generator is not None:
            self.generator = generator.to(device)
        else:
            self.generator = None
        if generator_test is not None:
            self.generator_test = generator_test.to(device)
        else:
            self.generator_test = None

    def forward(self, batch_size, **kwargs):
        '''
        默认使用Generator for Test网络，否则采用原生Generator
        '''
        gen = self.generator_test
        if gen is None:
            gen = self.generator
        return gen(batch_size=batch_size)

    def generate_test_images(self):
        '''
        利用生成网络生成测试图片
        '''
        gen = self.generator_test
        if gen is None:
            gen = self.generator
        return gen()

    def to(self, device):
        '''
        将模型送到训练设备上

        Params:
            device -> device : torch.device
        '''
        model = super().to(device)
        model._device = device
        return model
