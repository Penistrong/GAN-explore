#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File  : dataset.py
@Author: Penistrong
@Email : chen18296276027@gmail.com
@Date  : 2021-11-23 周二 18:05:40
@Desc  : GIRAFFE使用的数据集，由tools.config调用该模块加载cfg文件中的数据集
'''
import glob
import logging
import os
import random

import numpy as np
from PIL import Image, ImageFile
from torch.utils import data
from torchvision import transforms

# 启用修复破损图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

class ImagesDataset(data.Dataset):
    '''
    默认的图片数据集(CelebA)

    Params:
    dataset_folder -> str : 数据集本地路径
    size -> int: 数据集图片的尺寸
    celebA_center_crop -> bool : 在CelebA数据集上是否采用中心裁剪
    random_crop (bool): 是否启用随机裁剪
    use_tanh_range (bool): 是否要将图片像素放缩到tan的[-1,1]范围里
    '''

    def __init__(self, dataset_folder,  size=64, celebA_center_crop=False,
                 random_crop=False, use_tanh_range=False):

        self.size = size
        assert(not(celebA_center_crop and random_crop))
        if random_crop:
            self.transform = [
                transforms.Resize(size),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        elif celebA_center_crop:
            if size <= 128:     # celebA
                crop_size = 108
            else:               # celebA_HQ
                crop_size = 650
            self.transform = [
                transforms.CenterCrop(crop_size),
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        else:
            self.transform = [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        if use_tanh_range:
            self.transform += [
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(self.transform)

        self.data_type = os.path.basename(dataset_folder).split(".")[-1]
        assert(self.data_type in ["jpg", "png", "npy"])

        import time
        t0 = time.time()
        print('Start loading file addresses ...')
        images = glob.glob(dataset_folder)
        random.shuffle(images)
        t = time.time() - t0
        print('done! time:', t)
        print("Number of images found: %d" % len(images))

        self.images = images
        self.length = len(images)

    def __getitem__(self, idx):
        try:
            buf = self.images[idx]
            if self.data_type == 'npy':
                img = np.load(buf)[0].transpose(1, 2, 0)
                img = Image.fromarray(img).convert("RGB")
            else:
                img = Image.open(buf).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)
            data = {
                'image': img
            }
            return data
        except Exception as e:
            print(e)
            print("Warning: Error occurred when loading file %s" % buf)
            return self.__getitem__(np.random.randint(self.length))

    def __len__(self):
        return self.length
