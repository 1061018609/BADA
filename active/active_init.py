import os
from shutil import copyfile

import numpy as np
from PIL import Image

import torch

##### 将BDD100k中的夜间图像换成DZ和ACDC的格式
image_dir = '/home/jerry/xxz/DANNet/dataset/ACDC/rgb_anon_trainvaltest/rgb_anon/night/train/'
GT_dir = '/home/jerry/xxz/DANNet/dataset/ACDC/gt_trainval/gt/night/train/'
label_mask_dir = '/home/jerry/xxz/DANNet/dataset/ACDC/gt_trainval/active_gt/night/label_mask/'
indicator_dir = '/home/jerry/xxz/DANNet/dataset/ACDC/gt_trainval/active_gt/night/indicator/'

night_dir_names = os.listdir(image_dir)
print(night_dir_names)
for dir_name in night_dir_names:
    image_names = os.listdir(image_dir + '/' + dir_name)
    print(dir_name, len(image_names))
    for name in image_names:
        image_name = name.split('_rgb_anon')[0]
        # image = Image.open(DZ_image_dir + dir_name + '//' + name)
        label = Image.open(GT_dir + dir_name + '/' + image_name + '_gt_labelTrainIds.png')

        # image = np.asarray(image)
        label = np.asarray(label)
        label_mask = np.ones_like(label) * 255
        label_mask = Image.fromarray(label_mask)
        label_mask.save(label_mask_dir + dir_name + '/' + image_name + '_gt_labelTrainIds.png')

        origin_mask = torch.from_numpy(label).long()
        active_indicator = torch.zeros_like(origin_mask, dtype=torch.bool)
        active_selected = torch.zeros_like(origin_mask, dtype=torch.bool)
        indicator = {
            'active': active_indicator,
            'selected': active_selected
        }
        torch.save(indicator, indicator_dir + dir_name + '/' +image_name + '_indicator.pth')