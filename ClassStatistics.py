import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image

active_path = 'D:/xxz/DANNet/dataset/ACDC/gt_trainval/active_gt/night_nopair_featureuncertainty_005/label_mask'
gt_path = 'D:/xxz/DANNet/dataset/ACDC/gt_trainval/gt/night/train'
citynames = os.listdir(gt_path)
print(citynames)

gt_class_statistics = [(i-i) for i in range(19)]
active_class_statistics = [(i-i) for i in range(19)]

flag = 0
for cityname in citynames:
    active_city = active_path + '/' + cityname
    gt_city = gt_path + '/' + cityname
    label_name_list = os.listdir(active_city)
    
    for label_name in label_name_list:
        active_label_name = active_city + '/' + label_name
        gt_label_name = gt_city + '/' + label_name

        active_label = np.array(Image.open(active_label_name), dtype=np.uint8)
        gt_label = np.array(Image.open(gt_label_name), dtype=np.uint8)

        # print(np.unique(active_label), np.unique(gt_label))
        for class_id in range(19):
            active_class_mask = np.zeros_like(active_label)
            active_class_mask[active_label == class_id] = 1
            active_now_class_sum = np.sum(active_class_mask)
            active_class_statistics[class_id] = active_class_statistics[class_id] + active_now_class_sum

            gt_class_mask = np.zeros_like(gt_label)
            gt_class_mask[gt_label == class_id] = 1
            gt_now_class_sum = np.sum(gt_class_mask)
            gt_class_statistics[class_id] = gt_class_statistics[class_id] + gt_now_class_sum
        
        flag = flag + 1
        if flag%20 == 0:
            print('We have processed {} labels!'.format(flag))

gt_class_statistics = np.array(gt_class_statistics)
active_class_statistics = np.array(active_class_statistics)

print(gt_class_statistics)
print(active_class_statistics)
print('#'*50)
print(gt_class_statistics/np.sum(gt_class_statistics))
print(active_class_statistics/np.sum(active_class_statistics))
print('#'*50)
print((active_class_statistics/np.sum(active_class_statistics))/(gt_class_statistics/np.sum(gt_class_statistics)))
