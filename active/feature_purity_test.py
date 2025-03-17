from dis import dis
import math
from pyexpat import features
import torch

import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
import random
# from .floating_region import FloatingRegionScore
# from .spatial_purity import SpatialPurity

import seaborn as sns
import matplotlib.pyplot as plt

def EuclideanDistance(t1,t2):
    dim=len(t1.size())
    if dim==2:
        N,C=t1.size()
        M,_=t2.size()
        dist = -2 * torch.matmul(t1, t2.permute(1, 0))
        dist += torch.sum(t1 ** 2, -1).view(N, 1)
        dist += torch.sum(t2 ** 2, -1).view(1, M)
        dist=torch.sqrt(dist)
        return dist
    elif dim==3:
        B,N,_=t1.size()
        _,M,_=t2.size()
        dist = -2 * torch.matmul(t1, t2.permute(0, 2, 1))
        dist += torch.sum(t1 ** 2, -1).view(B, N, 1)
        dist += torch.sum(t2 ** 2, -1).view(B, 1, M)
        # dist=torch.sqrt(dist)
        dist = torch.clamp(dist, min=1e-12, max=None)
        return dist
    else:
        print('error!')

def cosin_similarity(t1, t2):
    t1 = F.normalize(t1, p=2, dim=2)
    t1 = t1.squeeze(0)
    t2 = F.normalize(t2, p=2, dim=2)
    t2 = t2.squeeze(0)
    print(t1.size(), t2.size())
    sim = torch.mm(t1, t2.t())
    print(sim.size())
    return sim

#####gather test
# a = torch.FloatTensor([i for i in range(12)]).view(1,2,2,3)
# indices = torch.Tensor([0,0,0,1,1,1]).view(1,1,2,3)
# b = torch.gather(a, dim=1, index=indices.long())
# print(b)

#####scatter test
# a = torch.zeros([1,16,1])
# b = torch.ones([1,4,1])
# # index = torch.LongTensor([1,2,10,14]).view(1,4,1)
# index = torch.LongTensor(random.sample(range(16), 4))
# print(index)
# index = torch.sort(index)[0]
# print(index)
# d = torch.index_select(a, 1, index)
# print(d)
# index = index.view(1,4,1)
# c = a.scatter(dim=1, index=index, src=b)
# print(a)
# print(c)

#####feature entropy test
disparity_compute_mode = 'E'
# full_row = 16#1080
# full_col = 16#1920
# row = 4#270
# col = 4#480
# minibatch_size = row*col

# features_tensor = torch.randn([1,512,full_row,full_col]).to(0)
# features_tensor = features_tensor.view(1, full_row*full_col, 512)

classes_prototypes = torch.load('D:/xxz/DANNet/prototype/prototypes_nopairactive_19000')
classes_prototypes = classes_prototypes.unsqueeze(0).to(0)
print(classes_prototypes.size())
print(torch.max(classes_prototypes), torch.min(classes_prototypes))

vectors_memory_name = 'vectors_memory_PSPNet_active_nopair_1000'
ACDC_class_vectors_list = torch.load('D:/xxz/DANNet/prototype/{}'.format(vectors_memory_name))
for i in range(len(ACDC_class_vectors_list)):
    if i == 0:
        ACDC_class_features_tensor = ACDC_class_vectors_list[i]
        label = torch.Tensor([i]*ACDC_class_vectors_list[i].size()[0]).unsqueeze(0)
        classes_num_count_list = [ACDC_class_vectors_list[i].size()[0]]
    else:
        ACDC_class_features_tensor = torch.cat([ACDC_class_features_tensor, ACDC_class_vectors_list[i]], 0)
        label = torch.cat([label, torch.Tensor([i]*ACDC_class_vectors_list[i].size()[0]).unsqueeze(0)], 1)
        classes_num_count_list.append(ACDC_class_vectors_list[i].size()[0])
label = label.to(0)
ACDC_class_features_tensor = ACDC_class_features_tensor.view(1, -1, 512).to(0)
print(classes_num_count_list)
print(ACDC_class_features_tensor.size())
print(torch.max(ACDC_class_features_tensor), torch.min(ACDC_class_features_tensor))

### 余弦相似度
if disparity_compute_mode == 'C':
    sim = cosin_similarity(ACDC_class_features_tensor, classes_prototypes)
    sim_descending, descending_index = torch.sort(sim, dim=1, descending=True)
elif disparity_compute_mode == 'E':
    sim = EuclideanDistance(ACDC_class_features_tensor, classes_prototypes)
    sim = sim.squeeze(0)
    sim_descending, descending_index = torch.sort(sim, dim=1, descending=False)
print('sim.size(): ', sim.size())
disparity_score = descending_index - label.permute(1,0)
disparity_score[disparity_score != 0] = 1
disparity_score = 1-disparity_score
disparity_score = torch.argmax(disparity_score, dim=1, keepdim=True)
print(disparity_score.size())

for i in range(18):
    disparity_score_count = disparity_score.clone()
    disparity_score_count[disparity_score_count != (i+1)] = 0
    disparity_score_count[disparity_score_count != 0] = 1
    print(torch.sum(disparity_score_count))
print(torch.max(disparity_score), torch.min(disparity_score))
print(torch.sum(disparity_score))


if disparity_compute_mode == 'C':
    max_value, max_index = torch.max(sim, dim=1)
elif disparity_compute_mode == 'E':
    max_value, max_index = torch.min(sim, dim=1)
wrong_match = max_index - label.squeeze(0)
wrong_match[wrong_match != 0] = 1
wrong_match_num = torch.sum(wrong_match)
print(wrong_match_num)

# 基于相似度的熵值计算
# p = F.softmax(sim, dim=1)
# print(p.size())
# print(torch.max(p), torch.min(p))
# pixel_entropy = torch.sum(-p * torch.log(p + 1e-6), dim=1) / math.log(19)
# print(pixel_entropy.size())
# print(torch.max(pixel_entropy), torch.min(pixel_entropy))

# wrong_match_entropy = torch.sum(pixel_entropy * wrong_match) / wrong_match_num
# right_match_entropy = torch.sum(pixel_entropy * (1-wrong_match)) / torch.sum((1-wrong_match))
# print(wrong_match_entropy, right_match_entropy)


### 欧氏距离
# feature_to_prototype_distance = EuclideanDistance(ACDC_class_features_tensor, classes_prototypes)
# print(feature_to_prototype_distance[:, :1, :])
# print(torch.max(feature_to_prototype_distance), torch.min(feature_to_prototype_distance))

# feature_to_prototype_distance = feature_to_prototype_distance/torch.sum(feature_to_prototype_distance, dim=2).unsqueeze(2)
# print(feature_to_prototype_distance[:, :1, :])

# p = F.softmax(feature_to_prototype_distance, dim=2)
# print(p[:, :1, :])
# print(torch.max(p), torch.min(p))