from dis import dis
import math
from pyexpat import features
from tkinter.messagebox import NO
import torch

import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
import random
# from .floating_region import FloatingRegionScore
# from .spatial_purity import SpatialPurity

import seaborn as sns
import matplotlib.pyplot as plt


class FeatureSpaceEntropy(nn.Module):

    def __init__(self, args, input_size=(1080, 1920), row_ratio=8, col_ratio=8, minibatch_size=2025, classes_prototypes=None):
        super(FeatureSpaceEntropy, self).__init__()
        self.row = int(input_size[0]/row_ratio)
        self.col = int(input_size[1]/col_ratio)
        self.full_row = input_size[0]
        self.full_col = input_size[1]
        self.minibatch_size = minibatch_size
        self.minibatch_iter = int((self.row*self.col)/minibatch_size)
        self.num_classes = args.num_classes
        self.feature_neibor_length = 9 #args.feature_neibor_length
        self.disparity_mode = args.disparity_mode

        # self.classes_prototypes = torch.load('D:/xxz/DANNet/prototype/prototypes_nopairactive_19000')
        # self.classes_prototypes = self.classes_prototypes.unsqueeze(0).to(0)
        self.classes_prototypes = classes_prototypes

    def forward(self, features_tensor, outputs, sample_index=None):
        b,c,h,w = features_tensor.size()
        feature_entropy_score = torch.zeros([b, 1, self.row*self.col, 1]).to(0)
        full_feature_entropy_score = torch.zeros([b, 1, self.full_row*self.full_col, 1]).to(0)
        classes_prototypes = self.classes_prototypes

        if sample_index is None:
            sample_index = torch.LongTensor(random.sample(range(h*w), self.row*self.col)).to(0)
            sample_index = torch.sort(sample_index)[0]

        features_tensor = features_tensor.permute(0,2,3,1).view(1,h*w,512)
        features_tensor = torch.index_select(features_tensor, 1, sample_index)

        class_ids = outputs.squeeze(dim=0)  # [19, h ,w]
        class_ids = torch.softmax(class_ids, dim=0)  # [19, h, w]
        class_ids = torch.argmax(class_ids, dim=0).view(h*w,1)  # [h*w, 1]
        class_ids = torch.index_select(class_ids, dim=0, index=sample_index)
        # class_ids = class_ids.repeat(1, self.minibatch_size, self.feature_neibor_length)

        for mini_batch in range(self.minibatch_iter):
            now_batch_features = features_tensor[:,self.minibatch_size*mini_batch:self.minibatch_size*(mini_batch+1),:]
            if self.disparity_mode == 'E':
                disparity = self.EuclideanDistance(now_batch_features, classes_prototypes)
                disparity = disparity.squeeze(0)
                _, descending_index = torch.sort(disparity, dim=1, descending=False)
            elif self.disparity_mode == 'C':
                disparity = self.cosin_similarity(now_batch_features, classes_prototypes)
                _, descending_index = torch.sort(disparity, dim=1, descending=False)

            disparity_score = descending_index - class_ids
            disparity_score[disparity_score != 0] = 1
            disparity_score = 1-disparity_score
            disparity_score = torch.argmax(disparity_score, dim=1, keepdim=True)
            disparity_score = (disparity_score/int(self.num_classes-1)).unsqueeze(0).unsqueeze(0)
            feature_entropy_score[:,:,self.minibatch_size*mini_batch:self.minibatch_size*(mini_batch+1),:] = disparity_score #[1,1,minibatch_size,1]

        full_feature_entropy_score = full_feature_entropy_score.scatter(dim=2, index=sample_index.view(b, 1, self.row*self.col, 1), src=feature_entropy_score)
        full_feature_entropy_score = full_feature_entropy_score.view(1, 1, self.full_row, self.full_col)

        return full_feature_entropy_score.squeeze(0).squeeze(0)
    
    def EuclideanDistance(self, t1,t2):
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
    
    def cosin_similarity(self, t1, t2):
        t1 = F.normalize(t1, p=2, dim=2)
        t1 = t1.squeeze(0)
        t2 = F.normalize(t2, p=2, dim=2)
        t2 = t2.squeeze(0)
        # print(t1.size(), t2.size())
        sim = torch.mm(t1, t2.t())
        # print(sim.size())
        return sim
    
