import math
from tkinter.messagebox import NO
from turtle import forward
from xml.sax.handler import feature_external_ges
import torch

import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
# from .floating_region import FloatingRegionScore
# from .spatial_purity import SpatialPurity

import seaborn as sns
import matplotlib.pyplot as plt
import random


class FeatureSpacePurityEntropyScore(nn.Module):

    def __init__(self, args, input_size=(1080, 1920), row_ratio=8, col_ratio=8, minibatch_size=2025, classes_prototypes=None):
        super(FeatureSpacePurityEntropyScore, self).__init__()
        self.row = int(input_size[0]/row_ratio)
        self.col = int(input_size[1]/col_ratio)
        self.full_row = input_size[0]
        self.full_col = input_size[1]
        self.minibatch_size = minibatch_size
        self.minibatch_iter = int((self.row*self.col)/minibatch_size)
        self.num_classes = args.num_classes
        self.feature_neibor_length = 9 #args.feature_neibor_length
        self.disparity_mode = args.disparity_mode #E means EuclideanDistance and C menas CosinSimilarity
        self.classes_prototypes = classes_prototypes

    def forward(self, features_tensor, outputs):
        b,c,h,w = features_tensor.size()
        #purity_score
        feature_purity_score = torch.zeros([b, 1, self.row*self.col, 1]).to(0)
        full_feature_purity_score = torch.zeros([b, 1, self.full_row*self.full_col, 1]).to(0)
        #entropy_score
        feature_entropy_score = torch.zeros([b, 1, self.row*self.col, 1]).to(0)
        full_feature_entropy_score = torch.zeros([b, 1, self.full_row*self.full_col, 1]).to(0)

        sample_index = torch.LongTensor(random.sample(range(h*w), self.row*self.col)).to(0)
        sample_index = torch.sort(sample_index)[0]
        
        features_tensor = features_tensor.permute(0,2,3,1).view(1,h*w,512)
        features_tensor = torch.index_select(features_tensor, 1, sample_index)
        

        class_ids = outputs.squeeze(dim=0)  # [19, h ,w]
        class_ids = torch.softmax(class_ids, dim=0)  # [19, h, w]
        class_ids = torch.argmax(class_ids, dim=0).view(h*w,1,1)  # [h*w, 1, 1]
        class_ids = torch.index_select(class_ids, dim=0, index=sample_index)
        class_ids_useinentropy = class_ids.clone()
        class_ids = class_ids.repeat(1, self.minibatch_size, self.feature_neibor_length)


        for mini_batch in range(self.minibatch_iter):
            now_batch_features = features_tensor[:,self.minibatch_size*mini_batch:self.minibatch_size*(mini_batch+1),:]

            distance = self.EuclideanDistance(now_batch_features, features_tensor)
            distance, sorted_indices = torch.sort(distance, descending=False, dim=2)

            distance = distance[:, :, :self.feature_neibor_length]
            sorted_indices = sorted_indices[:, :, :self.feature_neibor_length]

            batch_class_ids = torch.gather(class_ids, dim=0, index=sorted_indices.long())
            batch_class_one_hot = F.one_hot(batch_class_ids, num_classes=19).float().permute(0,3,1,2)
            summary = torch.sum(batch_class_one_hot, dim=3, keepdim=True)
            region_count = torch.sum(summary, dim=1, keepdim=True)
            dist = summary/region_count
            features_impurity = torch.sum(-dist * torch.log(dist+1e-6), dim=1, keepdim=True) / math.log(self.num_classes) #math.log(19)
            feature_purity_score[:,:,self.minibatch_size*mini_batch:self.minibatch_size*(mini_batch+1),:] = features_impurity

            if self.disparity_mode == 'E':
                disparity = self.EuclideanDistance(now_batch_features, self.classes_prototypes)
                disparity = disparity.squeeze(0)
                _, descending_index = torch.sort(disparity, dim=1, descending=False)
            elif self.disparity_mode == 'C':
                disparity = self.cosin_similarity(now_batch_features, self.classes_prototypes)
                _, descending_index = torch.sort(disparity, dim=1, descending=False)

            disparity_score = descending_index - class_ids_useinentropy
            disparity_score[disparity_score != 0] = 1
            disparity_score = 1-disparity_score
            disparity_score = torch.argmax(disparity_score, dim=1, keepdim=True)
            disparity_score = (disparity_score/int(self.num_classes-1)).unsqueeze(0).unsqueeze(0)
            feature_entropy_score[:,:,self.minibatch_size*mini_batch:self.minibatch_size*(mini_batch+1),:] = disparity_score #[1,1,minibatch_size,1]

        # feature_purity_score = feature_purity_score.view(1,1,row,col)
        full_feature_purity_score = full_feature_purity_score.scatter(dim=2, index=sample_index.view(b, 1, self.row*self.col, 1), src=feature_purity_score)
        full_feature_purity_score = full_feature_purity_score.view(1, 1, self.full_row, self.full_col)
        
        full_feature_entropy_score = full_feature_entropy_score.scatter(dim=2, index=sample_index.view(b, 1, self.row*self.col, 1), src=feature_entropy_score)
        full_feature_entropy_score = full_feature_entropy_score.view(1, 1, self.full_row, self.full_col)
        
        return full_feature_purity_score.squeeze(0).squeeze(0), full_feature_entropy_score.squeeze(0).squeeze(0)


    
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

    def CosinSimilarity(self, t1, t2):
        t1 = F.normalize(t1, p=2, dim=2)
        t1 = t1.squeeze(0)
        t2 = F.normalize(t2, p=2, dim=2)
        t2 = t2.squeeze(0)
        # print(t1.size(), t2.size())
        sim = torch.mm(t1, t2.t())
        # print(sim.size())
        return sim

# feature_tensors = torch.randn(1,512,1080,1920).to(0)
# outputs = torch.randn(1,19,1080,1920).to(0)

# featurespaceregionscore = FeatureSpaceRegionScore(args=None).to(0)

# full_feature_purity_score = featurespaceregionscore(feature_tensors, outputs)
# print(torch.max(full_feature_purity_score), torch.min(full_feature_purity_score))
# print(full_feature_purity_score.size())