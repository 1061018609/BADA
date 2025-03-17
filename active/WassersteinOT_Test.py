# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from cmath import cos
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torch.optim.lr_scheduler import _LRScheduler
# from models.sync_batchnorm import SynchronizedBatchNorm2d

def sinkhorn(class_distribution, out):
        Q = torch.exp(out / 0.05).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0] # how many prototypes
        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
    #     dist.all_reduce(sum_Q)
        Q /= sum_Q
        for it in range(3):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q *= class_distribution.unsqueeze(1)

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        # Q = torch.argmax(Q, 0)
        return Q.t()

vectors_memory_name = 'vectors_memory_PSPNet_active_nopair_19000'
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

class_distribution = np.load('D:/xxz/CPSL-main/CPSL-main/Pseudo/class_distribution.npy')
class_distribution = torch.Tensor(class_distribution).to(0)
print(class_distribution)


ACDC_class_features_tensor = F.normalize(ACDC_class_features_tensor, dim = 2, p=2)
print(ACDC_class_features_tensor.size())


classes_prototypes = torch.load('D:/xxz/DANNet/prototype/prototypes_nopairactive_19000')
classes_prototypes = classes_prototypes.to(0)
classes_prototypes = F.normalize(classes_prototypes, dim=1, p=2)
print(classes_prototypes.size())

for i in range(ACDC_class_features_tensor.shape[0]):
    proto = F.normalize(classes_prototypes, dim=1, p=2)
    out = torch.mm(ACDC_class_features_tensor[i], proto.t())
    with torch.no_grad():
        out_ = out.detach()
        q = sinkhorn(class_distribution, out_)
        print(q.size())

        sim_descending, descending_index = torch.sort(q, dim=1, descending=True)
        disparity_score = descending_index - label.permute(1,0)
        disparity_score[disparity_score != 0] = 1
        disparity_score = 1-disparity_score
        disparity_score = torch.argmax(disparity_score, dim=1)

        for i in range(18):
            disparity_score_count = disparity_score.clone()
            disparity_score_count[disparity_score_count != (i+1)] = 0
            print(torch.sum(disparity_score_count))
        print(torch.max(disparity_score), torch.min(disparity_score))
        print(torch.sum(disparity_score))