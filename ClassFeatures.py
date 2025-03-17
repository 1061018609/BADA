import os
import torch
import numpy as np

from PIL import Image
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F

from network import *
from dataset.zurich_night_dataset import zurich_night_DataSet
from configs.test_config import get_arguments as get_test_arguments
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.zurich_pair_dataset import zurich_pair_DataSet
# from configs.train_config import get_arguments as train_get_arguments
from dataset.ACDC_dataset import ACDCandBDD100K_Night_DataSet


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask
    

def prototypes_init(i_iter, train_args, lightnet, model, class_features):
    lightnet.eval()
    model.eval()

    args = get_test_arguments()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    #####
    # class_features = Class_Features(numbers=args.num_classes)
    

    # testloader = data.DataLoader(zurich_night_DataSet(args.data_dir, args.data_list, set=args.set))
    # #####
    # trainloader = data.DataLoader(
    #     cityscapesDataSet(train_args, train_args.data_dir, train_args.data_list, max_iters=1,#train_args.num_steps * train_args.iter_size * train_args.batch_size,
    #                       set=train_args.set),
    #     batch_size=train_args.batch_size, shuffle=True, num_workers=train_args.num_workers, pin_memory=True)
    # # trainloader_iter = enumerate(trainloader)

    # targetloader = data.DataLoader(zurich_pair_DataSet(train_args, train_args.data_dir_target, train_args.data_list_target,
    #                                                    max_iters=1,#train_args.num_steps * train_args.iter_size * train_args.batch_size,
    #                                                    set=train_args.set),
    #                                batch_size=train_args.batch_size, shuffle=True, num_workers=train_args.num_workers,
    #                                pin_memory=True)
    # # targetloader_iter = enumerate(targetloader)
    ACDCloader = data.DataLoader(ACDCandBDD100K_Night_DataSet(train_args, train_args.data_dir_acdc, train_args.data_list_acdc,
                                                                         max_iters=train_args.num_steps * train_args.iter_size * train_args.batch_size,
                                                                         set=train_args.set,
                                                                         active_mode=train_args.update_active_mode),
                                            batch_size=1, shuffle=False, num_workers=train_args.num_workers,
                                            pin_memory=True, drop_last=False)

    # if args.dataset == 'cityscapes':
    #     dataloader = trainloader
    # elif args.dataset == 'DarkZurich':
    #     dataloader = targetloader
    # elif args.dataset == 'ACDC':
    dataloader = ACDCloader

    interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)

    weights = torch.log(torch.FloatTensor(
        [0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341, 0.00207795, 0.0055127, 0.15928651,
         0.01157818, 0.04018982, 0.01218957, 0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
         0.00413907])).cuda()
    weights = (torch.mean(weights) - weights) / torch.std(weights) * train_args.std + 1.0

    for index, batch in enumerate(dataloader):
        if index % 10 == 0:
            print('%d processd' % index)
        # image, name = batch #if dataloader is from target test dataset
        # image, labels, _, _ = batch #if dataloader is from source train dataset
        # image_n, image_d, _, _, _ = batch #if dataloader is DarkZurich Night
        image, name = batch['image_n'], batch['name']

        image = image.to(0)
        with torch.no_grad():
            r = lightnet(image)
            enhancement = image + r
            if train_args.model == 'RefineNet':
                output2, features = model(enhancement)
            else:
                _, output2, features = model(enhancement, name=None)
                #####
                # output2, _ = model(enhancement, name=name)
                # _, output2 = model(enhancement)

        weights_prob = weights.expand(output2.size()[0], output2.size()[3], output2.size()[2], 19)
        weights_prob = weights_prob.transpose(1, 3)

        output2 = output2 * weights_prob

        #####
        vectors, ids = class_features.calculate_mean_vector(features, output2)
        for t in range(len(ids)):
            class_features.update_objective_SingleVector(ids[t], vectors[t].detach().cpu(), 'mean')

    torch.save(class_features.objective_vectors, '/home/jerry/xxz/DANNet/prototypes/prototypes_{}_{}_{}'.format('Training', i_iter, args.dataset))
    # torch.save(class_features.vectors_memory, '/home/jerry/xxz/DANNet/prototypes/vectors_memory_{}_{}_{}'.format(checkpoint_dir, j, args.dataset))
    # # print(class_features.road_memory.size())
    # for i in range(len(class_features.vectors_memory)):
    #     print(class_features.vectors_memory[i].size())



class Class_Features:
    def __init__(self, numbers = 19):
        self.class_numbers = numbers
        self.class_features = [[] for i in range(self.class_numbers)]
        self.num = np.zeros(numbers)
        self.objective_vectors = torch.zeros([self.class_numbers, 512])
        self.objective_vectors_num = torch.zeros([self.class_numbers])
        # self.road_memory = torch.zeros([1, 512])
        # self.vectors_memory = []
        # for class_num in range(19):
        #     self.vectors_memory.append(torch.zeros([1, 512]))

        # self.batch_vectors = torch.zeros([self.class_numbers, 512])
        # # self.batch_vectors.requires_grad = True
        # self.batch_vectors_num = torch.zeros([self.class_numbers])
        # # self.batch_vectors_num.requires_grad = True

        self.proto_momentum = 0.0001
        self.proto_temperature = 1

    def calculate_mean_vector_by_output(self, feat_cls, outputs, model):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = model.process_label(outputs_argmax.float())
        outputs_pred = outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def calculate_mean_vector(self, feat_cls, outputs, labels_val=None, model=None):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = self.process_label(outputs_argmax.float())
        if labels_val is None:
            outputs_pred = outputs_argmax
        else:
            labels_expanded = self.process_label(labels_val)
            outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []

        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids
    
    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, self.class_numbers+1, w, h).to(0)
        id = torch.where(label < self.class_numbers, label, torch.Tensor([self.class_numbers]).to(0))
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1
    
    def update_objective_SingleVector(self, id, vector, name='moving_average', start_mean=True):
        if vector.sum().item() == 0:
            return
        if start_mean and self.objective_vectors_num[id].item() < 100:
            name = 'mean'
        if name == 'moving_average':
            self.objective_vectors[id] = self.objective_vectors[id] * (1 - self.proto_momentum) + self.proto_momentum * vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
        elif name == 'mean':
            self.objective_vectors[id] = self.objective_vectors[id] * self.objective_vectors_num[id] + vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors[id] = self.objective_vectors[id] / self.objective_vectors_num[id]
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
            pass
        else:
            raise NotImplementedError('no such updating way of objective vectors {}'.format(name))

    def combine_batch_features(self, id, vector, batch_vectors, batch_vectors_num):
        if vector.sum().item() == 0:
            print('The vector.sum() == 0!')
            return
        batch_vectors[id] = batch_vectors[id] * batch_vectors_num[id] + vector.squeeze()
        batch_vectors_num[id] += 1
        batch_vectors[id] = batch_vectors[id] / batch_vectors_num[id]
        # self.batch_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
        # return self.batch_vectors
    def feat_prototype_distance(self, feat):
        N, C, H, W = feat.shape
        feat_proto_distance = -torch.ones((N, self.class_numbers, H, W)).to(feat.device)
        for i in range(self.class_numbers):
            #feat_proto_distance[:, i, :, :] = torch.norm(torch.Tensor(self.objective_vectors[i]).reshape(-1,1,1).expand(-1, H, W).to(feat.device) - feat, 2, dim=1,)
            feat_proto_distance[:, i, :, :] = torch.norm(self.objective_vectors[i].reshape(-1,1,1).expand(-1, H, W) - feat, 2, dim=1,)
        return feat_proto_distance

    def get_prototype_weight(self, feat, label=None, target_weak_params=None):
        # feat = self.full2weak(feat, target_weak_params)
        feat_proto_distance = self.feat_prototype_distance(feat)
        feat_nearest_proto_distance, feat_nearest_proto = feat_proto_distance.min(dim=1, keepdim=True)

        feat_proto_distance = feat_proto_distance - feat_nearest_proto_distance
        weight = F.softmax(-feat_proto_distance * self.proto_temperature, dim=1)
        return weight
