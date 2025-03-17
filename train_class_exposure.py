from multiprocessing import pool
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from network import *

from dataset.zurich_pair_dataset import zurich_pair_DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from configs.train_config import get_arguments


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def weightedMSE(D_out, D_label):
    return torch.mean((D_out - D_label).abs() ** 2)


def adjust_learning_rate(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

class L_exp_classes(nn.Module):
    def __init__(self, pool_flag=False, patch_size=None, batch_size=8, classes_num=19, input_size=512):
        super(L_exp_classes, self).__init__()
        self.classes_exposure = torch.load('/home/jerry/xxz/DANNet/prototypes/exposure_onlyGAN39')
        # self.classes_exposure = self.classes_exposure.view(batch_size, classes_num, 1, 1)
        self.interp = nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=True)
        self.pool_flag = pool_flag
        self.classes_num = classes_num
        if pool_flag:
            self.pool = nn.AvgPool2d(patch_size)

    def forward(self, x, label):
        #####首先处理label
        label = self.interp(label)
        label = torch.argmax(label, dim=1, keepdim=True)

        x = torch.mean(x, 1, keepdim=True)
        classes_mean_val = torch.zeros_like(x)
        for i in range(self.classes_num):
            classes_mean_val[label == i] = self.classes_exposure[i]
        if self.pool_flag:
            mean = self.pool(x)
            d = torch.mean(torch.pow(mean - torch.FloatTensor([mean_val]).cuda(), 2))
        else:
            d = torch.mean(torch.pow(x - classes_mean_val.cuda(), 2))
        return d


def main():
    args = get_arguments()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # torch.cuda.set_device(0)
    device = torch.device("cuda")
    # device=0

    cudnn.enabled = True
    cudnn.benchmark = True

    if args.model == 'PSPNet':
        model = PSPNet(num_classes=args.num_classes)
    if args.model == 'DeepLab':
        model = Deeplab(num_classes=args.num_classes)
    if args.model == 'RefineNet':
        model = RefineNet(num_classes=args.num_classes, imagenet=False)
    saved_state_dict = torch.load(args.restore_from)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0] == 'fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
    model.load_state_dict(new_params)

    model.train()
    model.to(device)

    lightnet = LightNet()
    if args.all_pretrained:
        saved_lightnet = torch.load(args.lightnet_restore_from)
        lightnet.load_state_dict(saved_lightnet)
    lightnet.train()
    lightnet.to(device)

    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    if args.all_pretrained:
        saved_d1 = torch.load(args.d1_restore_from)
        model_D1.load_state_dict(saved_d1)
    model_D1.train()
    model_D1.to(device)

    model_D2 = FCDiscriminator(num_classes=args.num_classes)
    if args.all_pretrained:
        saved_d2 = torch.load(args.d2_restore_from)
        model_D2.load_state_dict(saved_d2)
    model_D2.train()
    model_D2.to(device)
    #####如果要使用预训练好的全部模型，那么就把学习率都下调一个数量级
    if args.all_pretrained:
        args.learning_rate = args.learning_rate * 0.1
        args.learning_rate_D = args.learning_rate_D * 0.1

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    trainloader = data.DataLoader(
        cityscapesDataSet(args, args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                          set=args.set),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(zurich_pair_DataSet(args, args.data_dir_target, args.data_list_target,
                                                       max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                       set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)
    targetloader_iter = enumerate(targetloader)

    optimizer = optim.SGD(list(model.parameters())+list(lightnet.parameters()),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer.zero_grad()
    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()
    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    weights = torch.log(torch.FloatTensor([0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341,
                                           0.00207795, 0.0055127, 0.15928651, 0.01157818, 0.04018982, 0.01218957,
                                           0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
                                           0.00413907])).cuda()
    weights = (torch.mean(weights) - weights) / torch.std(weights) * args.std + 1.0

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weights)
    if args.use_forge_labels:
        static_loss = StaticLoss(num_classes=19, weight=weights)
    else:
        static_loss = StaticLoss(num_classes=11, weight=weights[:11])

    # loss_exp_z = L_exp_z(32)
    loss_exp_classes = L_exp_classes()
    loss_TV = L_TV()
    loss_SSIM = SSIM()

    interp = nn.Upsample(size=(args.input_size, args.input_size), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(args.input_size_target, args.input_size_target), mode='bilinear', align_corners=True)

    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):
        loss_seg_value = 0
        loss_adv_target_value = 0
        loss_pseudo = 0
        loss_D_value1 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(args, optimizer, i_iter)
        optimizer_D1.zero_grad()
        adjust_learning_rate_D(args, optimizer_D1, i_iter)
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(args, optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):
            # train G
            for param in model_D1.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False

            # train with target
            _, batch = targetloader_iter.__next__()
            if args.pseudo_mode == 'generate_previously':
                images_n, images_d, _, _, night_pseudo_labels = batch
                night_pseudo_labels = night_pseudo_labels.to(device)
            else:
                images_n, images_d, _, _, _ = batch
            images_d = images_d.to(device)

            mean_light = images_n.mean()
            # print(mean_light)
            # print(images_d.size())
            r = lightnet(images_d)
            enhanced_images_d = images_d + r
            loss_enhance = 10*loss_TV(r)+torch.mean(loss_SSIM(enhanced_images_d, images_d))#\
                           #+torch.mean(loss_exp_z(enhanced_images_d, mean_light))

            if args.model == 'RefineNet':
                pred_target_d = model(enhanced_images_d)
            else:
                _, pred_target_d = model(enhanced_images_d) #65x65
            
            #####计算基于类别的亮度一致性损失
            exposure_label = pred_target_d.detach()
            loss_exposure = torch.mean(loss_exp_classes(enhanced_images_d, exposure_label))
            loss_enhance = loss_enhance + loss_exposure
            #####计算基于类别的亮度一致性损失

            pred_target_d = interp_target(pred_target_d) #960x960
            D_out_d = model_D1(F.softmax(pred_target_d, dim=1))
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(source_label).to(device)
            loss_adv_target_d = weightedMSE(D_out_d, D_label_d)
            loss = 0.01 * loss_adv_target_d + 0.01 * loss_enhance
            loss = loss / args.iter_size
            loss.backward()


            images_n = images_n.to(device)
            r = lightnet(images_n)
            enhanced_images_n = images_n + r
            loss_enhance = 10*loss_TV(r)+torch.mean(loss_SSIM(enhanced_images_n, images_n))#\
                        #    + torch.mean(loss_exp_z(enhanced_images_n, mean_light))

            if args.model == 'RefineNet':
                pred_target_n = model(enhanced_images_n)
            else:
                _, pred_target_n = model(enhanced_images_n)
            pred_target_n = interp_target(pred_target_n)

            #####计算基于类别的亮度一致性损失
            exposure_label = pred_target_n.detach()
            loss_exposure = torch.mean(loss_exp_classes(enhanced_images_n, exposure_label))
            loss_enhance = loss_enhance + loss_exposure
            #####计算基于类别的亮度一致性损失
            
            if args.pseudo_mode == 'generate_previously':
                psudo_gt = night_pseudo_labels
                if not args.use_forge_labels:
                    psudo_gt[psudo_gt >= 11] = 255
            elif args.pseudo_mode == 'generate_with_training':
                psudo_prob = torch.zeros_like(pred_target_d)
                threshold = torch.ones_like(pred_target_d[:,:11,:,:])*0.2
                threshold[pred_target_d[:,:11,:,:]>0.4] = 0.8
                psudo_prob[:,:11,:,:] = threshold*pred_target_d[:,:11,:,:].detach() + (1-threshold)*pred_target_n[:,:11,:,:].detach()
                psudo_prob[:,11:,:,:] = pred_target_n[:,11:,:,:].detach()

                weights_prob = weights.expand(psudo_prob.size()[0], psudo_prob.size()[3], psudo_prob.size()[2], 19)
                weights_prob = weights_prob.transpose(1, 3)
                psudo_prob = psudo_prob*weights_prob
                psudo_gt = torch.argmax(psudo_prob.detach(), dim=1)
                psudo_gt[psudo_gt >= 11] = 255

            D_out_n_19 = model_D2(F.softmax(pred_target_n, dim=1))
            D_label_n_19 = torch.FloatTensor(D_out_n_19.data.size()).fill_(source_label).to(device)
            loss_adv_target_n_19 = weightedMSE(D_out_n_19, D_label_n_19,)

            if args.use_forge_labels:
                loss_pseudo = static_loss(pred_target_n, psudo_gt)
            else:
                loss_pseudo = static_loss(pred_target_n[:,:11,:,:], psudo_gt)
            loss = 0.01 * loss_adv_target_n_19 + loss_pseudo + 0.01 * loss_enhance
            loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value += loss_adv_target_n_19.item() / args.iter_size

            # train with source
            _, batch = trainloader_iter.__next__()

            images, labels, _, _ = batch
            images = images.to(device)
            labels = labels.long().to(device)
            r = lightnet(images)
            enhanced_images = images + r
            loss_enhance = 10 * loss_TV(r) + torch.mean(loss_SSIM(enhanced_images, images)) #\
                           #+ torch.mean(loss_exp_z(enhanced_images, mean_light))

            if args.model == 'RefineNet':
                pred_c = model(enhanced_images)
            else:
                _, pred_c = model(enhanced_images)

            #####计算基于类别的亮度一致性损失
            exposure_label = pred_c.detach()
            loss_exposure = torch.mean(loss_exp_classes(enhanced_images, exposure_label))
            loss_enhance = loss_enhance + loss_exposure
            #####计算基于类别的亮度一致性损失

            pred_c = interp(pred_c)
            loss_seg = seg_loss(pred_c, labels)
            loss = loss_seg + loss_enhance
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value += loss_seg.item() / args.iter_size

            # train D
            for param in model_D1.parameters():
                param.requires_grad = True
            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred_c = pred_c.detach()
            D_out1 = model_D1(F.softmax(pred_c, dim=1))
            D_label1 = torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device)
            loss_D1 = weightedMSE(D_out1, D_label1)
            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D1.backward()
            loss_D_value2 += loss_D1.item()

            pred_c = pred_c.detach()
            D_out2 = model_D2(F.softmax(pred_c, dim=1))
            D_label2 = torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device)
            loss_D2 = weightedMSE(D_out2, D_label2)
            loss_D2 = loss_D2 / args.iter_size /2
            loss_D2.backward()
            loss_D_value2 += loss_D2.item()

            # train with target
            pred_target_d = pred_target_d.detach()
            D_out1 = model_D1(F.softmax(pred_target_d, dim=1))
            D_label1 = torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(device)
            loss_D1 = weightedMSE(D_out1, D_label1)
            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D1.backward()
            loss_D_value1 += loss_D1.item()

            pred_target_n = pred_target_n.detach()
            D_out2 = model_D2(F.softmax(pred_target_n, dim=1))
            D_label2 = torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(device)
            loss_D2 = weightedMSE(D_out2, D_label2)
            loss_D2 = loss_D2 / args.iter_size / 2
            loss_D2.backward()
            loss_D_value2 += loss_D2.item()

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()

        print(
            'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv = {3:.3f}, loss_D1 = {4:.3f}, loss_D2 = {5:.3f}, loss_pseudo = {6:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value,
                loss_adv_target_value, loss_D_value1, loss_D_value2, loss_pseudo))

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'dannet' + str(i_iter) + '.pth'))
            torch.save(lightnet.state_dict(), os.path.join(args.snapshot_dir, 'dannet_light' + str(i_iter) + '.pth'))
            torch.save(model_D1.state_dict(), os.path.join(args.snapshot_dir, 'dannet_d1_' + str(i_iter) + '.pth'))
            torch.save(model_D2.state_dict(), os.path.join(args.snapshot_dir, 'dannet_d2_' + str(i_iter) + '.pth'))


if __name__ == '__main__':
    main()
