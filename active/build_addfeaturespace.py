import math
import torch

import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from .floating_region import FloatingRegionScore
from .spatial_purity import SpatialPurity

import seaborn as sns
import matplotlib.pyplot as plt

# palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
#            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
#            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
# zero_pad = 256 * 3 - len(palette)
# for i in range(zero_pad):
#     palette.append(0)


# def colorize_mask(mask):
#     new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
#     new_mask.putpalette(palette)
#     return new_mask


def RegionSelection(args, lightnet, seg_net, tgt_epoch_loader):

    lightnet.eval()
    seg_net.eval()
    if args.use_lightM:
        seg_net.cca.active_state = 'update'

    floating_region_score = FloatingRegionScore(in_channels=args.num_classes, size=2 * args.radius_k + 1).cuda()
    per_region_pixels = (2 * args.radius_k + 1) ** 2
    active_radius = args.radius_k
    mask_radius = args.radius_k * 2
    active_ratio = args.active_ratio / len(args.active_select_iter_list)

    weights = torch.log(torch.FloatTensor([0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341,
                                           0.00207795, 0.0055127, 0.15928651, 0.01157818, 0.04018982, 0.01218957,
                                           0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
                                           0.00413907])).cuda()
    weights = (torch.mean(weights) - weights) / torch.std(weights) * args.std + 1.0

    with torch.no_grad():
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask = tgt_data['image_n'], tgt_data['path_to_mask']
            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['label_size']
            active_indicator = tgt_data['active_indicator']
            selected_indicator = tgt_data['active_selected']
            path2indicator = tgt_data['path_to_indicator']
            if args.use_lightM:
                lightMask = tgt_data["lightMask"]
                lightMask = lightMask.to(0)

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            # tgt_feat = feature_extractor(tgt_input)
            r = lightnet(tgt_input)
            enhanced_image = r + tgt_input
            if args.model == 'RefineNet':
                tgt_out = seg_net(enhanced_image)
            else:
                if args.use_lightM:
                    _, tgt_out, _ = seg_net(enhanced_image, data_from='target', light_M=lightMask)
                else:
                    _, tgt_out = seg_net(enhanced_image)

            weights_prob = weights.expand(tgt_out.size()[0], tgt_out.size()[3], tgt_out.size()[2], 19)
            weights_prob = weights_prob.transpose(1, 3)
            tgt_out = tgt_out * weights_prob


            for i in range(len(origin_mask)):
                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                num_pixel_cur = size[0] * size[1]
                active = active_indicator[i]
                selected = selected_indicator[i]

                output = tgt_out[i:i + 1, :, :, :]
                output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
                
                ################
                # output2 = output.detach()
                # name = tgt_data['name'][0]
                # print(name)
                # if name.split('/')[0] != 'BDD100K':
                #     exit()
                # output_prob = F.softmax(output2, dim=1).cpu().data[0].numpy()
                # output2 = output2.cpu().data[0].numpy()

                # output2 = output2.transpose(1,2,0)
                # output_prob = output_prob.transpose(1,2,0)
                # maxprob = np.asarray(np.max(output_prob, axis=2))
                # #maxprob[maxprob > 0.9] = 1
                # sns.set()
                # sns.heatmap(maxprob)
                # plt.savefig('/home/jerry/xxz/DANNet/dataset/ACDC/gt_trainval/active_gt/night/output_visualization/prob/%s/%s_maxprob.png' % (name.split('/')[0], name.split('/')[1]))
                # plt.close()

                # output2 = np.asarray(np.argmax(output2, axis=2), dtype=np.uint8)

                # output_col = colorize_mask(output2)
                # output_col.save('/home/jerry/xxz/DANNet/dataset/ACDC/gt_trainval/active_gt/night/output_visualization/color/%s/%s_color.png' % (name.split('/')[0], name.split('/')[1]))
                
                ################

                score, purity, entropy = floating_region_score(output)

                score[active] = -float('inf')

                active_regions = math.ceil(num_pixel_cur * active_ratio / per_region_pixels) #per_region_pixels == 9

                for pixel in range(active_regions):
                    values, indices_h = torch.max(score, dim=0)
                    _, indices_w = torch.max(values, dim=0)
                    w = indices_w.item()
                    h = indices_h[w].item()

                    active_start_w = w - active_radius if w - active_radius >= 0 else 0
                    active_start_h = h - active_radius if h - active_radius >= 0 else 0
                    active_end_w = w + active_radius + 1
                    active_end_h = h + active_radius + 1

                    mask_start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    mask_start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    mask_end_w = w + mask_radius + 1
                    mask_end_h = h + mask_radius + 1

                    # mask out
                    score[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = -float('inf')
                    active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
                    selected[active_start_h:active_end_h, active_start_w:active_end_w] = True
                    # active sampling
                    active_mask[active_start_h:active_end_h, active_start_w:active_end_w] = \
                        ground_truth[active_start_h:active_end_h, active_start_w:active_end_w]

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    if args.use_lightM:
        seg_net.cca.active_state = 'train'
    lightnet.train()
    seg_net.train()

def Classification_difference_calculation(input_feature, seg_net, weight, batch_size):
    seg_net.eval()
    input_without_attention = input_feature.detach()
    input_attention = input_feature.detach()
    size = (input_without_attention.size()[2], input_without_attention.size()[3])

    with torch.no_grad():
        attention_out = seg_net.cca(input_attention, weight, batch_size, data_from = 'source', light_M=None)
        concat_out = torch.cat((input_without_attention, attention_out), dim=0)
        concat_out = seg_net.maxpool(concat_out)
        concat_out = seg_net.layer1(concat_out)
        concat_out = seg_net.layer2(concat_out)
        concat_out = seg_net.layer3(concat_out)
        concat_out = seg_net.layer4(concat_out)
        concat_out = seg_net.head(concat_out)

        weights_prob = weights.expand(concat_out.size()[0], concat_out.size()[3], concat_out.size()[2], 19)
        weights_prob = weights_prob.transpose(1, 3)
        concat_out = concat_out * weights_prob
        concat_out = F.interpolate(concat_out, size=size, mode='bilinear', align_corners=True)
        out_without_attention = concat_out[:batch_size, :, :, :]
        out_attention = concat_out[batch_size:batch_size*2, :, :, :]

        out_without_attention = torch.argmax(out_without_attention, dim=1, keepdim=True)
        out_attention = torch.argmax(out_attention, dim=1, keepdim=True)

        different_position = out_without_attention - out_attention
        different_position[different_position != 0] = 1

    seg_net.train()
    return different_position

# def RegionSelection(cfg, lightnet, seg_net, tgt_epoch_loader):
#
#     lightnet.eval()
#     seg_net.eval()
#
#     floating_region_score = FloatingRegionScore(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
#     per_region_pixels = (2 * cfg.ACTIVE.RADIUS_K + 1) ** 2
#     active_radius = cfg.ACTIVE.RADIUS_K
#     mask_radius = cfg.ACTIVE.RADIUS_K * 2
#     active_ratio = cfg.ACTIVE.RATIO / len(cfg.ACTIVE.SELECT_ITER)
#
#     with torch.no_grad():
#         for tgt_data in tqdm(tgt_epoch_loader):
#
#             tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
#             origin_mask, origin_label = \
#                 tgt_data['origin_mask'], tgt_data['origin_label']
#             origin_size = tgt_data['size']
#             active_indicator = tgt_data['active']
#             selected_indicator = tgt_data['selected']
#             path2indicator = tgt_data['path_to_indicator']
#
#             tgt_input = tgt_input.cuda(non_blocking=True)
#
#             tgt_size = tgt_input.shape[-2:]
#             # tgt_feat = feature_extractor(tgt_input)
#             r = lightnet(tgt_input)
#             enhanced_image = r + tgt_input
#             if args.model == 'RefineNet':
#                 tgt_out = seg_net(enhanced_image)
#             else:
#                 _, tgt_out, _ = seg_net(enhanced_image, data_from='target', light_M=lightMask)
#
#             for i in range(len(origin_mask)):
#                 active_mask = origin_mask[i].cuda(non_blocking=True)
#                 ground_truth = origin_label[i].cuda(non_blocking=True)
#                 size = (origin_size[i][0], origin_size[i][1])
#                 num_pixel_cur = size[0] * size[1]
#                 active = active_indicator[i]
#                 selected = selected_indicator[i]
#
#                 output = tgt_out[i:i + 1, :, :, :]
#                 output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
#                 score, purity, entropy = floating_region_score(output)
#
#                 score[active] = -float('inf')
#
#                 active_regions = math.ceil(num_pixel_cur * active_ratio / per_region_pixels)
#
#                 for pixel in range(active_regions):
#                     values, indices_h = torch.max(score, dim=0)
#                     _, indices_w = torch.max(values, dim=0)
#                     w = indices_w.item()
#                     h = indices_h[w].item()
#
#                     active_start_w = w - active_radius if w - active_radius >= 0 else 0
#                     active_start_h = h - active_radius if h - active_radius >= 0 else 0
#                     active_end_w = w + active_radius + 1
#                     active_end_h = h + active_radius + 1
#
#                     mask_start_w = w - mask_radius if w - mask_radius >= 0 else 0
#                     mask_start_h = h - mask_radius if h - mask_radius >= 0 else 0
#                     mask_end_w = w + mask_radius + 1
#                     mask_end_h = h + mask_radius + 1
#
#                     # mask out
#                     score[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = -float('inf')
#                     active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
#                     selected[active_start_h:active_end_h, active_start_w:active_end_w] = True
#                     # active sampling
#                     active_mask[active_start_h:active_end_h, active_start_w:active_end_w] = \
#                         ground_truth[active_start_h:active_end_h, active_start_w:active_end_w]
#
#                 active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
#                 active_mask.save(path2mask[i])
#                 indicator = {
#                     'active': active,
#                     'selected': selected
#                 }
#                 torch.save(indicator, path2indicator[i])
#
#     lightnet.train()
#     seg_net.train()

