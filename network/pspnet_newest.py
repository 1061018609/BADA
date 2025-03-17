import torch.nn as nn
from torch.nn import functional as F
import torch
# import seaborn as sns
import matplotlib.pyplot as plt
from .cc_attention_exposure import CrissCrossAttention

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

affine_par = True
BatchNorm2d = nn.BatchNorm2d
image_name = None


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(out_features),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        # bn = BatchNorm2d(out_features)
        # return nn.Sequential(prior, conv, bn)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        # print('bottle: ', bottle.size())
        # for i in range(512):
        #     sns.set_theme()
        #     temp_tensor = bottle[0,i,:,:].cpu().numpy()
        #     ax = sns.heatmap(temp_tensor)
        #     plt.savefig('/home/jerry/xxz/DANNet/heatmaps/heatmap_afterpsp_day/{}.png'.format(str(i)))
        #     plt.close()
        #     # plt.show()
        #特征向量保存
        # global image_name
        # save_name = image_name[0].split('/')[-1].split('.')[-2]
        # print(save_name)
        # torch.save(bottle, '/home/jerry/xxz/DANNet/tensors_saved/afterpsp_night/{}.pth'.format(save_name))
        return bottle


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.uselightconvs = True
        self.lightconv_inchannels = 1

        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.cca = CrissCrossAttention(128)
        #####lightM conv layers
        if self.uselightconvs:
            self.lightconv1 = conv3x3(self.lightconv_inchannels, 32, stride=2)
            self.lightbn1 = BatchNorm2d(32)
            self.lightrelu1 = nn.ReLU(inplace=False)
            self.lightconv2 = conv3x3(32, 32)
            self.lightbn2 = BatchNorm2d(32)
            self.lightrelu2 = nn.ReLU(inplace=False)
            self.lightconv3 = conv3x3(32, 1)
            self.lightbn3 = BatchNorm2d(1)
            self.lightsigmoid = nn.Sigmoid()
        #####lightM conv layers
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1,1,1))

        self.head = nn.Sequential(PSPModule(2048, 512),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        
        # self.transpose = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x, name=None, recurrence=2, data_from=None, light_M=None):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        # name = name[0].split('/')[-1]
        # lightM_save = light_M.detach()
        # lightM_save = lightM_save.squeeze(0).cpu().data[0].numpy()
        # sns.set()
        # sns.heatmap(lightM_save)
        # plt.savefig('/home/jerry/xxz/DANNet/heatmaps/lightM_beforeconv//{}_lightM.png'.format(name.split('.')[0]))
        # plt.close()

        if self.uselightconvs and data_from == 'target':
            light_M = self.lightrelu1(self.lightbn1(self.lightconv1(light_M)))
            light_M = self.lightrelu2(self.lightbn2(self.lightconv2(light_M)))
            light_M = self.lightsigmoid(self.lightbn3(self.lightconv3(light_M)))
        

        # lightM_save = light_M.squeeze(0).cpu().data[0].numpy()
        # sns.set()
        # sns.heatmap(lightM_save)
        # plt.savefig('/home/jerry/xxz/DANNet/heatmaps/lightM_afterconv//{}_lightM.png'.format(name.split('.')[0]))
        # plt.close()

        for i in range(recurrence):
            x = self.cca(x, data_from=data_from, light_M=light_M)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)
        # for i in range(512):
        #     sns.set_theme()
        #     temp_tensor = x[0,i,:,:].cpu().numpy()
        #     ax = sns.heatmap(temp_tensor)
        #     plt.savefig('/home/jerry/xxz/DANNet/heatmaps/heatmap_beforepsp_day/{}.png'.format(str(i)))
        #     plt.close()
        # global image_name
        # image_name = name
        #####
        # x = self.head(x)
        x_features = self.head[0](x)
        # x_features = self.transpose(x_features)
        x = self.head[1](x_features)
        ##参数打印
        # for name, para in self.head.named_parameters():
        #     if name == '1.weight':
        #         print(para.size())
        #         # print(para[0,:,:,:])
        #     if name == '1.bias':
        #         print(para.size())
        #         print(para)
        return x_dsn, x, x_features


def PSPNet(num_classes=19):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model

