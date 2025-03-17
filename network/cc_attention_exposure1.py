import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
     # return -torch.diag(torch.tensor(float(0)).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim, use_last_lightMinvs=False, train_state='eval'):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        self.use_last_lightMinvs = use_last_lightMinvs

        self.train_state = train_state

        if self.train_state == 'train':
            distance_weighted_h = 256
            distance_weighted_w = 256
            distance_weighted_h_M = torch.ones((1, 1, distance_weighted_h, distance_weighted_h))
            for i in range(distance_weighted_h):
                for j in range(distance_weighted_h):
                    distance_weighted_h_M[:,:,i,j] = j-i

            distance_weighted_w_M = torch.ones((1, 1, distance_weighted_w, distance_weighted_w))
            for i in range(distance_weighted_w):
                for j in range(distance_weighted_w):
                    distance_weighted_w_M[:,:,i,j] = j-i

        elif self.train_state == 'eval':
            distance_weighted_h = 270
            distance_weighted_w = 480
            distance_weighted_h_M = torch.ones((1, 1, distance_weighted_h, distance_weighted_h))
            for i in range(distance_weighted_h):
                for j in range(distance_weighted_h):
                    distance_weighted_h_M[:,:,i,j] = j-i

            distance_weighted_w_M = torch.ones((1, 1, distance_weighted_w, distance_weighted_w))
            for i in range(distance_weighted_w):
                for j in range(distance_weighted_w):
                    distance_weighted_w_M[:,:,i,j] = j-i

        # self.distance_weighted_h_M = nn.Parameter(torch.exp(-torch.pow(distance_weighted_h_M)/(2*torch.pow(distance_weighted_h/2))), requires_grad=False)
        # self.distance_weighted_w_M = nn.Parameter(torch.exp(-torch.pow(distance_weighted_w_M)/(2*torch.pow(distance_weighted_w/2))), requires_grad=False)
        self.distance_weighted_h_M = nn.Parameter(
            torch.exp(-(distance_weighted_h_M*distance_weighted_h_M) / (2 * (distance_weighted_h*distance_weighted_h / 4))),
            requires_grad=False)
        self.distance_weighted_w_M = nn.Parameter(
            torch.exp(-(distance_weighted_w_M*distance_weighted_w_M) / (2 * (distance_weighted_w*distance_weighted_w / 4))),
            requires_grad=False)

        # print(self.distance_weighted_h_M.size())
        # print(self.distance_weighted_w_M.size())
        # exit()

    def forward(self, x, data_from, light_M=None):
        m_batchsize, _, height, width = x.size()
        # data_from = 'source'
        # light_M = None

        if data_from == 'target':
            light_M_weighted_x = x * light_M
            light_M_invs = 1-light_M
        elif data_from == 'source':
            light_M_weighted_x = x

        proj_query = self.query_conv(light_M_weighted_x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        if data_from == 'target':
            proj_query_H_lightM = light_M_invs.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
            proj_query_H_lightM = proj_query_H_lightM.expand(m_batchsize*width, height, height).view(m_batchsize, width, height, height)#(8,64,32,32)
            proj_query_W_lightM = light_M_invs.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
            proj_query_W_lightM = proj_query_W_lightM.expand(m_batchsize*height, width, width).view(m_batchsize, height, width, width)

        proj_key = self.key_conv(light_M_weighted_x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        if data_from == 'source':
            energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height)#.permute(0,2,1,3)
            energy_H = (energy_H * self.distance_weighted_h_M).permute(0,2,1,3)
        elif data_from == 'target':
            energy_H = torch.bmm(proj_query_H, proj_key_H).view(m_batchsize,width,height,height)
            energy_H = ((energy_H * self.distance_weighted_h_M * proj_query_H_lightM)+self.INF(m_batchsize, height, width).view(m_batchsize,width,height,height)).permute(0,2,1,3)

        if data_from == 'source':
            energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
            energy_W = energy_W * self.distance_weighted_w_M
        elif data_from == 'target':
            energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
            energy_W = energy_W * self.distance_weighted_w_M * proj_query_W_lightM

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        # return self.gamma*(out_H + out_W) + x
        # print(self.gamma)
        if self.use_last_lightMinvs:
            return self.gamma*(out_H + out_W)*light_M_invs + x
        else:
            return self.gamma * (out_H + out_W) + x



if __name__ == '__main__':
    model = CrissCrossAttention(64)
    x = torch.randn(2, 64, 5, 6)
    out = model(x)
    print(out.shape)