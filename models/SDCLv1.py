import torch
from  torch import nn
import  torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import numpy as np
from models.backbone.resnet import *
from models.STAM128 import STAM
import sys
from torch.autograd import Variable
import numpy as np

import slayerSNN as snn
netParams = snn.params('models/network.yaml')

def getNeuronConfig(type: str='SRMALPHA',
                    theta: float=10.,
                    tauSr: float=1.,
                    tauRef: float=1.,
                    scaleRef: float=2.,
                    tauRho: float=0.3,  # Was set to 0.2 previously (e.g. for fullRes run)
                    scaleRho: float=1.):
    """
    :param type:     neuron type
    :param theta:    neuron threshold
    :param tauSr:    neuron time constant
    :param tauRef:   neuron refractory time constant
    :param scaleRef: neuron refractory response scaling (relative to theta)
    :param tauRho:   spike function derivative time constant (relative to theta)
    :param scaleRho: spike function derivative scale factor
    :return: dictionary
    """
    return {
        'type': type,
        'theta': theta,
        'tauSr': tauSr,
        'tauRef': tauRef,
        'scaleRef': scaleRef,
        'tauRho': tauRho,
        'scaleRho': scaleRho,
    }

class NetworkBasic(torch.nn.Module):
    def __init__(self, netParams,
                 theta=[30, 50, 100],
                 tauSr=[1, 2, 4],
                 tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1],
                 tauRho=[1, 1, 10],
                 scaleRho=[10, 10, 100]):
        super(NetworkBasic, self).__init__()

        self.neuron_config = []
        self.neuron_config.append(
            getNeuronConfig(theta=theta[0], tauSr=tauSr[0], tauRef=tauRef[0], scaleRef=scaleRef[0], tauRho=tauRho[0],
                            scaleRho=scaleRho[0]))
        self.neuron_config.append(
            getNeuronConfig(theta=theta[1], tauSr=tauSr[1], tauRef=tauRef[1], scaleRef=scaleRef[1], tauRho=tauRho[1],
                            scaleRho=scaleRho[1]))
        self.neuron_config.append(
            getNeuronConfig(theta=theta[2], tauSr=tauSr[2], tauRef=tauRef[2], scaleRef=scaleRef[2], tauRho=tauRho[2],
                            scaleRho=scaleRho[2]))

        self.slayer1 = snn.layer(self.neuron_config[0], netParams['simulation'])
        self.slayer2 = snn.layer(self.neuron_config[1], netParams['simulation'])
        self.slayer3 = snn.layer(self.neuron_config[2], netParams['simulation'])

        self.conv1 = self.slayer1.conv(8, 8, 5, padding=2)
        self.conv2 = self.slayer2.conv(8, 8, 3, padding=1)
        # self.upconv1 = self.slayer3.convTranspose(8, 2, kernelSize=2, stride=2)


    def forward(self, spikeInput):
        psp1 = self.slayer1.psp(spikeInput)
        spikes_layer_1 = self.slayer1.spike(self.conv1(psp1))
        spikes_layer_2 = self.slayer2.spike(self.conv2(self.slayer2.psp(spikes_layer_1)))
        # b, t, c, w, h = spikes_layer_2.size()    # [16,8,3,256,128] [b, t, c, w, h]
        # spikes_layer_2 = spikes_layer_2.contiguous().view(b * t, c, w, h)  # x=[128,3,256,128]
        # print('spikes_layer_2=',spikes_layer_2.shape)
        return spikes_layer_2



##################################################################################
##################################################################################
##################################################################################



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}









##########################################################################################
def init_pretrained_weight(model, model_url):
    """Initializes model with pretrained weight

    Layers that don't match with pretrained layers in name or size are kept unchanged
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


################################################################################################




class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = Variable(torch.cat([torch.arange(0, 2*N, 2), torch.arange(1, 2*N+1, 2)]), requires_grad=False).type_as(x).long()
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset



def get_offset(event):
    grid = torch.zeros(event.shape)
    for k in range(grid.shape[1]):
        for i in range(grid.shape[2]):
            for j in range(grid.shape[3]):
                if event[:,k.i.j] != 0:
                    grid[:,k,i,j] = 1
    
    return grid

from .utils.gelu import GELU
from .utils.layer_norm import LayerNorm
import math

class Cross_Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self):
        super().__init__()
        
        self.activation = GELU()

    def set_bn_momentum(self, momentum):
        pass
    def forward(self, event_f, img_f):


        query = self.conv_query(event_f)
        key = self.conv_key(img_f)
        value = self.conv_val(img_f)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        
        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, value)


class Cross_Deform(nn.Module):

    def __init__(self):
        super().__init__()
        self.offsets = nn.Conv2d(3, 18, kernel_size=3, padding=1)
        self.deform = DeformConv2D(3, 3, kernel_size=3, padding=1)

    def forward(self, event_f, img_f):
        event_offsets = get_offset(event_f)
        event_feature = F.relu(self.deformconv(img_f, event_offsets))     # x = [64,3,256,128] feat_map= [128, 2048, 16, 8]
        
        return event_feature


class cross_alignment(nn.Module):

    def __init__(self):
        super().__init__()


        self.att = Cross_Attention()
        self.deform = Cross_Deform()
        self.alpha = 0.1
        self.beta = 0.1

    def forward(self, event_f, img_f):

        event_f1 = Cross_Deform(event_f,img_f)
        img_f1 = Cross_Deform(img_f,event_f)

        event_f2 = Cross_Attention(event_f,img_f)
        img_f2 = Cross_Attention(img_f,event_f)

        event_f = event_f + self.alpha*event_f1 + self.beta *event_f2
        img_f = img_f + self.alpha*img_f1 + self.beta*img_f2

        return event_f, img_f
    



################################################################################################
class SDCLv1(nn.Module):

    def __init__(self, num_classes, model_name, pretrain_choice, seq_len=8):
        super(SDCLv1, self).__init__()
        self.inplanes = 64
        self.in_planes = 2048
        self.base = ResNet()

        if pretrain_choice == 'imagenet':
            init_pretrained_weight(self.base, model_urls[model_name])
            print('Loading pretrained ImageNet model ......')

        self.seq_len = seq_len
        self.num_classes = num_classes
        self.plances = 1024
        self.mid_channel = 256

        self.avg_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.down_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_planes, out_channels=self.plances, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.plances),
            self.relu
        )

        t = seq_len
        self.layer1 = STAM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, num= '1')

        t = t / 2
        self.layer2 = STAM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, num= '2')

        t = t / 2
        self.layer3 = STAM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, num= '3')


        self.bottleneck = nn.ModuleList([nn.BatchNorm1d(self.plances) for _  in range(3)])
        self.classifier = nn.ModuleList([nn.Linear(self.plances, num_classes) for _ in range(3)])

        self.bottleneck[0].bias.requires_grad_(False)
        self.bottleneck[1].bias.requires_grad_(False)
        self.bottleneck[2].bias.requires_grad_(False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weight_init_classifier)


        self.offsets = nn.Conv2d(3, 18, kernel_size=3, padding=1)
        self.deformconv = DeformConv2D(3, 3, kernel_size=3, padding=1)


        self.snn = NetworkBasic(netParams=netParams)


        self.cross_alignment = cross_alignment()
        ######################################################################
        self.res_conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        

    def _make_layer(self,block,planes,blocks,stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes * block.expansion,
                          kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes*block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))

        return nn.Sequential(*layers)


    def forward(self, x, pids=None, camid=None):    # x=[16,8,3,256,128]
        _, _t, _, _, _ = x.size()    # [16,8,3,256,128] [b, t, c, w, h]
        
        event = x[:,_t//2:,:,:,:]
        event = self.snn(event)
        # print('event=',event.shape)
        
        b, t, c, w, h = event.size()
        event = event.contiguous().view(b * t, c, w, h)


        offsets = get_offset(event)
        event_feature = F.relu(self.deformconv(event, offsets))     # x = [64,3,256,128] feat_map= [128, 2048, 16, 8]
        
        event_feature = event_feature.contiguous().view(b, t, c, w, h)
        # x[:,_t//2:,:,:,:] = event  
        
        
        _x = x[:,0:_t//2,:,:,:]

        b, t, c, w, h = _x.size()    # [16,8,3,256,128] [b, t, c, w, h]
        _x = _x.contiguous().view(b * t, c, w, h)  # x=[128,3,256,128]

        #############################################################################
        #############################################################################
        #############################################################################
        feat_map = self.base(_x)  # (b * t, c, 16, 8)  feat_map= torch.Size([128, 2048, 16, 8])  

        event_f, img_f = cross_alignment(event_feature,feat_map)

        feat_map = torch.concat((event_f,img_f),dim=1)

        #############################################################################
        #############################################################################
        #############################################################################

        w = feat_map.size(2)
        h = feat_map.size(3)

        feat_map = self.down_channel(feat_map)
        feat_map = feat_map.view(b, t, -1, w, h)    # [4, 8, 1024, 16, 8]
        # feat_map = feat_map.view(-1, t, 1024, w, h)
        # b,_,_,_,_ = feat_map.size()
        
        feature_list = []
        list = []
        # print('feat_map=',feat_map.shape)
            
        feat_map_1 = self.layer1(feat_map)
        feature_1 = torch.mean(feat_map_1, 1)
        feature1 = self.avg_2d(feature_1).view(b, -1)   # [4, 1024]
        feature_list.append(feature1)
        list.append(feature1)
        # print('feature1=',feature1.shape)

        feat_map_2 = self.layer2(feat_map_1)
        feature_2 = torch.mean(feat_map_2, 1)           # # [4, 1024]
        feature_2 = self.avg_2d(feature_2).view(b, -1)
        list.append(feature_2)
        # print('feature_2=',feature_2.shape)

        feature2 = torch.stack(list, 1)
        feature2 = torch.mean(feature2, 1)              # [4, 1024]
        feature_list.append(feature2)
        # print('feature2=',feature2.shape)

        feat_map_3 = self.layer3(feat_map_2)
        feature_3 = torch.mean(feat_map_3, 1)
        feature_3 = self.avg_2d(feature_3).view(b, -1)  # [4, 1024]
        list.append(feature_3)
        # print('feature_3=',feature_3.shape)

        feature3 = torch.stack(list, 1)
        feature3 = torch.mean(feature3, 1)          # [4, 1024]
        feature_list.append(feature3)
        # print('feature3=',feature3.shape)
        # sys.exit()

        BN_feature_list = []
        for i in range(len(feature_list)):
            BN_feature_list.append(self.bottleneck[i](feature_list[i]))
        torch.cuda.empty_cache()

        cls_score = []
        for i in range(len(BN_feature_list)):
            cls_score.append(self.classifier[i](BN_feature_list[i]))

        if self.training:
            return cls_score, BN_feature_list
        else:
            return BN_feature_list[2], pids, camid