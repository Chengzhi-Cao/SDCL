import torch
from  torch import nn
import  torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

from models.backbone.resnet import *

import sys
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

##################################################################################
##################################################################################
##################################################################################
class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

##################################################################################
##################################################################################
##################################################################################

class ConvRecurrent(nn.Module):
    """
    Convolutional recurrent cell (for direct comparison with spiking nets).
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation=None):
        super().__init__()

        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.rec = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.out = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        assert activation is None, "ConvRecurrent activation cannot be set (just for compatibility)"

    def forward(self, input_, prev_state=None):
        # generate empty prev_state, if None is provided
        if prev_state is None:
            batch, _, height, width = input_.shape
            state_shape = (batch, self.hidden_size, height, width)
            prev_state = torch.zeros(*state_shape, dtype=input_.dtype, device=input_.device)

        ff = self.ff(input_)
        rec = self.rec(prev_state)
        state = torch.tanh(ff + rec)
        out = self.out(state)
        out = torch.relu(out)

        return out, state


########################################################################################

class TSB(nn.Module):
    def __init__(self, in_channels, use_gpu=False, **kwargs):
        super(TSB, self).__init__()
        self.in_channels = in_channels
        self.use_gpu = use_gpu
        self.patch_size = 2
        
        self.W = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels,
                    kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.in_channels)
        )

        self.pool = nn.AvgPool3d(kernel_size=(1, self.patch_size, self.patch_size), 
                stride=(1, 1, 1), padding=(0, self.patch_size//2, self.patch_size//2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.constant_(self.W[1].weight.data, 0.0)
        nn.init.constant_(self.W[1].bias.data, 0.0)


    def forward(self, x):
        b, c, t, h, w = x.size()
        inputs = x

        query = x.view(b, c, t, -1).mean(-1) 
        query = query.permute(0, 2, 1) 

        memory = self.pool(x) 
        if self.patch_size % 2 == 0:
            memory = memory[:, :, :, :-1, :-1]

        memory = memory.contiguous().view(b, 1, c, t * h * w) 

        query = F.normalize(query, p=2, dim=2, eps=1e-12)
        memory = F.normalize(memory, p=2, dim=2, eps=1e-12)
        f = torch.matmul(query.unsqueeze(2), memory) * 5
        f = f.view(b, t, t, h * w) 

        # mask the self-enhance
        mask = torch.eye(t).type(x.dtype) 
        if self.use_gpu: mask = mask.cuda()
        mask = mask.view(1, t, t, 1)

        f = (f - mask * 1e8).view(b, t, t * h * w)
        f = F.softmax(f, dim=-1)

        y = x.view(b, c, t * h * w)
        y = torch.matmul(f, y.permute(0, 2, 1)) 
        y = self.W(y.view(b * t, c, 1, 1))
        y = y.view(b, t, c, 1, 1)
        y = y.permute(0, 2, 1, 3, 4)
        z = y + inputs

        return z




##################################################################################
##################################################################################
##################################################################################

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

class TCLNet(nn.Module):

    def __init__(self, num_classes, model_name, pretrain_choice, seq_len=8):
        super(TCLNet, self).__init__()

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


        self.bottleneck = nn.ModuleList([nn.BatchNorm1d(self.plances) for _  in range(3)])
        self.classifier = nn.ModuleList([nn.Linear(self.plances, num_classes) for _ in range(3)])

        self.bottleneck[0].bias.requires_grad_(False)
        self.bottleneck[1].bias.requires_grad_(False)
        self.bottleneck[2].bias.requires_grad_(False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weight_init_classifier)



#############################################################################
#############################################################################
#############################################################################

        self.LSTM_layer1 = TSB(self.plances)
        self.LSTM_layer2 = TSB(self.plances)
        self.LSTM_layer3 = TSB(self.plances)









    def _make_layer(
        self,
        block,
        layer,
        in_channels,
        out_channels,
        reduce_spatial_size,
        IN=False
    ):
        layers = []

        layers.append(block(in_channels, out_channels, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)


    def forward(self, x, pids=None, camid=None):    # x=[16,8,3,256,128]

        b, t, c, w, h = x.size()    # [16,8,3,256,128] [b, t, c, w, h]
        x = x.contiguous().view(b * t, c, w, h)  # x=[128,3,256,128]
        # 调用里面的模块，然后提取特征

        feat_map = self.base(x)  # (b * t, c, 16, 8)  feat_map= torch.Size([128, 2048, 16, 8])      
 
        w = feat_map.size(2)
        h = feat_map.size(3)

        feat_map = self.down_channel(feat_map)
        feat_map = feat_map.view(b, t, -1, w, h)    # [4, 8, 1024, 16, 8]
        feature_list = []
        list = []
        # print('feat_map=',feat_map.shape)
            
        feat_map_1 = self.LSTM_layer1(feat_map)
        feature_1 = torch.mean(feat_map_1, 1)
        feature1 = self.avg_2d(feature_1).view(b, -1)   # [4, 1024]
        feature_list.append(feature1)
        list.append(feature1)
        # print('feature1=',feature1.shape)

        feat_map_2 = self.LSTM_layer2(feat_map_1)
        feature_2 = torch.mean(feat_map_2, 1)           # # [4, 1024]
        feature_2 = self.avg_2d(feature_2).view(b, -1)
        list.append(feature_2)
        # print('feature_2=',feature_2.shape)

        feature2 = torch.stack(list, 1)
        feature2 = torch.mean(feature2, 1)              # [4, 1024]
        feature_list.append(feature2)
        # print('feature2=',feature2.shape)

        feat_map_3 = self.LSTM_layer3(feat_map_2)
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