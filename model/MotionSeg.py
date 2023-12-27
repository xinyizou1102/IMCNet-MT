from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.resnet import ResNet101Backbone
from model.backbone.swin import SwinTransformer
from model.base.correlation import Correlation
from model.learner import MotionLearner



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        
    def forward(self, x):
        x = self.conv_bn(x)
        return x


class DimReduce(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimReduce, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class MotionSOD(nn.Module):
    def __init__(self, option):
        super(MotionSOD, self).__init__()
        self.option = option
        if self.option['backbone'] == 'R101':
            self.backbone = ResNet101Backbone()
            self.DimReduce_xxx_128 = DimReduce(in_channel=2048, out_channel=128)
            self.DimReduce_xx_128 = DimReduce(in_channel=1024, out_channel=128)
            self.DimReduce_x_128 = DimReduce(in_channel=512, out_channel=128)
        elif self.option['backbone'] == 'swin':
            self.backbone = SwinTransformer(img_size=option['trainsize'], embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32], window_size=12)
            pretrained_dict = torch.load("model/swin/swin_base_patch4_window12_384.pth")["model"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.backbone.state_dict()}
            self.backbone.load_state_dict(pretrained_dict)
            self.DimReduce_xxx_128 = DimReduce(in_channel=1024, out_channel=128)
            self.DimReduce_xx_128 = DimReduce(in_channel=512, out_channel=128)
            self.DimReduce_x_128 = DimReduce(in_channel=256, out_channel=128)
            self.DimReduce_128 = DimReduce(in_channel=128, out_channel=128)

        self.motion_learner = MotionLearner([1, 1, 1])
    
    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)
    
    def feature_norm(self, feat):
        eps = 1e-5
        return (feat / (feat.norm(dim=1, p=2, keepdim=True) + eps))        

    def forward(self, query_img, support_img, flow):
        # Obtain the features of the two images and normalize the features
        query_feats = self.backbone(query_img)
        support_feats = self.backbone(support_img)
        query_feats_norm = [self.feature_norm(x) for x in query_feats]
        support_feats_norm = [self.feature_norm(x) for x in support_feats]
        # Swin: [8, 128, 96, 96], [8, 256, 48, 48], [8, 512, 24, 24], [8, 1024, 12, 12]
        # R101: [8, 256, 96, 96], [8, 512, 48, 48], [8, 1024, 24, 24], [8, 2048, 12, 12]

        # Calculate multi-scale correlation
        corr = Correlation.multilayer_correlation(query_feats_norm[1:], support_feats_norm[1:])

        logit_mask_list_out = []
        feat1_list = [self.DimReduce_128(query_feats_norm[0]), self.DimReduce_x_128(query_feats_norm[1]), self.DimReduce_xx_128(query_feats_norm[2]), self.DimReduce_xxx_128(query_feats_norm[3])]
        logit_mask_list, vis_feat_dict = self.motion_learner(corr, feat1_list)

        for logit_mask in logit_mask_list:
            logit_mask = F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True)
            logit_mask_list_out.append(logit_mask)

        return logit_mask_list_out, vis_feat_dict

