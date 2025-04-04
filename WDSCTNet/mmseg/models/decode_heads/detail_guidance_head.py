# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.runner import force_fp32
import torch.nn.functional as F
import math
import warnings
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.runner import BaseModule, ModuleList, Sequential
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,trunc_normal_init,normal_init)

from ..losses.OhemCEloss import OhemCELoss
from ..losses.detail_loss import DetailAggregateLoss

#TODO,inhe inter huancheng in chanel
@HEADS.register_module()
class DetailGuidanceHead(BaseDecodeHead):
    def __init__(self,
                **kwargs):
        super(DetailGuidanceHead,self).__init__(**kwargs)

        # self.conv_stdc_out2 = BiSeNetOutput(64, 64, 1)  #Small
        self.conv_stdc_out2 = BiSeNetOutput(128, 128, 1) #Base
        self.conv_stdc_out2.apply(self._weights_kaiming)



    def forward(self, inputs,decoder_feature,decoder_seg_logits):
        stdc_layer2_fea = inputs[1]
        feature_stdc_out2 = self.conv_stdc_out2(stdc_layer2_fea)

        return feature_stdc_out2

    

    def forward_train(self, inputs,decoder_feature,decoder_seg_logits,img_metas, gt_semantic_seg, *args):
        feature_stdc_out2 = self.forward(inputs,decoder_feature,decoder_seg_logits)
        losses = self.losses(feature_stdc_out2, gt_semantic_seg)
        return losses

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit,label):
        """Compute segmentation loss. 计算分割损失"""
        loss = dict()

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,label)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,label)

        return loss

    def _weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)




class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm_cfg=dict(type='SyncBN'),*args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        # self.bn = BatchNorm2d(out_chan)
        # self.bn = nn.BatchNorm2d(out_chan, activation='none')
        self.bn = build_norm_layer(norm_cfg, out_chan)[1]

        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)

        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)




