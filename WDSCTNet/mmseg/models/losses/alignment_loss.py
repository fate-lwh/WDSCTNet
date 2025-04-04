import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,trunc_normal_init,normal_init)
from ..builder import LOSSES



def attention_transform(feat):
    return F.normalize(feat.pow(2).mean(1).view(feat.size(0), -1))


def similarity_transform(feat):
    feat = feat.view(feat.size(0), -1)
    gram = feat @ feat.t()
    return F.normalize(gram)


_TRANS_FUNC = {"attention": attention_transform, "similarity": similarity_transform, "linear": lambda x : x}


def ChannelWiseDivergence(feat_t, feat_s):
    assert feat_s.shape[-2:] == feat_t.shape[-2:]
    N, C, H, W = feat_s.shape
    softmax_pred_T = F.softmax(feat_t.reshape(-1, W * H) / 4.0, dim=1)
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    loss = torch.sum(softmax_pred_T *
                     logsoftmax(feat_t.reshape(-1, W * H) / 4.0) -
                     softmax_pred_T *
                     logsoftmax(feat_s.reshape(-1, W * H) / 4.0)) * (
                         (4.0)**2)
    loss =  loss / (C * N)   
    return loss




@LOSSES.register_module()
class AlignmentLoss(nn.Module):

    def __init__(self, 
                loss_weight=1.0,
                loss_name='loss_guidance',
                inter_transform_type='linear'):
        super(AlignmentLoss, self).__init__()
        self.inter_transform_type=inter_transform_type
        self._loss_name = loss_name
        self.loss_weight = loss_weight
        # self.se320 = SEBlock(320)
        # self.se512 = SEBlock(512)

        # self.se320.apply(self._weights_kaiming)
        # self.se512.apply(self._weights_kaiming)


    # def _weights_kaiming(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_init(m.weight, std=.02)
    #         if m.bias is not None:
    #             constant_init(m.bias, val=0)
    #     elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
    #         constant_init(m.weight, val=1.0)
    #         constant_init(m.bias, val=0)
    #     elif isinstance(m, nn.Conv2d):
    #         kaiming_init(m.weight)
    #         if m.bias is not None:
    #             constant_init(m.bias, val=0)

#     def WeightChannelWise320(self,feat_t, feat_s,weight):
#         # assert feat_s.shape[-2:] == feat_t.shape[-2:]
#         N, C, H, W = feat_s.shape
#         # weight_stu, weight = self.se320(feat_s)
#         # weight = F.softmax(weight, dim=1)
#         weight_temperature = weight * C / 2 + 3.5

#         softmax_pred_T = F.softmax(feat_t.reshape(-1, W * H) / weight_temperature, dim=1)
#         logsoftmax = torch.nn.LogSoftmax(dim=1)
#         loss = torch.sum(softmax_pred_T *
#                          logsoftmax(feat_t.reshape(-1, W * H) / weight_temperature) -
#                          softmax_pred_T *
#                          logsoftmax(feat_s.reshape(-1, W * H) / weight_temperature)) * (
#                        (4.0) ** 2)
#         loss = loss / (C * N)
#         return loss

#     def WeightChannelWise512(self,feat_t, feat_s,weight):
#         # assert feat_s.shape[-2:] == feat_t.shape[-2:]
#         N, C, H, W = feat_s.shape
#         # weight_stu, weight = self.se512(feat_s)
#         # weight = F.softmax(weight, dim=1)
#         weight_temperature = weight * C / 2 + 3.5

#         softmax_pred_T = F.softmax(feat_t.reshape(-1, W * H) / weight_temperature, dim=1)
#         logsoftmax = torch.nn.LogSoftmax(dim=1)
#         loss = torch.sum(softmax_pred_T *
#                          logsoftmax(feat_t.reshape(-1, W * H) / weight_temperature) -
#                          softmax_pred_T *
#                          logsoftmax(feat_s.reshape(-1, W * H) / weight_temperature)) * (
#                        (4.0) ** 2)
#         loss = loss / (C * N)
#         return loss

       

    def forward(self, x_guidance_feature):
        loss_inter = x_guidance_feature[0][0].new_tensor(0.0)
        
#         feat_t_0 = x_guidance_feature[0][2]
#         feat_s_0 = x_guidance_feature[1][2][0]
#         weight_0 = x_guidance_feature[1][2][1]
        
#         # print(f"feat_s_0 type: {type(feat_s_0)}, feat_t_0 type: {type(feat_t_0)}")

        
#         loss_inter = loss_inter + self.loss_weight[2]*self.WeightChannelWise512(feat_t_0, feat_s_0,weight_0)

#         feat_t_1 = x_guidance_feature[0][3]
#         feat_s_1 = x_guidance_feature[1][3][0]
#         weight_1 = x_guidance_feature[1][3][1]
#         loss_inter = loss_inter + self.loss_weight[3]*self.WeightChannelWise320(feat_t_1, feat_s_1,weight_1)

        for i in range(4):
            feat_t = x_guidance_feature[0][i]
            feat_s = x_guidance_feature[1][i]
            loss_inter = loss_inter + self.loss_weight[i]*ChannelWiseDivergence(feat_t, feat_s)
        return loss_inter
    
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name