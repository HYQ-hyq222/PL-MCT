# -----------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Revised by Jilai Zheng, for USOT
# ------------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.models.repvgg import RepVGGBlock as RepBlock
import numpy as np
# from lib.models.prroi_pool import PrRoIPool2D
from mmcv.ops.prroi_pool import PrRoIPool
from qqspy.Confidence_assessment.conf_utils import kaiming_init, constant_init
from mmcv.ops.deform_conv import DeformConv2dPack
from mmcv.ops.deform_roi_pool import DeformRoIPoolPack
from mmcv.ops.prroi_pool import prroi_pool


class matrix(nn.Module):
    """
    Encode feature for multi-scale correlation
    """
    def __init__(self, in_channels, out_channels):
        super(matrix, self).__init__()

        # Same size: h, w
        self.matrix11_k = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.matrix11_s = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Dilated size: h/2, w
        self.matrix12_k = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(2, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.matrix12_s = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(2, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Dilated size: w/2, h
        self.matrix21_k = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(1, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.matrix21_s = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(1, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, z=None, x=None):

        if x is not None:
            x11 = self.matrix11_s(x)
            x12 = self.matrix12_s(x)
            x21 = self.matrix21_s(x)

        if z is not None:
            z11 = self.matrix11_k(z)
            z12 = self.matrix12_k(z)
            z21 = self.matrix21_k(z)

        if x is not None and z is not None:
            return [z11, z12, z21], [x11, x12, x21]
        elif z is not None:
            return [z11, z12, z21], None
        elif x is not None:
            return None, [x11, x12, x21]
        else:
            return None, None


class GroupDW(nn.Module):
    """
    Encode backbone feature
    """

    def __init__(self, in_channels=256):
        super(GroupDW, self).__init__()
        self.weight = nn.Parameter(torch.ones(3))

    def forward(self, z, x):
        z11, z12, z21 = z
        x11, x12, x21 = x

        re11 = xcorr_depthwise(x11, z11)
        re12 = xcorr_depthwise(x12, z12)
        re21 = xcorr_depthwise(x21, z21)
        re = [re11, re12, re21]

        # Weight
        weight = F.softmax(self.weight, 0)

        s = 0
        for i in range(3):
            s += weight[i] * re[i]

        return s

class Conf_Fusion(nn.Module):
    """
    Fusion N_mem memory features with confidence-value paradigm
    """

    def __init__(self, in_channels=256, out_channels=256):
        super(Conf_Fusion, self).__init__()

        self.conf_gen = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.value_gen = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        batch, mem_size, channel, h, w = x.shape
        x = x.view(-1, channel, h, w)

        # Calc confidence on each position
        confidence = self.conf_gen(x)
        confidence = torch.clamp(confidence, max=4, min=-6)
        # Softmax each confidence map across all confidence maps
        confidence = torch.exp(confidence)
        confidence = confidence.view(batch, mem_size, channel, h, w)
        confidence_sum = confidence.sum(dim=1).view(batch, 1, channel, h, w).repeat(1, mem_size, 1, 1, 1)
        confidence_norm = confidence / confidence_sum

        # The raw value for output (not weighted yet)
        value = self.value_gen(x)
        value = value.view(batch, mem_size, channel, h, w)

        # Weighted sum of the value maps, with confidence maps as element-wise weights
        out = confidence_norm * value
        out = out.sum(dim=1)

        return out


def xcorr_depthwise(x, kernel):
    """
    Depth-wise cross correlation
    """
    batch, channel, h_k, w_k = kernel.shape
    _, _, h_x, w_x = x.shape
    x = x.view(-1, batch*channel, h_x, w_x)
    kernel = kernel.view(batch*channel, 1, h_k, w_k)
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(-1, channel, out.size(2), out.size(3))
    return out


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, pr_pool=False):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )
        if pr_pool:
            # self.prpooling = PrRoIPool2D(7, 7, spatial_scale=1.0)
            self.prpooling = PrRoIPool(output_size=(7, 7), spatial_scale=1.0)

    def forward(self, x, crop=False, pr_pool=False, bbox=None):  # x(b,1024,15,15)  bbox(b,4)

        x_ori = self.downsample(x)        # x_roi(b,256,15,15)

        if not crop:
            return x_ori

        if crop:
            l = 4
            r = -4
            xf = x_ori[:, :, l:r, l:r]   # xf(b,256,7,7)

        if not pr_pool:
            return x_ori, xf
        else:
            # PrPool feature according to pseudo bbox
            batch_index = torch.arange(0, x.shape[0]).view(-1, 1).float().to(x.device)   # (b,1)
            bbox = torch.cat((batch_index, bbox), dim=1)    # bbox(b,5)

            xf_pr = self.prpooling(x_ori, bbox)    # xf_pr (b,256,7,7)
            return x_ori, xf_pr

class box_tower_reg(nn.Module):
    """
    Box tower for FCOS reg
    """
    def __init__(self, in_channels=512, out_channels=256, tower_num=1):
        super(box_tower_reg, self).__init__()
        tower = []
        cls_tower = []
        cls_memory_tower = []
        # Layers for multi-scale correlation
        self.cls_encode = matrix(in_channels=in_channels, out_channels=out_channels)
        self.reg_encode = matrix(in_channels=in_channels, out_channels=out_channels)
        self.cls_dw = GroupDW(in_channels=in_channels)
        self.reg_dw = GroupDW(in_channels=in_channels)
        # Layers for confidence-value integration
        self.conf_fusion = Conf_Fusion(in_channels=out_channels, out_channels=out_channels)
        self.conv = nn.Conv2d(in_channels=256, out_channels=256,
                             kernel_size=7, stride=1, padding=0)

        # Box pred tower
        for i in range(tower_num):
            if i == 0:
                tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            else:
                tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

            tower.append(nn.BatchNorm2d(out_channels))
            tower.append(nn.ReLU())

        # Cls tower
        for i in range(tower_num):
            if i == 0:
                cls_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            else:
                cls_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

            cls_tower.append(nn.BatchNorm2d(out_channels))
            cls_tower.append(nn.ReLU())

        # Memory cls tower
        for i in range(tower_num):
            if i == 0:
                cls_memory_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            else:
                cls_memory_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

            cls_memory_tower.append(nn.BatchNorm2d(out_channels))
            cls_memory_tower.append(nn.ReLU())

        self.add_module('bbox_tower', nn.Sequential(*tower))
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('cls_memory_tower', nn.Sequential(*cls_memory_tower))

        # Reg head
        self.bbox_pred = nn.Conv2d(out_channels, 4, kernel_size=3, stride=1, padding=1)
        # Cls head
        self.cls_pred = nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1)
        self.cls_memory_pred = nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1)

        # Adjust scale for reg
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())

    def forward(self, search, kernel=None, memory_kernel=None,
                memory_confidence=None, cls_x_store=None):

        if kernel is not None:
            # Since template kernel (pooled by PrPool from the template patch) is inputted,
            #         we should conduct tracking with offline Siamese module

            # Encode feature for correlation first
            cls_z, cls_x = self.cls_encode(kernel, search)   # [z11, z12, z13]
            reg_z, reg_x = self.reg_encode(kernel, search)  # [x11, x12, x13]

            # DW for cls and reg
            cls_dw = self.cls_dw(cls_z, cls_x)
            reg_dw = self.reg_dw(reg_z, reg_x)
            x_reg = self.bbox_tower(reg_dw)
            x_bbox = self.adjust * self.bbox_pred(x_reg) + self.bias
            x_bbox = torch.exp(x_bbox)

            # Cls tower
            c = self.cls_tower(cls_dw)
            cls = 0.1 * self.cls_pred(c)

            # 置信度分支的输入
            f_ = torch.cat((cls_dw, c), dim=1).detach()
            search_ = search.detach()

            if memory_kernel is None:
                # cache cls_x to prevent double processing of search area feature
                return x_bbox, cls, cls_x, reg_x, None, f_, search_

        # Cls memory
        if memory_kernel is not None:

            # Since memory queue is inputted, we should conduct tracking with online memory module
            if cls_x_store is None:
                cls_mem_zs, cls_x_store = self.cls_encode(memory_kernel, x=search)
            else:
                # Use the cached cls_x to prevent double processing of search area feature
                cls_mem_zs, _ = self.cls_encode(memory_kernel, x=None)

            # Multi-scale correlation, between the memory queue and the search area feature in the template frame
            batch, mem_size = memory_confidence.shape
            store_repeat = []
            for cls_x in cls_x_store:
                _, c, h, w = cls_x.shape
                cls_x_rep = cls_x.view(batch, 1, c, h, w)
                cls_x_rep = cls_x_rep.repeat(1, mem_size, 1, 1, 1).view(-1, c, h, w)
                store_repeat.append(cls_x_rep)

            cls_mem_dw = self.cls_dw(cls_mem_zs, store_repeat)
            _, c, h, w = cls_mem_dw.shape
            cls_mem_dw = cls_mem_dw.view(batch, mem_size, c, h, w)   # 1,7,256,25,25

            # Fuse memory correlation maps
            cls_mem_fusion = self.conf_fusion(cls_mem_dw)    # 1,256,25,25

            # Memory cls head
            c_mem = self.cls_memory_tower(cls_mem_fusion)   # 1,256,25,25
            cls_mem = 0.1 * self.cls_memory_pred(c_mem)   # 1,1,25,25



            if kernel is not None:
                return x_bbox, cls, cls_x, reg_x, cls_mem, f_, search_
            else:
                return None, None, None, None, cls_mem, None, None
        return None

# class Process_conf(nn.Module):
#     def __init__(self, in_channels):
#         super(Process_conf, self).__init__()
#
#         self.down_conf = nn.Sequential(
#             RepBlock(768, in_channels, kernel_size=3, padding=1),
#             RepBlock(in_channels, in_channels, kernel_size=3, padding=1),
#             RepBlock(in_channels, 4, kernel_size=3, padding=1))
#         self.conf_fc1 = nn.Linear(in_features=2500, out_features=512)
#         self.conf_fc2 = nn.Linear(in_features=512, out_features=1)
#         self.conf_act = nn.ReLU()
#
#         self.down_search = nn.Conv2d(256, 256, kernel_size=(7, 7))
#
#         # initialization
#         for modules in [self.down_conf, self.conf_fc1, self.conf_fc2]:
#             for l in modules.modules():
#                 if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear):
#                     torch.nn.init.normal_(l.weight, std=0.01)
#
#     def forward(self, f_, search_):   # x (b,512,25,25)   search(b,256,31,31)
#         search_ = self.down_search(search_)    # search(b,256,25,25)
#         x = torch.cat((f_, search_), dim=1)    # x(b,768,25,25)
#
#         conf = self.down_conf(x)    # conf (b,4,25,25)
#         conf = conf.flatten(1)      # conf (b,2500)
#         conf = self.conf_fc2(self.conf_act(self.conf_fc1(conf))).sigmoid().squeeze()    # conf tensor(0.9998, device'cuda0')
#         return conf

    # a = self.conf_fc1(conf)     # (b,512)
    # b = self.conf_act(a)        # (b,512)   把a张量中的负数变为0，其他元素不变
    # c = self.conf_fc2(b)        # (b,1)
    # conf = c.sigmoid().squeeze()

# Process_conf_new
class Process_conf(nn.Module):
    def __init__(self, in_channels):
        super(Process_conf, self).__init__()

        self.down_search = nn.Sequential(
            RepBlock(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=(7, 7)),
            RepBlock(256, 256, kernel_size=3, padding=1))

        self.down_conf = nn.Sequential(
            RepBlock(768, in_channels, kernel_size=3, padding=1),
            RepBlock(in_channels, in_channels, kernel_size=3, padding=1),
            RepBlock(in_channels, in_channels, kernel_size=3, padding=1),
            RepBlock(in_channels, 4, kernel_size=3, padding=1))

        self.fc_blocks = nn.Sequential(
            nn.Linear(in_features=2500, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1))

        # initialization
        for modules in [self.down_conf, self.fc_blocks, self.down_search]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear):
                    torch.nn.init.normal_(l.weight, std=0.01)

    def forward(self, f_, search_):   # x (b,512,25,25)   search(b,256,31,31)
        search_ = self.down_search(search_)    # search(b,256,25,25)
        x = torch.cat((f_, search_), dim=1)    # x(b,768,25,25)

        conf = self.down_conf(x)    # conf (b,4,25,25)
        conf = conf.flatten(1)      # conf (b,2500)
        conf = self.fc_blocks(conf).sigmoid().squeeze()    # conf tensor(0.9998, device'cuda0')
        return conf


# class box_tower_reg(nn.Module):
#     """
#     Box tower for FCOS reg
#     """
#     def __init__(self, in_channels=512, out_channels=256, tower_num=1):
#         super(box_tower_reg, self).__init__()
#         tower = []
#         cls_tower = []
#         cls_memory_tower = []
#         # Layers for multi-scale correlation
#         self.cls_encode = matrix(in_channels=in_channels, out_channels=out_channels)
#         self.reg_encode = matrix(in_channels=in_channels, out_channels=out_channels)
#         self.cls_dw = GroupDW(in_channels=in_channels)
#         self.reg_dw = GroupDW(in_channels=in_channels)
#         # Layers for confidence-value integration
#         self.conf_fusion = Conf_Fusion(in_channels=out_channels, out_channels=out_channels)
#         self.conv = nn.Conv2d(in_channels=256, out_channels=256,
#                              kernel_size=7, stride=1, padding=0)
#
#         # Box pred tower
#         for i in range(tower_num):
#             if i == 0:
#                 tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#             else:
#                 tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#
#             tower.append(nn.BatchNorm2d(out_channels))
#             tower.append(nn.ReLU())
#
#         # Cls tower
#         for i in range(tower_num):
#             if i == 0:
#                 cls_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#             else:
#                 cls_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#
#             cls_tower.append(nn.BatchNorm2d(out_channels))
#             cls_tower.append(nn.ReLU())
#
#         # Memory cls tower
#         for i in range(tower_num):
#             if i == 0:
#                 cls_memory_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#             else:
#                 cls_memory_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#
#             cls_memory_tower.append(nn.BatchNorm2d(out_channels))
#             cls_memory_tower.append(nn.ReLU())
#
#         self.add_module('bbox_tower', nn.Sequential(*tower))
#         self.add_module('cls_tower', nn.Sequential(*cls_tower))
#         self.add_module('cls_memory_tower', nn.Sequential(*cls_memory_tower))
#
#         # Reg head
#         self.bbox_pred = nn.Conv2d(out_channels, 4, kernel_size=3, stride=1, padding=1)
#         # Cls head
#         self.cls_pred = nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1)
#         self.cls_memory_pred = nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1)
#
#         # Adjust scale for reg
#         self.adjust = nn.Parameter(0.1 * torch.ones(1))
#         self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())
#
#     def forward(self, search, kernel=None, memory_kernel=None,
#                 memory_confidence=None, cls_x_store=None):
#
#         if kernel is not None:
#             # Since template kernel (pooled by PrPool from the template patch) is inputted,
#             #         we should conduct tracking with offline Siamese module
#
#             # Encode feature for correlation first
#             cls_z, cls_x = self.cls_encode(kernel, search)   # [z11, z12, z13]
#             reg_z, reg_x = self.reg_encode(kernel, search)  # [x11, x12, x13]
#
#             # DW for cls and reg
#             cls_dw = self.cls_dw(cls_z, cls_x)
#             reg_dw = self.reg_dw(reg_z, reg_x)
#             x_reg = self.bbox_tower(reg_dw)
#             x_bbox = self.adjust * self.bbox_pred(x_reg) + self.bias
#             x_bbox = torch.exp(x_bbox)
#
#             # Cls tower
#             c = self.cls_tower(cls_dw)
#             cls = 0.1 * self.cls_pred(c)
#
#             # 置信度分支的输入
#             # output = self.conv(search)
#             f_ = torch.cat((cls_dw, c), dim=1).detach()
#             search_ = search.detach()
#             kernel_ = kernel.detach()
#
#
#             if memory_kernel is None:
#                 # cache cls_x to prevent double processing of search area feature
#                 return x_bbox, cls, cls_x, reg_x, None, f_, search_, kernel_
#
#         # Cls memory
#         if memory_kernel is not None:
#
#             # Since memory queue is inputted, we should conduct tracking with online memory module
#             if cls_x_store is None:
#                 cls_mem_zs, cls_x_store = self.cls_encode(memory_kernel, x=search)
#             else:
#                 # Use the cached cls_x to prevent double processing of search area feature
#                 cls_mem_zs, _ = self.cls_encode(memory_kernel, x=None)
#
#             # Multi-scale correlation, between the memory queue and the search area feature in the template frame
#             batch, mem_size = memory_confidence.shape
#             store_repeat = []
#             for cls_x in cls_x_store:
#                 _, c, h, w = cls_x.shape
#                 cls_x_rep = cls_x.view(batch, 1, c, h, w)
#                 cls_x_rep = cls_x_rep.repeat(1, mem_size, 1, 1, 1).view(-1, c, h, w)
#                 store_repeat.append(cls_x_rep)
#
#             cls_mem_dw = self.cls_dw(cls_mem_zs, store_repeat)
#             _, c, h, w = cls_mem_dw.shape
#             cls_mem_dw = cls_mem_dw.view(batch, mem_size, c, h, w)
#
#             # Fuse memory correlation maps
#             cls_mem_fusion = self.conf_fusion(cls_mem_dw)
#
#             # Memory cls head
#             c_mem = self.cls_memory_tower(cls_mem_fusion)
#             cls_mem = 0.1 * self.cls_memory_pred(c_mem)
#
#             if kernel is not None:
#                 return x_bbox, cls, cls_x, reg_x, cls_mem, None
#             else:
#                 return None, None, None, None, cls_mem, None
#         return None
#
# class Process_conf(nn.Module):
#     def __init__(self, in_channels):
#         super(Process_conf, self).__init__()
#
#         self.down_conf = nn.Sequential(
#             RepBlock(1024, in_channels, kernel_size=3, padding=1),
#             RepBlock(in_channels, in_channels, kernel_size=3, padding=1),
#             RepBlock(in_channels, 4, kernel_size=3, padding=1))
#         self.conf_fc1 = nn.Linear(in_features=2500, out_features=512)
#         self.conf_fc2 = nn.Linear(in_features=512, out_features=1)
#         self.conf_act = nn.ReLU()
#
#         self.down_search = nn.Conv2d(256, 256, kernel_size=(7, 7))
#         self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#
#         # initialization
#         for modules in [self.down_conf, self.conf_fc1, self.conf_fc2]:
#             for l in modules.modules():
#                 if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear):
#                     torch.nn.init.normal_(l.weight, std=0.01)
#
#     def forward(self, f_, search_, kernel_):   # x (b,512,25,25)   search(b,256,31,31)
#         score_ = xcorr_depthwise(search_, kernel_)
#         score_1 = self.conv1(score_)
#         score_2 = self.conv2(score_1)
#
#         search_2 = self.down_search(search_)    # search(b,256,25,25)
#
#         x = torch.cat((f_, score_2, search_2), dim=1)    # x(b,768,25,25)
#
#         conf = self.down_conf(x)    # conf (b,4,25,25)
#         conf = conf.flatten(1)      # conf (b,2500)
#         conf = self.conf_fc2(self.conf_act(self.conf_fc1(conf))).sigmoid().squeeze()    # conf tensor(0.9998, device'cuda0')
#         return conf




#
# class box_tower_reg(nn.Module):
#     """
#     Box tower for FCOS reg
#     """
#     def __init__(self, in_channels=512, out_channels=256, tower_num=1):
#         super(box_tower_reg, self).__init__()
#         tower = []
#         cls_tower = []
#         cls_memory_tower = []
#         # Layers for multi-scale correlation
#         self.cls_encode = matrix(in_channels=in_channels, out_channels=out_channels)
#         self.reg_encode = matrix(in_channels=in_channels, out_channels=out_channels)
#         self.cls_dw = GroupDW(in_channels=in_channels)
#         self.reg_dw = GroupDW(in_channels=in_channels)
#         # Layers for confidence-value integration
#         self.conf_fusion = Conf_Fusion(in_channels=out_channels, out_channels=out_channels)
#         self.conv = nn.Conv2d(in_channels=256, out_channels=256,
#                              kernel_size=7, stride=1, padding=0)
#
#         # Box pred tower
#         for i in range(tower_num):
#             if i == 0:
#                 tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#             else:
#                 tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#
#             tower.append(nn.BatchNorm2d(out_channels))
#             tower.append(nn.ReLU())
#
#         # Cls tower
#         for i in range(tower_num):
#             if i == 0:
#                 cls_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#             else:
#                 cls_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#
#             cls_tower.append(nn.BatchNorm2d(out_channels))
#             cls_tower.append(nn.ReLU())
#
#         # Memory cls tower
#         for i in range(tower_num):
#             if i == 0:
#                 cls_memory_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#             else:
#                 cls_memory_tower.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#
#             cls_memory_tower.append(nn.BatchNorm2d(out_channels))
#             cls_memory_tower.append(nn.ReLU())
#
#         self.add_module('bbox_tower', nn.Sequential(*tower))
#         self.add_module('cls_tower', nn.Sequential(*cls_tower))
#         self.add_module('cls_memory_tower', nn.Sequential(*cls_memory_tower))
#
#         # Reg head
#         self.bbox_pred = nn.Conv2d(out_channels, 4, kernel_size=3, stride=1, padding=1)
#         # Cls head
#         self.cls_pred = nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1)
#         self.cls_memory_pred = nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1)
#
#         # Adjust scale for reg
#         self.adjust = nn.Parameter(0.1 * torch.ones(1))
#         self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())
#
#     def forward(self, search, kernel=None, memory_kernel=None,
#                 memory_confidence=None, cls_x_store=None):
#
#         if kernel is not None:
#             # Since template kernel (pooled by PrPool from the template patch) is inputted,
#             #         we should conduct tracking with offline Siamese module
#
#             # Encode feature for correlation first
#             cls_z, cls_x = self.cls_encode(kernel, search)   # [z11, z12, z13]
#             reg_z, reg_x = self.reg_encode(kernel, search)  # [x11, x12, x13]
#
#             # DW for cls and reg
#             cls_dw = self.cls_dw(cls_z, cls_x)
#             reg_dw = self.reg_dw(reg_z, reg_x)
#             x_reg = self.bbox_tower(reg_dw)
#             x_bbox = self.adjust * self.bbox_pred(x_reg) + self.bias
#             x_bbox = torch.exp(x_bbox)
#
#             # Cls tower
#             c = self.cls_tower(cls_dw)
#             cls = 0.1 * self.cls_pred(c)
#
#             # 置信度分支的输入
#             # output = self.conv(search)
#             f_ = torch.cat((cls_dw, c), dim=1).detach()
#             search_ = search.detach()
#             kernel_ = kernel.detach()
#
#
#             if memory_kernel is None:
#                 # cache cls_x to prevent double processing of search area feature
#                 return x_bbox, cls, cls_x, reg_x, None, f_, search_, kernel_
#
#         # Cls memory
#         if memory_kernel is not None:
#
#             # Since memory queue is inputted, we should conduct tracking with online memory module
#             if cls_x_store is None:
#                 cls_mem_zs, cls_x_store = self.cls_encode(memory_kernel, x=search)
#             else:
#                 # Use the cached cls_x to prevent double processing of search area feature
#                 cls_mem_zs, _ = self.cls_encode(memory_kernel, x=None)
#
#             # Multi-scale correlation, between the memory queue and the search area feature in the template frame
#             batch, mem_size = memory_confidence.shape
#             store_repeat = []
#             for cls_x in cls_x_store:
#                 _, c, h, w = cls_x.shape
#                 cls_x_rep = cls_x.view(batch, 1, c, h, w)
#                 cls_x_rep = cls_x_rep.repeat(1, mem_size, 1, 1, 1).view(-1, c, h, w)
#                 store_repeat.append(cls_x_rep)
#
#             cls_mem_dw = self.cls_dw(cls_mem_zs, store_repeat)
#             _, c, h, w = cls_mem_dw.shape
#             cls_mem_dw = cls_mem_dw.view(batch, mem_size, c, h, w)
#
#             # Fuse memory correlation maps
#             cls_mem_fusion = self.conf_fusion(cls_mem_dw)
#
#             # Memory cls head
#             c_mem = self.cls_memory_tower(cls_mem_fusion)
#             cls_mem = 0.1 * self.cls_memory_pred(c_mem)
#
#             if kernel is not None:
#                 return x_bbox, cls, cls_x, reg_x, cls_mem, None
#             else:
#                 return None, None, None, None, cls_mem, None
#         return None
#
# class Process_conf(nn.Module):
#     def __init__(self, in_channels):
#         super(Process_conf, self).__init__()
#         # self.feature_enhance = FeatureEnhance(in_channels=256, out_channels=256)
#         self.down_conf = nn.Sequential(
#             RepBlock(768, in_channels, kernel_size=3, padding=1),
#             RepBlock(in_channels, in_channels, kernel_size=3, padding=1),
#             RepBlock(in_channels, 4, kernel_size=3, padding=1))
#         self.conf_fc1 = nn.Linear(in_features=2500, out_features=512)
#         self.conf_fc2 = nn.Linear(in_features=512, out_features=1)
#         self.conf_act = nn.ReLU()
#         self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#
#         # self.down_search = nn.Conv2d(256, 256, kernel_size=(7, 7))
#
#         # initialization
#         for modules in [self.down_conf, self.conf_fc1, self.conf_fc2]:
#             for l in modules.modules():
#                 if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear):
#                     torch.nn.init.normal_(l.weight, std=0.01)
#
#     def forward(self, f_, search_, kernel_):   # x (b,512,25,25)   search(b,256,31,31)#
#         xf_ = xcorr_depthwise(search_, kernel_)
#         xf_1 = self.conv1(xf_)
#         xf_2 = self.conv2(xf_1)
#         x = torch.cat((f_, xf_2), dim=1)  # x(b,768,25,25)
#         conf = self.down_conf(x)    # conf (b,4,25,25)
#         conf = conf.flatten(1)      # conf (b,2500)
#         conf = self.conf_fc2(self.conf_act(self.conf_fc1(conf))).sigmoid().squeeze()    # conf tensor(0.9998, device'cuda0')
#         return conf



