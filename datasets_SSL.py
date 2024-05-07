# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Revised by Jilai Zheng, for USOT
# ------------------------------------------------------------------------------
from __future__ import division

import cv2
import json
import random
from os.path import join

import torch
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from lib.utils.image_utils import *
import imgaug as ia
import heapq
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from augtest.augtest import random_sys, disappear, random_translation, random_occ, random_background, rand, \
    image_augmentation



sample_random = random.Random()


# sample_random.seed(123456)

class USOTDataset(Dataset):
    def __init__(self, cfg, aug_settings, crop_settings, epoch, json_path_dict):
        super(USOTDataset, self).__init__()

        # Pair information
        self.template_size = cfg.USOT.TRAIN.TEMPLATE_SIZE
        self.search_size = cfg.USOT.TRAIN.SEARCH_SIZE
        # Response map size
        self.size = 25
        # Feature size of template patch
        self.tf_size = 15
        # Feature axis of search area (designed to be the same as response map size in USOT v1)
        self.sf_size = 25
        # Total stride of backbone
        self.stride = cfg.USOT.TRAIN.STRIDE

        # Aug information
        self.aug_settings = aug_settings
        self.crop_settings = crop_settings
        # Aug for template patch
        self.shift = cfg.USOT.DATASET.SHIFT
        self.scale = cfg.USOT.DATASET.SCALE

        # Aug for search areas
        self.shift_s = cfg.USOT.DATASET.SHIFTs
        self.scale_s = cfg.USOT.DATASET.SCALEs

        # Aug for memory search areas
        self.shift_m = cfg.USOT.DATASET.SHIFTm
        self.scale_m = cfg.USOT.DATASET.SCALEm

        # Threshold for video quality
        self.video_quality = cfg.USOT.DATASET.VIDEO_QUALITY
        # Number of memory frames in a single training instance
        self.memory_num = cfg.USOT.TRAIN.MEMORY_NUM
        # Parameter for sampling memory frames
        self.far_sample = cfg.USOT.DATASET.FAR_SAMPLE

        # Choices for training
        # Set self.cycle_memory = False for naive Siamese training
        # Set self.cycle_memory = True for cycle memory training
        self.cycle_memory = True
        self.json_path_dict = json_path_dict
        self.epoch = epoch
        # For testing dataloader, you can set self.loader_test to True
        # See the dataloader testing scripts at the bottom of this file
        self.loader_test = False

        self.grids()



        self.template_aug_seq = iaa.Sequential([
            iaa.PerspectiveTransform(scale=(0.01, 0.07)),  # 透视变换

        ])

        # Augmentation for search area
        self.search_aug_seq1 = iaa.Sequential([
            iaa.Fliplr(0.2),  # 左右翻转
            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),  # 多重色调和饱和度
            iaa.MotionBlur(k=(3, 9), angle=[-60, 60]),  # 运动模糊

        ])
        # Augmentation for search area
        self.search_aug_seq2 = iaa.Sequential([
            iaa.Flipud(0.1),  # 上下翻转
            iaa.MultiplyBrightness((0.5, 1.5)),  # 倍增亮度
            iaa.SaltAndPepper(0.05, per_channel=True),  # 盐和胡椒
        ])
        self.search_aug_seq3 = iaa.Sequential([
            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),  # 多重色调和饱和度
            iaa.CoarseDropout((0.0, 0.05), size_percent=0.15, per_channel=0.5),  # 随机遮挡
        ])
        self.memory_aug_seq = iaa.Sequential([
            iaa.Fliplr(0.4),
            iaa.Flipud(0.2),
            iaa.PerspectiveTransform(scale=(0.01, 0.15)),
            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
            iaa.MultiplyBrightness((0.5, 1.5)),
            iaa.MotionBlur(k=(3, 9), angle=[-60, 60]),
        ])

        print('train datas: {}'.format(cfg.USOT.TRAIN.WHICH_USE))
        # List of all training datasets
        self.train_datas = []
        start = 0
        self.num = 0
        for data_name in cfg.USOT.TRAIN.WHICH_USE:
            dataset = subData(cfg, data_name, start, self.memory_num,
                              self.video_quality, self.far_sample, self.epoch, self.json_path_dict)
            self.train_datas.append(dataset)
            # Real video number
            start += dataset.num
            # The number used for subset shuffling
            self.num += dataset.num_use

        self._shuffle()
        print(cfg)



    def __len__(self):
        return self.num

    def __getitem__(self, index):
        """
        The main logic for sampling training instances.
        Two sampling modes are provided:
           1. naive Siamese: sampling a template frame and crop both template and search area from it
           2. cycle memory: besides the naive Siamese pair, sampling N_mem memory search areas additionally
        Switch between two modes: self.cycle_memory should be set to True only when conducting cycle memory training.
        """
        # aug参数设置
        # cv2.setNumThreads(0)
        # cv2.ocl.setUseOpenCL(False)  ##关闭OpenCV的多线程
        neg = random_sys()
        pos = 1.0  # if neg >= self.aug_settings['neg_pair']['prob'] else 0.0
        pos_list = []
        type = 0.0
        type_list = []
        index = self.pick[index]

        dataset, index = self._choose_dataset(index)

        # For offline naive Siamese tracker (template patch and search area are picked in the same frame)
        # Note that the actual video index may be re-sampled in _get_instances() if the video is of low quality
        # Warning: cycle_memory should be set to True only when conducting cycle memory training.
        pair_info = dataset._get_instances(index, cycle_memory=self.cycle_memory)

        # Here only one template frame image is returned, and it will be utilized for both template and search
        search_image = cv2.imread(pair_info[0])
        ### 我加的
        avg_channels = np.mean(search_image, axis=(0, 1))
        search_image = self.crop_511(search_image, pair_info[1], instanc_size=511, padding=avg_channels)
        search_bbox = self._toBBox(search_image, pair_info[1])
        template_image = search_image
        template_bbox = search_bbox

        # 生成正负样本对
        # 步骤一：511的输入图裁剪成127 255
        template_img, template_box, dag_param_t = self._511crop127255(template_image, template_bbox, 127)
        search_img, search_box, dag_param_s = self._511crop127255(search_image, search_bbox, 255, search=True)
        search_box2 = np.array(search_box)
        # 步骤二：生成负样本对   0-0.5
        if neg < self.aug_settings['neg_pair']['prob']:

            # 负样本对1,来自不同帧  0-0.1
            if neg < self.aug_settings['different_seq']['prob']:
                # 随机抽取一幅图像，作为模板图
                index = random.randint(max(0, index - 20), min(dataset.num - 1, index + 20))
                pair = dataset._get_instances(index, cycle_memory=self.cycle_memory)
                template_image = cv2.imread(pair[0])
                template_bbox = pair[1]
                avg_channels = np.mean(template_image, axis=(0, 1))
                template_image = self.crop_511(template_image, template_bbox, instanc_size=511, padding=avg_channels)
                template_bbox = self._toBBox(template_image, template_bbox)
                # 裁剪成127*127
                template_img, template_box, _ = self._511crop127255(template_image, template_bbox, 127)
                pos = 0.0
                type = 1.0
            # 负样本对2，让目标消失，搜索区域没有目标  原图search_image 0.1-0.2
            else:
                if neg < self.aug_settings['disappear']['prob']:
                    # 随机地在搜索图像中找到一片背景区域作为搜索patch，要求其中不能有目标
                    outs = disappear(search_image, search_bbox, size=self.search_size,
                                     settings=self.crop_settings['val'])
                    if outs[0] is not None:
                        search_img, search_box, mix_boxes = outs
                        pos = 0.0
                        type = 2.0

                else:
                    # 负样本对3，背景挡到目标box上 原图search_image 0.2-0.3
                    if neg < self.aug_settings['occ_background']['prob']:
                        # 在当前图像中找到一块背景区域作为遮挡块
                        occ_image = search_image
                        occ_image_, occ_box_, _ = self._511crop127255(occ_image, search_bbox, 255, search=True)
                        occ_bbox = random_background(occ_image_, occ_box_,
                                                     min_rate=self.aug_settings['occ_background']['crop_rate_min'],
                                                     max_rate=self.aug_settings['occ_background']['crop_rate_max'],
                                                     protect_settings=rand(
                                                         self.aug_settings['occ_background']['protect_rate_min'],
                                                         self.aug_settings['occ_background']['protect_rate_max']))
                        if occ_bbox is not None:
                            box = list(map(int, occ_bbox))
                            # 裁剪出遮挡块
                            occ = occ_image_[box[1]: box[3] + 1, box[0]: box[2] + 1]
                            # 将遮挡块覆盖到目标框位置上，并计算box被遮挡部分的占比
                            search_img_, occed_box, overlap = random_occ(search_img, search_box2, obj=occ,
                                                                         center_rate=self.aug_settings['center_rate'],
                                                                         try_num=self.aug_settings['try_num'],
                                                                         overlap_thresh=self.aug_settings[
                                                                             'overlap_thresh'])
                            if occed_box is not None and overlap > self.aug_settings['overlap_thresh']:
                                search_img = search_img_
                                pos = 0.0
                                type = 3.0

                    else:
                        # 负样本对4，其他序列目标遮挡 0.3-0.5
                        index = random.randint(max(0, index - 20), min(dataset.num - 1, index + 20))
                        pair = dataset._get_instances(index, cycle_memory=self.cycle_memory)
                        random_image = cv2.imread(pair[0])
                        random_bbox = pair[1]
                        random_img, random_box, _ = self.crop127255(random_image, random_bbox, 255)
                        occ_image, occ_bbox = random_img, random_box
                        if occ_bbox is not None:
                            box = list(map(int, occ_bbox))
                            # 裁剪出遮挡块
                            occ = occ_image[box[1]: box[3] + 1, box[0]: box[2] + 1]
                            # 将遮挡块覆盖到目标框位置上，并计算box被遮挡部分的占比
                            search_img_, occed_box, overlap = random_occ(search_img, search_box2, obj=occ,
                                                                         center_rate=self.aug_settings['center_rate'],
                                                                         try_num=self.aug_settings['try_num'],
                                                                         overlap_thresh=self.aug_settings[
                                                                             'overlap_thresh'])
                            if occed_box is not None and overlap > self.aug_settings['overlap_thresh']:
                                search_img = search_img_
                                pos = 0.0
                                type = 4.0

        # 正样本对，复制粘贴目标，造成干扰
        else:
            copy_mix = random_sys()  ##0.24
            if copy_mix < self.aug_settings['copy_mix']['prob']:
                # 将目标区域（带一定大小的背景）从搜索图像中裁剪出来。记录下box的放大倍数，以便移动后精确计算
                scale_w, scale_h = self.roi_rate(), self.roi_rate()
                search_area = box2roi(search_box2, rate_w=scale_w, rate_h=scale_h, boundary=[255, 255])
                box = list(map(int, search_area))
                target = search_img[box[1]: box[3] + 1, box[0]: box[2] + 1]
                # 粘到目标周围
                search_img, mixed_box = random_translation(search_img, search_box2, obj=target,
                                                           min_rate=self.aug_settings['copy_mix']['trans_rate_min'],
                                                           max_rate=self.aug_settings['copy_mix']['trans_rate_max'])

        pos_list.append(pos)
        type_list.append(type)
        # 步骤三：只对正样本对增强
        template_aug, bbox_t = self._aug(template_img, template_box)
        if pos == 1.0:
            search_aug, bbox_s = self._aug(search_img, search_box, search=True)
        else:
            search_aug = search_img
            bbox_s = search_box

        # # ###可视化单帧图片
        # loader_test_name = "{:06d}".format(random.randint(0, 999999))
        # self._draw(search_aug, bbox_s, "/home/zcy/code/hyqcode/autodlusot/USOT-main/show/dan/" + loader_test_name + "_s.jpg")
        # self._draw(template_aug, bbox_t, "/home/zcy/code/hyqcode/autodlusot/USOT-main/show/dan/" + loader_test_name + "_t.jpg")

        # Now begin to retrieve memory search areas for cycle memory training
        search_memory = []
        a = 0
        if self.cycle_memory:
            search_images_nearby = []
            for image_path in pair_info[2]:
                search_image = cv2.imread(image_path)
                avg_channels = np.mean(search_image, axis=(0, 1))
                search_image = self.crop_511(search_image, pair_info[3][a], instanc_size=511, padding=avg_channels)
                a += 1
                search_images_nearby.append(search_image)
            search_bbox_nearby = [self._toBBox(search_images_nearby[i], pair_info[3][i])
                                  for i in range(len(search_images_nearby))]

            for i in range(len(search_images_nearby)):
                crop_nearby, bbox_nearby, _ = self._augmentation(search_images_nearby[i], search_bbox_nearby[i],
                                                                 self.search_size, search=True,
                                                                 cycle_memory=self.cycle_memory)
                # ####可视化多帧
                # self._draw(crop_nearby, bbox_nearby, "/home/zcy/code/hyqcode/autodlusot/USOT-main/show/dan/" + loader_test_name + "_n_{:02d}.jpg".format(i))
                crop_nearby = np.array(crop_nearby)
                crop_nearby = np.transpose(crop_nearby, (2, 0, 1)).astype(np.float32)
                search_memory.append(crop_nearby)
            search_memory = np.stack(search_memory)

        # From PIL image to numpy
        template = np.array(template_aug)
        search = np.array(search_aug)

        template, search = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search])

        # Pseudo classification labels for offline naive tracker
        out_label = self._dynamic_label([self.size, self.size], dag_param_s.shift)

        # Pseudo regression labels for offline naive tracker
        reg_label, reg_weight = self.reg_label(bbox_s)

        # Template pseudo bbox label for PrROIPooling
        bbox_t = self.pool_label_template(bbox_t)
        bbox_t = np.array(bbox_t, np.float32)
        pos_list = torch.tensor(pos_list)
        type_list = torch.Tensor(type_list)
        if len(search_memory) > 0:
            # Search area bbox label for PrROIPooling
            bbox_s = self.pool_label_search(bbox_s)
            bbox_s = np.array(bbox_s, np.float32)
            # Additionally return memory frames and bbox_s for Siamese search areas
            return template, search, out_label, reg_label, reg_weight, bbox_t, \
                pos_list, type_list, search_memory, bbox_s

        return template, search, out_label, reg_label, reg_weight, bbox_t, pos_list, type_list

    def _shuffle(self):
        """
        Random shuffle
        """
        pick = []
        m = 0
        while m < self.num:
            p = []
            for subset in self.train_datas:
                sub_p = subset.pick
                p += sub_p
            sample_random.shuffle(p)

            pick += p
            m = len(pick)
        self.pick = pick
        print("dataset length {}".format(self.num))

    def _choose_dataset(self, index):
        for dataset in self.train_datas:
            if dataset.start + dataset.num > index:
                return dataset, index - dataset.start

    def grids(self):
        """
        Each element of feature map on template patch and response map
        :return: H*W*2 (position for each element)
        """
        # Response map grid
        sz = self.size
        sz_x = sz // 2
        sz_y = sz // 2
        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))
        self.grid_to_search = {}
        self.grid_to_search_x = x * self.stride + self.search_size // 2
        self.grid_to_search_y = y * self.stride + self.search_size // 2

        # Template feature map grid
        tf_sz = self.tf_size
        sz_x_t = tf_sz // 2
        sz_y_t = tf_sz // 2
        x, y = np.meshgrid(np.arange(0, tf_sz) - np.floor(float(sz_x_t)),
                           np.arange(0, tf_sz) - np.floor(float(sz_y_t)))

        self.grid_to_template = {}
        self.grid_to_template_x = x * self.stride + self.template_size // 2
        self.grid_to_template_y = y * self.stride + self.template_size // 2

        # Search area feature map grid
        sf_sz = self.sf_size
        sf_x_s = sf_sz // 2
        self.search_area_x_axis = (np.arange(0, sf_sz) - np.floor(float(sf_x_s))) \
                                  * self.stride + self.search_size // 2

    def reg_label(self, bbox):
        """
        Generate regression label
        :param bbox: [x1, y1, x2, y2]
        :return: [l, t, r, b]
        """
        x1, y1, x2, y2 = bbox
        l = self.grid_to_search_x - x1
        t = self.grid_to_search_y - y1
        r = x2 - self.grid_to_search_x
        b = y2 - self.grid_to_search_y

        l, t, r, b = map(lambda x: np.expand_dims(x, axis=-1), [l, t, r, b])
        reg_label = np.concatenate((l, t, r, b), axis=-1)
        reg_label_min = np.min(reg_label, axis=-1)
        inds_nonzero = (reg_label_min > 0).astype(float)

        return reg_label, inds_nonzero

    def pool_label_template(self, bbox):
        """
        Get pseudo bbox for PrPool on the template patch
        """

        reg_min = self.grid_to_template_x[0][0]
        reg_max = self.grid_to_template_x[-1][-1]

        bbox = np.array(bbox, np.float32)
        bbox = np.clip(bbox, a_max=reg_max, a_min=reg_min)

        sz = 2 * (self.tf_size // 2)
        slope = sz / (reg_max - reg_min)

        return (bbox - reg_min) * slope

    def pool_label_search(self, bbox):
        """
        Get pseudo bbox for PrPool on the search area
        """

        reg_min = self.search_area_x_axis[0]
        reg_max = self.search_area_x_axis[-1]

        bbox = np.array(bbox, np.float32)
        bbox = np.clip(bbox, a_max=reg_max, a_min=reg_min)

        sz = 2 * (self.sf_size // 2)
        slope = sz / (reg_max - reg_min)

        return (bbox - reg_min) * slope

    def _posNegRandom(self):
        """
        Get a random number from [-1, 1]
        """
        return random.random() * 2 - 1.0

    def _toBBox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.template_size

        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def crop_511(self, image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=511, padding=(0, 0, 0)):
        target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
        target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]  # width, height
        wc_z = target_size[1] + context_amount * sum(target_size)
        hc_z = target_size[0] + context_amount * sum(target_size)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        d_search = (instanc_size - exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad
        x = self._crop_hwc(image, self.pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
        return x

    def pos_s_2_bbox(self, pos, s):
        return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]

    def _crop_hwc(self, image, bbox, out_sz, padding=(0, 0, 0)):
        """
        Crop image
        """
        bbox = [float(x) for x in bbox]
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        return crop

    def _draw(self, image, box, name):
        """
        Draw loaded image for debugging
        """
        draw_image = np.array(image.copy()[:, :, ::-1])
        if box is not None:
            x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
            cv2.rectangle(draw_image, (x1, y1), (x2, y2), (255, 215, 0), 2)
            # cv2.circle(draw_image, (int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)), 3, (0, 0, 255))
            # cv2.putText(draw_image, '[x: {}, y: {}]'.format(int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)),
            #            (int(round(x1 + x2) / 2) - 3, int(round(y1 + y2) / 2) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
            #            (255, 255, 255), 1)
        cv2.imwrite(name, draw_image[:, :, ::-1])

    def clip_number(self, num, _max=127.0, _min=0.0):

        if num >= _max:
            return _max
        elif num <= _min:
            return _min
        else:
            return num

    def roi_rate(self):
        return rand(self.aug_settings['roi_rate_min'], self.aug_settings['roi_rate_max'])

    def _511crop127255(self, image, bbox, size, search=False, cycle_memory=False):
        """
        Data augmentation for input frames
        """
        shape = image.shape
        crop_bbox = center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        if not search:
            # The shift and scale for template
            param.shift = (self._posNegRandom() * self.shift, self._posNegRandom() * self.shift)  # shift
            param.scale = (
                (1.0 + self._posNegRandom() * self.scale), (1.0 + self._posNegRandom() * self.scale))  # scale change
        elif not cycle_memory:
            # The shift and scale for search area
            param.shift = (self._posNegRandom() * self.shift_s, self._posNegRandom() * self.shift_s)  # shift
            param.scale = (
                (1.0 + self._posNegRandom() * self.scale_s),
                (1.0 + self._posNegRandom() * self.scale_s))  # scale change
        else:
            # The shift and scale for memory search areas
            param.shift = (self._posNegRandom() * self.shift_m, self._posNegRandom() * self.shift_m)  # shift
            param.scale = (
                (1.0 + self._posNegRandom() * self.scale_m),
                (1.0 + self._posNegRandom() * self.scale_m))  # scale change

        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)

        x1, y1 = crop_bbox.x1, crop_bbox.y1
        # bbox = BBox(bbox.x1 - x1, bbox.y1 - y1, bbox.x2 - x1, bbox.y2 - y1)
        bbox = BBox(bbox[0] - x1, bbox[1] - y1, bbox[2] - x1, bbox[3] - y1)

        scale_x, scale_y = param.scale
        bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale

        return image, bbox, param

    def _aug(self, image, bbox, search=False, cycle_memory=False):
        """
        Data augmentation for input frames
        """

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2),
        ], shape=image.shape)

        if not search:
            # Augmentation for template
            image, bbs_aug = self.template_aug_seq(image=image, bounding_boxes=bbs)
        elif not cycle_memory:  # 搜索帧
            a = random_sys()
            if a < 0.3:
                image, bbs_aug = self.search_aug_seq1(image=image, bounding_boxes=bbs)
            elif a >= 0.3 and a <= 0.6:
                image, bbs_aug = self.search_aug_seq2(image=image, bounding_boxes=bbs)
            else:
                image, bbs_aug = self.search_aug_seq3(image=image, bounding_boxes=bbs)
        else:
            # Augmentation for memory search areas
            image, bbs_aug = self.memory_aug_seq(image=image, bounding_boxes=bbs)

        bbox = Corner(self.clip_number(bbs_aug[0].x1, _max=image.shape[0]),
                      self.clip_number(bbs_aug[0].y1, _max=image.shape[1]),
                      self.clip_number(bbs_aug[0].x2, _max=image.shape[0]),
                      self.clip_number(bbs_aug[0].y2, _max=image.shape[1]))

        return image, bbox

    def crop127255(self, image, bbox, size, search=False, cycle_memory=False):
        """
        Data augmentation for input frames
        """
        shape = image.shape
        crop_bbox = center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        if not search:
            # The shift and scale for template
            param.shift = (self._posNegRandom() * self.shift, self._posNegRandom() * self.shift)  # shift
            param.scale = (
                (1.0 + self._posNegRandom() * self.scale), (1.0 + self._posNegRandom() * self.scale))  # scale change
        elif not cycle_memory:
            # The shift and scale for search area
            param.shift = (self._posNegRandom() * self.shift_s, self._posNegRandom() * self.shift_s)  # shift
            param.scale = (
                (1.0 + self._posNegRandom() * self.scale_s),
                (1.0 + self._posNegRandom() * self.scale_s))  # scale change
        else:
            # The shift and scale for memory search areas
            param.shift = (self._posNegRandom() * self.shift_m, self._posNegRandom() * self.shift_m)  # shift
            param.scale = (
                (1.0 + self._posNegRandom() * self.scale_m),
                (1.0 + self._posNegRandom() * self.scale_m))  # scale change

        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)

        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = BBox(bbox[0] - x1, bbox[1] - y1, bbox[2] - x1, bbox[3] - y1)

        scale_x, scale_y = param.scale
        bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
        ], shape=image.shape)

        bbox = Corner(self.clip_number(bbs[0].x1, _max=image.shape[0]),
                      self.clip_number(bbs[0].y1, _max=image.shape[1]),
                      self.clip_number(bbs[0].x2, _max=image.shape[0]),
                      self.clip_number(bbs[0].y2, _max=image.shape[1]))

        return image, bbox, param

    def _augmentation(self, image, bbox, size, search=False, cycle_memory=False):
        """
        Data augmentation for input frames
        """
        shape = image.shape
        crop_bbox = center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        if not search:
            # The shift and scale for template
            param.shift = (self._posNegRandom() * self.shift, self._posNegRandom() * self.shift)  # shift
            param.scale = (
                (1.0 + self._posNegRandom() * self.scale), (1.0 + self._posNegRandom() * self.scale))  # scale change
        elif not cycle_memory:
            # The shift and scale for search area
            param.shift = (self._posNegRandom() * self.shift_s, self._posNegRandom() * self.shift_s)  # shift
            param.scale = (
                (1.0 + self._posNegRandom() * self.scale_s),
                (1.0 + self._posNegRandom() * self.scale_s))  # scale change
        else:
            # The shift and scale for memory search areas
            param.shift = (self._posNegRandom() * self.shift_m, self._posNegRandom() * self.shift_m)  # shift
            param.scale = (
                (1.0 + self._posNegRandom() * self.scale_m),
                (1.0 + self._posNegRandom() * self.scale_m))  # scale change

        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)

        x1, y1 = crop_bbox.x1, crop_bbox.y1
        # bbox = BBox(bbox.x1 - x1, bbox.y1 - y1, bbox.x2 - x1, bbox.y2 - y1)
        bbox = BBox(bbox[0] - x1, bbox[1] - y1, bbox[2] - x1, bbox[3] - y1)

        scale_x, scale_y = param.scale
        bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale   255

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2),
        ], shape=image.shape)

        if not search:
            # Augmentation for template
            image, bbs_aug = self.template_aug_seq(image=image, bounding_boxes=bbs)
        elif not cycle_memory:
            # Augmentation for search area
            image, bbs_aug = self.search_aug_seq(image=image, bounding_boxes=bbs)
        else:
            # Augmentation for memory search areas
            image, bbs_aug = self.memory_aug_seq(image=image, bounding_boxes=bbs)

        bbox = Corner(self.clip_number(bbs_aug[0].x1, _max=image.shape[0]),
                      self.clip_number(bbs_aug[0].y1, _max=image.shape[1]),
                      self.clip_number(bbs_aug[0].x2, _max=image.shape[0]),
                      self.clip_number(bbs_aug[0].y2, _max=image.shape[1]))

        return image, bbox, param

    def _dynamic_label(self, fixedLabelSize, c_shift, rPos=2, rNeg=0):
        """
        Generating classification label
        """
        if isinstance(fixedLabelSize, int):
            fixedLabelSize = [fixedLabelSize, fixedLabelSize]

        assert (fixedLabelSize[0] % 2 == 1)

        d_label = self._create_dynamic_logisticloss_label(fixedLabelSize, c_shift, rPos, rNeg)

        return d_label

    def _create_dynamic_logisticloss_label(self, label_size, c_shift, rPos=2, rNeg=0):
        if isinstance(label_size, int):
            sz = label_size
        else:
            sz = label_size[0]

        sz_x = sz // 2 + int(-c_shift[0] / 8)  # 8 is the stride
        sz_y = sz // 2 + int(-c_shift[1] / 8)

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        dist_to_center = np.abs(x) + np.abs(y)  # Block metric
        label = np.where(dist_to_center <= rPos,
                         np.ones_like(y),
                         np.where(dist_to_center < rNeg,
                                  0.5 * np.ones_like(y),
                                  np.zeros_like(y)))
        return label


class subData(object):
    """
    Sub dataset class for training USOT with multi dataset
    """

    def __init__(self, cfg, data_name, start, memory_num, video_quality, far_sample, epoch, json_path_dict):
        self.data_name = data_name
        self.start = start
        # Dataset info
        info = cfg.USOT.DATASET[data_name]
        self.root = info.PATH


        if epoch >= cfg.USOT.TRAIN.MEMORY_EPOCH:
            json_path = json_path_dict[data_name]
            with open(json_path) as fin:
                self.labels = json.load(fin)
                self._clean()
                # Video number
                self.num = len(self.labels)
        else:
            with open(info.ANNOTATION) as fin:
                self.labels = json.load(fin)
                self._clean()
                # Video number
                self.num = len(self.labels)


        # Number of training instances used in each epoch for a certain dataset
        self.num_use = info.USE
        # Number of memory frames in a single training instance
        self.memory_num = memory_num
        # The threshold to filter videos
        self.video_quality = video_quality
        # When sampling memory frames, first sample (memory_num + far_sample) frames in the video fragment,
        #             and then pick (memory_num) frames "most far from" the template frame
        self.far_sample = far_sample

        self._shuffle()

    def _clean(self):
        """
        Remove empty videos/frames/annos in dataset
        """
        # No frames
        to_del = []
        for video in self.labels:
            frames = self.labels[video]
            if len(frames) <= 0:
                print("warning {} has no frames.".format(video))
                to_del.append(video)

        for video in to_del:
            try:
                del self.labels[video]
            except:
                pass

        print(self.data_name)

        self.videos = list(self.labels.keys())
        print('{} loaded.'.format(self.data_name))

    def _shuffle(self):
        """
        Shuffle to get random pairs index (video)
        """
        lists = list(range(self.start, self.start + self.num))
        m = 0
        pick = []
        while m < self.num_use:
            sample_random.shuffle(lists)
            pick += lists
            m += self.num

        self.pick = pick[:self.num_use]
        return self.pick

    def _get_siamese_image_anno(self, index):
        image_path_s = ""
        track_info = []
        if 'VID' in self.data_name:
            video_name = self.videos[index]
            video = self.labels[video_name]
            video_keys = list(video.keys())
            frames = video_keys[0]  # '00'
            video_info = video[frames]
            track_id = list(video_info.keys())[0]
            track_info = video_info[track_id]
            image_path_s = join(self.root, video_name, "{}.JPEG".format(track_id))
        elif 'GOT10K' in self.data_name:
            video_name = self.videos[index]
            video = self.labels[video_name]
            track_id = list(video.keys())[0]
            track_info = video[track_id]
            image_path_s = join(self.root, video_name, "{}.jpg".format(track_id))
        elif 'LASOT' in self.data_name:
            video_name = self.videos[index]
            video = self.labels[video_name]
            video_keys = list(video.keys())
            frames = video_keys[0]  # frames == '00'
            video_info = video[frames]
            track_id = list(video_info.keys())[0]    #'000000'
            track_info = video_info[track_id]      #'000000'对应的bbox
            track_id_6 = str(int(track_id) + 1)   #'000001'
            track_id_8 = track_id_6.zfill(8)     #'00000001'
            image_path_s = join(self.root, video_name, 'img', "{}.jpg".format(track_id_8))
        elif 'YTVOS' in self.data_name:
            video_name = self.videos[index]
            video = self.labels[video_name]
            track_id = list(video.keys())[0]
            track_info = video[track_id]
            image_path_s = join(self.root, video_name, "{}.jpg".format(track_id))
        # Return the single frame for template-search pair
        return image_path_s, track_info[:4]

    def _get_cycle_memory_image_anno(self, index):
        # 第一帧
        # video_name = 'c/ILSVRC2015_train_00574000'
        video_name = self.videos[index]
        video_info = self.labels[video_name]
        frames = video_info["frames"]
        track_id = list(frames.keys())[0]
        track_info = frames[track_id]
        suffix = ""
        if 'VID' in self.data_name:
            suffix = "{}.JPEG".format(track_id)
        elif 'GOT10K' in self.data_name:
            suffix = "{}.jpg".format(track_id)
        elif 'LASOT' in self.data_name:
            suffix = join('img', "{}.jpg".format(track_id))
        elif 'YTVOS' in self.data_name:
            suffix = "{}.jpg".format(track_id)
        image_path_s = join(self.root, video_name, suffix)
        # 后续帧抽m帧

        lost = video_info['lost']
        if not lost:  # lost=false,没丢
            my_list = list(frames.keys())[1:]
        else:  # lost=true丢了
            my_list = list(frames.keys())[1:-1]
        conf_list = video_info['conf']
        n = self.memory_num
        if len(conf_list) == 0:
            frames_list = list(frames.keys())
            frame_id_nearby_s = frames_list * n
        else:
            if n >= len(conf_list):
                largest_values = conf_list * n
                my_list = my_list * n
            else:
                largest_values = heapq.nlargest(n, conf_list)
            largest_indexes = []
            for value in largest_values:
                index = conf_list.index(value)
                largest_indexes.append(index)
            largest_new_indexes = []
            for index in largest_indexes:
                if index >= len(my_list):
                    index_new = len(my_list) - 1
                else:
                    index_new = index
                largest_new_indexes.append(index_new)
            frame_id_nearby_s = [my_list[index] for index in largest_new_indexes]

        bbox_nearby_s = [frames[frame_id][:4] for frame_id in frame_id_nearby_s]
        image_path_nearby_s = []
        for frame_id in frame_id_nearby_s:
            if 'VID' in self.data_name:
                suffix = "{}.JPEG".format(frame_id)
            elif 'GOT10K' in self.data_name:
                suffix = "{}.jpg".format(frame_id)
            elif 'LASOT' in self.data_name:
                suffix = join('img', "{}.jpg".format(track_id))
            elif 'YTVOS' in self.data_name:
                suffix = "{}.jpg".format(track_id)
            image_path_nearby_s.append(join(self.root, video_name, suffix))
        # 返回第一帧和分阶段抽的几帧
        return image_path_s, track_info[:4], image_path_nearby_s, bbox_nearby_s

    def _get_instances(self, index, cycle_memory=False):
        """
        get training instances
        """

        if cycle_memory:
            # For cycle memory training (returning one search frame and several memory frames)
            return self._get_cycle_memory_image_anno(index)
        else:
            # For offline naive Siamese tracker (one template and one search area picked in the same frame)
            return self._get_siamese_image_anno(index)


if __name__ == '__main__':

    import os
    from torch.utils.data import DataLoader
    from lib.config.config_usot import config

    # For testing and visualizing dataloader, you can set self.loader_test to True
    # Then example training instances can be found in $USOT_PATH/var/loader
    vis_dataloader_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       "..", "..", "var", "loader")
    if not os.path.exists(vis_dataloader_path):
        os.makedirs(vis_dataloader_path)

    # Cycle the dataloader
    train_set = USOTDataset(config)
    train_set.loader_test = True
    train_loader = DataLoader(train_set, batch_size=4, num_workers=1, pin_memory=False)

    for iter, input in enumerate(train_loader):
        template = input[0]
        search = input[1]
