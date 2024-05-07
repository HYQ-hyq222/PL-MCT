# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Revised by Jilai Zheng, for USOT
# ------------------------------------------------------------------------------
from __future__ import absolute_import
import os
import cv2
import random
import argparse
import numpy as np
import torch
import sys
from os.path import exists, join, dirname, realpath, abspath

import lib.models.models as models

from lib.tracker.SSL_tracker import USOTTracker
from easydict import EasyDict as edict
from lib.utils.train_utils import load_pretrain
from lib.utils.test_utils import cxy_wh_2_rect, get_axis_aligned_bbox, poly_iou
from lib.dataset_loader.benchmark import load_dataset
from lib.eval_toolkit.pysot.utils import success_overlap, success_error
from lib.eval_toolkit.pysot.utils.region import vot_overlap, vot_float2str

def parse_args():
    """
    args for USOT testing.
    """
    #### 还要看是离线还是在线更新
    parser = argparse.ArgumentParser(description='USOT testing')
    parser.add_argument('--arch', dest='arch', default='USOT', help='backbone architecture')
    parser.add_argument('--resume', default="/home/zcy/code/hyqcode/autodlusot/USOT-main/scripts/var_vid/var106/snapshot/checkpoint_e10.pth", type=str, help='pretrained model')
    parser.add_argument('--dataset', default='VOT2016', help='dataset test')
    # parser.add_argument('--dataset', default='VOT2018', help='dataset test')
    # parser.add_argument('--dataset', default='OTB100', help='dataset test')
    # parser.add_argument('--dataset', default='LaSOT', help='dataset test')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    parser.add_argument('--version', default='v1', help='testing style version')
    parser.add_argument('--gpu_id', default=0, type=int, help="gpu id")
    args = parser.parse_args()
    return args

def track(tracker, net, video, args):
    start_frame, toc = 0, 0

    # Save result to evaluate
    if args.epoch_test:
        suffix = args.resume.split('/')[-1]
        suffix = suffix.split('.')[0]
        tracker_path = os.path.join('var_vid/var1228/result', args.dataset, args.arch + suffix)
    else:
        tracker_path = os.path.join('var_vid/var1228/result', args.dataset, args.arch)

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in args.dataset:
        baseline_path = os.path.join(tracker_path, 'baseline')
        video_path = os.path.join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    elif 'GOT' in args.dataset:
        video_path = os.path.join(tracker_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
        time_path = os.path.join(video_path, '{}_time.txt'.format(video['name']))
    else:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

    # if os.path.exists(result_path):
    #     return

    regions = []
    track_times = []
    image_files, gt = video['image_files'], video['gt']
    for f, image_file in enumerate(image_files):

        im = cv2.imread(image_file)
        if len(im.shape) == 2:
            # Align with training
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        tic = cv2.getTickCount()

        # Init procedure
        if f == start_frame:
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            # Init tracker
            state = tracker.init(im, target_pos, target_sz, net)
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(1 if 'VOT' in args.dataset else gt[f])

        # Tracking procedure
        elif f > start_frame:
            state = tracker.track(state, im)
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            b_overlap = poly_iou(gt[f], location) if 'VOT' in args.dataset else 1
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append(2)
                start_frame = f + 5
        else:
            regions.append(0)

        toc += cv2.getTickCount() - tic
        if 'GOT' in args.dataset:
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())

    with open(result_path, "w") as fin:
        if 'VOT' in args.dataset:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')
        else:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')

    if 'GOT' in args.dataset:
        with open(time_path, 'w') as file_handle:
            for x in track_times:
                file_handle.write("{:.6f}\n".format(x))

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps'.format(video['name'], toc, f / toc))


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    # Prepare model
    net = models.__dict__[args.arch]()
    net = load_pretrain(net, args.resume)
    net.eval()
    net = net.cuda()

    # Prepare video
    dataset = load_dataset(args.dataset)
    video_keys = list(dataset.keys()).copy()

    # Prepare tracker
    info = edict()
    info.arch = args.arch
    info.dataset = args.dataset
    info.epoch_test = args.epoch_test
    info.version = args.version

    if info.arch == 'USOT':
        # 有在线更新
        tracker = USOTTracker(info)
        # # 没有在线更新
        # tracker = USOTORITracker(info)
    else:
        assert False, "Warning: Model should be USOT, but currently {}.".format(info.arch)

    # Tracking all videos in benchmark
    for video in video_keys:
        track(tracker, net, dataset[video], args)

def test(tracker, name, dataset, test_video='', save_path='results', visual=False, test_name=''):
    total_lost = 0
    dataset_name = dataset.name
    base_name = dataset.base_name

    """
    假设从VOT2018中挑选出25个受参数影响较大的序列，组成新的测试集VOT2018-hard
    VOT2018为base dataset，VOT2018-hard为derive dataset
    save manner为 base 时，结果保存在results/VOT2018中（但是只有挑选出的25个序列的结果）；
    save manner为 derive 时，结果保存在results/VOT2018-hard中；
    save manner为 all 时，结果同时保存在results/VOT2018和results/VOT2018-hard中

    应用场景：
    1.挑选困难、参数敏感序列进行调参，选择 derive save方式
    2.LaSOT等速度很慢的超大型数据集，可将其分为若干个子集，分配到各个显卡上并行测试
      选择base的save方式，子数据集的结果会保存在同一文件夹中，最后组成完整的结果
    """
    if test_name != '':
        dataset_name_ = test_name + '-' + dataset_name
        base_name_ = test_name + '-' + base_name
    else:
        dataset_name_ = test_name
        base_name_ = test_name

    if dataset.save == 'base':
        save_name = [base_name_]
    elif dataset.save == 'derive':
        save_name = [dataset_name_]
    elif dataset.save == 'all' and dataset_name_ != base_name_:
        save_name = [dataset_name_, base_name_]

    # if dataset.name in ['VOT2016', 'VOT2018', 'VOT2019']:
    if 'VOT20' in base_name and 'VOT2018-LT' != base_name and 'VOT2020' != base_name:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if test_video != '':
                # test one special video
                if video.name != test_video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    if len(pred_bboxes) > 0 and pred_bboxes[-1] == 0:  #
                        cx, cy, w, h = get_axis_aligned_bbox(gt_bbox)
                        target_pos = np.array([cx, cy])
                        target_sz = np.array([w, h])
                        # Init tracker
                        state = tracker.init(img, target_pos, target_sz)
                        gt_bbox_ = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

                    else:
                        cx, cy, w, h = get_axis_aligned_bbox(gt_bbox)
                        target_pos = np.array([cx, cy])
                        target_sz = np.array([w, h])
                        if 'VOT2017-TIR' != base_name or 'SiamDCA' in name:
                            cx, cy, w, h = get_axis_aligned_bbox(gt_bbox)
                        gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                        state = tracker.init(img, target_pos, target_sz)

                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    state = tracker.track(state, img)
                    pred_bbox = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5  # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic

                if idx == 0:
                    cv2.destroyAllWindows()
                if visual and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    k = cv2.waitKey(15) & 0xff
                    if k == 27:
                        cv2.destroyWindow(video.name)
                        break

            toc /= cv2.getTickFrequency()
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                v_idx + 1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number

            # if visual and dataset.name != 'VOT2018-LT' and k != 27:
            #     tracker_traj = np.array(pred_bboxes[1:])
            #     gt_traj = np.array(video.gt_traj)
            #     n_frame = len(gt_traj)
            #     a_o = success_overlap(gt_traj[1:, :], tracker_traj, n_frame)
            #     thresholds = np.arange(0, 51, 1)
            #     gt_center = convert_bb_to_center(gt_traj)
            #     tracker_center = convert_bb_to_center(tracker_traj)
            #     a_p = success_error(gt_center[1:, :], tracker_center, thresholds, n_frame)
            #     print("precision: %.4f, AUC: %.4f" % (a_p[20], np.mean(a_o)))

            if save_path:

                for name_ in save_name:
                    # save results
                    video_path = os.path.join(save_path, name_, name, 'baseline', video.name)
                    if not os.path.isdir(video_path):
                        os.makedirs(video_path)
                    result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            if isinstance(x, int):
                                f.write("{:d}\n".format(x))
                            else:
                                f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')

        print("{:s} total lost: {:d}".format(name, total_lost))

    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if test_video != '':
                # test one special video
                if video.name != test_video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(gt_bbox)
                    target_pos = np.array([cx, cy])
                    target_sz = np.array([w, h])
                    # Init tracker
                    state = tracker.init(img, target_pos, target_sz)
                    gt_bbox_ = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                    pred_bbox = gt_bbox_
                    # scores.append(None)
                    if 'VOT2018-LT' == base_name:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    state = tracker.track(state, img)
                    pred_bbox = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                    pred_bboxes.append(pred_bbox)
                    # scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if visual and idx > 0:
                    if len(gt_bbox) == 4 and not np.isnan(gt_bbox[0]) and gt_bbox[2] != 0. and gt_bbox[3] != 0.:
                        gt_bbox = list(map(int, gt_bbox))
                        cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                      (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
                    else:
                        cv2.putText(img, 'NAN', (-40, -40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    k = cv2.waitKey(15) & 0xff
                    if k == 27:
                        cv2.destroyWindow(video.name)
                        break

            toc /= cv2.getTickFrequency()
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video.name, toc, idx / toc))

            if visual and base_name != 'VOT2018-LT' and k != 27:
                tracker_traj = np.array(pred_bboxes)
                gt_traj = np.array(video.gt_traj)
                n_frame = len(gt_traj)
                a_o = success_overlap(gt_traj, tracker_traj, n_frame)
                thresholds = np.arange(0, 51, 1)
                gt_center = convert_bb_to_center(gt_traj)
                tracker_center = convert_bb_to_center(tracker_traj)
                a_p = success_error(gt_center, tracker_center, thresholds, n_frame)
                print("precision: %.4f, AUC: %.4f" % (a_p[20], np.mean(a_o)))

            if save_path:
                # save results
                for name_ in save_name:
                    if 'VOT2018-LT' == base_name:
                        video_path = os.path.join(save_path, name_, name, 'longterm', video.name)
                        if not os.path.isdir(video_path):
                            os.makedirs(video_path)
                        result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                        with open(result_path, 'w') as f:
                            for x in pred_bboxes:
                                f.write(','.join([str(i) for i in x]) + '\n')
                        result_path = os.path.join(video_path, '{}_001_confidence.value'.format(video.name))
                        with open(result_path, 'w') as f:
                            for x in scores:
                                f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                        result_path = os.path.join(video_path, '{}_time.txt'.format(video.name))
                        with open(result_path, 'w') as f:
                            for x in track_times:
                                f.write("{:.6f}\n".format(x))
                    elif 'GOT-10k' == base_name:
                        video_path = os.path.join(save_path, name_, name, video.name)
                        if not os.path.isdir(video_path):
                            os.makedirs(video_path)
                        result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                        with open(result_path, 'w') as f:
                            for x in pred_bboxes:
                                f.write(','.join([str(i) for i in x]) + '\n')
                        result_path = os.path.join(video_path, '{}_time.txt'.format(video.name))
                        with open(result_path, 'w') as f:
                            for x in track_times:
                                f.write("{:.6f}\n".format(x))
                    else:
                        model_path = os.path.join(save_path, name_, name)
                        if not os.path.isdir(model_path):
                            os.makedirs(model_path)
                        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                        with open(result_path, 'w') as f:
                            for x in pred_bboxes:
                                f.write(','.join([str(i) for i in x]) + '\n')
def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T


def convert_bb_to_norm_center(bboxes, gt_wh):
    return convert_bb_to_center(bboxes) / (gt_wh+1e-16)
if __name__ == '__main__':
    main()
