import _init_paths
import os
import argparse
import time
import shutil

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from dataset.grasp_positive import GraspData
from tools.nms import nms2
from tools.coverage_vs_precision import coverage_vs_precision


parser = argparse.ArgumentParser()
parser.add_argument('--GPU', dest='GPUid',
                    help='set GPU id', default=0, type=str)
parser.add_argument('-pd', '--pred_data', type=str, default=None, help='data root dir')
parser.add_argument('-gd', '--gt_data', type=str, default=None, help='data root dir')
parser.add_argument('--top10', dest='top10', default=False, help='whether use top10', action='store_true')


opt = parser.parse_args()


def main():
    data_path = opt.gt_data
    dataset = GraspData(data_path, split='test')
    dataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    pred_data_path = opt.pred_data
    isdir = os.path.isdir(pred_data_path)
    if not isdir:
        shape_poses = readShapePoses(pred_data_path)

    keep_num = []
    pp = []
    rr = []
    n = 21
    th = 0.025
    q_th = np.pi / 6
    all_coverage = []
    all_precision = []
    all_number = []
    for i, data in enumerate(dataLoader):
        gt_grasps, shape = data
        shape = shape[0]
        # if shape == '5db63af675507081118ddfdb81cc6068':
        #     continue
        print(shape, i)
        gt_grasps = gt_grasps[0].float()

        # seen_num += 1
        if isdir:
            if not os.path.exists(os.path.join(pred_data_path, shape + '.npz')):
                continue
            f = np.load(os.path.join(pred_data_path, shape + '.npz'))
            pred_cent = f['centers']
            pred_quat = f['quaternions']
            # widths = f['widths']
            scores = f['scores']
            # pred_correct = f['pred_label']
            posi_idx = scores > 0.5
            pred_cent = pred_cent[posi_idx]
            pred_quat = pred_quat[posi_idx]
            scores = scores[posi_idx]

            keep = nms2(pred_cent, pred_quat, scores,
                        cent_th=0.04, ang_th=np.pi/3)
            keep = np.array(keep, dtype=np.int32)
            keep_num.append(keep.shape[0])
            print('keep_num', keep.shape)

            pred_cent = pred_cent[keep]
            pred_quat = pred_quat[keep]
        # pred_correct = pred_correct[keep]
        else:
            if shape not in shape_poses:
                continue
            grasp = shape_poses[shape]
            grasp = np.concatenate(grasp, 0)
            pred_cent = grasp[:, :3]
            pred_quat = grasp[:, 3:]

        print(gt_grasps.size(0), pred_cent.shape[0])

        # if gt_grasps.size(0) < 800:
        #     continue
        p = coverage_vs_precision(gt_grasps.numpy())
        coverage = np.zeros(4)
        precision = np.zeros(4)
        number = np.zeros(4)
        for i, per in enumerate([0.1, 0.3, 0.5, 1.0]):
            s, c, k = p.precision_and_recall_at_k_percent(
                pred_cent, pred_quat, per, th, q_th, gpu=True)
            coverage[i] = c
            precision[i] = s
            number[i] = k
        all_coverage.append(coverage)
        all_precision.append(precision)
        all_number.append(number)
    all_coverage = np.array(all_coverage)
    all_precision = np.array(all_precision)
    all_number = np.array(all_number)
    print(np.mean(all_coverage, 0))
    print(np.mean(all_precision, 0))
    print(np.mean(all_number, 0))
    print(len(dataset))


def readShapePoses(fname):
    shape_poses = {}
    f = open(fname, 'r')
    lines = f.readlines()
    num = 0
    for l in lines:
        if not ',' in l:
            shape = l.strip()
            shape_poses[shape] = []
        else:
            grasp = np.array(l.strip().split(',')).astype(float).reshape(1, -1)
            shape_poses[shape].append(grasp)

    return shape_poses



if __name__ == '__main__':
    main()
