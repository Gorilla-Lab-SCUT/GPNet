#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-03-10 21:04:24
# @Author  : Chaozheng Wu (eewuchaozheng@mail.scut.edu.cn)
# @Version : $Id$

import os
import numpy as np
import argparse
import time
import shutil
# import pybullet

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tools.nms import nms2

def readShapePoses(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    num = len(lines)
    shape_poses = {}
    for l in range(0, num, 11):
        shape = lines[l].strip()
        grasps = lines[l+1: l+11]
        grasps = np.array([g.strip().split(',') for g in grasps]).astype(float).reshape(-1, 8)
        shape_poses[shape] = grasps
        # print(shape, grasps.shape, grasps.dtype)
    return shape_poses


parser = argparse.ArgumentParser()
parser.add_argument('--GPU', dest='GPUid', help='set GPU id', default=0, type=str)
parser.add_argument('-dp', '--data_path', type=str, default=None, help='data root dir')
parser.add_argument('--posi_only', dest='posi_only', default=False, help='whether save all prediction', action='store_true')
parser.add_argument('--anti_score', dest='anti_score', default=False, help='whether to use antipodal score', action='store_true')


opt = parser.parse_args()


def main():

    pred_data_path = opt.data_path

    outputdir = os.path.dirname(pred_data_path)
    if opt.posi_only:
        if opt.anti_score:
            LOG_FOUT = open(os.path.join(outputdir, 'test_top10_poses_%s_anti_score_posi_only.txt'%(os.path.basename(pred_data_path))), 'w')
            log_all = open(os.path.join(outputdir, 'test_all_poses_%s_anti_score_posi_only.txt'%(os.path.basename(pred_data_path))), 'w')
        else:
            LOG_FOUT = open(os.path.join(outputdir, 'test_top10_poses_%s_posi_only.txt'%(os.path.basename(pred_data_path))), 'w')
            log_all = open(os.path.join(outputdir, 'test_all_poses_%s_posi_only.txt'%(os.path.basename(pred_data_path))), 'w')
    else:
        if opt.anti_score:
            LOG_FOUT = open(os.path.join(outputdir, 'test_top10_poses_%s_anti_score.txt'%(os.path.basename(pred_data_path))), 'w')
            log_all = open(os.path.join(outputdir, 'test_all_poses_%s_anti_score.txt'%(os.path.basename(pred_data_path))), 'w')
        else:
            LOG_FOUT = open(os.path.join(outputdir, 'test_top10_poses_%s.txt'%(os.path.basename(pred_data_path))), 'w')
            log_all = open(os.path.join(outputdir, 'test_all_poses_%s.txt'%(os.path.basename(pred_data_path))), 'w')


    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        print(out_str)

    keep_num = []
    pp = []
    rr = []
    n = 21
    th = 0.03
    q_th = np.pi/4
    files = os.listdir(opt.data_path)
    for i, f in enumerate(files):
        shape = f.split('.')[0]
        # if shape == '64d97464f86b591caf17412e945e52f4':
        #     continue
        print(shape, i)
        
        # seen_num += 1
        if not os.path.exists(os.path.join(pred_data_path, shape+'.npz')):
            continue
        f = np.load(os.path.join(pred_data_path, shape+'.npz'))
        pred_cent = f['centers']
        pred_quat = f['quaternions']
        # widths = f['widths']
        scores = f['scores']
        if opt.anti_score:
            anti_scores = f['anti_scores']
            print('antipodal scores max-min :', anti_scores.max(), anti_scores.min())
        print('grasp scores max-min :', scores.max(), scores.min())
        if opt.anti_score:
            idx = anti_scores > 0.5
            if idx.sum() == 0:
                idx = scores > 0
            scores = scores[idx]
            pred_cent = pred_cent[idx]
            pred_quat = pred_quat[idx]
        if opt.posi_only:
            idx = scores > 0.5
            if idx.sum() == 0:
                idx = scores > 0
            scores = scores[idx]
            pred_cent = pred_cent[idx]
            pred_quat = pred_quat[idx]

        keep = nms2(pred_cent, pred_quat, scores, cent_th=0.04, ang_th=np.pi/3)
        keep = np.array(keep, dtype=np.int32)

        pred_cent = pred_cent[keep]
        pred_quat = pred_quat[keep]
        
        log_all.write(shape + '\n')
        for i in range(pred_cent.shape[0]):
            c = pred_cent[i]
            q = pred_quat[i]
            log_all.write('%f,%f,%f,%f,%f,%f,%f\n'%(c[0], c[1], c[2], q[0], q[1], q[2], q[3]))
        log_all.flush()

        keep_num = keep.shape[0]
        print('keep_num: ', keep_num)
        if keep_num < 10:
            idx = np.concatenate([np.arange(keep_num), np.random.choice(keep_num, 10 - keep_num, replace=True)], 0)
            pred_cent = pred_cent[idx]
            pred_quat = pred_quat[idx]
        log_string(shape)
        for i in range(10):
            c = pred_cent[i]
            q = pred_quat[i]
            log_string('%f,%f,%f,%f,%f,%f,%f'%(c[0], c[1], c[2], q[0], q[1], q[2], q[3]))





def check_pred_pose(pred_poses, gt_poses, cent_th=0.035, ang_th=5*np.pi/18):
    k = pred_poses.size(0)
    topk_correct = []
    topk_ang_correct = []
    topk_cent_correct = []
    gt_cent = gt_poses[:, 1:4]
    gt_quat = gt_poses[:, 4:]
    cover = []
    for i in range(k):
        p = pred_poses[i]
        cent = p[1:4]
        quat = p[4:]
        # print(p.size())
        cent_correct = (torch.sum((gt_cent - cent)**2, 1) <= cent_th**2).float()
        quat_diff = torch.abs(gt_quat.matmul(quat))
        # print(quat_diff.max().item(), quat_diff.min().item())
        quat_correct = (2 * torch.acos(quat_diff) <= ang_th).float()
        all_correct = cent_correct * quat_correct
        correct_idx = torch.nonzero(quat_correct).view(-1)
        cover.append(correct_idx)
        if all_correct.sum()>0:
            topk_correct.append(True)
        else:
            topk_correct.append(False)
        if quat_correct.sum()>0:
            topk_ang_correct.append(True)
        else:
            topk_ang_correct.append(False)
        if cent_correct.sum()>0:
            topk_cent_correct.append(True)
        else:
            topk_cent_correct.append(False)
    cover = torch.unique(torch.cat(cover, 0))
    return topk_correct, topk_ang_correct, topk_cent_correct, cover


def dist_matrix_torch(x, y):
    x2 = torch.sum(x**2, -1, keepdim=True)
    y2 = torch.sum(y**2, -1, keepdim=True)
    xy = torch.matmul(x, y.transpose(-1, -2))
    matrix = x2 - 2*xy + y2.transpose(-1, -2)
    matrix[matrix<=0] = 1e-10
    return torch.sqrt(matrix)


def check_pred_pose_batch(pred_poses, gt_poses, cent_th=0.035, ang_th=5*np.pi/18, GPU=False):
    if GPU:
        pred_poses = pred_poses.cuda(0)
        gt_poses = gt_poses.cuda(0)
    gt_cent = gt_poses[:, 1:4]
    gt_quat = gt_poses[:, 4:]

    pred_cent = pred_poses[:, :3]
    print(pred_cent.size(), gt_cent.size())
    pred_quat = pred_poses[:, 3:]

    pred_num = pred_cent.size(0)
    if pred_num > 50000:
        pred_correct = []
        pred_cent_correct = []
        pred_quat_correct = []
        num = pred_num // 20000 + 1
        delta = pred_num // num
        for i in range(num):
            s = delta * i
            e = delta * (i+1)
            if i+1 == num:
                e = max(e, pred_num)
            cent_dist_ = dist_matrix_torch(pred_cent[s:e], gt_cent)
            quat_diff_ = torch.abs(torch.matmul(pred_quat[s:e], gt_quat.t()))
            cent_correct = (cent_dist_ <= cent_th).float()
            quat_correct = (2 * torch.acos(quat_diff_) <= ang_th).float()
            all_correct = cent_correct * quat_correct 
            pred_correct_ = (all_correct.sum(-1) > 0).cpu().float().numpy()
            pred_cent_correct_ = (cent_correct.sum(-1) > 0).cpu().float().numpy()
            pred_quat_correct_ = (quat_correct.sum(-1) > 0).cpu().float().numpy()
            
            pred_correct.append(pred_correct_)
            pred_cent_correct.append(pred_cent_correct_)
            pred_quat_correct.append(pred_quat_correct_)
        pred_correct = np.concatenate(pred_correct, 0)
        pred_cent_correct = np.concatenate(pred_cent_correct, 0)
        pred_quat_correct = np.concatenate(pred_quat_correct, 0)
        assert pred_correct.shape[0] == pred_num

    else:
        cent_dist = dist_matrix_torch(pred_cent, gt_cent)
        quat_diff = torch.abs(torch.matmul(pred_quat, gt_quat.t()))
        cent_correct = (cent_dist <= cent_th).float()
        quat_correct = (2 * torch.acos(quat_diff) <= ang_th).float()
        all_correct = cent_correct * quat_correct
        pred_correct = (all_correct.sum(-1) > 0).cpu().float().numpy()
        pred_cent_correct = (cent_correct.sum(-1) > 0).cpu().float().numpy()
        pred_quat_correct = (quat_correct.sum(-1) > 0).cpu().float().numpy()

    return pred_correct, pred_cent_correct, pred_quat_correct



def EulerPose2QuaternionPose(poses):
    poses[:,4] = poses[:, 4] * 2*np.pi - np.pi
    poses[:,5] = poses[:, 5] * np.pi - np.pi/2
    poses[:,6] = poses[:, 6] * 2*np.pi - np.pi
    q_poses = []
    for p in poses:
        Q = pybullet.getQuaternionFromEuler([p[4], p[5], p[6]])
        Q = np.array([Q[3], Q[0], Q[1], Q[2]])
        pose = np.zeros(8)
        pose[:4] = p[:4]
        pose[4:] = Q
        q_poses.append(pose)
    q_poses = torch.FloatTensor(np.array(q_poses)).view(1, -1, 8)
    return q_poses


if __name__ == '__main__':
    main()









