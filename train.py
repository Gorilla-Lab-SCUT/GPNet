import _init_paths
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

from dataset.grasp_dataset import GraspData
from tools.logger import Logger
from tools.proposal import *
from lib.gpnet import GraspPoseNet
from loss import angle_loss
# from lib.loss import Loss
# from lib.loss_refiner import Loss_refine
# from lib.utils import setup_logger
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='/data/wuchaozheng/dataset/shapenetSemGrasp/nips2020/new_9cls/', help='dataset root dir')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--optimizer', default='sgd', type=str, help='training optimizer')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum in sgd')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--wd', default=0.0001, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume', type=str, default=None,  help='resume GPNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--logdir', default='log_exp', type=str, metavar='SPATH', help='path to save checkpoint (default: log)')
parser.add_argument('-p', '--print_freq', default=1, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--tanh', dest='tanh', default=False, help='whether use tanh', action='store_true')
parser.add_argument('--grid', dest='grid', default=False, help='whether use grid coordinate', action='store_true')
parser.add_argument('--lamb', default=0.01, type=float, help='lambda for multi-angle loss')
parser.add_argument('--grid_len', default=22, type=float, help='grid space length (cm)')
parser.add_argument('--grid_num', type=int, default=10, help='number of grids')
parser.add_argument('--ratio', default=1.0, type=float, help='grasp ratio use for training')
parser.add_argument('--posi_ratio', default=0.3, type=float, help='positive grasp ratio use for training')
opt = parser.parse_args()


def main():
    # opt.manualSeed = random.randint(1, 10000)
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.tanh:
        opt.logdir = opt.logdir + '_tanh'
    if opt.grid:
        opt.logdir = opt.logdir + '_grid'

    outputdir = os.path.join(opt.logdir, 'gridlen%r_gridnum%d'%(opt.grid_len, opt.grid_num), 
        'bs%d_wd%s_lr%r_lamb%r_ratio%r_posi%r_%s'%(opt.batch_size, opt.wd, opt.lr, opt.lamb, opt.ratio, \
            opt.posi_ratio, opt.optimizer))
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    LOG_FOUT = open(os.path.join(outputdir, 'log.txt'), 'w')
    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        print(out_str)

    log_train = Logger(outputdir)
    tb_log = Logger(outputdir)

    log_string(str(opt)+'\n')

    grid_len = opt.grid_len / 100
    grid_num = opt.grid_num
    dataset = GraspData(opt.dataset_root, sample_ratio=opt.ratio, posi_ratio=opt.posi_ratio, 
                        grid_len=grid_len, grid_num=grid_num)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, 
                        num_workers=opt.workers, pin_memory=True)

    net = GraspPoseNet(tanh=opt.tanh, grid=opt.grid, bn=False).cuda()

    lr = opt.lr
    params = []
    for item in net.named_parameters():
        key, value = item[0], item[1]
        if value.requires_grad:
            params += [{'params': [value], 'lr': lr, 'weight_decay': opt.wd}]

    if opt.optimizer == "adam":
        optimizer = torch.optim.Adam(params)
    elif opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=opt.momentum)

    if opt.resume is not None:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']+1
            best_loss = checkpoint['best_loss']
            model_dict = net.state_dict()
            pretrained_dict = checkpoint['state_dict']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            # lr = checkpoint['lr']
            print("\n=> loaded checkpoint '{}' (epoch {})" .format(opt.resume, checkpoint['epoch']))
            del checkpoint
        else:
            assert False, 'WRONG RESUME PATH!'
    else:
        start_epoch = opt.start_epoch
        best_loss = 100000.0

    lr = opt.lr

    # criterion = Loss(opt.num_points_mesh, opt.sym_list)
    score_criterion = nn.BCELoss().cuda()
    reg_criterion = nn.MSELoss(reduce=False).cuda()

    best_test = np.Inf

    st_time = time.time()

    for epoch in range(start_epoch, opt.nepoch+1):
        lr = adjust_learning_rate(optimizer, epoch, lr)
        net.train()
        loss_sum = 0
        prop_loss_sum = 0
        score_loss_sum = 0
        ang_loss_sum = 0
        off_loss_sum = 0
        anti_acc_sum = 0
        anti_recall_sum = 0
        grasp_acc_sum = 0
        grasp_recall_sum = 0
        prop_acc_sum = 0
        prop_recall_sum = 0

        loss_epoch = 0
        prop_loss_epoch = 0
        score_loss_epoch = 0
        ang_loss_epoch = 0
        off_loss_epoch = 0
        anti_acc_epoch = 0
        anti_recall_epoch = 0
        grasp_acc_epoch = 0
        grasp_recall_epoch = 0
        prop_acc_epoch = 0
        prop_recall_epoch = 0

        for i, data in enumerate(dataLoader):
            pc_, grids_, contact_, center_, contact_index_, scores_, grasps_idx_, angles_, posi_mask_, \
            angles_scorer_, posi_nega_idx_ = data
            print(grids_.size(), contact_.size(), pc_.size(), angles_.size())
            if contact_index_.size(1) == 1 or pc_.size(1) > 20000 or pc_.size(1) < 10:
                continue

            st = time.time()
            # Due to the limit of GPU memory, we need two GPU. One for model training, another for grasp proposal.
            pc1, grids1, contact_index1, center1, scores1 = pc_.float().cuda(1), grids_.float().cuda(1), \
                    contact_index_.long().cuda(1), center_.float().cuda(1), scores_.float().cuda(1)

            pc, grids, angles, contact_index, center, scores, grasps_idx, posi_mask = \
                            pc_.float().cuda(0), grids_.float().cuda(0), angles_.float().cuda(0), \
                            contact_index_.long().cuda(0), center_.float().cuda(0), \
                            scores_.float().cuda(0), grasps_idx_.long().cuda(0), posi_mask_.float().cuda(0)

            data_index = torch.arange(contact_index_.size(1)).long().cuda()

            radius = grid_len / grid_num * np.sqrt(3)
            pairs_all_, scores_all_, offsets_all_, local_points_, data_index_, prop_label_, posi_prop_idx_, \
            nega_prop_idx_, posi_idx_, nega_idx_ = getProposals(pc1, grids1, center1, contact_index1, \
                                                                scores1, data_index, radius=radius)

            del (grids1, center1, contact_index1, scores1)
            pairs_all, scores_all, offsets_all, local_points, data_index, prop_label, posi_prop_idx, nega_prop_idx, \
            posi_idx, nega_idx = pairs_all_.cuda(0), scores_all_.cuda(0), offsets_all_.cuda(0), local_points_.cuda(0), \
                                data_index_.cuda(0), prop_label_.cuda(0), posi_prop_idx_.cuda(0), nega_prop_idx_.cuda(0), \
                                posi_idx_.cuda(0), nega_idx_.cuda(0)
            
            print('proposal time: ', time.time()-st, 'posi-nega num: ', posi_idx.size(0), nega_idx.size(0))
            if scores_all.max() == 0 or scores_all.min() > 0 or posi_idx.size(0) == 0 or nega_idx.size(0) == 0 or \
                nega_prop_idx.size(0) == 0 or posi_prop_idx.size(0) == 0:
                continue

            grasp_center_ = center_[:, posi_nega_idx_[0]].float()
            grasp_contact_ = contact_[:, posi_nega_idx_[0]].float()
            grasp_angle_ = angles_scorer_[:, posi_nega_idx_[0]].float()
            grasp_center1 = grasp_center_.cuda(1)
            grasp_contact1 = grasp_contact_.cuda(1)
            grasp_local_points_ = getLocalPoints(pc1, grasp_contact1, grasp_center1)
            grasp_local_points = grasp_local_points_.cuda(0).long()
            grasp_center, grasp_angle = grasp_center_.cuda(0), grasp_angle_.cuda(0).unsqueeze(-1)
            grasp_label = scores_[:, posi_nega_idx_[0]].float().cuda(0)
            
            del (pc_, grids_, contact_, center_, contact_index_, scores_, grasps_idx_, angles_, local_points_, \
                offsets_all_, scores_all_, pairs_all_, data_index_, posi_prop_idx_, nega_prop_idx_, posi_idx_, \
                nega_idx_, grasp_local_points_, grasp_center1, grasp_contact1)

            prop_score, pred_score, pred_offset, pred_angle, posi_prop_idx, nega_prop_idx, posi_idx, nega_idx\
             = net(pc, local_points, pairs_all, posi_prop_idx, nega_prop_idx, posi_idx, nega_idx, grasp_center,
                    grasp_angle, grasp_local_points)
            
            prop_label = prop_label[:, torch.cat([posi_prop_idx.view(-1), nega_prop_idx.view(-1)], 0)]
            select_idx = torch.cat([posi_idx.view(-1), nega_idx.view(-1)], 0)
            select_data = data_index[:, select_idx].view(-1)
            gt_score = scores_all[:, select_idx]
            gt_offset = offsets_all[:, select_idx]
            gt_angle = angles[:, select_data].squeeze(0)
            grasps_idx = grasps_idx[:, select_data].squeeze(0)
            posi_mask = posi_mask[:, select_data].squeeze(0)

            prop_acc, prop_recall = cal_accuracy(prop_label, prop_score, recall=True)
            gt_label = (gt_score>0).float()
            grasp_acc, grasp_recall = cal_accuracy(grasp_label, pred_score, recall=True)

            prop_loss = score_criterion(prop_score, prop_label)
            print('proposal score: ', prop_score.max().item(), prop_score.min().item(), prop_label.max().item(), prop_label.min().item())
            print('grasp score: ', pred_score.max().item(), pred_score.min().item(), gt_score.max().item(), gt_score.min().item())
            score_loss = score_criterion(pred_score, grasp_label)
            print('angle: %6f  %6f'%(pred_angle.min().item(), pred_angle.max().item()), 
                'offsets: %6f  %6f'%(pred_offset.min().item(), pred_offset.max().item()))

            posi_gt = torch.nonzero(gt_score.view(-1)).view(-1)
            posi_score = gt_score[0, posi_gt]
            ang_loss = angle_loss(pred_angle[0][posi_gt].unsqueeze(-1), gt_angle[posi_gt].unsqueeze(-1), posi_mask[posi_gt].unsqueeze(-1))
            ang_loss = torch.sum(posi_score*ang_loss)/posi_score.sum()
            off_loss = torch.sum(gt_score*reg_criterion(pred_offset, gt_offset).sum(-1))/gt_score.sum()

            all_loss = prop_loss + score_loss + opt.lamb * ang_loss + off_loss

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            # print(time.time() - st)

            loss_sum += all_loss.item()
            prop_loss_sum += prop_loss.item()
            score_loss_sum += score_loss.item()
            ang_loss_sum += ang_loss.item()
            off_loss_sum += off_loss.item()
            prop_acc_sum += prop_acc
            prop_recall_sum += prop_recall
            grasp_acc_sum += grasp_acc
            grasp_recall_sum += grasp_recall

            loss_epoch += all_loss.item()
            prop_loss_epoch += prop_loss.item()
            score_loss_epoch += score_loss.item()
            ang_loss_epoch += ang_loss.item()
            off_loss_epoch += off_loss.item()
            prop_acc_epoch += prop_acc
            prop_recall_epoch += prop_recall
            grasp_acc_epoch += grasp_acc
            grasp_recall_epoch += grasp_recall

            del (all_loss, prop_loss, score_loss, ang_loss, off_loss)

            if i % opt.print_freq == 0:
                loss_sum /= opt.print_freq
                prop_loss_sum /= opt.print_freq
                score_loss_sum /= opt.print_freq
                ang_loss_sum /= opt.print_freq
                off_loss_sum /= opt.print_freq
                prop_acc_sum /= opt.print_freq
                grasp_acc_sum /= opt.print_freq
                prop_recall_sum /= opt.print_freq
                grasp_recall_sum /= opt.print_freq
                log_string('Epoch: [{0}][{1}/{2}]\t'
                        'all_loss: {Loss:.4f}  '
                        'prop_loss: {prop_loss:.4f}  '
                        'score_loss: {score_loss:.4f}  '
                        'ang_loss: {ang_loss:.4f}  '
                        'off_loss: {off_loss:.4f}\t'
                        'prop_acc: {prop_acc:.4f}  '
                        'prop_recall: {prop_recall:.4f}  '
                        'grasp_acc: {grasp_acc:.4f}  '
                        'grasp_recall: {grasp_recall:.4f}\t'
                        'lr: {lr:.5f}\t'.format(
                        epoch, i, len(dataLoader), Loss=loss_sum, prop_loss=prop_loss_sum, score_loss=score_loss_sum, \
                        ang_loss=ang_loss_sum, off_loss=off_loss_sum, prop_acc=prop_acc_sum, grasp_acc=grasp_acc_sum, \
                        prop_recall=prop_recall_sum, grasp_recall=grasp_recall_sum, lr=lr))
                loss_sum = 0
                prop_loss_sum = 0
                score_loss_sum = 0
                ang_loss_sum = 0
                off_loss_sum = 0
                prop_acc_sum = 0
                grasp_acc_sum = 0
                prop_recall_sum = 0
                grasp_recall_sum = 0
        loss_epoch /= len(dataLoader)
        prop_loss_epoch /= len(dataLoader)
        score_loss_epoch /= len(dataLoader)
        ang_loss_epoch /= len(dataLoader)
        off_loss_epoch /= len(dataLoader)
        prop_acc_epoch /= len(dataLoader)
        grasp_acc_epoch /= len(dataLoader)
        prop_recall_epoch /= len(dataLoader)
        grasp_recall_epoch /= len(dataLoader)
        tb_log.scalar_summary('train_loss/all_loss', loss_epoch, epoch)
        tb_log.scalar_summary('train_loss/prop_loss', prop_loss_epoch, epoch)
        tb_log.scalar_summary('train_loss/score_loss', score_loss_epoch, epoch)
        tb_log.scalar_summary('train_loss/angle_loss', ang_loss_epoch, epoch)
        tb_log.scalar_summary('train_loss/offset_loss', off_loss_epoch, epoch)

        tb_log.scalar_summary('train_acc/prop_accuracy', prop_acc_epoch, epoch)
        tb_log.scalar_summary('train_acc/grasp_accuracy', grasp_acc_epoch, epoch)
        tb_log.scalar_summary('train_acc/prop_recall', prop_recall_epoch, epoch)
        tb_log.scalar_summary('train_acc/grasp_recall', grasp_recall_epoch, epoch)

        best_loss = 1000.0
        is_best = False
        checkpoint_dict = {'epoch': epoch, 
                        'state_dict': net.state_dict(), 
                        'best_loss': best_loss, 
                        'lr': lr,
                        'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint_dict, is_best, outputdir, epoch)



def cal_accuracy(gt_label, pred_score, th=0.5, recall=False, posi_num=None):
    pred_label = (pred_score > th).float()
    correct = (gt_label == pred_label).float().view(-1)
    acc = correct.sum() / correct.size(0)
    if not recall:
        return acc.item()
    else:
        posi_correct = correct * gt_label
        if posi_num is None:
            recall = posi_correct.sum() / gt_label.sum()
        else:
            recall = posi_correct.sum() / (posi_num + 1e-8)
        return acc.item(), recall.item()


def save_checkpoint(state, is_best, dir, epoch, filename='checkpoint_%s.pth.tar'):
    torch.save(state, os.path.join(dir, filename%(epoch)))
    if is_best:
        shutil.copyfile(os.path.join(dir, filename%(epoch)), os.path.join(dir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr, lamb=0.1, step=200):
    """Sets the learning rate to the initial LR decayed by 10 every 200 epochs"""
    lr = lr * (lamb ** (epoch // step))
    lr = max(0.00001, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()
