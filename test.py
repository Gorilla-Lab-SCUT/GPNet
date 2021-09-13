import _init_paths
import argparse
import os
import random
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable

from dataset.grasp_dataset import GraspData
from tools.logger import Logger
from tools.proposal import *
from tools.transformTool import getOrientation2
from tools.transformTool import matrix2quaternion2
from lib.gpnet import GraspPoseNet
from loss import angle_loss
from tools.nms import nms, nms2
from get_contact_cos import zMove


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='/data/wuchaozheng/dataset/shapenetSemGrasp/nips2020/new_9cls/', help='dataset root dir')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--optimizer', default='sgd', type=str, help='training optimizer')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum in sgd')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--wd', default=0.0001, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--nepoch', type=int, default=50, help='max number of epochs to train')
parser.add_argument('--resume', type=str, default=None,  help='resume PoseNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--tanh', dest='tanh', default=False, help='whether use tanh', action='store_true')
parser.add_argument('--grid', dest='grid', default=False, help='whether use grid coordinate', action='store_true')
parser.add_argument('--lamb', default=1.0, type=float, help='lambda for multi-angle loss')
parser.add_argument('--grid_len', default=22, type=float, help='grid space length (cm)')
parser.add_argument('--grid_num', type=int, default=10, help='number of grids')
parser.add_argument('--save_all', dest='save_all', default=False, help='whether save all prediction', action='store_true')
parser.add_argument('--view', type=int, default=0, help='view id for testing.')
parser.add_argument('--grid_th', type=float, default=0.075, help='proposal grid number id for testing.')
parser.add_argument('--epoch', type=str, default=None,  help='resume epochs')
opt = parser.parse_args()


def main():
    # opt.manualSeed = random.randint(1, 10000)
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    print(opt)

    outputdir = os.path.join(os.path.dirname(opt.resume), 'test')

    grid_len = opt.grid_len / 100
    grid_num = opt.grid_num
    dataset = GraspData(opt.dataset_root, split='test', view=opt.view, grid_len=grid_len, grid_num=grid_num)

    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.workers, 
                        pin_memory=True)

    net = GraspPoseNet(tanh=opt.tanh, grid=opt.grid, training=False, bn=False, use_angle=True).cuda()

    epochs = [int(s) for s in opt.epoch.split(',')]

    for e in epochs:
        resume_path = os.path.join(opt.resume, 'checkpoint_%d.pth.tar'%e)

        if os.path.isfile(resume_path):
                print("=> loading checkpoint '{}'".format(resume_path))
                checkpoint = torch.load(resume_path)
                pretrained_dict = checkpoint['state_dict']
                net.load_state_dict(pretrained_dict)
                # lr = checkpoint['lr']
                epoch = checkpoint['epoch']
                print("\n=> loaded checkpoint '{}' (epoch {})" .format(resume_path, checkpoint['epoch']))
                del checkpoint
        else:
            assert False, 'WRONG RESUME PATH!'

        if 'epoch' in outputdir.split('/')[-1]:
            outputdir = os.path.dirname(outputdir)
        all_grasps_dir = os.path.join(outputdir, 'epoch%d'%epoch)

        if opt.save_all:
            all_grasps_dir = all_grasps_dir + '_all'

        outputdir = all_grasps_dir

        all_grasps_dir = os.path.join(all_grasps_dir, 'view%d'%(opt.view))

        if not os.path.exists(all_grasps_dir):
            os.makedirs(all_grasps_dir)

        log = open(os.path.join(outputdir, 'log.txt'), 'a')
        log.write('view%d\n'%(opt.view))

        log_all = open(os.path.join(outputdir, 'nms_poses_view%s.txt'%(opt.view)), 'w')

        lr = opt.lr

        score_criterion = nn.BCELoss().cuda()
        reg_criterion = nn.MSELoss(reduce=False).cuda()

        best_grasp_acc = -1.0
        best_grasp_epoch = -1

        st_time = time.time()

        net.eval()
        time_list = []
        
        with torch.no_grad():

            for i, data in enumerate(dataLoader):
                st = time.time()
                pc_, grids_, contact_index_, shape = data
                shape = shape[0]
                # print(shape, i)
                print('\n<================================>', e, shape, i, '<================================>\n')

                print(grids_.size(), contact_index_.size(), pc_.size())

                pc1, grids1, contact_index1 = pc_.float().cuda(1), grids_.float().cuda(1), contact_index_.long().cuda(1)

                pc, grids, contact_index = pc_.float().cuda(0), grids_.float().cuda(0), contact_index_.long().cuda(0)

                data_index = torch.arange(contact_index_.size(1)).long().cuda()

                radius = grid_len / grid_num * np.sqrt(3)
                # pairs_all_, local_points_ = getTestProposals(pc1, grids1, contact_index1)
                pairs_all_, local_points_ = getTestProposalsV3(pc1, grids1, contact_index1, grid_th=opt.grid_th)
                print(pairs_all_.size())

                del (pc1, grids1, contact_index1)

                pairs_all, local_points = pairs_all_.cuda(0), local_points_.cuda(0)

                del (pc_, grids_, contact_index_, local_points_, pairs_all_)

                prop_score, pred_score, pred_offset, pred_angle, prop_posi_idx \
                = net(pc, local_points, pairs_all, scale=radius)

                print('\nforward time: ', time.time() - st)

                print('pred_score: ', pred_score.max().item(), pred_score.min().item())
                if opt.save_all:
                    posi_idx = torch.nonzero(pred_score.view(-1)>=0.).view(-1)
                else:
                    posi_idx = torch.nonzero(pred_score.view(-1)>=0.5).view(-1)
                if posi_idx.size(0) == 0:
                    continue
                posi_scores = pred_score.view(-1)[posi_idx]
                prop_score = prop_score.view(-1)[posi_idx]
                prop_posi_idx = prop_posi_idx[posi_idx]
                pred_offset = pred_offset[0, posi_idx]
                pred_angle = pred_angle[0, posi_idx]
                pred_pairs = pairs_all[0, prop_posi_idx]

                centers, widths, quaternions = get7dofPoses(pred_pairs, pred_offset, pred_angle, scale=radius)
                time_list.append(time.time() - st)
                z = centers[:, 2]
                select = torch.nonzero((widths < 0.085)*(z>0)).view(-1)
                if select.size(0) == 0:
                    continue
                centers, widths, quaternions = centers[select], widths[select], quaternions[select]
                select_pred_pairs = pred_pairs[select]
                posi_scores = posi_scores[select]
                prop_score = prop_score[select]
                pred_angle = pred_angle[select].view(-1)

                posi_contacts = select_pred_pairs[:, 0]
                posi_contacts_cpu = posi_contacts.cpu().numpy()
                posi_scores_cpu = posi_scores.cpu().numpy()
                centers = centers.cpu().numpy()
                widths = widths.cpu().numpy()
                quaternions = quaternions.cpu().numpy()
                prop_score_cpu = prop_score.cpu().numpy()
                pred_angle_cpu = pred_angle.cpu().numpy()
                assert pred_angle_cpu.shape[0] == centers.shape[0]
                assert posi_contacts_cpu.shape[0] == centers.shape[0]
                print('posi grasp num:', posi_scores.size(0))
                centers = zMove(quaternions, centers, zMoveLength=0.015)

                all_grasps_path = os.path.join(all_grasps_dir, shape+'.npz')
                np.savez(all_grasps_path, widths=widths, centers=centers, quaternions=quaternions, 
                    scores=posi_scores_cpu, contacts=posi_contacts_cpu, angles=pred_angle_cpu)

                st = time.time()
                keep = nms2(centers, quaternions, posi_scores_cpu, cent_th=0.04, ang_th=np.pi/3)
                keep = np.array(keep, dtype=np.int32)
                print('nms time: ', time.time() - st)
                centers = centers[keep]
                widths = widths[keep]
                quaternions = quaternions[keep]
                
                log_all.write(shape + '\n')
                for i in range(centers.shape[0]):
                    w = widths[i]
                    c = centers[i]
                    q = quaternions[i]
                    log_all.write('%f,%f,%f,%f,%f,%f,%f\n'%(c[0], c[1], c[2], q[0], q[1], q[2], q[3]))
                log_all.flush()

        print(max(time_list), min(time_list), np.mean(time_list))
        log.write('min time%f\n'%(min(time_list)))
        log.write('max time%f\n'%(max(time_list)))
        log.write('mean time%f\n'%(np.mean(time_list)))
        log.write(str(time_list)+'\n')


def get7dofPoses(pairs, offsets, angles, scale=0.022*np.sqrt(3)):
    offsets = offsets * scale
    contacts = pairs[:, 0]
    grids = pairs[:, 1]
    centers = grids - offsets
    widths = 2 * torch.sqrt(((contacts - centers)**2).sum(1))
    matrix = getOrientation2(contacts, centers, angles).contiguous().view(-1, 9)
    quaternions = matrix2quaternion2(matrix)
    return centers, widths, quaternions


if __name__ == '__main__':
    main()
