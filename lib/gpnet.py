import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from tools.proposal import *
from etw_pytorch_utils import pytorch_utils as pt_utils
from pointnet2.utils.pointnet2_modules import (
    PointnetSAModule, PointnetFPModule, PointnetSAModuleMSG
)


class PointNetFeat(nn.Module):
    """docstring for PointNetFeat"""
    def __init__(self, input_channels=0, use_xyz=True, bn=True):
        super(PointNetFeat, self).__init__()
        
        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        # print(c_in)
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=True,
                bn=bn
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=use_xyz,
                bn=bn
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=use_xyz,
                bn=bn
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=16,
                radii=[0.4, 0.8],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=use_xyz,
                bn=bn
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(
            PointnetFPModule(mlp=[256 + input_channels, 128, 128], bn=bn)
        )
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256], bn=bn))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512], bn=bn))
        self.FP_modules.append(
            PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512], bn=bn)
        )

        self.FC_layer = nn.Sequential(
            pt_utils.Conv1d(128, 128, bn=bn)
        )
        self.sigmoid = nn.Sigmoid()

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]



class PosePredictBatch(nn.Module):
    """docstring for PosePredict"""
    def __init__(self, input_dim=128, point_num=50, bn=True, tanh=False):
        super(PosePredictBatch, self).__init__()
        self.input_dim = input_dim
        # self.conv1 = nn.Conv2d(input_dim, 512, 1)
        self.max_pool = nn.MaxPool2d((1, point_num), stride=(1, 1))
        self.mean_pool = nn.AvgPool2d((1, point_num), stride=(1, 1))
        self.weighted_mean = pt_utils.Conv2d(input_dim, input_dim, kernel_size=(1, point_num), padding=0, bn=bn)
        self.conv1 = pt_utils.Conv1d(input_dim, 512, 1, bn=bn)
        self.score = nn.Sequential(pt_utils.Conv1d(512, 256, 1, bn=bn), pt_utils.Conv1d(256, 1, 1, activation=None))
        self.offset = nn.Sequential(pt_utils.Conv1d(512, 256, 1, bn=bn), pt_utils.Conv1d(256, 3, 1, activation=None))
        self.angle = nn.Sequential(pt_utils.Conv1d(512, 256, 1, bn=bn), pt_utils.Conv1d(256, 1, 1, activation=None))
        self.sigmoid = nn.Sigmoid()
        self.use_tanh = tanh
        self.tanh = nn.Tanh()

    def forward(self, gather_feat, xyz=None):
        # max_feat = self.max_pool(gather_feat).squeeze(-1)
        # mean_feat = self.mean_pool(gather_feat).squeeze(-1)
        # feat = torch.cat([max_feat, mean_feat], 1)
        feat = self.weighted_mean(gather_feat).squeeze(-1)
        feat = feat.transpose(0, 2)
        feat = self.conv1(feat)
        score = self.sigmoid(self.score(feat)).transpose(0, 2) #(B, 1, num_proposal)
        offset = self.offset(feat).transpose(0, 2)
        angle = self.angle(feat).transpose(0, 2)
        if self.use_tanh:
            offset = self.tanh(offset)
            angle = self.tanh(angle)
        return score.transpose(2, 1).squeeze(-1), offset.transpose(2, 1), angle.transpose(2, 1).squeeze(-1)


class GraspClassifier(nn.Module):
    """docstring for PosePredict"""
    def __init__(self, input_dim=128, point_num=50, bn=True):
        super(GraspClassifier, self).__init__()
        self.input_dim = input_dim
        # self.conv1 = nn.Conv2d(input_dim, 512, 1)
        self.max_pool = nn.MaxPool2d((1, point_num), stride=(1, 1))
        self.mean_pool = nn.AvgPool2d((1, point_num), stride=(1, 1))
        self.weighted_mean = pt_utils.Conv2d(input_dim, input_dim, kernel_size=(1, point_num), padding=0, bn=bn)
        self.conv1 = pt_utils.Conv1d(input_dim, 512, 1, bn=bn)
        self.score = nn.Sequential(pt_utils.Conv1d(512, 256, 1, bn=bn), pt_utils.Conv1d(256, 1, 1, activation=None))
        self.sigmoid = nn.Sigmoid()

    def forward(self, gather_feat):
        # max_feat = self.max_pool(gather_feat).squeeze(-1)
        # mean_feat = self.mean_pool(gather_feat).squeeze(-1)
        # feat = torch.cat([max_feat, mean_feat], 1)
        feat = self.weighted_mean(gather_feat).squeeze(-1)
        feat = feat.transpose(0, 2)
        feat = self.conv1(feat)
        score = self.sigmoid(self.score(feat)).transpose(0, 2) #(B, 1, num_proposal)
        return score.transpose(2, 1).squeeze(-1)



class AntipodalPredictBatch(nn.Module):
    """docstring for PosePredict"""
    def __init__(self, input_dim=128, point_num=50, bn=True):
        super(AntipodalPredictBatch, self).__init__()
        self.input_dim = input_dim
        # self.conv1 = nn.Conv2d(input_dim, 512, 1)
        # self.max_pool = nn.MaxPool2d((1, point_num), stride=(1, 1))
        # self.mean_pool = nn.AvgPool2d((1, point_num), stride=(1, 1))
        self.weighted_mean = pt_utils.Conv2d(input_dim, input_dim, kernel_size=(1, point_num), padding=0, bn=bn)
        self.conv1 = pt_utils.Conv1d(input_dim, 512, 1, bn=bn)
        self.score = nn.Sequential(pt_utils.Conv1d(512, 256, 1, bn=bn), pt_utils.Conv1d(256, 1, 1, activation=None))
        self.sigmoid = nn.Sigmoid()

    def forward(self, gather_feat, xyz=None):
        # max_feat = self.max_pool(gather_feat).squeeze(-1)
        # mean_feat = self.mean_pool(gather_feat).squeeze(-1)
        # feat = torch.cat([max_feat, mean_feat], 1)
        feat = self.weighted_mean(gather_feat).squeeze(-1)
        feat = feat.transpose(0, 2)
        feat = self.conv1(feat)
        score = self.sigmoid(self.score(feat)).transpose(0, 2) #(B, 1, num_proposal)

        return score.transpose(2, 1).squeeze(-1)


class GridFeat(nn.Module):
    """docstring for gridFeat"""
    def __init__(self, in_dim=3, out_dim=128, bn=True):
        super(GridFeat, self).__init__()
        self.out_dim = out_dim
        self.layers = nn.Sequential(pt_utils.Conv1d(in_dim, 64, 1, bn=bn), pt_utils.Conv1d(64, out_dim, 1, bn=bn))

    def forward(self, grid):
        grid_feat = self.layers(grid)
        return grid_feat



class GraspPoseNet(nn.Module):
    def __init__(self, local_point_num=100, training=True, tanh=False, grid=False, 
        bn=False, posi_rate=0.5, use_angle=True):
        super(GraspPoseNet, self).__init__()
        self.local_point_num = local_point_num
        self.grid = grid
        self.bn = bn
        self.posi_rate = posi_rate
        self.use_angle = use_angle

        self.point_feat = PointNetFeat(bn=bn)
        dim = 128
        self.dim = dim

        self.ap = AntipodalPredictBatch(dim, local_point_num, bn=bn)
        self.grasp_cls = GraspClassifier(dim, local_point_num, bn=bn)
        self.posePred = PosePredictBatch(dim, local_point_num, tanh=tanh, bn=bn)
        if grid:
            self.gridFeat = GridFeat(out_dim=dim, bn=bn)
        if self.use_angle:
            self.gridAngleFeat = GridFeat(in_dim=4, out_dim=dim, bn=bn)
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                if not bn:
                    for p in m.parameters(): p.requires_grad = False


    def forward(self, pc, prop_local_points, prop_pair, posi_prop_idx=None, nega_prop_idx=None, posi_idx=None, \
        nega_idx=None, grasp_centers=None, grasp_angles=None, grasp_local_points=None, scale=0.022*np.sqrt(3)):
        '''
        pc: (B, N, 3)
        '''
        st = time.time()
        bs = pc.size(0)
        pc_feat = self.point_feat(pc)  #(B, C, N)
        ap_x = pc_feat
        C = ap_x.size(1)
        ft = time.time() - st

        if self.training:
            local_points = prop_local_points
            posi_prop_idx, nega_prop_idx, posi_idx, nega_idx = \
                selectProposal(posi_prop_idx, nega_prop_idx, posi_idx, nega_idx, ratio2=self.posi_rate)
            local_points = local_points[:, torch.cat([posi_prop_idx, nega_prop_idx], 0)]
            posi_prop_num = posi_idx.size(0) + nega_idx.size(0)
            prop_pair = prop_pair[:, torch.cat([posi_prop_idx, nega_prop_idx], 0)]

            if self.grid:
                prop_grid = prop_pair[:, :, 1].transpose(1, 2)
                grid_feat = self.gridFeat(prop_grid)
            if self.use_angle:
                cent_ang = torch.cat([grasp_centers, grasp_angles], -1).contiguous().transpose(1, 2)
                grid_angle_feat = self.gridAngleFeat(cent_ang)
            local_points = local_points.contiguous()
            num_proposal = local_points.size(1)

            local_point_num = local_points.size(-1)

            feat_list = []
            for i in range(num_proposal):
                points = local_points[:, i].contiguous()
                gather_feat = torch.gather(ap_x, 2, points.view(bs, 1, -1).expand(-1, C, -1)) #(bs, C, num_local_point)
                if self.grid:
                    grid_feat_ = grid_feat[:, :, i].view(bs, -1, 1)
                    gather_feat = grid_feat_.expand(-1, C, local_point_num) + gather_feat
                feat_list.append(gather_feat)
            gather_feat_batch = torch.stack(feat_list, 2).contiguous()
            # print(gather_feat_batch.is_contiguous())
            local_pc_batch = None
            prop_score = self.ap(gather_feat_batch, local_pc_batch)
            _, pred_offset_batch, pred_angle_batch = self.posePred(gather_feat_batch[:, :, :posi_prop_num], local_pc_batch)
            
            num_proposal = grasp_local_points.size(1)
            grasp_feat_list = []
            for i in range(num_proposal):
                points = grasp_local_points[:, i].contiguous()
                # print(ap_x.size(), points.view(bs, 1, -1).expand(-1, C, -1).size())
                gather_feat = torch.gather(ap_x, 2, points.view(bs, 1, -1).expand(-1, C, -1)) #(bs, C, num_local_point)
                # gather_feat_batch[:,:,i] = gather_feat
                grid_feat_ = grid_angle_feat[:, :, i].view(bs, -1, 1)
                gather_feat = grid_feat_.expand(-1, C, local_point_num) + gather_feat
                grasp_feat_list.append(gather_feat)
            grasp_feat_batch = torch.stack(grasp_feat_list, 2).contiguous()
            grasp_score = self.grasp_cls(grasp_feat_batch)

            return prop_score, grasp_score, pred_offset_batch, pred_angle_batch, posi_prop_idx, nega_prop_idx, posi_idx, nega_idx

        else:
            contacts = prop_pair[:, :, 0]
            prop_grid = prop_pair[:, :, 1]
            if self.grid:
                grid_feat = self.gridFeat(prop_grid.transpose(1, 2))
            local_points = prop_local_points.contiguous()
            num_proposal = local_points.size(1)

            local_point_num = local_points.size(-1)

            feat_list = []
            local_pc_list = []
            grasp_feat_list = []
            prop_posi_idx_list = []
            prop_score_list = []
            local_pc_batch = None
            prop_num = 0
            pred_score_list = []
            pred_offset_list = []
            pred_angle_list = []
            pred_contact_list = []
            _output = sys.stdout
            _second = 0.1
            for i in range(num_proposal):
                points = local_points[:, i].contiguous()
                gather_feat = torch.gather(ap_x, 2, points.view(bs, 1, -1).expand(-1, C, -1))

                if self.grid:
                    grid_feat_ = grid_feat[:, :, i].view(bs, -1, 1)
                    gather_feat = grid_feat_.expand(-1, C, local_point_num) + gather_feat
                feat_list.append(gather_feat)
                if (i + 1) % 5000 == 0 or (i + 1) == num_proposal:
                    gather_feat_batch = torch.stack(feat_list, 2).contiguous()
                    prop_score = self.ap(gather_feat_batch, local_pc_batch)
                    prop_posi_idx = torch.nonzero(prop_score.view(-1)>0.5).view(-1)
                    prop_score_list.append(prop_score)
                    prop_posi_idx_list.append(prop_posi_idx)
                    feat_list = []
                    prop_num += prop_posi_idx.size(0)
                    time.sleep(_second)
                    _output.write('\rcomplete percent: %d/%d\t proposal positive percent: %d/%d'%(i+1, num_proposal, prop_num, num_proposal))
                    if prop_posi_idx.size(0) == 0:
                        continue
                    pred_score_batch, pred_offset_batch, pred_angle_batch = \
                    self.posePred(gather_feat_batch[:, :, prop_posi_idx], local_pc_batch)
                    pred_score_list.append(pred_score_batch)
                    pred_offset_list.append(pred_offset_batch)
                    pred_angle_list.append(pred_angle_batch)
                    # pred_contact_list.append(contacts[:, anti_posi_idx])

            del (gather_feat, gather_feat_batch, feat_list)
            # anti_posi_idx = torch.cat(anti_posi_idx_list, 0)
            prop_score = torch.cat(prop_score_list, 1)
            prop_posi_idx = torch.nonzero(prop_score.view(-1)>0.5).view(-1)
            assert prop_score.size(1) == contacts.size(1)
            if len(pred_score_list) == 0:
                pred_score = torch.FloatTensor([[0.0]]).cuda(0)
                pred_offset = torch.FloatTensor([[[0.0, 0.0, 0.0]]]).cuda(0)
                pred_angle = torch.FloatTensor([[0.0]]).cuda(0)
            else:
                pred_score_ = torch.cat(pred_score_list, 1)
                assert prop_posi_idx.size(0) == pred_score_.size(1)
                pred_offset = torch.cat(pred_offset_list, 1)
                pred_angle = torch.cat(pred_angle_list, 1)
                pred_contact = contacts[:, prop_posi_idx]
                pred_grid = prop_grid[:, prop_posi_idx]
                assert pred_grid.size(1) == pred_offset.size(1)
                offsets = pred_offset * scale
                centers = pred_grid - offsets
                cent_ang = torch.cat([centers, pred_angle.unsqueeze(-1)], -1).contiguous().transpose(1, 2)
                grid_angle_feat = self.gridAngleFeat(cent_ang)
                # grid_feat = self.gridFeat(centers.transpose(1, 2))
                num_pred = pred_offset.size(1)
                # local_points_list = []
                # interval = 1000
                local_points = getLocalPointsV2(pc, pred_contact, centers)
                # grid_feat_list = []
                grid_angle_feat_list = []
                pred_score_list = []
                # pred_anti_score_list = []
                for i in range(num_pred):
                    points = local_points[:, i].contiguous()
                    gather_feat = torch.gather(ap_x, 2, points.view(bs, 1, -1).expand(-1, C, -1))

                    grid_angle_feat_ = grid_angle_feat[:, :, i].view(bs, -1, 1)
                    # grid_feat_ = grid_feat[:, :, i].view(bs, -1, 1)
                    # gather_grid_feat = grid_feat_.expand(-1, C, local_point_num) + gather_feat
                    # grid_feat_list.append(gather_grid_feat)
                    gather_grid_angle_feat = grid_angle_feat_.expand(-1, C, local_point_num) + gather_feat
                    grid_angle_feat_list.append(gather_grid_angle_feat)
                    if (i + 1) % 5000 == 0 or (i + 1) == num_pred:
                        # gather_grid_feat_batch = torch.stack(grid_feat_list, 2).contiguous()
                        # anti_score = self.ap(gather_grid_feat_batch)
                        # grid_feat_list = []
                        gather_grid_angle_feat_batch = torch.stack(grid_angle_feat_list, 2).contiguous()
                        pred_score_batch = self.grasp_cls(gather_grid_angle_feat_batch)
                        grid_angle_feat_list = []
                        time.sleep(_second)
                        _output.write('\rcomplete percent: %d/%d'%(i+1, num_pred))
                        pred_score_list.append(pred_score_batch)
                        # pred_anti_score_list.append(anti_score)
                # anti_score = torch.cat(pred_anti_score_list, 1)
                pred_score = torch.cat(pred_score_list, 1)
                assert pred_score.size(1) == pred_offset.size(1), str(pred_score.size())

            return prop_score, pred_score, pred_offset, pred_angle, prop_posi_idx


    def scorer(self, emb_feat, pc, contacts, centers, angles):
        bs = pc.size(0)
        C = emb_feat.size(1)
        cent_ang = torch.cat([centers, angles.unsqueeze(-1)], -1).contiguous().transpose(1, 2)
        grid_angle_feat = self.gridAngleFeat(cent_ang)
        # grid_feat = self.gridFeat(centers.transpose(1, 2))
        num_pred = centers.size(1)
        # local_points_list = []
        # interval = 1000
        local_points = getLocalPointsV2(pc, contacts, centers)
        # grid_feat_list = []
        grid_angle_feat_list = []
        pred_score_list = []
        # pred_anti_score_list = []
        for i in range(num_pred):
            points = local_points[:, i].contiguous()
            gather_feat = torch.gather(emb_feat, 2, points.view(bs, 1, -1).expand(-1, C, -1))

            grid_angle_feat_ = grid_angle_feat[:, :, i].view(bs, -1, 1)
            # grid_feat_ = grid_feat[:, :, i].view(bs, -1, 1)
            # gather_grid_feat = grid_feat_.expand(-1, C, local_point_num) + gather_feat
            # grid_feat_list.append(gather_grid_feat)
            gather_grid_angle_feat = grid_angle_feat_.expand(-1, C, self.local_point_num) + gather_feat
            grid_angle_feat_list.append(gather_grid_angle_feat)
            if (i + 1) % 5000 == 0 or (i + 1) == num_pred:
                # gather_grid_feat_batch = torch.stack(grid_feat_list, 2).contiguous()
                # anti_score = self.ap(gather_grid_feat_batch)
                # grid_feat_list = []
                gather_grid_angle_feat_batch = torch.stack(grid_angle_feat_list, 2).contiguous()
                pred_score_batch = self.grasp_cls(gather_grid_angle_feat_batch)
                grid_angle_feat_list = []
                pred_score_list.append(pred_score_batch)
                # pred_anti_score_list.append(anti_score)
        # anti_score = torch.cat(pred_anti_score_list, 1)
        pred_score = torch.cat(pred_score_list, 1)
        assert pred_score.size(1) == centers.size(1), str(pred_score.size())
        return pred_score


    @torch.no_grad()
    def emb_feat(self, pc):
        pc_feat = self.point_feat(pc)
        return pc_feat


    def train(self, mode=True):
        nn.Module.train(self, mode)
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        if not self.bn :
            self.apply(set_bn_eval)
            # self.point_feat.apply(set_bn_eval)
            # self.gridFeat.apply(set_bn_eval)
            # self.ap.apply(set_bn_eval)
            # self.posePred.apply(set_bn_eval)


def get_centers(grids, offsets, scale=0.022*np.sqrt(3)):
    offsets = offsets * scale
    centers = grids - offsets
    return centers

