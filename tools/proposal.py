import time

import cv2
import math
import numpy as np
import torch
import torch.nn as nn
from lib.pointnet2.utils import pointnet2_utils as pu
from tools.Quaternion import *


def get_dist_matrix(x, y):
    x2 = np.sum(x**2, 1, keepdims=True)
    y2 = np.sum(y**2, 1, keepdims=True)
    xy = x.dot(y.transpose())
    matrix = x2 - 2*xy + y2.transpose()
    matrix[matrix<=0] = 1e-10
    # print(matrix.min())
    return np.sqrt(matrix)


def dist_matrix_torch(x, y):
    x2 = torch.sum(x**2, -1, keepdim=True)
    y2 = torch.sum(y**2, -1, keepdim=True)
    xy = torch.matmul(x, y.transpose(-1, -2))
    matrix = x2 - 2*xy + y2.transpose(-1, -2)
    matrix += 1e-10
    return torch.sqrt(matrix)


def dist_matrix_torch_v2(x, y):
    y2 = torch.sum(y**2, -1, keepdim=True)
    x_num = x.size(0)
    # print(x_num)
    delta = 1e4
    num = int((x_num + delta - 1) // delta)
    matrix_list = []
    for i in range(num):
        s = int(delta * i)
        e = int(delta * (i+1))
        e = min(e, x_num)
        # print(s, e)
        x2 = torch.sum(x[s:e]**2, -1, keepdim=True)
        xy = torch.matmul(x[s:e], y.transpose(-1, -2))
        matrix = x2 - 2*xy + y2.transpose(-1, -2) + 1e-10
        matrix_list.append(matrix)
    matrix = torch.cat(matrix_list, 0)
    assert matrix.size(0) == x_num
    return torch.sqrt(matrix)


def score_decay(score, dist, sigma=0.002):
    score_new = score*np.exp(-(dist)**2/sigma)
    return score_new

def score_decay_pytorch(score, dist, sigma=0.002):
    score_new = score * torch.exp(-(dist)**2/sigma)
    return score_new


def getContactPoint(gripperLength, gripperCenter, gripperOrientation):
    rotation = Quaternion(
        w=gripperOrientation[0],
        x=gripperOrientation[1],
        y=gripperOrientation[2],
        z=gripperOrientation[3]
    )
    contact1 = [- gripperLength / 2, 0, 0]
    contact2 = [gripperLength / 2, 0, 0]
    contact1 = rotation.applyRotation(contact1)
    contact2 = rotation.applyRotation(contact2)
    contact1 = np.array([
        gripperCenter[0] + contact1[0],
        gripperCenter[1] + contact1[1],
        gripperCenter[2] + contact1[2]
    ])
    contact2 = np.array([
        gripperCenter[0] + contact2[0],
        gripperCenter[1] + contact2[1],
        gripperCenter[2] + contact2[2]
    ])
    return contact1, contact2


def getCosAngle(gripperOrientation):
    rotation = Quaternion()
    rotation.listInit(gripperOrientation)
    gripperX = np.array(rotation.applyRotation([1, 0, 0]))
    gripperX = getUniqueGripperX(gripperX)
    gripperZ = np.array(rotation.applyRotation([0, 0, 1]))
    gripperZ = gripperZ / getNorm(gripperZ)
    tangentY = - np.cross(gripperX, (0, 0, 1))
    tangentY = tangentY / getNorm(tangentY)
    cosAngle = np.dot(gripperZ, tangentY)
    if cosAngle > 1.0:
        cosAngle = 1.00
    if cosAngle < -1.0:
        cosAngle = -1.0
    return cosAngle    # return cos of elevation


def getUniqueGripperX(gripperX):
    if gripperX[2] < 0:
        gripperX = -gripperX
    gripperX = gripperX / getNorm(gripperX)
    if abs(gripperX[0] * gripperX[1]) < 1e-6:
        gripperX[0] += 0.001
        gripperX = gripperX / getNorm(gripperX)
    return gripperX


def getContactCenter(grasps, obj_pc, positive=None, contact_th=0.0035):
    index_dict = {}
    count = 0
    for j, g in enumerate(grasps):
        w, cent, ori = g[-3], g[-2], g[-1]
        c1, c2 = getContactPoint(w, cent, ori)
        ang = getCosAngle(ori.tolist())
        dist1 = np.sqrt(np.sum((obj_pc-c1)**2, 1))
        dist2 = np.sqrt(np.sum((obj_pc-c2)**2, 1))

        idx1 = np.argmin(dist1)
        d1_min = dist1[idx1]
        idx2 = np.argmin(dist2)
        d2_min = dist2[idx2]

        if d1_min <= contact_th or d2_min <= contact_th:
            count += 1
            if d1_min < d2_min:
                idx = idx1
                c = c1
            else:
                idx = idx2
                c = c2
            if idx not in index_dict:
                s = [g[0]]
                angles = [ang]
                center = [cent]
                con = [c]
                g_idx = [j]
                index_dict[idx] = (s, angles, center, con, g_idx)
            elif (index_dict[idx][2][0] - cent).sum() == 0 and len(index_dict[idx][0]) < 9:
                index_dict[idx][0].append(g[0])
                index_dict[idx][1].append(ang)
                index_dict[idx][2].append(cent)
                index_dict[idx][3].append(c)
                index_dict[idx][4].append(j)
    # return index_dict
    index = []
    scores = []
    angles = []
    center = []
    grasp_index = []
    posi_mask = []
    contact = []
    for k, v in index_dict.items():
        index.append(k)
        s, ang, cent, con, gi = v[0], v[1], v[2], v[3], v[4]
        cent = np.array(cent)
        mean_center = cent.mean(0)
        dist = ((mean_center - cent[0])**2).sum()
        assert dist < 0.0001
        s = np.array(s)
        ang = np.array(ang)
        gi = np.array(gi)
        posi_idx = np.where(s>0)[0]
        ang_exp = np.zeros(9)
        gi_exp = np.zeros(9).astype(np.int32)
        mask = np.zeros(9)
        num = posi_idx.shape[0]
        assert num <= 9, str(num)
        # if num > 9:
        #     print(k, v)
        center.append(cent[0])
        contact.append(con[0])
        if num > 0:
            scores.append(1.0)
            mask[:num] = 1
            ang_exp[:num] = ang[posi_idx]
            gi_exp[:num] = gi[posi_idx]
        else:
            scores.append(0.0)
        posi_mask.append(mask)
        grasp_index.append(gi_exp)
        angles.append(ang_exp)
    contact = np.array(contact)
    center = np.array(center)
    contact_index = np.array(index)
    scores = np.array(scores)
    grasps_select = np.array(grasp_index)
    angles = np.array(angles)
    posi_mask = np.array(posi_mask)
    return contact, center, contact_index, scores, grasps_select, angles, posi_mask


def getContactCenterV2(grasps, obj_pc, contacts, angle, contact_th=0.0035):
    index_dict = {}
    count = 0
    for j, g in enumerate(grasps):
        w, cent, ori = g[2], g[3:6], g[6:]
        # c1, c2 = getContactPoint(w, cent, ori)
        # ang = getCosAngle(ori.tolist())
        cont = contacts[j]
        c1, c2 = cont[0], cont[1]
        ang = angle[j]
        dist1 = np.sqrt(np.sum((obj_pc-c1)**2, 1))
        dist2 = np.sqrt(np.sum((obj_pc-c2)**2, 1))

        idx1 = np.argmin(dist1)
        d1_min = dist1[idx1]
        idx2 = np.argmin(dist2)
        d2_min = dist2[idx2]

        if d1_min <= contact_th or d2_min <= contact_th:
            count += 1
            if d1_min < d2_min:
                idx = idx1
                c = c1
            else:
                idx = idx2
                c = c2
            if idx not in index_dict:
                s = [g[0]]
                angles = [ang]
                center = [cent]
                con = [c]
                g_idx = [j]
                index_dict[idx] = (s, angles, center, con, g_idx)
            elif (index_dict[idx][2][0] - cent).sum() == 0 and len(index_dict[idx][0]) < 9:
                index_dict[idx][0].append(g[0])
                index_dict[idx][1].append(ang)
                index_dict[idx][2].append(cent)
                index_dict[idx][3].append(c)
                index_dict[idx][4].append(j)
    # return index_dict
    index = []
    scores = []
    angles_exp = []
    center = []
    grasp_index = []
    posi_mask = []
    contact = []
    angles = []
    for k, v in index_dict.items():
        index.append(k)
        s, ang, cent, con, gi = v[0], v[1], v[2], v[3], v[4]
        cent = np.array(cent)
        mean_center = cent.mean(0)
        dist = ((mean_center - cent[0])**2).sum()
        assert dist < 0.0001
        s = np.array(s)
        ang = np.array(ang)
        gi = np.array(gi)
        posi_idx = np.where(s>0)[0]
        ang_exp = np.zeros(9)
        gi_exp = np.zeros(9).astype(np.int32)
        mask = np.zeros(9)
        num = posi_idx.shape[0]
        assert num <= 9, str(num)
        # if num > 9:
        #     print(k, v)
        center.append(cent[0])
        contact.append(con[0])
        angles.append(ang[0])
        if num > 0:
            scores.append(1.0)
            mask[:num] = 1
            ang_exp[:num] = ang[posi_idx]
            gi_exp[:num] = gi[posi_idx]
        else:
            scores.append(0.0)
        posi_mask.append(mask)
        grasp_index.append(gi_exp)
        angles_exp.append(ang_exp)
    contact = np.array(contact)
    center = np.array(center)
    contact_index = np.array(index)
    scores = np.array(scores)
    grasps_select = np.array(grasp_index)
    angles_exp = np.array(angles_exp)
    posi_mask = np.array(posi_mask)
    angles = np.array(angles)
    return contact, center, contact_index, scores, grasps_select, angles_exp, posi_mask, angles


def getContactCenterCosAngle(grasps, obj_pc, contact_th=0.0035):
    index_dict = {}
    count = 0
    contact_idx_list = []
    angle_list = []
    score_list = []
    contact_list = []
    center_list = []
    grasp_idx_list = []
    for j, g in enumerate(grasps):
        w, cent, ori = g[2], g[3:6], g[6:]
        c1, c2 = getContactPoint(w, cent, ori)
        ang = getCosAngle(ori.tolist())
        dist1 = np.sqrt(np.sum((obj_pc-c1)**2, 1))
        dist2 = np.sqrt(np.sum((obj_pc-c2)**2, 1))

        idx1 = np.argmin(dist1)
        d1_min = dist1[idx1]
        idx2 = np.argmin(dist2)
        d2_min = dist2[idx2]

        if d1_min <= contact_th or d2_min <= contact_th:
            count += 1
            if d1_min < d2_min:
                idx = idx1
                c = c1
            else:
                idx = idx2
                c = c2
            contact_idx_list.append(idx)
            contact_list.append(c)
            angle_list.append(ang)
            score_list.append(g[0])
            center_list.append(cent)
            grasp_idx_list.append(j)

    contact = np.array(contact_list)
    center = np.array(center_list)
    contact_index = np.array(contact_idx_list)
    scores = np.array(score_list)
    grasps_select = np.array(grasp_idx_list)
    angles = np.array(angle_list)
    return contact, center, contact_index, scores, grasps_select, angles


def getProposals(obj_pc, grids, center, index, scores, data_index, radius=0.022*np.sqrt(3), \
    local_th=0.011, local_pn=100):
    center = center.squeeze(0)
    index = index.squeeze(0)
    scores = scores.squeeze(0)
    grids = grids.squeeze(0)
    obj_pc = obj_pc.squeeze(0) * torch.FloatTensor([0.22/2, 0.22/2, 0.22]).to(obj_pc.device)
    contact = obj_pc[index]
    
    cent_grid_dist_matrix = dist_matrix_torch(center, grids)
    point_dist = dist_matrix_torch(contact, obj_pc)

    con_num = index.size(0)
    grid_num = grids.size(0)
    pn_num = obj_pc.size(0)

    # get contact-grid pairs
    contact_exp = contact.view(-1, 1, 3).expand(-1, grid_num, -1)
    grids_exp = grids.view(1, -1, 3).expand(con_num, -1, -1)
    pairs_ = torch.stack([contact_exp, grids_exp], 2).view(-1, 2, 3).unsqueeze(0)
    pairs = pairs_.cpu()
    del pairs_

    # get positive and negative proposals
    select = (cent_grid_dist_matrix < radius).float()
    posi_prop_idx_ = torch.nonzero(select.view(-1)).view(-1)
    nega_prop_idx_ = torch.nonzero(select.view(-1) == 0).view(-1)
    posi_prop_idx, nega_prop_idx = posi_prop_idx_.cpu(), nega_prop_idx_.cpu()

    offsets_ = (grids_exp - center.view(-1, 1, 3)).view(1, -1, 3) / radius
    offsets = (offsets_ * select.view(1, -1, 1)).cpu()
    del offsets_
    # get proposals scores
    scores_all_ = scores.view(-1, 1) * select
    scores_all_ = select * scores_all_
    scores_all = scores_all_.view(1, -1).cpu()
    posi_prop_scores = scores_all_.view(-1)[posi_prop_idx_]
    posi_idx_ = torch.nonzero(posi_prop_scores).view(-1)  # positive proposals associated to positive grasps
    nega_idx_ = torch.nonzero(posi_prop_scores == 0).view(-1)  # positive proposals associated to negative grasps
    posi_idx, nega_idx = posi_idx_.cpu(), nega_idx_.cpu()
    anti_label = select.view(1, -1).cpu() # proposals labels
    del (scores_all_, posi_prop_idx_, nega_prop_idx_, posi_idx_, nega_idx_)

    # get local points 
    pg_vec = contact_exp - grids_exp # vectors from grids to contacts
    pg_vec = pg_vec / torch.sqrt(torch.sum(pg_vec**2, -1, keepdim=True))
    obj_pc_exp = obj_pc.view(1, -1, 3) #.expand(con_num, -1, -1)
    pp_vec = obj_pc_exp - contact.view(-1, 1, 3) # vectors from contacts to other points
    point_dist_view = point_dist.view(con_num, -1, 1)
    pp_vec = pp_vec / point_dist_view
    del (obj_pc_exp, contact_exp, grids_exp, cent_grid_dist_matrix, point_dist)
    
    data_num = grid_num * con_num * pn_num
    num = data_num // 5e8 + 1
    num = int(num)
    delta = (con_num + num - 1) // num
    local_points_list = []
    # in case of out of memory
    for i in range(num):
        s = delta * i
        e = delta * (i+1)
        if i+1 == num:
            e = max(e, con_num)
        dist_ = point_dist_view[s:e].transpose(1,2) * (1.0 + torch.abs(pg_vec[s:e].matmul(pp_vec[s:e].transpose(1,2))))
        dist = dist_.to('cuda:0')
        local_points = pu.matrix_k_min(local_th, local_pn, dist).long()
        local_points_list.append(local_points)
        del (dist_, dist)
    local_points = torch.cat(local_points_list, 0).view(1, -1, local_pn).long()

    assert local_points.size(1) == con_num * grid_num, local_points.size(1)
    data_index = data_index.new(con_num, grid_num).zero_() + data_index.view(-1, 1)
    data_index = data_index.view(1, -1)
    del (select, pg_vec, pp_vec)

    return pairs, scores_all, offsets, local_points, data_index, anti_label, posi_prop_idx, nega_prop_idx, posi_idx, nega_idx


def getTestProposals(obj_pc, grids, contact_index, radius=0.022*np.sqrt(3), \
    local_th=0.011, local_pn=100, return_time=False):
    st = time.time()
    contact_index = contact_index.squeeze(0)
    grids = grids.squeeze(0)
    obj_pc = obj_pc.squeeze(0) * torch.FloatTensor([0.22/2, 0.22/2, 0.22]).to(obj_pc.device)
    # print('pc max-min:', obj_pc.max(0), obj_pc.min(0))
    contact = obj_pc[contact_index]
    # dist_matrix = dist_matrix_torch(center, grids)
    point_dist = dist_matrix_torch(contact, obj_pc)

    con_num = contact_index.size(0)
    grid_num = grids.size(0)
    pn_num = obj_pc.size(0)
    contact_exp = contact.view(-1, 1, 3).expand(-1, grid_num, -1)
    grids_exp = grids.view(1, -1, 3).expand(con_num, -1, -1)
    pairs_ = torch.stack([contact_exp, grids_exp], 2).view(-1, 2, 3).unsqueeze(0)
    pairs = pairs_.cpu()
    del pairs_

    pg_vec = contact_exp - grids_exp
    pg_vec = pg_vec / torch.sqrt(torch.sum(pg_vec**2, -1, keepdim=True))
    obj_pc_exp = obj_pc.view(1, -1, 3)#.expand(con_num, -1, -1)
    pp_vec = obj_pc_exp - contact.view(-1, 1, 3)
    point_dist_view = point_dist.view(con_num, -1, 1)
    pp_vec = pp_vec / point_dist_view
    del (obj_pc_exp, contact_exp, grids_exp, point_dist)
    
    data_num = grid_num * con_num * pn_num
    num = data_num // 7e8 + 1
    num = int(num)
    # print(data_num, num)
    delta = (con_num + num - 1) // num
    local_points_list = []
    t1 = time.time() - st
    st = time.time()
    for i in range(num):
        s = delta * i
        e = delta * (i+1)
        if i+1 == num:
            e = max(e, con_num)
            # print(data_num, num, e)
        dist_ = point_dist_view[s:e].transpose(1,2) * (1.0 + torch.abs(pg_vec[s:e].matmul(pp_vec[s:e].transpose(1,2))))
        dist = dist_.to('cuda:0')
        local_points = pu.matrix_k_min(local_th, local_pn, dist).long()
        local_points_list.append(local_points.cpu())
        del (dist_, dist)
    t2 = time.time() - st
    t = t1 + t2 / num
    local_points = torch.cat(local_points_list, 0).view(1, -1, local_pn).long()
    assert local_points.size(1) == con_num * grid_num, local_points.size(1)
    del (pg_vec, pp_vec)
    if not return_time:
        return pairs, local_points
    else:
        return pairs, local_points, t


def getTestProposalsV2(obj_pc, grids, contact_index, radius=0.022*np.sqrt(3), \
    local_th=0.011, local_pn=100, grid_th=0.0425+0.022*np.sqrt(3), grid_num=150):
    contact_index = contact_index.squeeze(0)
    grids = grids.squeeze(0)
    obj_pc = obj_pc.squeeze(0) * torch.FloatTensor([0.22/2, 0.22/2, 0.22]).to(obj_pc.device)
    contact = obj_pc[contact_index]
    point_dist = dist_matrix_torch(contact, obj_pc)
    contact_grid_dist_ = dist_matrix_torch(contact, grids)
    contact_grid_dist = contact_grid_dist_.to('cuda:0')
    del (contact_grid_dist_)
    st = time.time()
    grids_idx_ = pu.matrix_k_min(grid_th, grid_num, contact_grid_dist.unsqueeze(0))
    print('grids select time: ', time.time() - st)
    grids_idx = grids_idx_[0].to(obj_pc.device)
    grids_idx = grids_idx.unsqueeze(-1).expand(-1, -1, 3).long()
    del(contact_grid_dist, grids_idx_)

    con_num = contact_index.size(0)
    # grid_num = grids.size(0)
    pn_num = obj_pc.size(0)
    contact_exp = contact.view(-1, 1, 3).expand(-1, grid_num, -1)
    grids_exp = grids.view(1, -1, 3).expand(con_num, -1, -1)
    grids_exp = torch.gather(grids_exp, 1, grids_idx)
    pairs_ = torch.stack([contact_exp, grids_exp], 2).view(-1, 2, 3).unsqueeze(0)
    pairs = pairs_.cpu()
    del pairs_

    pg_vec = contact_exp - grids_exp
    pg_vec = pg_vec / torch.sqrt(torch.sum(pg_vec**2, -1, keepdim=True))
    obj_pc_exp = obj_pc.view(1, -1, 3) #.expand(con_num, -1, -1)
    pp_vec = obj_pc_exp - contact.view(-1, 1, 3)
    point_dist_view = point_dist.view(con_num, -1, 1)
    pp_vec = pp_vec / point_dist_view
    del (obj_pc_exp, contact_exp, grids_exp, point_dist)
    
    data_num = grid_num * con_num * pn_num

    num = data_num // 6e8 + 1
    # num = int(num)
    num = int(num)
    # print(data_num, num)
    delta = (con_num + num - 1) // num
    local_points_list = []
    for i in range(num):
        s = delta * i
        e = delta * (i+1)
        if i+1 == num:
            e = max(e, con_num)
            # print(data_num, num, e)
        dist_ = point_dist_view[s:e].transpose(1,2) * (1.0 + torch.abs(pg_vec[s:e].matmul(pp_vec[s:e].transpose(1,2))))
        dist = dist_.to('cuda:0')
        local_points = pu.matrix_k_min(local_th, local_pn, dist).long()
        local_points_list.append(local_points)
        del (dist_, dist)
    local_points = torch.cat(local_points_list, 0).view(1, -1, local_pn).long()
    assert local_points.size(1) == con_num * grid_num, local_points.size(1)
    del (pg_vec, pp_vec)

    return pairs, local_points


def getTestProposalsV3(obj_pc, grids, contact_index, local_th=0.011, local_pn=100, 
    grid_th=0.0425+0.022*np.sqrt(3)):
    contact_index = contact_index.squeeze(0)
    grids = grids.squeeze(0)
    grid_num = grids.size(0)
    obj_pc = obj_pc.squeeze(0) * torch.FloatTensor([0.22/2, 0.22/2, 0.22]).to(obj_pc.device)
    contact = obj_pc[contact_index]
    point_dist = dist_matrix_torch(contact, obj_pc)
    contact_grid_dist = dist_matrix_torch(contact, grids)
    # contact_grid_dist = contact_grid_dist_.to('cuda:0')
    # del (contact_grid_dist_)
    con_num = contact_index.size(0)
    pn_num = obj_pc.size(0)
    
    local_points_list = []
    pairs_list = []
    for i in range(grid_num):
        g = grids[i]
        c_idx = torch.nonzero(contact_grid_dist[:, i]<grid_th).view(-1)
        if len(c_idx) == 0:
            continue
        cons = contact[c_idx] # (c num, 3)
        point_dist_ = point_dist[c_idx].unsqueeze(-1) # (c num, pc num, 1)
        pp_vec = obj_pc.view(1, -1, 3) - cons.view(-1, 1, 3) # (c num, pc num, 3)
        pp_vec = pp_vec / point_dist_ 
        pg_vec = cons - g # (c num, 3)
        pg_vec = pg_vec / torch.sqrt(torch.sum(pg_vec**2, -1, keepdim=True))
        dist_ = point_dist_.transpose(1, 2) * (1.0+torch.abs(pg_vec.unsqueeze(1).matmul(pp_vec.transpose(1,2))))
        dist = dist_.to('cuda:0')
        local_points = pu.matrix_k_min(local_th, local_pn, dist).long()
        local_points_list.append(local_points)
        del (dist_, dist)
        pair = torch.stack([cons, g.view(1, -1).expand(cons.size(0), -1)], 1)
        pairs_list.append(pair)
    pairs_ = torch.cat(pairs_list, 0)
    pairs = pairs_.cpu().unsqueeze(0)
    # print('pairs size', pairs.size())
    local_points = torch.cat(local_points_list, 0).view(1, -1, local_pn).long()
    del (pg_vec, pp_vec, pairs_, point_dist_, point_dist)
    assert pairs.size(1) == local_points.size(1)
    return pairs, local_points


def selectProposal(posi_prop_idx, nega_prop_idx, posi_idx, nega_idx, max_prop=4000, 
                ratio1=0.5, max_grasp=2000, ratio2=0.5):
    posi_prop_idx = posi_prop_idx.view(-1)
    nega_prop_idx = nega_prop_idx.view(-1)
    posi_idx = posi_idx.view(-1)
    nega_idx = nega_idx.view(-1)

    posi_num = posi_idx.size(0)
    nega_num = nega_idx.size(0)
    posi_prop_num = posi_prop_idx.size(0)
    nega_prop_num = nega_prop_idx.size(0)

    posi_num_exp = int(max_grasp * ratio2)
    nega_num_exp = max_grasp - posi_num_exp
    posi_prop_num_exp = int(max_prop * ratio1)
    nega_prop_num_exp = max_prop - posi_prop_num_exp

    if posi_num < posi_num_exp:
        choice = torch.cat([torch.arange(0, posi_num).cuda().long(), torch.randint(posi_num, (posi_num_exp-posi_num,)).cuda().long()], 0)
        posi_idx = posi_idx[choice.long()]
    else:
        choice = torch.LongTensor(np.random.choice(posi_num, posi_num_exp, replace=False)).cuda()
        posi_idx = posi_idx[choice]

    if nega_num < nega_num_exp:
        choice = torch.cat([torch.arange(0, nega_num).cuda().long(), torch.randint(nega_num, (nega_num_exp-nega_num,)).cuda().long()], 0)
        nega_idx = nega_idx[choice.long()]
    else:
        choice = torch.LongTensor(np.random.choice(nega_num, nega_num_exp, replace=False)).cuda()
        nega_idx = nega_idx[choice]

    if nega_prop_num < nega_prop_num_exp:
        choice = torch.cat([torch.arange(0, nega_prop_num).cuda().long(), torch.randint(nega_prop_num, (nega_prop_num_exp-nega_prop_num,)).cuda().long()], 0)
        nega_prop_idx = nega_prop_idx[choice.long()]
    else:
        choice = torch.LongTensor(np.random.choice(nega_prop_num, nega_prop_num_exp, replace=False)).cuda()
        nega_prop_idx = nega_prop_idx[choice]

    if posi_prop_num_exp > max_grasp:
        select = posi_prop_idx.new(posi_prop_num).zero_()
        select[posi_idx] = 1
        select[nega_idx] = 1
        un_select = torch.nonzero(select == 0).view(-1)
        choice = torch.LongTensor(np.random.choice(un_select.size(0), posi_prop_num_exp-max_grasp)).cuda()
        un_select = un_select[choice]
        posi_idx = posi_prop_idx[posi_idx]
        nega_idx = posi_prop_idx[nega_idx]
        posi_prop_idx = torch.cat([posi_idx, nega_idx, posi_prop_idx[un_select]], 0)
    else:
        posi_idx = posi_prop_idx[posi_idx]
        nega_idx = posi_prop_idx[nega_idx]
        posi_prop_idx = torch.cat([posi_idx, nega_idx], 0)

    return posi_prop_idx, nega_prop_idx, posi_idx, nega_idx


def getLocalPoints(pc, contacts, centers, local_th=0.011, local_pn=100):
    pc = pc.squeeze(0) * torch.FloatTensor([0.22/2, 0.22/2, 0.22]).to(pc.device)
    point_num = pc.size(0)
    contacts = contacts.squeeze(0)
    con_num = contacts.size(0)
    centers = centers.squeeze(0)
    con_cen_vec = nn.functional.normalize(contacts - centers, dim=-1)
    con_pc_dist = dist_matrix_torch(contacts, pc)
    con_pc_vec = contacts.view(-1, 1, 3) - pc.view(1, -1, 3)
    con_pc_vec /= con_pc_dist.view(con_num, point_num, 1)
    dist_ = con_pc_dist.unsqueeze(1) * (1.0 + torch.abs(con_cen_vec.unsqueeze(1).matmul(con_pc_vec.transpose(1,2))))
    dist = dist_.to('cuda:0')
    local_points = pu.matrix_k_min(local_th, local_pn, dist).long()
    # print(dist.size(), local_points.size())
    # print(dist.device, local_points.device)
    del (dist_, dist)
    return local_points.view(1, -1, local_pn)


def getLocalPointsV2(pc, contacts, centers, local_th=0.011, local_pn=100):
    pc = pc.squeeze(0) * torch.FloatTensor([0.22/2, 0.22/2, 0.22]).to(pc.device)
    point_num = pc.size(0)
    contacts = contacts.squeeze(0)
    con_num = contacts.size(0)
    centers = centers.squeeze(0)
    interval = 1e3
    num = con_num // interval + 1
    num = int(num)
    delta = (con_num + num - 1) // num
    local_points_list = []
    for i in range(num):
        s = delta * i
        e = delta * (i+1)
        if i+1 == num:
            e = max(e, con_num)
            # print(con_num, num, e)
        contacts_ = contacts[s:e]
        centers_ = centers[s:e]
        con_cen_vec = nn.functional.normalize(contacts_ - centers_, dim=-1)
        con_pc_dist = dist_matrix_torch(contacts_, pc)
        con_pc_vec = contacts_.view(-1, 1, 3) - pc.view(1, -1, 3)
        con_pc_vec /= con_pc_dist.view(-1, point_num, 1)
        dist_ = con_pc_dist.unsqueeze(1) * (1.0 + torch.abs(con_cen_vec.unsqueeze(1).matmul(con_pc_vec.transpose(1,2))))
        dist = dist_.to('cuda:0')
        local_points_ = pu.matrix_k_min(local_th, local_pn, dist).long()
        local_points_list.append(local_points_.view(1, -1, local_pn))
        del (dist_, dist, local_points_, con_cen_vec, con_pc_dist, con_pc_vec)
    local_points = torch.cat(local_points_list, 1)
    return local_points


if __name__ == '__main__':
    pc = torch.randn(1, 10, 3)
    matrix = dist_matrix_torch(pc, pc)
    # matrix[matrix==math.nan] = 0.0
    print(matrix)
    print(matrix.min(), matrix.max())




