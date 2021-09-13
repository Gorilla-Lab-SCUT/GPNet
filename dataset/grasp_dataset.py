import os
import time
import random

import numpy as np
import torch
import torch.utils.data as data
import OpenEXR
import Imath
import cv2
from skimage import io
from skimage.transform import resize

from tools.depth2points import Depth2PointCloud
from tools.Quaternion import *
from tools.proposal import *

random.seed(1)


def read_camera_info(camera_info_file):
    camera_info_array = np.load(camera_info_file)
    cameraInfoDict = {}
    for item in camera_info_array:
        cameraInfoDict[item['id'].decode()] = (item['position'],
                                              item['orientation'],
                                              item['calibration_matrix'])
    return cameraInfoDict


def render_image(obj_id, camera_pose):
    pass


def random_camera_view(cameraInfoDict=None):
    view_num = len(cameraInfoDict)
    view = np.random.choice(view_num, 1)[0]
    return cameraInfoDict['view%d'%(view)], view


def exr2tiff(exrpath):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    rgb = [np.frombuffer(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
    r = np.reshape(rgb[0], (Size[1], Size[0]))
    mytiff = r
    return mytiff


def read_image(img_root, view):
    img_path = img_root
    # img = io.imread(img_path+'/render%d.png'%(view))
    depth = exr2tiff(img_path+'/render%dDepth0001.exr'%(view))
    return depth


def noramlize_depth(depth):
    pass


def generate_grid(max_value, grid_num):
    x = np.linspace(0, max_value, grid_num+1)[:-1]
    x, y, z = np.meshgrid(x, x, x)
    x = x - max_value/2
    y = y - max_value/2
    x = x.reshape(-1, 1) + max_value/(grid_num)/2
    y = y.reshape(-1, 1) + max_value/(grid_num)/2
    z = z.reshape(-1, 1) + max_value/(grid_num)/2
    grids = np.concatenate([x, y, z], -1).reshape(-1, 3)
    return grids


def travel_image_dir(img_root):
    folders = os.listdir(img_root)
    shape_dir = {}
    shapes = os.listdir(img_root)
    for s in shapes:
        shape_path = os.path.join(img_root, s)
        if os.path.isdir(shape_path):
            shape_dir[s] = shape_path
    return shape_dir


def read_BBox(data_path):
    f = np.load(data_path)
    bbox = {}
    for item in f:
        value = item['aabbValue']
        bbox[item['objId'].decode()] = [value[:3], value[3:]]
    return bbox



class GraspData(data.Dataset):
    def __init__(self, dataroot, img_size=(224,224), seg_th=0.002, split='train',
        grid_len=0.22, grid_num=10, part=None, return_shape=False, sample_ratio=1.0, 
        sample_num=1000, posi_ratio=0.5, view=0):
        self.dataroot = dataroot
        self.split = split
        self.seg_th = seg_th
        self.img_size = img_size
        self.grid_len = grid_len
        self.grid_num = grid_num
        self.grids = generate_grid(grid_len, grid_num)
        self.return_shape = return_shape
        self.posi_ratio = posi_ratio
        self.sample_num = sample_num
        self.view = view

        file_path = os.path.join('./dataset/%s_set.csv'%split)
        file = open(file_path, 'r')
        lines = file.readlines()
        self.datasamples = []
        for l in lines:
            shape = l.strip().split('.')[0]
            if shape == '4eefe941048189bdb8046e84ebdc62d2':
                continue
            self.datasamples.append(shape)
        file.close()
        if part is not None and split == 'train':
            num = round(len(self.datasamples) / part)
            self.datasamples = self.datasamples[:num]

        print(split, len(self.datasamples))
        self.img_root = os.path.join(dataroot, 'images')
        self.shape_dir = travel_image_dir(self.img_root)
        self.annoRoot = os.path.join(dataroot, 'annotations')

        self.bbox = read_BBox('./dataset/aabbValue.npy')
        self.sample_ratio = sample_ratio


    def __len__(self):
        return len(self.datasamples)


    def __getitem__(self, index):
        shape = self.datasamples[index]
        grasps, cam_info, contacts, angles = self.read_grasps(shape)
        if self.split == 'train':
            camera_pose, view = random_camera_view(cam_info)
        else:
            view = self.view
            camera_pose = cam_info['view%d'%view]
        # print(view)
        ca_loc = camera_pose[0]
        ca_ori = camera_pose[1]
        intrinsic = camera_pose[2].reshape(3, 3)
        img_path = self.shape_dir[shape]
        depth = read_image(img_path, view)

        org_size = depth.shape
        depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_NEAREST)

        pc = Depth2PointCloud(depth, intrinsic, ca_ori, ca_loc, org_size=org_size).transpose()
        inf_idx = (pc != pc) | (np.abs(pc) > 100)
        pc[inf_idx] = 0.0
        pc_index = np.nonzero(pc[:, 2]>self.seg_th)[0]
        pc = pc[pc_index]
        # off = self.offset[shape][0]
        # pc[:, 2] += off[2]
        pc_x = pc[:, 0]
        pc_y = pc[:, 1]
        pc_z = pc[:, 2]
        del_idx = (pc_x<-0.22/2) | (pc_x>0.22/2) | (pc_y<-0.22/2) | (pc_y>0.22/2) | (pc_z>0.22)
        pc = pc[del_idx==False]
        pc_index = pc_index[del_idx==False]

        assert len(grasps)>0, 'no grasps: %s_%s'%(shape, view)
        if pc.shape[0] == 0:
            pc = np.array([[0, 0, 0]])
            grids, contact, center, contact_index, scores, grasps_idx, angles_exp, \
            posi_mask, angles, posi_nega_idx, unanti_center \
            = np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0]), \
                np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0])
            return pc, grids, contact, center, contact_index, scores, grasps_idx, angles_exp, posi_mask, \
                    angles, posi_nega_idx, unanti_center

        if self.split == 'test':
            xyz_max = pc.max(0)
            xyz_min = pc.min(0)
            xyz_diff = xyz_max - xyz_min
            xyz_idx = np.where(xyz_diff<self.grid_len/(self.grid_num))[0]
            if xyz_idx.shape[0] > 0:
                xyz_max[xyz_idx] += (self.grid_len/(self.grid_num))
                xyz_min[xyz_idx] -= (self.grid_len/(self.grid_num))
            grid_choose = (xyz_min.reshape(1,-1) <= self.grids) * (self.grids <= xyz_max.reshape(1, -1))
            grid_choose = (grid_choose.sum(1) == 3)
            grids = self.grids[grid_choose]
            if grids.shape[0] > 400:
                idx = np.random.choice(np.arange(grids.shape[0]), 400, replace=False)
                grids = grids[idx]
            contact_index = np.arange(pc.shape[0])
            pc = pc/np.array([0.22/2, 0.22/2, 0.22])
            return pc, grids, contact_index, shape

        bbox = self.bbox[shape]
        xyz_min, xyz_max = bbox[0], bbox[1]
        # xyz_max = pc.max(0)
        # xyz_min = pc.min(0)
        xyz_diff = xyz_max - xyz_min
        xyz_idx = np.where(xyz_diff<self.grid_len/(self.grid_num))[0]
        # print(self.grid_len/self.grid_num, self.grids.min(0))
        if xyz_idx.shape[0] > 0:
            xyz_max[xyz_idx] += (self.grid_len/(self.grid_num))
            xyz_min[xyz_idx] -= (self.grid_len/(self.grid_num))
            # print(shape, xyz_idx, xyz_max, xyz_min)
        grid_choose = (xyz_min.reshape(1,-1) <= self.grids) * (self.grids <= xyz_max.reshape(1, -1))
        grid_choose = (grid_choose.sum(1) == 3)
        grids = self.grids[grid_choose]
        assert grids.shape[0] > 0, 'grids is empty: %s_%s '%(shape, view) + str(xyz_max) + str(xyz_min)
        grid_un_choose = np.where(grid_choose==False)[0]
        np.random.shuffle(grid_un_choose)
        out_box_grids = self.grids[grid_un_choose][:grids.shape[0]]
        if self.split == 'train':
            if grids.shape[0]<400:
                grids = np.concatenate([grids, out_box_grids[:50]], 0)
            else:
                idx = np.random.choice(np.arange(grids.shape[0]), 400, replace=False)
                grids = grids[idx]

        contact, center, contact_index, scores, grasps_idx, angles_exp, posi_mask, angles \
        = getContactCenterV2(grasps, pc, contacts, angles, 0.0035)

        if contact_index.shape[0] == 0 or np.sum(scores) == 0 or np.sum(scores) == scores.shape[0]:
            contact = np.array([0])
            center = np.array([0])
            contact_index = torch.Tensor([-1])
            scores = np.array([0])
            grasps_idx = np.array([0])
            angles_exp = np.array([0])
            posi_mask = np.array([0])
            angles = np.array([0])
            posi_nega_idx = np.array([0])
        else:
            posi_nega_idx = self.select_posi_nega(scores)
        pc = pc/np.array([0.22/2, 0.22/2, 0.22])
        return pc, grids, contact, center, contact_index, scores, grasps_idx, \
                angles_exp, posi_mask, angles, posi_nega_idx


    def read_grasps(self, shape):
        if self.split != 'test':
            centers = np.load(os.path.join(self.annoRoot, 'candidate_contact_v2', shape+'_c.npy'))
            quaternions = np.load(os.path.join(self.annoRoot, 'candidate', shape+'_q.npy'))
            widths = np.load(os.path.join(self.annoRoot, 'candidate', shape+'_d.npy'))
            contacts = np.load(os.path.join(self.annoRoot, 'candidate_contact_v2', shape+'_contact.npy'))
            angles = np.load(os.path.join(self.annoRoot, 'candidate_contact_v2', shape+'_cos.npy'))
            labels = np.load(os.path.join(self.annoRoot, 'simulateResult', shape+'.npy'))
            qualities = np.zeros(centers.shape[0])
            assert labels.shape[0] == centers.shape[0]
            
            all_grasps = np.concatenate([labels.reshape(-1, 1), qualities.reshape(-1, 1), widths.reshape(-1, 1), 
                                        centers, quaternions], -1)

            posi_idx = np.nonzero(labels.reshape(-1))[0].reshape(-1)
            nega_idx = np.nonzero((labels==False).reshape(-1))[0].reshape(-1)
            posi_num = posi_idx.shape[0]
            nega_num = nega_idx.shape[0]
            posi_idx = posi_idx[:int(posi_num*self.sample_ratio)]
            nega_idx = nega_idx[:int(nega_num*self.sample_ratio)]

            posi_num = posi_idx.shape[0]
            nega_num = nega_idx.shape[0]
            sample_num = 5000
            half_num = sample_num // 2
            if posi_num > half_num:
                posi_idx = np.random.choice(posi_idx, half_num, replace=False)
            else:
                posi_idx = np.random.choice(posi_idx, half_num, replace=True)
            if nega_num > half_num:
                nega_idx = np.random.choice(nega_idx, half_num, replace=False)
            else:
                nega_idx = np.random.choice(nega_idx, half_num, replace=True)

            all_idx = np.concatenate([posi_idx, nega_idx], 0)
            np.random.shuffle(all_idx)
            grasps = all_grasps[all_idx]
            contacts = contacts[all_idx]
            angles = angles[all_idx]
        else:
            grasps = np.zeros((1, 10))
            contacts = np.zeros((1, 2, 3))
            angles = np.zeros((1,))
        cam_info_path = os.path.join(self.img_root, shape, 'CameraInfo.npy')
        cam_info = read_camera_info(cam_info_path)

        return grasps, cam_info, contacts, angles


    def select_posi_nega(self, scores):
        sel_posi_num = int(self.sample_num*self.posi_ratio)
        sel_nega_num = self.sample_num - sel_posi_num
        posi_idx = np.nonzero(scores.reshape(-1))[0].reshape(-1)
        nega_idx = np.nonzero((scores==0).reshape(-1))[0].reshape(-1)

        if posi_idx.shape[0] >= sel_posi_num:
            idx = np.random.choice(posi_idx.shape[0], sel_posi_num, replace=False)
            posi_idx = posi_idx[idx]
        else:
            idx = np.random.choice(posi_idx.shape[0], sel_posi_num-posi_idx.shape[0])
            idx = np.concatenate([np.arange(posi_idx.shape[0]), idx], 0)
            posi_idx = posi_idx[idx]

        if nega_idx.shape[0] >= sel_nega_num:
            idx = np.random.choice(nega_idx.shape[0], sel_nega_num, replace=False)
            nega_idx = nega_idx[idx]
        else:
            idx = np.random.choice(nega_idx.shape[0], sel_nega_num-nega_idx.shape[0])
            idx = np.concatenate([np.arange(nega_idx.shape[0]), idx], 0)
            nega_idx = nega_idx[idx]
        idx = np.concatenate([posi_idx, nega_idx])
        return idx

