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
    def __init__(self, dataroot, split='test'):
        self.dataroot = dataroot
        self.split = split

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
        
        print(split, len(self.datasamples))
        self.annoRoot = os.path.join(dataroot, 'annotations')


    def __len__(self):
        return len(self.datasamples)


    def __getitem__(self, index):
        shape = self.datasamples[index]
        grasps = self.read_grasps(shape)
        grasps = grasps[:, 2:]
        return grasps, shape


    def read_grasps(self, shape):
        centers = np.load(os.path.join(self.annoRoot, 'candidate', shape+'_c.npy'))
        quaternions = np.load(os.path.join(self.annoRoot, 'candidate', shape+'_q.npy'))
        widths = np.load(os.path.join(self.annoRoot, 'candidate', shape+'_d.npy'))
        contacts = np.load(os.path.join(self.annoRoot, 'candidate', shape+'_contact.npy'))
        angles = np.load(os.path.join(self.annoRoot, 'candidate', shape+'_cos.npy'))
        labels = np.load(os.path.join(self.annoRoot, 'simulateResult', shape+'.npy'))
        qualities = np.zeros(centers.shape[0])
        assert labels.shape[0] == centers.shape[0]
        
        all_grasps = np.concatenate([labels.reshape(-1, 1), qualities.reshape(-1, 1), widths.reshape(-1, 1), 
                                    centers, quaternions], -1)

        posi_idx = np.nonzero(labels.reshape(-1))[0].reshape(-1)
        return all_grasps[posi_idx]

