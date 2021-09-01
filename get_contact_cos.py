import os
import time
import numpy as np
from tools.proposal import *


def read_grasps(root, shape):
    centers = np.load(os.path.join(root, shape+'_c.npy'))
    quaternions = np.load(os.path.join(root, shape+'_q.npy'))
    widths = np.load(os.path.join(root, shape+'_d.npy'))
    return centers, quaternions, widths


def save_contact_angle(centers, contacts, cos_angle, root, shape):
    cen_file = os.path.join(root, shape+'_c.npy')
    np.save(cen_file, centers)
    con_file = os.path.join(root, shape+'_contact.npy')
    np.save(con_file, contacts)
    cos_file = os.path.join(root, shape+'_cos.npy')
    np.save(cos_file, cos_angle)


def get_shape_id(root):
    all_files = os.listdir(root)
    shapes = []
    for f in all_files:
        if '_c.npy' in f:
            s = f.split('_')[0]
            shapes.append(s)
    return shapes


# pytorch version
# quaternion : torch tensor of size [bs, 4]
# return matrix: torch tensor of size [bs, 9]
def quaternion2matrix(quaternion):
    sw = quaternion[:, 0] * quaternion[:, 0]
    sx = quaternion[:, 1] * quaternion[:, 1]
    sy = quaternion[:, 2] * quaternion[:, 2]
    sz = quaternion[:, 3] * quaternion[:, 3]

    m00 = (sx - sy - sz + sw)
    m11 = (-sx + sy - sz + sw)
    m22 = (-sx - sy + sz + sw)

    tmp1 = quaternion[:, 1] * quaternion[:, 2]
    tmp2 = quaternion[:, 3] * quaternion[:, 0]
    m10 = 2.0 * (tmp1 + tmp2)
    m01 = 2.0 * (tmp1 - tmp2)

    tmp1 = quaternion[:, 1] * quaternion[:, 3]
    tmp2 = quaternion[:, 2] * quaternion[:, 0]
    m20 = 2.0 * (tmp1 - tmp2)
    m02 = 2.0 * (tmp1 + tmp2)

    tmp1 = quaternion[:, 2] * quaternion[:, 3]
    tmp2 = quaternion[:, 1] * quaternion[:, 0]
    m21 = 2.0 * (tmp1 + tmp2)
    m12 = 2.0 * (tmp1 - tmp2)
    return torch.stack([m00, m01, m02, m10, m11, m12, m20, m21, m22], dim=1)


def zMove(quaternion, gripperPosition, zMoveLength=-0.015):
    quaternion_ = torch.tensor(quaternion)
    rotationMatrix = quaternion2matrix(quaternion_).numpy().reshape(-1, 3, 3)
    rotationZ = rotationMatrix[:, :, 2]
    moveZ = rotationZ * zMoveLength
    gripperPosition = gripperPosition + moveZ  # move away from object: avoid collision
    return gripperPosition



if __name__ == '__main__':
    data_root = '/data/wuchaozheng/dataset/shapenetSemGrasp/jrnl_data/annotations/candidate'
    shapes = get_shape_id(data_root)
    np.random.shuffle(shapes)
    for s in shapes[::-1]:
        print(s)
        if os.path.exists(os.path.join(data_root, s+'_cos.npy')):
            continue
        st = time.time()
        centers, quaternions, widths = read_grasps(data_root, s)
        centers = zMove(quaternions, centers)
        contacts = []
        cos_angles = []
        for i in range(centers.shape[0]):
            cent, quat, w = centers[i], quaternions[i], widths[i]
            c1, c2 = getContactPoint(w, cent, quat)
            cos = getCosAngle(quat.tolist())
            contacts.append(np.stack([c1, c2], 0))
            cos_angles.append(cos)
        contacts = np.array(contacts)
        cos_angles = np.array(cos_angles)
        assert contacts.shape[0] == centers.shape[0]
        assert cos_angles.shape[0] == centers.shape[0]
        save_contact_angle(centers, contacts, cos_angles, data_root, s)
        print('time: ', time.time()-st)





