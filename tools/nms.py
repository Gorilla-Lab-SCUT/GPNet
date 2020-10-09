import numpy as np



def dist_fn(contact1, contact2):
    dist = np.sum((contact1 - contact2)**2, 1)
    return np.sqrt(dist)


def nms(contact, scores, th=0.005):
    assert contact.shape[0] == scores.shape[0]
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        contact1 = contact[order[0], ...]
        contact2 = contact[order[1:], ...]

        dist = dist_fn(contact1, contact2)

        # supress the high thresh
        inds = (dist > th).nonzero()[0]
        order = order[inds + 1]
    return keep


def dist_fn2(cent1, cent2, quat1, quat2):
    cent_dist = np.sum((cent1 - cent2)**2, 1)
    quat_dist = 2 * np.arccos(np.abs(quat2.dot(quat1.T)))
    return np.sqrt(cent_dist), quat_dist


def nms2(center, quaternion, scores, cent_th=0.05, ang_th=np.pi/3):
    assert center.shape[0] == quaternion.shape[0] and center.shape[0] == scores.shape[0]
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        cent1 = center[order[0], ...]
        cent2 = center[order[1:], ...]

        quat1 = quaternion[order[0], ...]
        quat2 = quaternion[order[1:], ...]

        cent_dist, quat_dist = dist_fn2(cent1, cent2, quat1, quat2)

        # print('quat_dist: ', quat_dist.max(), quat_dist.min(), quat_dist.shape)
        cent_out = (cent_dist > cent_th)
        quat_out = (quat_dist > ang_th)
        # print('cent_dist:', cent_dist.max(), cent_dist.min(), cent_dist.shape, cent_out.sum(),
        #     'quat_dist: ', quat_dist.max(), quat_dist.min(), quat_dist.shape, quat_out.sum())
        inds = (cent_out | quat_out).nonzero()[0]
        order = order[inds + 1]
    return keep
