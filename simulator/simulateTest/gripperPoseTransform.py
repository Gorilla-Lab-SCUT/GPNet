import torch
import torch.nn.functional as F
import math
import numpy as np

from simulateTest.gripperSimpleCollision import GripperSimipleCollision


# pytorch version
#   q1, q2: torch tensor of size [bs, 4]
def quaternionMultiplication(q1, q2):
    w = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    x = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    y = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    z = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
    # print(w)
    return torch.stack([w, x, y, z], dim=1)


# pytorch version
#   axis:   torch tensor of size [bs, 3], normalized
#   angle:  torch tensor of size [bs],  radians
def axisAngle2Quaternion(axis, angle):
    bs = angle.shape[0]
    w = torch.cos(angle / 2)
    halfSin = torch.sin(angle / 2)
    xyz = axis * halfSin[:, None].expand(bs, 3)
    return torch.cat([w[:, None], xyz], dim=1)


# pytorch version
# matrix : torch tensor of size [bs, 9]
# return quaternion: torch tensor of size [bs, 4]
def matrix2quaternion(matrix):
    fourWSquaredMinus1 = matrix[:, 0] + matrix[:, 4] + matrix[:, 8]
    fourXSquaredMinus1 = matrix[:, 0] - matrix[:, 4] - matrix[:, 8]
    fourYSquaredMinus1 = matrix[:, 4] - matrix[:, 0] - matrix[:, 8]
    fourZSquaredMinus1 = matrix[:, 8] - matrix[:, 0] - matrix[:, 4]
    temp = torch.stack([fourWSquaredMinus1, fourXSquaredMinus1, fourYSquaredMinus1, fourZSquaredMinus1], dim=1)
    fourBiggestSquaredMinus1, biggestIndex = torch.max(temp, dim=1)
    biggestVal = torch.sqrt(fourBiggestSquaredMinus1 + 1) * 0.5
    mult = 0.25 / biggestVal
    temp0 = biggestVal
    temp1 = (matrix[:, 7] - matrix[:, 5]) * mult
    temp2 = (matrix[:, 2] - matrix[:, 6]) * mult
    temp3 = (matrix[:, 3] - matrix[:, 1]) * mult
    temp4 = (matrix[:, 7] + matrix[:, 5]) * mult
    temp5 = (matrix[:, 2] + matrix[:, 6]) * mult
    temp6 = (matrix[:, 3] + matrix[:, 1]) * mult

    quaternion = torch.empty(size=[matrix.shape[0], 4], dtype=torch.float)
    quaternionBiggestIndex0 = torch.clone(torch.stack([temp0, temp1, temp2, temp3], dim=1))
    quaternionBiggestIndex1 = torch.clone(torch.stack([temp1, temp0, temp6, temp5], dim=1))
    quaternionBiggestIndex2 = torch.clone(torch.stack([temp2, temp6, temp0, temp4], dim=1))
    quaternionBiggestIndex3 = torch.clone(torch.stack([temp3, temp5, temp4, temp0], dim=1))

    # biggestIndex0Map = torch.ne(biggestIndex, 0)
    # biggestIndex0Map = biggestIndex0Map[:, None].expand_as(quaternion)
    biggestIndex1Map = torch.ne(biggestIndex, 1)
    biggestIndex1Map = biggestIndex1Map[:, None].expand_as(quaternion)
    biggestIndex2Map = torch.ne(biggestIndex, 2)
    biggestIndex2Map = biggestIndex2Map[:, None].expand_as(quaternion)
    biggestIndex3Map = torch.ne(biggestIndex, 3)
    biggestIndex3Map = biggestIndex3Map[:, None].expand_as(quaternion)

    quaternion = quaternionBiggestIndex0
    quaternion = quaternion.where(biggestIndex1Map, quaternionBiggestIndex1)
    quaternion = quaternion.where(biggestIndex2Map, quaternionBiggestIndex2)
    quaternion = quaternion.where(biggestIndex3Map, quaternionBiggestIndex3)
    return quaternion


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


# pytorch matrix version
# matrix: torch tensor of size [bs, 3, 3]
# return: torch tensor of size [bs]
def getCosAngle(matrix):
    gripperX = matrix[:, :, 0]
    gripperX = getUniqueGripperX(gripperX)
    gripperZ = matrix[:, :, 2]
    gripperZ = F.normalize(gripperZ, dim=1)
    zAxis = torch.tensor([0, 0, 1], dtype=torch.float)
    zAxis = zAxis.expand_as(gripperX)
    tangentY = - gripperX.cross(zAxis)
    tangentY = F.normalize(tangentY, dim=1)
    cosAngle = torch.sum(gripperZ * tangentY, dim=1)
    return cosAngle


# pytorch version
# gripperX: torch tensor of size [bs, 3]
# return: torch tensor of size [bs, 3]
def getUniqueGripperX(gripperX):
    negGripperX = torch.clone(torch.neg(gripperX))
    gripperXwithZgt0Flag = gripperX[:, 2].gt(0)
    gripperXwithZgt0Flag = gripperXwithZgt0Flag[:, None].expand_as(gripperX)
    gripperX = gripperX.where(gripperXwithZgt0Flag, negGripperX)
    gripperX = F.normalize(gripperX, dim=1)
    gripperXOffset = torch.clone(gripperX + 0.001)
    gripperXNotPerpendicularToXoYFlag = torch.abs(torch.mul(gripperX[:, 0], gripperX[:, 1])).gt(1e-6)
    gripperXNotPerpendicularToXoYFlag = gripperXNotPerpendicularToXoYFlag[:, None].expand_as(gripperX)
    gripperX = gripperX.where(gripperXNotPerpendicularToXoYFlag, gripperXOffset)
    gripperX = F.normalize(gripperX, dim=1)
    return gripperX


# pytorch version
# contact1: torch tensor of size [bs, 3]
# contact2: torch tensor of size [bs, 3]
# cosAngle: torch tensor of size [bs]
# return: torch tensor of size [bs, 3, 3]
def getOrientation(contact1, contact2, cosAngle):
    gripperX = contact1 - contact2
    gripperX = getUniqueGripperX(gripperX)
    # print('gripperX\t', gripperX)

    zAxis = torch.tensor([0, 0, 1], dtype=torch.float)
    zAxis = zAxis.expand_as(gripperX)
    tangentY = zAxis.cross(gripperX)
    tangentY = F.normalize(tangentY, dim=1)
    # print('tangentY\t', tangentY)

    tangentZ = gripperX.cross(tangentY)
    # print('tangentZ\t', tangentZ)
    sinAngle = torch.sqrt(1 - cosAngle * cosAngle)

    gripperZ = cosAngle[:, None].expand_as(tangentY) * tangentY + sinAngle[:, None].expand_as(tangentY) * tangentZ
    gripperZ = F.normalize(gripperZ, dim=1)
    gripperY = gripperZ.cross(gripperX)
    gripperY = F.normalize(gripperY, dim=1)
    return torch.transpose(torch.stack([gripperX, gripperY, gripperZ], dim=1), dim0=-1, dim1=-2)


if __name__ == "__main__":
    # pass
    contact1 = torch.tensor([0.040, 0.025, -0.010])
    contact2 = torch.tensor([-0.040, -0.025, 0.010])
    bs = 9
    contact1 = contact1.expand(bs, 3)
    contact2 = contact2.expand(bs, 3)
    angle = np.linspace(0, math.pi, bs)
    cos = np.cos(angle)
    print(cos)
    cos = torch.as_tensor(cos, dtype=torch.float)
    orientation = getOrientation(contact1, contact2, cos)
    cos = getCosAngle(orientation)
    # print(orientation)
    print(cos)
    gripper = GripperSimipleCollision()
    for rm in orientation:
        m = np.eye(4)
        m[:3, :3] = rm.numpy()
        gripper.setOpeningWidth()
        gripper.setOrientationAndTranslation2(orientation=m)
        gripper.visualization()