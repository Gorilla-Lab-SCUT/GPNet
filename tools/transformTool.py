from Quaternion import *
# from GripperSimpleCollision import GripperSimipleCollision
import torch
import torch.nn.functional as F
GRADS = {}


# pytorch version
#   q1, q2: torch tensor of size [bs, 9]
def quaternionMultiplication2(q1, q2):
    w = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    x = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    y = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    z = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
    # print(w)
    return torch.stack([w, x, y, z], dim=1)


# pytorch version
# matrix : torch tensor of size [bs, 9]
# return quaternion: torch tensor of size [bs, 4]
def matrix2quaternion2(matrix):
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
    quaternionBiggestIndex0 = torch.stack([temp0, temp1, temp2, temp3], dim=1)
    quaternionBiggestIndex1 = torch.stack([temp1, temp0, temp6, temp5], dim=1)
    quaternionBiggestIndex2 = torch.stack([temp2, temp6, temp0, temp4], dim=1)
    quaternionBiggestIndex3 = torch.stack([temp3, temp5, temp4, temp0], dim=1)

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
def quaternion2matrix2(quaternion):
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


# solution of the symmetry of gripper shape
# make the last element of y axis (the second column) of the rotation matrix to be positive
# matrix: list or ndarray of size [9]
def matrixYAxisLastElementPositive(matrix):
    if matrix[7] < 0:
        matrix[1] = - matrix[1]
        matrix[2] = - matrix[2]
        matrix[4] = - matrix[4]
        matrix[5] = - matrix[5]
        matrix[7] = - matrix[7]
        matrix[8] = - matrix[8]
    return matrix


# if loss is directly applied on the matrix, we need to solve the problem of symmetry of gripper shape
# solution of the symmetry of gripper shape
# pytorch version
# make the last element of y axis (the second column) of the rotation matrix to be positive
# matrix: torch tensor of size [bs, 9]
def matrixYAxisLastElementPositive2(matrix):
    negYZMask = torch.empty([1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=torch.uint8)
    negYZMatrix = matrix.where(negYZMask, -matrix)
    YAxisLastElementGT0Flag = matrix[:, 7].gt(0)
    YAxisLastElementGT0Flag = YAxisLastElementGT0Flag[:, None].expand_as(matrix)
    matrix = matrix.where(YAxisLastElementGT0Flag, negYZMatrix)
    return matrix


def getContactPoint(gripperLength, gripperCenter, gripperOrientation):
    rotation = Quaternion()
    rotation.listInit(gripperOrientation)
    # print(rotation.getMatrix())
    contact1Relative = [- gripperLength / 2, 0, 0]
    contact2Relative = [gripperLength / 2, 0, 0]
    contact1Relative = rotation.applyRotation(contact1Relative)
    contact2Relative = rotation.applyRotation(contact2Relative)
    contact1 = [c + r for c, r in zip(gripperCenter, contact1Relative)]
    contact2 = [c + r for c, r in zip(gripperCenter, contact2Relative)]
    return contact1, contact2


# def getCosAngle(gripperOrientation):
#     rotation = Quaternion()
#     rotation.listInit(gripperOrientation)
#     gripperY = np.array(rotation.applyRotation([0, 1, 0]))
#     gripperY = getUniqueGripperY(gripperY)
#     tangentX = np.cross(gripperY, (0, 0, 1))
#     tangentX = tangentX / getNorm(tangentX)
#     gripperX = np.array(rotation.applyRotation([1, 0, 0]))
#     gripperX = gripperX / getNorm(gripperX)
#     cosAngle = np.dot(gripperX, tangentX)
#     return cosAngle    # return cos of elevation


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
    return cosAngle    # return cos of elevation


# pytorch matrix version
# matrix: torch tensor of size [bs, 3, 3]
def getCosAngle2(matrix):
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


# def getOrientation(contact1, center, cosAngle):
#     gripperY = np.array(contact1) - np.array(center)
#     gripperY = getUniqueGripperY(gripperY)
#     tangentX = np.cross(gripperY, (0, 0, 1))
#     tangentX = tangentX / getNorm(tangentX)
#     tangentZ = np.cross(tangentX, gripperY)
#     sinAngle = - sqrt(1 - pow(cosAngle, 2))
#     gripperX = cosAngle * tangentX + sinAngle * tangentZ
#     # gripperX = gripperX / getNorm(gripperX)
#     gripperZ = np.cross(gripperX, gripperY)
#     # gripperZ = gripperZ / getNorm(gripperZ)
#     return np.vstack((gripperX, gripperY, gripperZ)).T


# pytorch version
# contact1: torch tensor of size [bs, 3]
# center: torch tensor of size [bs, 3]
# cosAngle: torch tensor of size [bs]
# return: torch tensor of size [bs, 9]
def getOrientation2(contact1, center, cosAngle):
    gripperX = contact1 - center
    gripperX = getUniqueGripperX2(gripperX)
    # print('gripperX\t', gripperX)

    zAxis = torch.tensor([0, 0, 1], dtype=torch.float).to(center.device)
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



# pytorch version
# contact1: torch tensor of size [bs, 3]
# contact2: torch tensor of size [bs, 3]
# cosAngle: torch tensor of size [bs]
# return: torch tensor of size [bs, 3, 3]
def getOrientation3(contact1, contact2, cosAngle):
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


# def getUniqueGripperY(gripperY):
#     if gripperY[2] < 0:
#         gripperY = -gripperY
#     gripperY = gripperY / getNorm(gripperY)
#     if abs(gripperY[0] * gripperY[1]) < 1e-6:
#         gripperY[0] += 0.001
#         gripperY = gripperY / getNorm(gripperY)
#     return gripperY


def getUniqueGripperX(gripperX):
    if gripperX[2] < 0:
        gripperX = -gripperX
    gripperX = gripperX / getNorm(gripperX)
    if abs(gripperX[0] * gripperX[1]) < 1e-6:
        gripperX[0] += 0.001
        gripperX = gripperX / getNorm(gripperX)
    return gripperX


# pytorch version
# gripperY: torch tensor of size [bs, 3]
def getUniqueGripperY2(gripperY):
    # gripperYSignZ = torch.sign(gripperY[:, 2])
    # gripperYMapping = gripperYSignZ.expand_as(gripperY)
    # gripperY = gripperY * gripperYMapping
    negGripperY = torch.neg(gripperY)
    gripperYwithZgt0Flag = gripperY[:, 2].gt(0)
    gripperYwithZgt0Flag = gripperYwithZgt0Flag[:, None].expand_as(gripperY)
    gripperY = gripperY.where(gripperYwithZgt0Flag, negGripperY)
    gripperY = F.normalize(gripperY, dim=1)
    gripperYOffset = gripperY + 0.001
    gripperYNotPerpendicularToXoYFlag = torch.abs(torch.mul(gripperY[:, 0], gripperY[:, 1])).gt(1e-6)
    gripperYNotPerpendicularToXoYFlag = gripperYNotPerpendicularToXoYFlag[:, None].expand_as(gripperY)
    gripperY = gripperY.where(gripperYNotPerpendicularToXoYFlag, gripperYOffset)
    gripperY = F.normalize(gripperY, dim=1)
    return gripperY


# pytorch version
# gripperX: torch tensor of size [bs, 3]
# return: torch tensor of size [bs, 3]
def getUniqueGripperX2(gripperX):
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






# input
#   gripperOrientation : np.array of size 9
#   preRotation : np.array of size 9
def dis4DoF(gripperOrientation, simpleL1=True):
    gripperQ = Quaternion()
    gripperQ.rotationMatrixInit(gripperOrientation)
    
    if simpleL1:
        return abs(gripperQ.x) + abs(gripperQ.y)
    else:
        # return sqrt(dof4Q.x * dof4Q.x + dof4Q.y * dof4Q.y)
        # return acos(sqrt(dof4Q.w * dof4Q.w + dof4Q.z * dof4Q.z))
        return 1 - sqrt(gripperQ.w * gripperQ.w + gripperQ.z * gripperQ.z)


# pytorch version
#   gripperOrientation: torch tensor of size [bs, 9], the output of function getOrientation2(contact1, center, cosAngle)
def dis4DoF2(gripperOrientation, sum=True):
    gripperQ = matrix2quaternion2(gripperOrientation)
    # only calculate the w and z component
    # initDof4Q = quaternionMultiplication2(gripperQ, invPreQ)
    w = gripperQ[:, 0]
    z = gripperQ[:, 3]
    temp = 1 - torch.sqrt(w * w + z * z)
    return torch.sum(temp) if sum else temp
    # return temp


def save_grad(name):
    def hook(grad):
        GRADS[name] = grad
    return hook


if __name__ == "__main__":
    # # contact cosAngle loss test
    # gripperCenter = [0.02564412, -0.00385468, 0.07458372]
    # gripperOrientation = [0.3105045, - 0.18513325, 0.06153022, 0.93033683]
    #
    # contact1, contact2 = getContactPoint(
    #     gripperLength=0.085,
    #     gripperCenter=gripperCenter,
    #     gripperOrientation=gripperOrientation
    # )
    # cosAngle = getCosAngle(gripperOrientation)
    #
    # gripperRotationMatrix2 = getOrientation(
    #     contact1=contact1,
    #     center=gripperCenter,
    #     cosAngle=cosAngle
    # )
    # gripperOrientation2 = matrix2quaternion(gripperRotationMatrix2, is4x4=False)
    #
    # gripper = GripperSimipleCollision()
    # gripper.setOpeningWidth()
    # gripper.setOrientationAndTranslation(orientation=gripperOrientation, translation=gripperCenter)
    # gripper.visualization()
    #
    # gripper2 = GripperSimipleCollision()
    # gripper2.setOpeningWidth()
    # gripper2.setOrientationAndTranslation(orientation=gripperOrientation2, translation=gripperCenter)
    # gripper2.visualization()
    # tensorIn = torch.tensor([0.02564412, -0.00385468, 0.07458372], dtype=torch.float)

    # gripperCenter = [0.02564412, -0.00385468, 0.07458372]
    # gripperOrientation = [0.3105045, - 0.18513325, 0.06153022, 0.93033683]
    #
    # gripper = GripperSimipleCollision()
    # gripper.setOpeningWidth()
    # gripper.setOrientationAndTranslation(orientation=gripperOrientation, translation=gripperCenter)
    # # gripper.visualization()
    #
    # q = Quaternion()
    # q.listInit(gripperOrientation)
    # a = q.getMatrix()
    # print(q.getMatrix())
    # contact1, contact2 = getContactPoint(0.010, gripperCenter, gripperOrientation)
    # cosAngle = getCosAngle(gripperOrientation)
    # matrix = getOrientation(contact1, gripperCenter, cosAngle)
    #
    # q.rotationMatrixInit(matrix.reshape(-1))
    # gripper = GripperSimipleCollision()
    # gripper.setOpeningWidth()
    # gripper.setOrientationAndTranslation(orientation=q.getList(), translation=gripperCenter)
    # # gripper.visualization()
    # print(matrix)

    # # contact cosAngle pytorch loss test
    # q1 = Quaternion()
    # batchSize = 32
    # matrixBatch = np.empty(shape=[batchSize, 9], dtype=np.float)
    # gripperCenter = np.empty(shape=[batchSize, 3], dtype=np.float)
    # contact1 = np.empty(shape=[batchSize, 3], dtype=np.float)
    # gripperLength = 0.060
    # for i in range(batchSize):
    #     q1.axisAngleInit(axis=[1, 1, 0.5], angle=1.5 + i * 0.01)
    #     gripperCenter[i] = np.array([0.2 + i * 0.05, 0.2 + i * 0.03, 0.2 + i * 0.01])
    #     matrixBatch[i] = np.array(q1.getMatrix())
    #     contact, _ = getContactPoint(gripperLength, gripperCenter[i], q1.getList())
    #     contact1[i] = np.array(contact)
    # contact1 = torch.as_tensor(contact1, dtype=torch.float)
    # gripperCenter = torch.as_tensor(gripperCenter, dtype=torch.float)
    # matrixBatch = torch.as_tensor(matrixBatch, dtype=torch.float)
    # print(matrixBatch)
    # cosAngle = getCosAngle2(matrixBatch)
    # matrix2 = getOrientation2(contact1, gripperCenter, cosAngle)
    # print(matrix2)
    # print(1)
    # gripperY = torch.tensor([
    #     [1, 2, 3],
    #     [4, -5, -6],
    #     [-7, 8, 9],
    #     [0.01, 0.01, 12]
    # ])
    # cosAngle = getUniqueGripperY2(gripperY)

    # # quaternion matrix transformation test
    # q1 = Quaternion()
    # q2 = Quaternion()
    # batchSize = 10
    # input1 = np.empty(shape=[batchSize, 4], dtype=np.float)
    # input2 = np.empty(shape=[batchSize, 4], dtype=np.float)
    # for i in range(batchSize):
    #     q1.axisAngleInit(axis=[1, 1, 2], angle=0.5 + i * 0.01)
    #     q2.axisAngleInit(axis=[-1, 1, 1.5], angle=0.5 + i * 0.015)
    #     input1[i, :] = q1.getList()
    #     input2[i, :] = q2.getList()
    #
    # intensor1 = torch.as_tensor(input1, dtype=torch.float)
    # intensor2 = torch.as_tensor(input2, dtype=torch.float)
    # intensor1.requires_grad = True
    # intensor2.requires_grad = True
    #
    # matrix1 = quaternion2matrix2(intensor1)
    # # matrix1.register_hook(save_grad('matrix1'))
    # matrix2 = quaternion2matrix2(intensor2)
    # # matrix2.register_hook(save_grad('matrix2'))
    #
    # quaternion1 = matrix2quaternion2(matrix1)
    # # quaternion1.register_hook(save_grad('quaternion1'))
    # quaternion2 = matrix2quaternion2(matrix2)
    # # quaternion2.register_hook(save_grad('quaternion2'))
    # #
    # # loss = torch.mean(torch.abs(quaternion1 - quaternion2))
    # # loss.backward()
    # # print('matrix1\n', GRADS['matrix1'])
    # # print('matrix2\n', GRADS['matrix2'])
    # # print('quaternion1\n', GRADS['quaternion1'])
    # # print('quaternion2\n', GRADS['quaternion2'])
    #
    # print(intensor1[1])
    # print(matrix1[1])
    # q = Quaternion()
    # q.listInit(input1[1])
    # matrixRaw = np.array(q.getMatrix())
    # print(matrixRaw)
    # print(quaternion1[1])
    # q2 = Quaternion()
    # q2.rotationMatrixInit(matrixRaw)
    # print(q2.getList())
    #
    # print(1)

    # 4dof Loss test
    # preRotation = Quaternion(sqrt(0.5), 0, sqrt(0.5), 0)
    # testQ0 = Quaternion()
    # # testQ0.axisAngleInit(axis=(0, 0, 1), angle=dof4Angle)
    # DoF7Rotation = Quaternion()
    # sampleNum = 4
    # # DoF7Rotation.axisAngleInit(axis=(1, 0, 0), angle=dof7Angle)
    # dof4AngleList = np.arange(sampleNum + 1) / sampleNum * pi
    # dof7AngleList = np.arange(sampleNum + 1) / sampleNum * (pi / 2)
    # map = np.empty(shape=(sampleNum + 1, sampleNum + 1))
    # gripper = GripperSimipleCollision()
    # gripper.setOpeningWidth()
    # gripper.setOrientationAndTranslation()
    # matrixList = np.empty(((sampleNum + 1) * (sampleNum + 1), 9), dtype=np.float)
    # for i, dof4Angle in enumerate(dof4AngleList):
    #     for j, dof7Angle in enumerate(dof7AngleList):
    #         testQ0.axisAngleInit(axis=(0, 0, 1), angle=dof4Angle)
    #         DoF7Rotation.axisAngleInit(axis=(cos(dof4Angle + pi / 2), sin(dof4Angle + pi / 2), 0), angle=dof7Angle)
    #         # DoF7Rotation.axisAngleInit(axis=(0, 1, 0), angle=dof7Angle)
    #         q = DoF7Rotation * testQ0 * preRotation
    #         # print(q.getList())
    #         matrixq = q.getMatrix()
    #         dis = dis4DoF(matrixq, simpleL1=False)
    #         matrixList[i * (sampleNum + 1) + j] = matrixq
    #         # if j == 4:
    #         #     gripper.setOpeningWidth()
    #         #     gripper.setOrientationAndTranslation(orientation=q.getList())
    #         #     gripper.visualization()
    #         #     print(dof4Angle, dof7Angle, dis)
    #         # if j == 2:
    #         #     print(dof4Angle, dis)
    #         map[i, j] = dis

    # for i in map:
    #     for j in i:
    #         print(j)

    # quaternionTensor = torch.as_tensor(matrixList, dtype=torch.float)
    # quaternionTensor.requires_grad = True
    # loss = dis4DoF2(quaternionTensor)
    # print(loss)
    # loss.backward()


    # quaternion multiplication test
    # q1 = Quaternion()
    # q2 = Quaternion()
    # batchSize = 10
    # input1 = np.empty(shape=[batchSize, 4], dtype=np.float)
    # input2 = np.empty(shape=[batchSize, 4], dtype=np.float)
    # for i in range(batchSize):
    #     q1.axisAngleInit(axis=[1, 1, 2], angle=0.5 + i * 0.01)
    #     q2.axisAngleInit(axis=[-1, 1, 1.5], angle=0.5 + i * 0.015)
    #     input1[i, :] = q1.getList()
    #     input2[i, :] = q2.getList()
    #
    # intensor1 = torch.as_tensor(input1, dtype=torch.float)
    # intensor2 = torch.as_tensor(input2, dtype=torch.float)
    # a = quaternionMultiplication2(intensor1, intensor2)
    # print(a)

    # overall
    gripperCenter = np.array([0.02564412, -0.00385468, 0.07458372])
    gripperOrientation = np.array([0.3105045, - 0.18513325, 0.06153022, 0.93033683])
    # print(gripperOrientation)
    contact1, contact2 = getContactPoint(
        gripperLength=0.085,
        gripperCenter=gripperCenter,
        gripperOrientation=gripperOrientation
    )
    # gripper = GripperSimipleCollision()
    # gripper.setOpeningWidth()
    # gripper.setOrientationAndTranslation(orientation=gripperOrientation, translation=gripperCenter)
    # gripper.visualization()

    cosAngle = getCosAngle(gripperOrientation)
    contact1T = torch.as_tensor(np.stack((contact1, contact2)), dtype=torch.float)
    cosAngleT = torch.as_tensor([cosAngle, cosAngle], dtype=torch.float)
    centerT = torch.as_tensor(np.stack((gripperCenter, gripperCenter)), dtype=torch.float)
    rotationMatrixT = getOrientation2(contact1T, centerT, cosAngleT)
    # print(rotationMatrixT[0].reshape(3,3))
    # print(rotationMatrixT[1].reshape(3,3))
    # print(rotationMatrixT)
    quaternionT = matrix2quaternion2(rotationMatrixT)

    # print(quaternionT)
    #
    # gripper = GripperSimipleCollision()
    # gripper.setOpeningWidth()
    # gripper.setOrientationAndTranslation(orientation=quaternionT[0], translation=gripperCenter)
    # gripper.visualization()

    pass




