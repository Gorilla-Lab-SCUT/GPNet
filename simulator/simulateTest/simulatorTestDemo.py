import os
import numpy as np
import torch
import argparse
from simulateTest.AutoGraspShapeCoreUtil import AutoGraspUtil

# parser = argparse.ArgumentParser(description='ShapeNetSem Grasp testing')
# parser.add_argument('-t', '--testFile', default='prediction/500ntop10.txt', type=str, metavar='FILE', help='testFile path')
# # parser.add_argument('-n', '--samplePerObject', default=10, type=int, metavar='N', help='sample num per object')
# parser.add_argument('-p', '--processNum', default=18, type=int, metavar='N', help='process num using')
# parser.add_argument('-w', '--haveWidth', default=1, type=int, metavar='N', help='0 : no width ; 1 : have width')
# parser.add_argument('--gripperFile', default='/data/shapeNet/annotator2/parallel_simple.urdf', type=str, metavar='FILE', help='gripper file')
# parser.add_argument('--objMeshRoot', default='/data/shapeNet/urdf', type=str, metavar='PATH', help='obj mesh path')

parser = argparse.ArgumentParser(description='ShapeNetSem Grasp testing')
parser.add_argument('-t', '--testFile', default='gpnet_data/prediction/nms_poses_view0.txt', type=str, metavar='FILE', help='testFile path')
# parser.add_argument('-n', '--samplePerObject', default=10, type=int, metavar='N', help='sample num per object')
parser.add_argument('-p', '--processNum', default=10, type=int, metavar='N', help='process num using')
# parser.add_argument('-w', '--haveWidth', default=0, type=int, metavar='N', help='0 : no width ; 1 : have width')
parser.add_argument('-w', "--width", action="store_true", dest="width", default=False, help="turn on this param if test file contains width.")
parser.add_argument('--gripperFile', default='gpnet_data/gripper/parallel_simple.urdf', type=str, metavar='FILE', help='gripper file')
parser.add_argument('--objMeshRoot', default='gpnet_data/urdf', type=str, metavar='PATH', help='obj mesh path')


def getObjStatusAndAnnotation(testFile, haveWidth=False):
    with open(testFile, 'r') as testData:
        lines = testData.readlines()
        objIdList = []
        quaternionDict = {}
        centerDict = {}
        # 0: scaling    1~3: position   4~7: orientation    8: staticFrictionCoeff
        objId = 'invalid'
        objCounter = -1
        annotationCounter = -1
        for line in lines:
            # new object
            msg = line.strip()
            if len(msg.split(',')) < 2 :
                objId = msg.strip()
                # skip invalid
                # begin
                objCounter += 1
                objIdList.append(objId)
                quaternionDict[objId] = np.empty(shape=(0, 4), dtype=np.float)
                centerDict[objId] = np.empty(shape=(0, 3), dtype=np.float)
                annotationCounter = -1
            # read annotation
            else:
                # skip invalid object
                if objId == 'invalid':
                    continue
                # begin
                annotationCounter += 1
                pose = msg.split(',')
                # print(objId, annotationCounter)
                if haveWidth:
                    length = float(pose[0]) * 0.085     # arbitrary value, will not be used in AutoGrasp
                    length = length if length < 0.085 else 0.085
                    position = np.array([float(pose[1]), float(pose[2]), float(pose[3])])
                    quaternion = np.array([float(pose[4]), float(pose[5]), float(pose[6]), float(pose[7])])
                    # quaternion = quaternion[[1, 2, 3, 0]]
                else:
                    length = 0.000  # arbitrary value, will not be used in AutoGrasp
                    position = np.array([float(pose[0]), float(pose[1]), float(pose[2])])
                    quaternion = np.array([float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])])
                    # quaternion = quaternion[[1, 2, 3, 0]]
                # print(objCounter, annotationCounter)
                quaternionDict[objId] = np.concatenate((quaternionDict[objId], quaternion[None, :]), axis=0)
                centerDict[objId] = np.concatenate((centerDict[objId], position[None, :]), axis=0)
    return quaternionDict, centerDict, objIdList


if __name__ == "__main__":

    cfg = parser.parse_args()
    objMeshRoot = cfg.objMeshRoot
    processNum = cfg.processNum
    gripperFile = cfg.gripperFile
    haveWidth = cfg.width
    testInfoFile = cfg.testFile
    logFile = cfg.testFile[:-4] + '_log.csv'

    open(logFile, 'w').close()
    simulator = AutoGraspUtil()
    quaternionDict, centerDict, objIdList = getObjStatusAndAnnotation(testInfoFile, haveWidth)

    for objId in objIdList:
        q = quaternionDict[objId]
        c = centerDict[objId]
        simulator.addObject2(
            objId=objId,
            quaternion=q,
            translation=c
        )

    simulator.parallelSimulation(
        logFile=logFile,
        objMeshRoot=objMeshRoot,
        processNum=processNum,
        gripperFile=gripperFile,
    )

    annotationSuccessDict = simulator.getSuccessData(logFile=logFile)
    # print(top 10% 30% 50% 100%)
    top10, top30, top50, top100 = simulator.getStatistic(annotationSuccessDict)
    print('top10:\t', top10, '\ntop30:\t', top30, '\ntop50:\t', top50, '\ntop100:\t', top100)