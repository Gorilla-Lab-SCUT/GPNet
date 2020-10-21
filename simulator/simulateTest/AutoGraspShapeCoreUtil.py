import numpy as np
import torch
import gc
import pybullet
import os
from joblib import Parallel, delayed
from math import *

from simulateTest.AutoGraspSimpleShapeCore import AutoGraspSimple
from simulateTest.gripperPoseTransform import matrix2quaternion

class AutoGraspUtil(object):
    def __init__(self):
        self.__memoryInit()

    def __memoryInit(self):
        self.objIdList = []
        self.annotationDict = {}

    def __annotationMemoryReallocate(self):
        del self.annotationDict
        gc.collect()
        self.annotationDict = {}

    # objId: str (len <= 32)
    # rotation: torch.tensor [N, 3, 3]
    # translation: torch.tensor [N, 3]
    def addObject(self, objId, rotation, translation):
        # add object
        self.objIdList.append(objId)
        # add annotation
        annotationNum = rotation.shape[0]
        quaternion = matrix2quaternion(rotation.reshape(annotationNum, 9))
        # fit the pybullet environment
        quaternion = quaternion[:, [1, 2, 3, 0]]
        fakeLength = torch.zeros((annotationNum, 1))
        annotation = torch.cat((fakeLength, translation, quaternion), dim=1)
        self.annotationDict[objId] = annotation.numpy()

    # objId: str (len <= 32)
    # quaternion: ndarray [N, 4]
    # translation: ndarray [N, 3]
    def addObject2(self, objId, quaternion, translation):
        # add object
        self.objIdList.append(objId)

        # add annotation
        annotationNum = quaternion.shape[0]
        # fit the pybullet environment
        quaternion = quaternion[:, [1, 2, 3, 0]]
        fakeLength = np.zeros((annotationNum, 1))
        annotation = np.concatenate((fakeLength, translation, quaternion), axis=1)
        self.annotationDict[objId] = annotation

    # used to 
    def addObjectSplit(self, objId, quaternion, translation, logFile, objMeshRoot, processNum, gripperFile, splitLen=50):
        # add object
        self.objIdList.append(objId)
        objNum = len(self.objIdList)
        if objNum <= 1:
            # erase log file
            open(logFile, 'w').close()


        # add annotation
        annotationNum = quaternion.shape[0]
        # fit the pybullet environment
        quaternion = quaternion[:, [1, 2, 3, 0]]
        fakeLength = np.zeros((annotationNum, 1))
        annotation = np.concatenate((fakeLength, translation, quaternion), axis=1)
        self.annotationDict[objId] = annotation
        if (objNum % splitLen) == 0:
            splitIndex = objNum // splitLen
            runningObjIdList, \
            runningObjIndexList, \
            runningAnnotaionList, \
            runningAnnotaionIndexList = self.__getRunningListSplit(
                objIdListSplit=self.objIdList[(splitIndex - 1) * splitLen: splitIndex * splitLen],
                objIndexShift=(splitIndex - 1) * splitLen
            )
            objMeshRoot = objMeshRoot
            with Parallel(n_jobs=processNum, backend='multiprocessing') as parallel:
                parallel(delayed(AutoGraspUtil.testAnnotation)
                     (objId, objIndex, annotation, annotationIndex, gripperFile, logFile, objMeshRoot)
                     for (objId, objIndex, annotation, annotationIndex) in
                     zip(runningObjIdList, runningObjIndexList, runningAnnotaionList,
                         runningAnnotaionIndexList)
                )
            self.__annotationMemoryReallocate()

    def __getRunningListSplit(self, objIdListSplit, objIndexShift):
        runningObjIdList = []
        runningObjIndexList = []
        runningAnnotaionList = []
        runningAnnotaionIndexList = []

        for objIndex, objId in enumerate(objIdListSplit):
            for annotationIndex, annotation in enumerate(self.annotationDict[objId]):
                runningObjIdList.append(objId)
                runningObjIndexList.append(objIndex + objIndexShift)
                runningAnnotaionList.append(annotation)
                runningAnnotaionIndexList.append(annotationIndex)

        return runningObjIdList, runningObjIndexList, runningAnnotaionList, \
               runningAnnotaionIndexList


    def parallelSimulation(self, logFile, objMeshRoot, processNum, gripperFile):
        runningObjIdList, \
        runningObjIndexList, \
        runningAnnotaionList, \
        runningAnnotaionIndexList = self.__getRunningList()
        # erase log file
        open(logFile, 'w').close()
        objMeshRoot = objMeshRoot
        with Parallel(n_jobs=processNum, backend='multiprocessing') as parallel:
            parallel(delayed(AutoGraspUtil.testAnnotation)
                 (objId, objIndex, annotation, annotationIndex, gripperFile, logFile, objMeshRoot)
                 for (objId, objIndex, annotation, annotationIndex) in
                 zip(runningObjIdList, runningObjIndexList, runningAnnotaionList,
                     runningAnnotaionIndexList)
            )

    def __getRunningList(self):
        # annotationNum = annotationDict.shape[1]
        runningObjIdList = []
        runningObjIndexList = []
        runningAnnotaionList = []
        runningAnnotaionIndexList = []
        for objIndex, objId in enumerate(self.objIdList):
            for annotationIndex, annotation in enumerate(self.annotationDict[objId]):
                runningObjIdList.append(objId)
                runningObjIndexList.append(objIndex)
                runningAnnotaionList.append(annotation)
                runningAnnotaionIndexList.append(annotationIndex)

        return runningObjIdList, runningObjIndexList, runningAnnotaionList, \
               runningAnnotaionIndexList

    @staticmethod
    def testAnnotation(objId, objIndex, annotation, annotationIndex, gripperFile, logfile,
                       objMeshRoot):
        status = AutoGraspUtil.annotationSimulation(
            objId=objId,
            annotation=annotation,
            objMeshRoot=objMeshRoot,
            gripperFile=gripperFile,
        )
        print('objId\t', objId, '\tobjIdx\t', str(objIndex), '\tannotationIdx\t', str(annotationIndex), '\tstatus\t', str(status))
        simulatorParamStrList = [str(i) for i in annotation]
        logInfo = [objId, str(annotationIndex), str(status)]
        simulatorParamStrList = logInfo + simulatorParamStrList
        # print(simulatorParamStrList)
        simulatorParamStr = ','.join(simulatorParamStrList)
        # logLine = objId + ',' + str(annotationIndex) + ',' + str(status) + '\n'
        logLine = simulatorParamStr + '\n'
        # print(logLine)
        with open(logfile, 'a') as logger:
            logger.write(logLine)

    @staticmethod
    def getSuccessData(logFile):
        annotationSuccessDict = {}
        with open(logFile, 'r') as logReader:
            lines = logReader
            for line in lines:
                msg = line.strip()
                # objId, annotationId, status = msg.split(',')
                msgList = msg.split(',')
                objId, annotationId, status = msgList[0], int(msgList[1]), int(msgList[2])
                # initialize
                if objId not in annotationSuccessDict.keys():
                    annotationSuccessDict[objId] = np.full(shape=[annotationId + 1], fill_value=False)
                # # if success
                # if status == 0:
                #     logDict[objId][annotationId, gripperId] = True

                # if overflow (keep rank): reallocate memory
                if annotationId + 1 > len(annotationSuccessDict[objId]):
                    temp = np.full(shape=[annotationId + 1], fill_value=False)
                    temp[:len(annotationSuccessDict[objId])] = annotationSuccessDict[objId]
                    annotationSuccessDict[objId] = temp

                # if success
                if status == 0:
                    annotationSuccessDict[objId][annotationId] = True

        return annotationSuccessDict

    @staticmethod
    def getCollisionData(logFile):
        annotationSuccessDict = {}
        with open(logFile, 'r') as logReader:
            lines = logReader
            for line in lines:
                msg = line.strip()
                msgList = msg.split(',')
                objId, annotationId, status = msgList[0], int(msgList[1]), int(msgList[2])

                # initialize
                if objId not in annotationSuccessDict.keys():
                    annotationSuccessDict[objId] = np.full(shape=[annotationId + 1], fill_value=False)

                # # if success
                # if status == 0:
                #     logDict[objId][annotationId, gripperId] = True

                # if overflow (keep rank): reallocate memory
                if annotationId + 1 > len(annotationSuccessDict[objId]):
                    temp = np.full(shape=[annotationId + 1], fill_value=False)
                    temp[:len(annotationSuccessDict[objId])] = annotationSuccessDict[objId]
                    annotationSuccessDict[objId] = temp

                # if success
                if status == 2 or status == 1:
                    annotationSuccessDict[objId][annotationId] = True

        return annotationSuccessDict

    @staticmethod
    def annotationVisualization(logFile, objIdv, annotationIdv, objMeshRoot, gripperFile):
        with open(logFile, 'r') as logReader:
            lines = logReader
            for line in lines:
                msg = line.strip()
                # objId, annotationId, status = msg.split(',')
                msgList = msg.split(',')
                objId, annotationId, status = msgList[0], int(msgList[1]), int(msgList[2])
                simulatorParamStr = msgList[3:]
                simulatorParam = [float(i) for i in simulatorParamStr]
                annotation = np.array(simulatorParam[:8])
                if objId == objIdv and annotationId == annotationIdv:
                    status = AutoGraspUtil.annotationSimulation(
                        objId=objId,
                        annotation=annotation,
                        objMeshRoot=objMeshRoot,
                        gripperFile=gripperFile,
                        visual=True
                    )
                    return status

    @staticmethod
    def annotationSimulation(objId, annotation, objMeshRoot, gripperFile, visual=False):
        length = annotation[0]
        position = annotation[1:4]
        quaternion = annotation[4:8]
        autoGraspInstance = AutoGraspSimple(
            # clientId = gripperIndex,
            objectURDFFile=os.path.join(objMeshRoot, objId + ".urdf"),
            gripperLengthInit=length,
            gripperBasePosition=position,
            gripperURDFFile=gripperFile,
            gripperBaseOrientation=quaternion,
            serverMode=pybullet.GUI if visual else pybullet.DIRECT,
            # serverMode=pybullet.GUI,
        )
        return autoGraspInstance.startSimulation()


    @staticmethod
    def getStatistic(annotationSuccessDict):
        currentObjNum = len(annotationSuccessDict.keys())
        top10Success = np.empty(shape=currentObjNum, dtype=np.float)
        top30Success = np.empty(shape=currentObjNum, dtype=np.float)
        top50Success = np.empty(shape=currentObjNum, dtype=np.float)
        top100Success = np.empty(shape=currentObjNum, dtype=np.float)

        for index, annotationSuccess in enumerate(annotationSuccessDict.values()):
            annotationNum = len(annotationSuccess)
            # print(annotationSuccess)
            top10_num = int(0.1 * annotationNum)
            if top10_num == 0:
                top10_num = 1
            top10Success[index] = np.sum(
                annotationSuccess[:top10_num]) / top10_num
            top30_num = round(0.3 * annotationNum)
            if top30_num == 0:
                top30_num = 1
            top30Success[index] = np.sum(
                annotationSuccess[:top30_num]) / top30_num
            top50_num = round(0.5 * annotationNum)
            if top50_num == 0:
                top50_num = 1
            top50Success[index] = np.sum(
                annotationSuccess[:top50_num]) / top50_num
            top100Success[index] = np.sum(annotationSuccess) / annotationNum
        return top10Success.mean(), top30Success.mean(), top50Success.mean(), top100Success.mean()

    @staticmethod
    def listSplit(listToSplit, splitLen=48000*19*50):
        return [listToSplit[splitLen * i: splitLen * (i + 1)] for i in range(len(listToSplit) // splitLen + 1)]