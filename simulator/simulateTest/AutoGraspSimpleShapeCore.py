import pybullet
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import numpy as np
import math
#
# from Antipodal import *
# from FerrariCannyL1 import *
# from coordinateTransformation import *
MU = 0.5
SPINNING_FRICTION = 0.1
ROLLING_FRICTION = 0.1
MAXIMUM_SIMULATED_STEP = 15000
COLLISION_DETECTION_INDENTATION_DEPTH=0.002
FINGER_REACH_INDENTATION_DEPTH=0.003
# EXTRA_CLOSING = 0.002


class AutoGraspSimple(object):
    def __init__(self, objectURDFFile, gripperURDFFile, gripperLengthInit, gripperBasePosition, gripperBaseOrientation, serverMode=pybullet.GUI, mu=MU, spinningFriction=SPINNING_FRICTION, rollingFriction=ROLLING_FRICTION):
        self.serverMode = serverMode

        self.objectURDFFile = objectURDFFile

        self.gripperURDFFile = gripperURDFFile
        self.gripperBasePosition = gripperBasePosition
        self.gripperBaseOrientation = gripperBaseOrientation

        self.mu = mu
        self.spinningFriction = spinningFriction
        self.rollingFriction = rollingFriction

        self.gripperLengthInit = gripperLengthInit
        self.gripperLengthList = self.__getGripperLengthList()

        self.SUCCESS = 0
        self.COLLIDE_WITH_GROUND = 1
        self.COLLIDE_WITH_OBJECT = 2
        self.UNTOUCHED = 3
        self.INCORRECT_CONTACT = 4
        self.OBJECT_FALLEN = 5
        self.TIME_OUT = 6

    def startSimulation(self):
        self.__initializeTheWorld()
        self.objectID = self.__loadObject()
        # objectPosition, objectOrientation = pybullet.getBasePositionAndOrientation(self.objectID)
        # a = pybullet.getDynamicsInfo(self.objectID, -1)
        self.gripperID = self.__loadGripper()
        self.__gripperControlInit()
        pybullet.changeDynamics(
            self.objectID,
            -1,
            lateralFriction=self.mu,
            spinningFriction=self.spinningFriction,
            rollingFriction=self.rollingFriction
        )
        pybullet.changeDynamics(
            self.gripperID,
            self.robotiq_85_left_finger_tip_joint_index,
            lateralFriction=self.mu,
            spinningFriction=self.spinningFriction,
            rollingFriction=self.rollingFriction
        )
        pybullet.changeDynamics(
            self.gripperID,
            self.robotiq_85_right_finger_tip_joint_index,
            lateralFriction=self.mu,
            spinningFriction=self.spinningFriction,
            rollingFriction=self.rollingFriction
        )

        pybullet.stepSimulation()
        if self.__isCollide(self.gripperID, self.planeID, - COLLISION_DETECTION_INDENTATION_DEPTH):
            pybullet.disconnect()
            return self.COLLIDE_WITH_GROUND

        if self.__isCollide(self.gripperID, self.objectID, - COLLISION_DETECTION_INDENTATION_DEPTH):
            pybullet.disconnect()
            return self.COLLIDE_WITH_OBJECT

        untouched = True
        stableGripperLength = 0.085
        try:
            for gripperLength in self.gripperLengthList:
                self.__gripperClosing(gripperLength=gripperLength)
                # contactListLeft = pybullet.getContactPoints(bodyA=self.gripperID,
                #                                             bodyB=self.objectID,
                #                                             linkIndexA=self.robotiq_85_left_finger_tip_joint_index)
                # contactListRight = pybullet.getContactPoints(bodyA=self.gripperID,
                #                                              bodyB=self.objectID,
                #                                              linkIndexA=self.robotiq_85_right_finger_tip_joint_index)
                # if (len(contactListLeft) >=1) and (len(contactListRight) >= 1) and (len(contactListLeft) + len(contactListRight) >= 3):
                #     untouched = False
                #
                #     self.__gripperClosing(gripperLength=gripperLength - EXTRA_CLOSING)
                #     stableGripperLenth = gripperLength
                #     break
                reachFlag = self.__fingerReach(
                    gripperId=self.gripperID,
                    objectId=self.objectID,
                    finger1LinkId=self.robotiq_85_left_finger_tip_joint_index,
                    finger2LinkId=self.robotiq_85_right_finger_tip_joint_index,
                    indentationDepth=- FINGER_REACH_INDENTATION_DEPTH
                )
                if reachFlag:
                    stableGripperLength = gripperLength
                    untouched = False
                    break

            if untouched:
                pybullet.disconnect()
                return self.UNTOUCHED
            self.gripperLengthInit = stableGripperLength
            self.__gripperLifting(stableGripperLength)
        except RuntimeError:
            pybullet.disconnect()
            return self.TIME_OUT

        contactListLeft = pybullet.getContactPoints(bodyA=self.gripperID,
                                                    bodyB=self.objectID,
                                                    linkIndexA=self.robotiq_85_left_finger_tip_joint_index)
        contactListRight = pybullet.getContactPoints(bodyA=self.gripperID,
                                                     bodyB=self.objectID,
                                                     linkIndexA=self.robotiq_85_right_finger_tip_joint_index)
        if len(contactListLeft) >=1 and len(contactListRight) >= 1:
            pybullet.disconnect()
            return self.SUCCESS
        else:
            pybullet.disconnect()
            return self.OBJECT_FALLEN

    def __getGripperLengthList(self):
        temp = [0.085 - x * 0.001 for x in range(85)]
        gripperLengthList = []
        for item in temp:
            if (item > 0) and (item < 0.085):
                gripperLengthList.append(item)
        return gripperLengthList

    def __initializeTheWorld(self):
        pybullet.connect(self.serverMode)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(0, 0, -9.8)
        self.planeID = pybullet.loadURDF("plane.urdf")

    def __loadObject(self):
        return pybullet.loadURDF(fileName=self.objectURDFFile)

    # return
    #   objectPosition
    #   objectOrientation
    def __getObjectState(self):
        objectInfo = pybullet.getBasePositionAndOrientation(self.objectID)
        return objectInfo[0], objectInfo[1]

    def __getLinkPositionAndOrientation(self, linkId):
        linkInfo = pybullet.getLinkState(self.gripperID, linkId)
        return linkInfo[0], linkInfo[1]

    def __stablizedFlag(self, currentObjectPosition, currentObjectOrientation, previousObjectPosition, previousObjectOrientation, threshold=1e-5):
        previousObjectInfo = np.hstack((previousObjectPosition, previousObjectOrientation))
        currentObjectInfo = np.hstack((currentObjectPosition, currentObjectOrientation))
        return np.linalg.norm(previousObjectInfo - currentObjectInfo) < threshold

    def __loadGripper(self):
        return pybullet.loadURDF(fileName=self.gripperURDFFile,
                                 basePosition=self.gripperBasePosition,
                                 baseOrientation=self.gripperBaseOrientation)

    def __gripperControlInit(self):
        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        numJoints = pybullet.getNumJoints(self.gripperID)
        jointInfo = namedtuple("jointInfo",
                               ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity"])

        self.joints = AttrDict()
        self.dummy_center_indicator_link_index = 0

        # get jointInfo and index of dummy_center_indicator_link
        for i in range(numJoints):
            info = pybullet.getJointInfo(self.gripperID, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            singleInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce,
                                   jointMaxVelocity)
            self.joints[singleInfo.name] = singleInfo
            # register index of dummy center link
            if jointName == "gripper_roll":
                self.dummy_center_indicator_link_index = i
            if jointName == "box_left_joint":
                self.robotiq_85_left_finger_tip_joint_index = i
            if jointName == "box_right_joint":
                self.robotiq_85_right_finger_tip_joint_index = i

        self.gripper_main_control_joint_name = "box_left_joint"
        self.mimic_joint_name = ["box_right_joint"]
        self.mimic_multiplier = [1]

        self.position_control_joint_name = [
            "center_x",
            "center_y",
            "center_z",
            "gripper_roll",
            "gripper_pitch",
            "gripper_yaw"
        ]

    def __isCollide(self, robotID1, robotID2, indentationDepth):
        contactList = pybullet.getContactPoints(robotID1, robotID2)
        for contact in contactList:
            if contact[8] < indentationDepth:
                return True
            # print('contact distance\t', contact[8]) # contact Distance
        return False

    def __fingerReach(self, gripperId, objectId, finger1LinkId, finger2LinkId, indentationDepth):
        contactFinger1 = pybullet.getContactPoints(
            bodyA=gripperId,
            bodyB=objectId,
            linkIndexA=finger1LinkId
        )
        contactFinger2 = pybullet.getContactPoints(
            bodyA=gripperId,
            bodyB=objectId,
            linkIndexA=finger2LinkId
        )
        if len(contactFinger1) < 1 or len(contactFinger2) < 1:
            return False
        contactList = contactFinger1 + contactFinger2
        for contact in contactList:
            if contact[8] < indentationDepth:
                return True
        return False

    def __gripperClosing(self, gripperLength):
        # gripper control
        gripper_opening_para = 0.0415 - gripperLength / 2
        leftTipLinkPosition, leftTipLinkOrientation = self.__getLinkPositionAndOrientation(self.robotiq_85_left_finger_tip_joint_index)
        rightTipLinkPosition, rightTipLinkOrientation = self.__getLinkPositionAndOrientation(self.robotiq_85_right_finger_tip_joint_index)
        simulatedStep = 0
        while(1):
            jointPose = pybullet.calculateInverseKinematics(self.gripperID,
                                                            self.dummy_center_indicator_link_index,
                                                            self.gripperBasePosition,
                                                            self.gripperBaseOrientation)
            for jointName in self.joints:
                if jointName in self.position_control_joint_name:
                    joint = self.joints[jointName]
                    pybullet.setJointMotorControl2(self.gripperID, joint.id, pybullet.POSITION_CONTROL,
                                                   targetPosition=jointPose[joint.id], force=joint.maxForce,
                                                   maxVelocity=joint.maxVelocity)

            pybullet.setJointMotorControl2(self.gripperID,
                                           self.joints[self.gripper_main_control_joint_name].id,
                                           pybullet.POSITION_CONTROL,
                                           targetPosition=gripper_opening_para,
                                           force=self.joints[self.gripper_main_control_joint_name].maxForce,
                                           maxVelocity=self.joints[self.gripper_main_control_joint_name].maxVelocity)
            # print(self.joints[self.gripper_main_control_joint_name].maxForce)
            for i in range(len(self.mimic_joint_name)):
                joint = self.joints[self.mimic_joint_name[i]]
                pybullet.setJointMotorControl2(self.gripperID, joint.id, pybullet.POSITION_CONTROL,
                                               targetPosition=gripper_opening_para * self.mimic_multiplier[i],
                                               force=joint.maxForce,
                                               maxVelocity=joint.maxVelocity)
                # print(joint.maxForce)
            simulatedStep = simulatedStep + 1
            if simulatedStep > MAXIMUM_SIMULATED_STEP:
                raise RuntimeError()
            pybullet.stepSimulation()

            currentLeftTipLinkPosition, currentLeftTipLinkOrientation = self.__getLinkPositionAndOrientation(self.robotiq_85_left_finger_tip_joint_index)
            currentRightTipLinkPosition, currentRightTipLinkOrientation = self.__getLinkPositionAndOrientation(self.robotiq_85_right_finger_tip_joint_index)

            leftStablizedFlag = self.__stablizedFlag(currentLeftTipLinkPosition, currentLeftTipLinkOrientation, leftTipLinkPosition, leftTipLinkOrientation, threshold=1e-4)
            rightStablizedFlag = self.__stablizedFlag(currentRightTipLinkPosition, currentRightTipLinkOrientation, rightTipLinkPosition, rightTipLinkOrientation, threshold=1e-4)
            if leftStablizedFlag and rightStablizedFlag:
                return True
            else:
                leftTipLinkPosition, leftTipLinkOrientation = self.__getLinkPositionAndOrientation(self.robotiq_85_left_finger_tip_joint_index)
                rightTipLinkPosition, rightTipLinkOrientation = self.__getLinkPositionAndOrientation(self.robotiq_85_right_finger_tip_joint_index)

    def __gripperLifting(self, gripperLength):
        basePosition = np.array(self.gripperBasePosition)
        # basePosition = self.gripperBasePosition.copy()
        basePosition[2] = basePosition[2] + 0.05
        # gripper control
        gripper_opening_para = 0.0415 - gripperLength / 2
        simulatedStep = 0
        while (1):
            jointPose = pybullet.calculateInverseKinematics(self.gripperID,
                                                            self.dummy_center_indicator_link_index,
                                                            basePosition,
                                                            self.gripperBaseOrientation)
            for jointName in self.joints:
                if jointName in self.position_control_joint_name:
                    joint = self.joints[jointName]
                    pybullet.setJointMotorControl2(self.gripperID, joint.id, pybullet.POSITION_CONTROL,
                                                   targetPosition=jointPose[joint.id], force=joint.maxForce,
                                                   maxVelocity=joint.maxVelocity)

            pybullet.setJointMotorControl2(self.gripperID,
                                           self.joints[self.gripper_main_control_joint_name].id,
                                           pybullet.POSITION_CONTROL,
                                           targetPosition=gripper_opening_para,
                                           force=self.joints[self.gripper_main_control_joint_name].maxForce,
                                           maxVelocity=self.joints[self.gripper_main_control_joint_name].maxVelocity)
            # print(self.joints[self.gripper_main_control_joint_name].maxForce)
            for i in range(len(self.mimic_joint_name)):
                joint = self.joints[self.mimic_joint_name[i]]
                pybullet.setJointMotorControl2(self.gripperID, joint.id, pybullet.POSITION_CONTROL,
                                               targetPosition=gripper_opening_para * self.mimic_multiplier[i],
                                               force=joint.maxForce,
                                               maxVelocity=joint.maxVelocity)

                # print(joint.maxForce)
            pybullet.stepSimulation()
            simulatedStep = simulatedStep + 1
            if simulatedStep > MAXIMUM_SIMULATED_STEP:
                raise RuntimeError()
            currentBasePosition, _ = self.__getLinkPositionAndOrientation(self.dummy_center_indicator_link_index)
            if math.fabs(currentBasePosition[2] - basePosition[2]) < 1e-4:
                return True


if __name__ == "__main__":
    pass