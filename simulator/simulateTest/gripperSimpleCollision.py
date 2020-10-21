import trimesh
from simulateTest.Quaternion import Quaternion
from copy import deepcopy
from simulateTest.visualization import getWorldMeshList
import numpy as np
import math

# for a gripper :
# firstly init :    gripper = GripperSimipleCollision()     (you can take one gripper instance and do multiple settings)
# then set width :  gripper.setOpeningWidth(x)
# lastly set translation and orientaion:    gripper.setOrientationAndTranslation(xxxx, xxx)

class GripperSimipleCollision(object):
    def __init__(self):
        self.__ROOT = trimesh.primitives.Box(
            extents=[0.075, 0.075, 0.090]
        )
        self.__ROOT.apply_translation([0, 0, 0.0691])

        self.__FINGER1 = trimesh.primitives.Box(
            extents=[0.030, 0.022, 0.048]
        )
        self.__FINGER1.apply_translation([0.0575, 0, 0])

        self.__FINGER2 = trimesh.primitives.Box(
            extents=[0.030, 0.022, 0.048]
        )
        self.__FINGER2.apply_translation([-0.0575, 0, 0])

    def setOpeningWidth(self, width=0.083, margin=0.002):
        self.root = deepcopy(self.__ROOT)
        self.finger1 = deepcopy(self.__FINGER1)
        self.finger2 = deepcopy(self.__FINGER2)
        self.finger1.apply_translation([-(0.0425 - width / 2 - margin / 2), 0, 0])
        self.finger2.apply_translation([0.0425 - width / 2 - margin / 2, 0, 0])

    # orientation : [w, x, y, z]    (quaternion)
    def setOrientationAndTranslation(self, orientation=(1, 0, 0, 0), translation=(0, 0, 0)):
        quaternion = Quaternion()
        quaternion.listInit(orientation)
        rotationMatrix = np.mat(quaternion.getMatrix(is4x4=True)).reshape([4,4])
        self.root.apply_transform(rotationMatrix)
        self.finger1.apply_transform(rotationMatrix)
        self.finger2.apply_transform(rotationMatrix)

        self.root.apply_translation(translation)
        self.finger1.apply_translation(translation)
        self.finger2.apply_translation(translation)

    # orientation : rotation matrix 4x4
    def setOrientationAndTranslation2(self, orientation, translation=(0, 0, 0)):
        self.root.apply_transform(orientation)
        self.finger1.apply_transform(orientation)
        self.finger2.apply_transform(orientation)

        self.root.apply_translation(translation)
        self.finger1.apply_translation(translation)
        self.finger2.apply_translation(translation)


    # def isCollideWithObject(self, obj):
    #     collisionManager = trimesh.collision.CollisionManager()
    #     collisionManager.add_object(name='obj', mesh=obj)
    #     collisionManager.add_object(name='root', mesh=self.root)
    #     collisionManager.add_object(name='finger1', mesh=self.finger1)
    #     collisionManager.add_object(name='finger2', mesh=self.finger2)
    #     flag = collisionManager.in_collision_internal()
    #     return flag

    def isCollideWithGround(self):
        for vertex in self.root.vertices:
            if vertex[2] < 0:
                return True
        for vertex in self.finger1.vertices:
            if vertex[2] < 0:
                return True
        for vertex in self.finger2.vertices:
            if vertex[2] < 0:
                return True
        return False

    # obj: trimesh obj (mesh)
    def visualization(self, obj=None):
        gripperColor = [128, 255, 255, 200]
        objColor = [255, 128, 255, 200]
        self.root.visual.face_colors = gripperColor
        self.finger1.visual.face_colors = gripperColor
        self.finger2.visual.face_colors = gripperColor
        worldMeshList = getWorldMeshList()
        worldMeshList.append(self.root)
        worldMeshList.append(self.finger1)
        worldMeshList.append(self.finger2)
        if not obj is None:
            obj.visual.face_colors = objColor
            worldMeshList.append(obj)
        scene = trimesh.Scene(worldMeshList)
        scene.show()


if __name__ == "__main__":
    gripper = GripperSimipleCollision()
    gripper.setOpeningWidth()
    # q1 = Quaternion(w=math.sqrt(0.5), x=0, y=math.sqrt(0.5), z=0)
    # q2 = Quaternion(w=math.sqrt(0.5), x=0, y=0, z=math.sqrt(0.5))
    # q = q2 * q1
    #
    # gripper.setOrientationAndTranslation(translation=(0, 0, -0.024), orientation=q.getList())
    gripper.visualization()

    # q = Quaternion()
    # q.axisAngleInit(axis=[1, 1, 1], angle=0.8)
    # gripper.setOpeningWidth(0.085)
    # gripper.setOrientationAndTranslation(orientation=q.getList(), translation=(0, 0, 0.25))      # orientation=(0, 0, 0.707106781,  0.707106781),
    # obj = trimesh.load_mesh("transformed_mesh/1c73c8ff52c31667c8724d5673a063a6.obj")
    # gripper.visualization(obj)
    # print(gripper.isCollideWithObject(obj))
    # [-3.31416396e-001, -4.99987225e-001, 8.00109959e-001, 3.65608578e-322],
    # [-1.91336827e-001, 8.66032779e-001, 4.61927964e-001, 3.65608578e-322],
    # [-9.23879533e-001, 0.00000000e+000, 3.82683432e-001, 3.65608578e-322],
    # [3.65608578e-322, 3.65608578e-322, 3.65608578e-322, 1.00000000e+000]
    # rotationMatrix = np.matrix([
    #     [0., 1., -0., 0.],
    #     [-0.38268343, 0., 0.92387953, 0.],
    #     [-0.92387953, 0., 0.38268343, 0.],
    #     [0., 0., 0., 1.]
    # ])
    # gripper.setOpeningWidth(0.05)
    # gripper.setOrientationAndTranslation2(orientation=rotationMatrix, translation=(0.1, 0.1, 0.1))
    # gripper.visualization()
    # gripper.setOpeningWidth()
    # gripper.setOrientationAndTranslation()

    finger1Points, _ = trimesh.sample.sample_surface_even(gripper.finger1, 40)
    print('finger1:\t', len(finger1Points))

    rootPoints, _ = trimesh.sample.sample_surface_even(gripper.root, 300)
    for rootPoint in rootPoints:
        if rootPoint[0] < 0:
            finger1Points = np.vstack((finger1Points, rootPoint))
    # print(finger1Points)
    print('add root:\t', len(finger1Points))
    negativeYsidePoints = finger1Points.copy()
    negativeYsidePoints[:, 0] = - negativeYsidePoints[:, 0]
    points = np.vstack((finger1Points, negativeYsidePoints))
    print('image:\t', len(points))
    points = np.vstack((np.array(gripper.finger2.vertices), points))
    print('add finger2 vertices:\t', len(points))
    points = np.vstack((np.array(gripper.finger1.vertices), points))
    print('add finger1 vertices:\t', len(points))
    points = np.vstack((np.array(gripper.root.vertices), points))
    print('add root vertices:\t', len(points))
    print(points)

    np.save('gs.npy', points)

    # print(gripper.finger1.vertices)
    # a = np.array(gripper.finger1.vertices)
