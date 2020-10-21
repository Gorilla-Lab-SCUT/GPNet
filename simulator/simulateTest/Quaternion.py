from math import *
import numpy as np
# import numbers


def getNorm(vector):
    sum = 0
    for item in vector:
        sum += item * item
    return sqrt(sum)


def quaternion2Matrix(q, is4x4=False):
    r = Quaternion()
    r.listInit(q)
    matrix = r.getMatrix(is4x4=is4x4)
    matrix = np.array(matrix).reshape(4, 4) if is4x4 else np.array(matrix).reshape(3, 3)
    return matrix


# m : np.array of size [3,3] or [4,4]
def matrix2quaternion(m, is4x4=True):
    r = Quaternion()
    if is4x4:
        m = m[:3, :3]
    r.rotationMatrixInit(m.reshape(-1))
    return r.getList()



def crossProduct3d(vector1, vector2):
    x1 = vector1[0]
    x2 = vector2[0]
    y1 = vector1[1]
    y2 = vector2[1]
    z1 = vector1[2]
    z2 = vector2[2]
    return y1 * z2 - y2 * z1, x2 * z1 - x1 * z2, x1 * y2 - x2 * y1


def coordinateTransformation(gripperCenter, gripperOrientation, cameraTranslation, cameraRotation):
    rotation = Quaternion(w=cameraRotation[0], x=cameraRotation[1], y=cameraRotation[2], z=cameraRotation[3])
    gripperCenterAfterTranslation = [gripperCenter[0] - cameraTranslation[0],
                                     gripperCenter[1] - cameraTranslation[1],
                                     gripperCenter[2] - cameraTranslation[2]]
    gripperCenterAfterRotation = rotation.conjugate().applyRotation(gripperCenterAfterTranslation)
    gripperQuaternion = Quaternion()
    gripperQuaternion.listInit(gripperOrientation)
    gripperOrientationAfterTranform = rotation.conjugate() * gripperQuaternion
    return gripperCenterAfterRotation, gripperOrientationAfterTranform.getList()


def pointInverseTransformation(point, coordinateCenter, coordinateOrientation, isTranslationFirst=False):
    rotation = Quaternion(
        w=coordinateOrientation[0],
        x=coordinateOrientation[1],
        y=coordinateOrientation[2],
        z=coordinateOrientation[3]
    )
    # do reverse transformation
    if isTranslationFirst:
        pointAfterTranslation = [
            point[0] - coordinateCenter[0],
            point[1] - coordinateCenter[1],
            point[2] - coordinateCenter[2]]
        pointAfterTransformation = rotation.conjugate().applyRotation(pointAfterTranslation)
    else:
        pointAfterRotation = rotation.conjugate().applyRotation(point)
        pointAfterTransformation = [
            pointAfterRotation[0] - coordinateCenter[0],
            pointAfterRotation[1] - coordinateCenter[1],
            pointAfterRotation[1] - coordinateCenter[1],
        ]

    return pointAfterTransformation


# def coordinateTransformation2(gripperCenter, gripperOrientation, cameraTranslation, cameraRotation):
#     rotation = Quaternion(w=cameraRotation[0], x=cameraRotation[1], y=cameraRotation[2], z=cameraRotation[3])
#     gripperCenterAfterRotation = rotation.applyRotation(gripperCenter)
#     gripperCenterAfterTranslation = [gripperCenterAfterRotation[0] - cameraTranslation[0],
#                                      gripperCenterAfterRotation[1] - cameraTranslation[1],
#                                      gripperCenterAfterRotation[2] - cameraTranslation[2]]
#     gripperQuaternion = Quaternion()
#     gripperQuaternion.listInit(gripperOrientation)
#     gripperOrientationAfterTranform = gripperQuaternion * rotation
#     return gripperCenterAfterTranslation, gripperOrientationAfterTranform.getList()


def invCoordinateTransformation(gripperCenterAfterTransform, gripperOrientationAfterTransform, cameraTranslation, cameraRotation):
    rotation = Quaternion(w=cameraRotation[0], x=cameraRotation[1], y=cameraRotation[2], z=cameraRotation[3])
    gripperCenterAfterRotation = rotation.applyRotation(gripperCenterAfterTransform)
    gripperCenter = [gripperCenterAfterRotation[0] + cameraTranslation[0],
                     gripperCenterAfterRotation[1] + cameraTranslation[1],
                     gripperCenterAfterRotation[2] + cameraTranslation[2]]

    gripperQuaternionAfterTransform = Quaternion()
    gripperQuaternionAfterTransform.listInit(gripperOrientationAfterTransform)
    gripperOrientation = rotation * gripperQuaternionAfterTransform
    return gripperCenter, gripperOrientation.getList()


class Quaternion(object):
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    # axis: list of size [3]
    # angle: float
    def axisAngleInit(self, axis, angle):
        self.w = cos(angle / 2)
        norm = getNorm(axis)
        self.x = sin(angle / 2) * axis[0] / norm
        self.y = sin(angle / 2) * axis[1] / norm
        self.z = sin(angle / 2) * axis[2] / norm

    # position: list of size [3]
    def pureQuaternionInit(self, position):
        self.w = 0
        self.x, self.y, self.z = position

    # l: list of size 4
    def listInit(self, l):
        self.w = l[0]
        self.x = l[1]
        self.y = l[2]
        self.z = l[3]

    # matrix: list of size [9]
    def rotationMatrixInit(self, matrix33):
        fourXSquaredMinus1 = matrix33[0] - matrix33[4] - matrix33[8]
        fourYSquaredMinus1 = matrix33[4] - matrix33[0] - matrix33[8]
        fourZSquaredMinus1 = matrix33[8] - matrix33[0] - matrix33[4]
        fourWSquaredMinus1 = matrix33[0] + matrix33[4] + matrix33[8]
        biggestIndex = 0
        fourBiggestSquaredMinus1 = fourWSquaredMinus1
        if (fourXSquaredMinus1 > fourBiggestSquaredMinus1):
            fourBiggestSquaredMinus1 = fourXSquaredMinus1
            biggestIndex = 1

        if (fourYSquaredMinus1 > fourBiggestSquaredMinus1):
            fourBiggestSquaredMinus1 = fourYSquaredMinus1
            biggestIndex = 2
        if (fourZSquaredMinus1 > fourBiggestSquaredMinus1):
            fourBiggestSquaredMinus1 = fourZSquaredMinus1
            biggestIndex = 3

        biggestVal = sqrt(fourBiggestSquaredMinus1 + 1) * 0.5
        mult = 0.25 / biggestVal

        if biggestIndex == 0:
            # return Quaternion(biggestVal, (matrix33[5] - matrix33[7]) * mult, (matrix33[6] - matrix33[2]) * mult, (matrix33[1] - matrix33[3]) * mult)
            self.w = biggestVal
            self.x = (matrix33[7] - matrix33[5]) * mult
            self.y = (matrix33[2] - matrix33[6]) * mult
            self.z = (matrix33[3] - matrix33[1]) * mult
        if biggestIndex == 1:
            # return Quaternion((matrix33[5] - matrix33[7]) * mult, biggestVal, (matrix33[1] + matrix33[3]) * mult, (matrix33[6] + matrix33[2]) * mult)
            self.w = (matrix33[7] - matrix33[5]) * mult
            self.x = biggestVal
            self.y = (matrix33[3] + matrix33[1]) * mult
            self.z = (matrix33[2] + matrix33[6]) * mult
        if biggestIndex == 2:
            # return Quaternion((matrix33[6] - matrix33[2]) * mult, (matrix33[1] + matrix33[3]) * mult, biggestVal, (matrix33[5] + matrix33[7]) * mult)
            self.w = (matrix33[2] - matrix33[6]) * mult
            self.x = (matrix33[3] + matrix33[1]) * mult
            self.y = biggestVal
            self.z = (matrix33[7] + matrix33[5]) * mult


        if biggestIndex == 3:
            # return Quaternion((matrix33[1] - matrix33[3]) * mult, (matrix33[6] + matrix33[2]) * mult, (matrix33[5] + matrix33[7]) * mult, biggestVal)
            self.w = (matrix33[3] - matrix33[1]) * mult
            self.x = (matrix33[2] + matrix33[6]) * mult
            self.y = (matrix33[7] + matrix33[5]) * mult
            self.z = biggestVal


    def getPositionFromPureQuaternion(self):
        return self.x, self.y, self.z


    # q and -q represent the same rotation
    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __add__(self, quaternion2):
        if isinstance(quaternion2, Quaternion):
            return Quaternion(self.w + quaternion2.w, self.x + quaternion2.x, self.y + quaternion2.y, self.z + quaternion2.z)
        else:
            return NotImplemented

    # multiplication, "other" on the right side
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        elif isinstance(other, float) or isinstance(other, int):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        else:
            return NotImplemented

    # multiplication, "other" on the left side
    def __rmul__(self, other):
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.x * other.w + self.w * other.x + self.z * other.y - self.y * other.z
            y = self.y * other.w - self.z * other.x + self.w * other.y + self.x * other.z
            z = self.z * other.w + self.y * other.x - self.x * other.y + self.w * other.z
            return Quaternion(w, x, y, z)
        elif isinstance(other, float) or isinstance(other, int):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        else:
            return NotImplemented

    #  quaternion2 : Quaternion
    def innerProduct(self, quaternion2):
        return self.w * quaternion2.w + self.x * quaternion2.x + self.y * quaternion2.y + self.z * quaternion2.z

    # def leftProduct(self, quaternion2):
    #     w = self.w * quaternion2.w - self.x * quaternion2.x - self.y * quaternion2.y - self.z * quaternion2.z
    #     x = self.x * quaternion2.w + self.w * quaternion2.x + self.z * quaternion2.y - self.y * quaternion2.z
    #     y = self.y * quaternion2.w - self.z * quaternion2.x + self.w * quaternion2.y + self.x * quaternion2.z
    #     z = self.z * quaternion2.w + self.y * quaternion2.x - self.x * quaternion2.y + self.w * quaternion2.z
    #     return Quaternion(w, x, y, z)
    #
    # def rightProduct(self, quaternion2):
    #     w = self.w * quaternion2.w - self.x * quaternion2.x - self.y * quaternion2.y - self.z * quaternion2.z
    #     x = self.w * quaternion2.x + self.x * quaternion2.w + self.y * quaternion2.z - self.z * quaternion2.y
    #     y = self.w * quaternion2.y - self.x * quaternion2.z + self.y * quaternion2.w + self.z * quaternion2.x
    #     z = self.w * quaternion2.z + self.x * quaternion2.y - self.y * quaternion2.x + self.z * quaternion2.w
    #     return Quaternion(w, x, y, z)

    # def scalarProduct(self, alpha):
    #     return Quaternion(self.w * alpha, self.x * alpha, self.y * alpha, self.z * alpha)

    # get the inverse of quaternion (q^-1 = q*)
    def conjugate(self):
        return Quaternion(w=self.w, x=-self.x, y=-self.y, z=-self.z)

    # # position: list of size [3]
    # def applyTransform(self, position):
    #     positionQuaternion = Quaternion()
    #     positionQuaternion.pureQuaternionInit(position)
    #     positionAfterTransformQuaternion = self.rightProduct(positionQuaternion).rightProduct(self.conjugate())
    #     return positionAfterTransformQuaternion.getPositionFromPureQuaternion()

    # position: list of size [3]
    def applyRotation(self, position):
        positionQuaternion = Quaternion()
        positionQuaternion.pureQuaternionInit(position)
        positionAfterTransformQuaternion = self * positionQuaternion * self.conjugate()
        return positionAfterTransformQuaternion.getPositionFromPureQuaternion()

    # get the difference of two rotation (angle difference)
    def __sub__(self, other):
        if isinstance(other, Quaternion):
            innerProduct = self.innerProduct(other)
            return 2 * acos(innerProduct)
        else:
            return NotImplemented
        # relativeRotation = self.leftProduct(quaternion2.conjugate())
        # angle = 2 * acos(relativeRotation.w)
        # return angle

    # normalized linear interpolation
    # t : float in [0, 1]
    def Nlerp(self, quaternion2, t):
        # due to double cover problem. shortest path interpolation
        if self.innerProduct(quaternion2) < 0:
            quaternion2 = - quaternion2

        qt = (1-t) * self + t * quaternion2
        qt.normalize()
        return qt

    # Spherical Linear Interpolation
    def Slerp(self, quaternion2, t):
        # due to double cover problem. shortest path interpolation
        if self.innerProduct(quaternion2) < 0:
            quaternion2 = - quaternion2

        angleDifference = self - quaternion2
        scalar1 = sin((1-t) * angleDifference) / sin(angleDifference)
        scalar2 = sin(t * angleDifference) / sin(angleDifference)
        qt = scalar1 * self + scalar2 * quaternion2
        qt.normalize()
        return qt

    def getList(self):
        return [self.w, self.x, self.y, self.z]

    def getPybulletList(self):
        return [self.x, self.y, self.z, self.w]

    def getNorm(self):
        return sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        norm = self.getNorm()
        self.w = self.w / norm
        self.x = self.x / norm
        self.y = self.y / norm
        self.z = self.z / norm

    # for double cover problem
    def upperSemisphereLimit(self):
        if self.w < 0:
            self.w = -self.w
            self.x = -self.x
            self.y = -self.y
            self.z = -self.z
        elif abs(self.w) < 1e-4 and self.x < 0:
            self.w = 0
            self.x = -self.x
            self.y = -self.y
            self.z = -self.z
        elif abs(self.w) < 1e-4 and abs(self.x) < 1e-4 and self.y < 0:
            self.w = 0
            self.x = 0
            self.y = -self.y
            self.z = -self.z
        elif abs(self.w) < 1e-4 and abs(self.x) < 1e-4 and abs(self.y) < 1e-4:
            self.w = 0
            self.x = 0
            self.y = 0
            self.z = 1

    # you can use numpy to resize it
    def getMatrix(self, is4x4=False):
        self.normalize()
        sw = self.w * self.w
        sx = self.x * self.x
        sy = self.y * self.y
        sz = self.z * self.z

        m00 = (sx - sy - sz + sw)
        m11 = (-sx + sy - sz + sw)
        m22 = (-sx - sy + sz + sw)

        tmp1 = self.x * self.y
        tmp2 = self.z * self.w
        m10 = 2.0 * (tmp1 + tmp2)
        m01 = 2.0 * (tmp1 - tmp2)

        tmp1 = self.x * self.z
        tmp2 = self.y * self.w
        m20 = 2.0 * (tmp1 - tmp2)
        m02 = 2.0 * (tmp1 + tmp2)

        tmp1 = self.y * self.z
        tmp2 = self.x * self.w
        m21 = 2.0 * (tmp1 + tmp2)
        m12 = 2.0 * (tmp1 - tmp2)
        if not is4x4:
            return [m00, m01, m02, m10, m11, m12, m20, m21, m22]
        else:
            return [m00, m01, m02, 0, m10, m11, m12, 0, m20, m21, m22, 0, 0, 0, 0, 1]

    def __str__(self):
        return '(' + str(self.w) + ', ' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')'


def getRotationMatrix(xAxis, yAxis):
    zAxis = np.cross(xAxis, yAxis)
    return np.vstack((xAxis, yAxis, zAxis))


if __name__ == "__main__":
    pass
    # "1" element
    # p = Quaternion(1.0, 0.0, 0.0, 0.0)
    # # rotate around y-axis of pi / 2
    # p2 = Quaternion(0.7071067811865476, 0.0, 0.7071067811865475, 0.0)
    # # composition of two rotation (left product are different from right product)
    # p3 = p * p2
    # # the angle between two quaternion
    # difference = p - p2
    # # q and -q represent the same rotation
    # p1 = -p
    #
    # rotationMatrix = getRotationMatrix([0.7071067811865476, 0.7071067811865476, 0.0],
    #                                    [0.7071067811865476, -0.7071067811865476, 0.0])
    # pr = Quaternion()
    # pr.rotationMatrixInit(rotationMatrix.reshape(-1).tolist())
    # rotationMatrix2 = pr.getMatrix()
    # pr.rotationMatrixInit(rotationMatrix2)
    # rotationMatrix3 = pr.getMatrix()
    #
    # # apply rotation
    # b = p2.applyRotation([1, 0, 0])
    # # print(b)
    # # do interpolation
    # print(p.Nlerp(p2, 0.5))
    # print(p.Slerp(p2, 0.5))
    q = Quaternion(w=0.141, x=0.854, y=0.182, z=0.433)
    a = q.getMatrix(is4x4=False)
    q2 = Quaternion()
    q2.rotationMatrixInit(a)
    print(q2)
    print(getNorm(q.getList()))
