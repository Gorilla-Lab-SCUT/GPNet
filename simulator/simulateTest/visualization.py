import trimesh
import numpy as np

def getWorldMeshList(planeWidth=0.2, axisHeight=0.05, axisRadius=0.001):
    groundColor = [220, 220, 220, 255]    # face_colors: [R, G, B, transparency]
    xColor = [255, 0, 0, 128]
    yColor = [0, 255, 0, 128]
    zColor = [0, 0, 255, 128]

    ground = trimesh.primitives.Box(
        center=[0, 0, -0.0001],
        extents=[planeWidth, planeWidth, 0.0002]
    )
    ground.visual.face_colors = groundColor
    xAxis = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    xAxis.apply_transform(matrix=np.mat(
        ((  0,  0, 1, axisHeight / 2),
        (   0,  1, 0, 0),
        (   -1, 0, 0, 0),
        (   0,  0, 0, 1))
    ))
    xAxis.visual.face_colors = xColor
    yAxis = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    yAxis.apply_transform(matrix=np.mat(
        ((  1, 0, 0, 0),
        (   0, 0, -1, axisHeight / 2),
        (   0, 1, 0, 0),
        (   0, 0, 0, 1))
    ))
    yAxis.visual.face_colors = yColor
    zAxis = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    zAxis.apply_transform(matrix=np.mat(
        ((1, 0, 0, 0),
         (0, 1, 0, 0),
         (0, 0, 1, axisHeight / 2),
         (0, 0, 0, 1))
    ))
    zAxis.visual.face_colors = zColor
    xBox = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3]
    )
    xBox.apply_translation((axisHeight, 0, 0))
    xBox.visual.face_colors = xColor
    yBox = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3]
    )
    yBox.apply_translation((0, axisHeight, 0))
    yBox.visual.face_colors = yColor
    zBox = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3]
    )
    zBox.apply_translation((0, 0, axisHeight))
    zBox.visual.face_colors = zColor
    worldMeshList = [ground, xAxis, yAxis, zAxis, xBox, yBox, zBox]
    return worldMeshList

def getNewCoordinate(axisHeight=0.05, axisRadius=0.001):
    xColor = [200, 50, 0, 128]
    yColor = [0, 200, 50, 128]
    zColor = [50, 0, 200, 128]

    xAxis2 = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    xAxis2.apply_transform(matrix=np.mat(
        ((  0,  0, 1, axisHeight / 2),
        (   0,  1, 0, 0),
        (   -1, 0, 0, 0),
        (   0,  0, 0, 1))
    ))
    xAxis2.visual.face_colors = xColor
    yAxis2 = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    yAxis2.apply_transform(matrix=np.mat(
        ((  1, 0, 0, 0),
        (   0, 0, -1, axisHeight / 2),
        (   0, 1, 0, 0),
        (   0, 0, 0, 1))
    ))
    yAxis2.visual.face_colors = yColor
    zAxis2 = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    zAxis2.apply_transform(matrix=np.mat(
        ((1, 0, 0, 0),
         (0, 1, 0, 0),
         (0, 0, 1, axisHeight / 2),
         (0, 0, 0, 1))
    ))
    zAxis2.visual.face_colors = zColor
    xBox2 = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3]
    )
    xBox2.apply_translation((axisHeight, 0, 0))
    xBox2.visual.face_colors = xColor
    yBox2 = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3]
    )
    yBox2.apply_translation((0, axisHeight, 0))
    yBox2.visual.face_colors = yColor
    zBox2 = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3]
    )
    zBox2.apply_translation((0, 0, axisHeight))
    zBox2.visual.face_colors = zColor

    return 1


def meshVisualization(mesh):
    worldMeshList = getWorldMeshList(planeWidth=0.2, axisHeight=0.05, axisRadius=0.001)
    if isinstance(mesh, trimesh.base.Trimesh):
        mesh.visual.face_colors = [255, 128, 255, 200]
        worldMeshList.append(mesh)
    elif isinstance(mesh, list):
        for meshItem in mesh:
            meshItem.visual.face_colors = [255, 128, 255, 200]
            worldMeshList.append(meshItem)
    scene = trimesh.Scene(worldMeshList)
    scene.show()


def meshPairVisualization(mesh1, mesh2):
    worldMeshList = getWorldMeshList(planeWidth=0.2, axisHeight=0.05, axisRadius=0.001)
    mesh1.visual.face_colors = [255, 128, 255, 200]
    mesh2.visual.face_colors = [255, 255, 128, 200]

    worldMeshList.append((mesh1, mesh2))
    scene = trimesh.Scene(worldMeshList)
    scene.show()


# pointCloud : [pointNum, 3]
def pointCloudViewer(pointCloud, boxScale=0.0005):
    worldMeshList = getWorldMeshList(planeWidth=0.2, axisHeight=0.05, axisRadius=0.001)
    boxColor = [255, 128, 255, 180]
    for point in pointCloud:
        transform = np.eye(4)
        transform[:3, 3] = point
        temp = trimesh.primitives.Box(
            extents=[boxScale, boxScale, boxScale],
            transform=transform
        )
        temp.visual.face_colors = boxColor
        worldMeshList.append(temp)

    scene = trimesh.Scene(worldMeshList)
    scene.show()


def contactVisualization(mesh, contact1, contact2):
    worldMeshList = getWorldMeshList(planeWidth=2, axisHeight=0.5, axisRadius=0.01)
    contact1Mesh = trimesh.primitives.Box(
        extents=[0.005, 0.005, 0.005]
    )
    contact1Mesh.apply_translation(contact1)
    contact2Mesh = trimesh.primitives.Box(
        extents=[0.005, 0.005, 0.005]
    )
    contact2Mesh.apply_translation(contact2)
    mesh.visual.face_colors = [255, 128, 255, 200]
    worldMeshList.append(mesh)
    worldMeshList.append(contact1Mesh)
    worldMeshList.append(contact2Mesh)
    scene = trimesh.Scene(worldMeshList)
    scene.show()



# if __name__ == '__main__':
#     meshVisualization(trimesh.convex.convex_hull(np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0]])))
