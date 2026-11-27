# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: visualization.py
# Details:  UR5 CAD meshes, detailed mecanum rollers and coordinate arrows.
# Authors:  Peter Manzl; Martin Sereinig; Michael Pieber (created with Claude Code)
# Date:     2026-11-04 (MoveIt geometry: 2026-11-05); 2026-11-12 moved into labRobots/robots/kairos (from LabCourse/MobileRobot/fahrzeug); the parts of the lab scene stay there
# Notes:    Mesh attribution is retained in mektcommon2/data/Exudyn/robotics/URDF/kairos_description/ur5_visual_assets/LICENSE.
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Graphics derived from the preparation model."""
from functools import lru_cache
import numpy as np
import exudyn.graphics as graphics
from exudyn.rigidBodyUtilities import (HTrotateX, HTrotateZ, HTtranslate, RotationMatrixX, RotationMatrixY,
                                      RotationMatrixZ)
from labRobots.robots.kairos import parameters as par


def RealUR5Colors(colors):
    """Map the dark material colors of the UR5 meshes (as in ur_description/RViz) to the real CB3 robot:
    blue-grey caps -> light blue, dark grey housings -> light grey, grey tubes and base -> aluminium;
    black seals are kept.

    Args:
        colors: Flat RGBA list of the mesh points.

    Returns:
        Flat RGBA array.
    """
    #colors as on the photos of the lab robot
    rgba = np.asarray(colors, dtype=float).reshape(-1, 4).copy()
    blueish = rgba[:, 2] - rgba[:, 0] > par.ur5BlueThreshold
    black = rgba[:, :3].max(axis=1) < par.ur5BlackThreshold
    tube = ~blueish & ~black & (rgba[:, :3].max(axis=1) >= par.ur5TubeThreshold)
    housing = ~blueish & ~black & ~tube
    for mask, color in ((blueish, par.ur5CapColor), (tube, par.ur5TubeColor), (housing, par.ur5HousingColor)):
        rgba[mask, :3] = color[:3]
    return rgba.reshape(-1)


def LoadArmGraphics():
    """Return UR5 base graphics in platform coordinates and six DH-local link graphics."""
    names = ('base', 'shoulder', 'upperarm', 'forearm', 'wrist1', 'wrist2', 'wrist3')
    frameTransforms = (np.eye(4), HTrotateX(-np.pi/2.), HTtranslate([0.425, 0., 0.]),
                       HTtranslate([0.39225, 0., 0.]), HTrotateX(-np.pi/2.), HTrotateX(np.pi/2.), np.eye(4))
    meshOffsets = (HTrotateZ(np.pi), HTrotateZ(np.pi),
                   HTtranslate([0., 0., 0.13585]) @ HTrotateZ(-np.pi/2.) @ HTrotateX(np.pi/2.),
                   HTtranslate([0., 0., 0.0165]) @ HTrotateZ(-np.pi/2.) @ HTrotateX(np.pi/2.),
                   HTtranslate([0., 0., -0.093]) @ HTrotateX(np.pi/2.),
                   HTtranslate([0., 0., -0.095]),
                   #wrist3's mesh axis is +y; align its flange with DH/tool +z.
                   HTtranslate([0., 0., -0.0818]) @ HTrotateX(np.pi/2.))
    result = []
    with np.load(par.ur5VisualMeshFile, allow_pickle=False) as data:
        for name, frame, offset in zip(names, frameTransforms, meshOffsets):
            mesh = {'type': 'TriangleList', **{key: data[name + '_' + key].reshape(-1)
                                             for key in ('points', 'normals', 'colors', 'triangles')}}
            mesh['colors'] = RealUR5Colors(mesh['colors'])
            transform = frame @ offset
            if name == names[0]:
                transform = HTtranslate(par.mountPosition) @ HTrotateZ(par.mountYaw) @ offset
            result.append([graphics.Move(mesh, transform[:3, 3], transform[:3, :3])])
    return result[0], result[1:]


def RollerContour():
    """Return the fitted generatrix of the preparation's 45-degree mecanum roller."""
    rollerRadius = 0.019
    rollerLength = par.wheelWidth*0.924
    distance = par.wheelRadius - rollerRadius
    angle = np.pi/4.
    parameter = np.linspace(-np.pi/4., np.pi/4., 1001)
    position = distance*np.cos(angle)**2/np.sin(angle)*np.tan(parameter) + par.wheelRadius*np.sin(angle)*np.sin(parameter)
    radius = np.sqrt(np.cos(angle)**2*np.tan(parameter)**2 + 1.)*(par.wheelRadius*np.cos(parameter) - distance)
    selected = np.abs(position) <= rollerLength/2.
    points = np.linspace(-rollerLength/2., rollerLength/2., 21)
    #An even polynomial retains the original smooth roller profile rather than a cylindrical substitute.
    coeffs = np.polynomial.polynomial.polyfit(position[selected], radius[selected], np.arange(0, 14, 2))
    values = np.polynomial.polynomial.polyval(points, coeffs)
    return [[-rollerLength/2., 0.]] + np.column_stack((points, values)).tolist() + [[rollerLength/2., 0.]]


#! Roller axis angles in rad of the two mecanum wheel types (number of cached wheel graphics).
mecanumRollerAngles = [np.pi/4., -np.pi/4.]


def WheelGraphics(wheelIndex):
    """Return a wheel with twelve shaped rollers in its rigid body's local frame (new list of shared, read-only items)."""
    return list(MecanumWheelGraphics(mecanumRollerAngles[1] if wheelIndex in (1, 2) else mecanumRollerAngles[0]))


#built once per roller angle (52 solids of revolution took 0.9 s of every model build,
#also of every reset in VR); the items are only read (kinematic tree graphics, GLB export).
@lru_cache(maxsize=len(mecanumRollerAngles))
def MecanumWheelGraphics(angle):
    """Return the graphics items (tuple) of a wheel whose rollers are inclined by angle in rad."""
    numberOfRollers = 12
    rollerRadius = 0.019
    rollerLength = par.wheelWidth*0.924
    items = [graphics.Cylinder([-rollerLength/2., 0., 0.], [rollerLength, 0., 0.],
                              radius=par.wheelRadius*0.7, color=par.rubberColor, nTiles=32)]
    contour = RollerContour()
    for rollerIndex in range(numberOfRollers):
        rotation = rollerIndex/numberOfRollers*2.*np.pi
        position = [0., -(par.wheelRadius - rollerRadius)*np.sin(rotation),
                    -(par.wheelRadius - rollerRadius)*np.cos(rotation)]
        direction = RotationMatrixX(-rotation) @ RotationMatrixZ(angle) @ [0.5*rollerLength, 0., 0.]
        items.append(graphics.SolidOfRevolution(position, direction, contour=contour,
                                               color=par.rubberColor, nTiles=16))
    #metallic hub caps and fasteners distinguish rollers from the wheel carrier.
    for side in (-1., 1.):
        items.append(graphics.Cylinder([side*0.048, 0., 0.], [side*0.006, 0., 0.],
                                       radius=0.063, color=par.metalColor, nTiles=32))
        items.append(graphics.Cylinder([side*0.054, 0., 0.], [side*0.003, 0., 0.],
                                       radius=0.024, color=par.accentColor, nTiles=24))
        for boltIndex in range(6):
            boltAngle = boltIndex*np.pi/3.
            items.append(graphics.Cylinder([side*0.055, 0.043*np.cos(boltAngle), 0.043*np.sin(boltAngle)],
                                           [side*0.003, 0., 0.], radius=0.004, color=par.rubberColor, nTiles=8))
    return tuple(items)


def UrdfRotationMatrix(rpy):
    """Return the rotation matrix of URDF roll/pitch/yaw angles in rad (fixed axes x, y, z: Rz*Ry*Rx)."""
    return RotationMatrixZ(rpy[2]) @ RotationMatrixY(rpy[1]) @ RotationMatrixX(rpy[0])


def KairosPartGraphics(part, color):
    """Return one URDF mesh of par.kairosParts in platform coordinates.

    Args:
        part: Entry of par.kairosParts: STL file, footprint position in m and URDF rpy in rad.
        color: RGBA color of the mesh.
    """
    mesh = LoadUrdfMesh(part['file'], color)
    position = np.asarray(part['position']) + par.footprintOffset
    return graphics.Move(mesh, position.tolist(), UrdfRotationMatrix(part['rpy']))


def LoadUrdfMesh(fileName, color, scale=1.):
    """Load an STL mesh of the URDF description with its own orientation.

    The URDF STL files are standard STL (counterclockwise seen from outside), which is also the Exudyn and glTF
    convention; FromSTLfile inverts triangles and normals by default (for STL files with inward orientation),
    which made faces dark in the renderer and black in WebXR (three.js decides the front side by the winding).
    """
    return graphics.FromSTLfile(str(par.kairosMeshDirectory / fileName), color=color, scale=scale,
                                invertNormals=False, invertTriangles=False)


def ColorTriangles(mesh, ColorFunction):
    """Return a triangle list with separate corners per triangle, colored per triangle (sharp color edges;
    shared corners would blend the colors of e.g. top and side faces).

    Args:
        mesh: Exudyn TriangleList graphics with outward (counterclockwise) triangles, see LoadUrdfMesh.
        ColorFunction: Maps triangle centers (n,3) and outward normals (n,3) to RGBA colors (n,4).
    """
    points = np.asarray(mesh['points']).reshape(-1, par.dimension3D)
    triangles = np.asarray(mesh['triangles'], dtype=int).reshape(-1, par.dimension3D)
    corners = points[triangles]
    faceNormals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    faceNormals /= np.maximum(np.linalg.norm(faceNormals, axis=1, keepdims=True), 1e-15)
    centers = corners.mean(axis=1)
    colors = np.asarray(ColorFunction(centers, faceNormals), dtype=float)
    numberOfCorners = len(triangles)*par.dimension3D
    return {'type': 'TriangleList', 'points': corners.reshape(-1).tolist(),
            'normals': np.repeat(faceNormals, par.dimension3D, axis=0).reshape(-1).tolist(),
            'colors': np.repeat(colors, par.dimension3D, axis=0).reshape(-1).tolist(),
            'triangles': list(range(numberOfCorners))}


def LidarColors(centers, part):
    """Return Velodyne colors as on the real sensor: silver bottom and top ring, dark blue window, black cable."""
    axisPoint = np.asarray(part['position']) + par.footprintOffset
    height = centers[:, 2] - axisPoint[2]
    radius = np.linalg.norm(centers[:, :2] - axisPoint[:2], axis=1)
    window = (height > par.lidarWindowHeights[0]) & (height < par.lidarWindowHeights[1])
    colors = np.where(window[:, None], par.lidarWindowColor, par.lidarHousingColor)
    return np.where((radius > par.lidarRadius)[:, None], par.chassisSideColor, colors)


def PlatformGraphics():
    """Return chassis, sensor mast, Velodyne lidar, two SICK S300 laser scanners and camera of the MoveIt/URDF
    description in platform coordinates, plus lights and emergency button on the chassis housing."""
    #replaces bodykairos.stl and its fitted panels by the URDF meshes used by MoveIt.
    partColors = {'chassis': par.chassisSideColor, 'rearVelodyneStructure': par.chassisSideColor,
                  'velodyne': par.lidarHousingColor, 'frontLaser': par.accentColor, 'rearLaser': par.accentColor}
    items = []
    for part in par.kairosParts:
        mesh = KairosPartGraphics(part, partColors[part['name']])
        if part['name'] == 'chassis':
            #as on the real robot: silver top plate, black sides
            topPlate = par.baseLinkHeight + par.chassisTopHeight + par.footprintOffset[2] - par.chassisTopPlateTolerance
            mesh = ColorTriangles(mesh, lambda centers, normals: np.where(
                (normals[:, [2]] > par.upwardNormalThreshold) & (centers[:, [2]] > topPlate),
                par.chassisTopColor, par.chassisSideColor))
        elif part['name'] == 'velodyne':
            mesh = ColorTriangles(mesh, lambda centers, normals, part=part: LidarColors(centers, part))
        items.append(mesh)
    #details on the upper housing (URDF chassis: x +-0.3595 m, y +-0.2893 m, top 0.3916 m above base_link)
    housingFront, housingTop = 0.3595, par.baseLinkHeight + par.chassisTopHeight + par.footprintOffset[2]
    items += FrontCameraGraphics()
    #as on the lab robot (Michael Pieber): no front and rear lights; a panel in the color of
    #the top plate on the rear wall (a band below the top plate, not over the whole width) with the emergency button
    #on the right side of the platform
    rearPanelHalfWidth, rearPanelTopGap, rearPanelHeight = 0.23, 0.04, 0.075
    items.append(graphics.Brick([-housingFront - 0.001, 0., housingTop - rearPanelTopGap - 0.5*rearPanelHeight],
                                [0.002, 2.*rearPanelHalfWidth, rearPanelHeight], color=par.chassisTopColor))
    #where the right rear light was drawn, on the center line of the band
    buttonPosition = [-housingFront - 0.002, -0.18, housingTop - rearPanelTopGap - 0.5*rearPanelHeight]
    items.append(graphics.Cylinder(buttonPosition, [-0.006, 0., 0.], radius=0.023,
                                   color=[0.92, 0.65, 0.08, 1.], nTiles=24))
    items.append(graphics.Cylinder([buttonPosition[0] - 0.006, buttonPosition[1], buttonPosition[2]], [-0.016, 0., 0.],
                                   radius=0.014, color=[0.75, 0.06, 0.05, 1.], nTiles=24))
    return items


#Bumblebee2 stereo camera of the real robot instead of the URDF Orbbec Astra
def FrontCameraGraphics():
    """Return the Bumblebee2 housing and its two lenses (looking along platform x) in platform coordinates."""
    center = np.asarray(par.frontCameraPosition) + par.footprintOffset
    items = [graphics.Brick(center.tolist(), par.frontCameraSize, color=par.frontCameraColor)]
    lensDepth = 0.004
    for side in (-0.5, 0.5):
        lens = center + [0.5*par.frontCameraSize[0], side*par.frontCameraBaseline, 0.]
        items.append(graphics.Cylinder(lens.tolist(), [lensDepth, 0., 0.], radius=par.frontCameraLensRadius,
                                       color=par.frontCameraLensColor, nTiles=16))
    return items


def GripperHousingGraphics():
    """Return the WSG50 gripper body (WSG50_110.stl) in the last UR5 link frame (= URDF tool0)."""
    #URDF summit_xl_wsg50_base_link is fixed at tool0 without offset.
    return [LoadUrdfMesh('WSG50_110.stl', par.metalColor)]


def GripperFingerGraphics():
    """Return the WSG50 guide and finger (mm meshes scaled to m) in the frame of one finger carriage;
    the finger mesh starts at summit_xl_wsg50_center, i.e. wsg50CenterOffset along carriage z."""
    #URDF summit_xl_wsg50_gripper_left/right (guide) and finger_left/right links.
    guide = LoadUrdfMesh('GUIDE_WSG50_110.stl', par.accentColor, par.millimeterScale)
    finger = LoadUrdfMesh('WSG-FMF.stl', par.metalColor, par.millimeterScale)
    return [guide, graphics.Move(finger, [0., 0., par.wsg50CenterOffset], np.eye(par.dimension3D))]


def CoordinateGraphics(length=par.frameLength):
    """Return exportable RGB arrow meshes for a local coordinate system."""
    result = []
    for axisIndex in range(par.dimension3D):
        direction = np.eye(par.dimension3D)[axisIndex]*length
        result.append(graphics.Arrow([0., 0., 0.], direction, par.frameRadius, color=par.axisColors[axisIndex]))
    return result
