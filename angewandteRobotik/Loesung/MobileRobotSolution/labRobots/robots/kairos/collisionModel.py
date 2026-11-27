# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: collisionModel.py
# Details:  Kairos-specific collision geometry for arm path planning: UR5 link capsules
#           and WSG50 boxes fitted to the visual meshes, platform obstacles (chassis,
#           mast, Velodyne, SICK scanners, camera, UR5 base, wheels, floor) from the
#           URDF meshes, all in the arm base frame (summit_xl_ur5_base).
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-05; 2026-11-12 moved into labRobots/robots/kairos (from LabCourse/MobileRobot/fahrzeug); the parts of the lab scene stay there
# Notes:    The shapes are derived from the same meshes as the graphics, so geometry has a single source;
#           checked by Exudyn (planning.KairosArmPlanner, CollisionModel).
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Collision shapes of the Kairos mobile manipulator."""
import numpy as np
from exudyn.rigidBodyUtilities import HTtranslate, HTrotateZ
from exudyn.robotics import StdDH2HT
from labRobots.robotics.collision.shapes import (CreateBox, CreateCapsule, BoxFromPoints, ShapesFromPoints,
                                      SampleTriangleSurface, TransformShape)
from labRobots.robots.kairos import parameters as par
from labRobots.robots.kairos.visualization import LoadArmGraphics, FrontCameraGraphics, GripperHousingGraphics, GripperFingerGraphics, KairosPartGraphics, UrdfRotationMatrix


#! Name suffix of the housing capsule of the next joint of a link (ShapesFromPoints with two axis housings).
nextHousingSuffix = 'J1'
#! Platform shapes on the axis of joint 1 (UR5 base cylinder), attached to link 0 for the collision model.
armBaseAxisShapes = ('armBaseJ0',)
#! Homogeneous transformation of the arm base (summit_xl_ur5_base) in footprint coordinates.
armBaseInFootprint = HTtranslate(par.armBaseFootprintPosition) @ HTrotateZ(par.mountYaw)


def GraphicsPoints(graphicsList):
    """Return all triangle-list points of a graphics list as array (n,3)."""
    return np.vstack([np.asarray(item['points']).reshape(-1, par.dimension3D) for item in graphicsList])


def GraphicsSurfacePoints(graphicsList):
    """Return points sampled on all triangle surfaces of a graphics list with par.collisionSampleSpacing."""
    return np.vstack([SampleTriangleSurface(item['points'], item['triangles'], par.collisionSampleSpacing)
                      for item in graphicsList])


def DHTransformation(dhParameters, jointAngle):
    """Return the standard DH transformation [theta, d, a, alpha] for a joint angle as 4x4 array (Claude Code,
    2026-11-20: StdDH2HT of exudyn.robotics, as the kinematics of the model; before a formula of its own)."""
    theta, offset, length, twist = dhParameters
    return np.array(StdDH2HT([theta + jointAngle, offset, length, twist]))


def ArmLinkPoses(jointAngles):
    """Return the six DH link frames of the UR5 in the arm base frame (frame 6 = URDF tool0)."""
    poses = []
    pose = np.eye(4)
    for dhParameters, jointAngle in zip(par.armDH, jointAngles):
        pose = pose @ DHTransformation(dhParameters, jointAngle)
        poses.append(pose)
    return poses


def JointHousings(linkIndex):
    """Return the joint axes bounding a UR5 link as (axisPoint, axisDirection, housingRadius) in the
    link's DH frame: the axis of its own joint (z of the previous frame) and of the next joint (own z)."""
    toPreviousFrame = np.linalg.inv(DHTransformation(par.armDH[linkIndex], 0.))
    housings = [(toPreviousFrame[:3, 3], toPreviousFrame[:3, 2], par.armJointHousingRadii[linkIndex])]
    if linkIndex + 1 < par.numberOfArmJoints:
        housings.append((np.zeros(par.dimension3D), np.eye(par.dimension3D)[2], par.armJointHousingRadii[linkIndex + 1]))
    return housings


def ArmLinkShapes():
    """Return (linkIndex, shape) of the UR5 links and the WSG50 gripper (boxes, fingers open); a link is
    covered by capsules along its joint axes (housings) and capsules/boxes fitted to slices of the rest."""
    unusedBaseGraphics, linkGraphics = LoadArmGraphics()
    shapes = [(linkIndex, shape) for linkIndex, graphics in enumerate(linkGraphics)
              for shape in ShapesFromPoints(GraphicsSurfacePoints(graphics), par.armLinkShapeCounts[linkIndex],
                                            name=par.armLinkNames[linkIndex], axisHousings=JointHousings(linkIndex))]
    lastLink = par.numberOfArmJoints - 1
    shapes.append((lastLink, BoxFromPoints(GraphicsPoints(GripperHousingGraphics()), name='gripper')))
    #fingers at the widest opening of the simulation, as on the finger carriages of model.FingerPose
    carriageOffset = np.clip(par.gripperOpenWidth/2. + par.gripperFingerFaceOffset, *par.gripperCarriageLimits)
    fingerPoints = GraphicsPoints(GripperFingerGraphics())
    for fingerIndex, name in enumerate(('fingerLeft', 'fingerRight')):
        carriage = HTrotateZ(np.pi*fingerIndex) @ HTtranslate([-carriageOffset, 0., 0.])
        points = fingerPoints @ carriage[:3, :3].T + carriage[:3, 3]
        shapes.append((lastLink, BoxFromPoints(points, rotation=carriage[:3, :3], name=name)))
    #the housing of the next joint (capsule on its axis, the same in every joint angle) on the link
    #turning about it, and the UR5 base cylinder (on the axis of joint 1) on link 0: the collision model of Exudyn
    #excludes whole link pairs (neighbors, overlapping in collisionReferenceAngles); on the link before, the housings were
    #always closer than the safety distance to the next but one link (forearm housing - wrist 2: 3 mm, base cylinder -
    #upper arm: overlapping), so these link pairs would never have been checked
    shapes = [(linkIndex + 1, TransformShape(shape, np.linalg.inv(DHTransformation(par.armDH[linkIndex + 1], 0.))))
              if linkIndex < lastLink and shape['name'] == par.armLinkNames[linkIndex] + nextHousingSuffix else (linkIndex, shape)
              for linkIndex, shape in shapes]
    toLink0 = np.linalg.inv(DHTransformation(par.armDH[0], 0.)) @ np.linalg.inv(armBaseInFootprint)
    shapes += [(0, TransformShape(shape, toLink0)) for shape in PlatformObstaclesFootprint() if shape['name'] in armBaseAxisShapes]
    return shapes


def PlatformObstaclesFootprint():
    """Return the platform obstacles in footprint coordinates (URDF summit_xl_base_footprint)."""
    obstacles = []
    for part in par.kairosParts:
        points = GraphicsPoints([KairosPartGraphics(part, par.metalColor)]) - par.footprintOffset
        rotation = UrdfRotationMatrix(part['rpy'])
        if part['name'] == 'chassis':
            #lower drive body and wider upper housing as two boxes
            upper = points[:, 2] >= par.baseLinkHeight + par.chassisHousingHeight
            obstacles.append(BoxFromPoints(points[~upper], rotation, name='chassisLower'))
            obstacles.append(BoxFromPoints(points[upper], rotation, name='chassisUpper'))
        else:
            obstacles.append(BoxFromPoints(points, rotation, name=part['name']))
    #Bumblebee2 front camera (generated graphics, not a URDF mesh)
    obstacles.append(BoxFromPoints(GraphicsPoints(FrontCameraGraphics()) - par.footprintOffset, np.eye(par.dimension3D),
                                   name='frontCamera'))
    #UR5 base: cylinder around the joint 1 axis plus the cable outlet
    baseGraphics, unusedLinkGraphics = LoadArmGraphics()
    jointAxis = (np.asarray(par.armBaseFootprintPosition), np.eye(par.dimension3D)[2], par.armBaseHousingRadius)
    obstacles += ShapesFromPoints(GraphicsSurfacePoints(baseGraphics) - par.footprintOffset, 1, name='armBase',
                                  axisHousings=[jointAxis])
    for wheelIndex in range(par.numberOfWheels):
        longitudinal = par.wheelBase/2.*(1. if wheelIndex < 2 else -1.)
        lateral = par.wheelTrack/2.*(1. if wheelIndex in (0, 2) else -1.)
        obstacles.append(CreateCapsule([longitudinal, lateral - par.wheelWidth/2., par.wheelRadius],
                                       [longitudinal, lateral + par.wheelWidth/2., par.wheelRadius],
                                       par.wheelRadius, name='wheel%d' % wheelIndex))
    obstacles.append(CreateBox([0., 0., -par.floorThickness/2.],
                               [par.floorHalfLength, par.floorHalfLength, par.floorThickness/2.], name='floor'))
    return obstacles


def PlatformObstacles():
    """Return the platform obstacles in the arm base frame (without the UR5 base cylinder, which is on link 0 of
    ArmLinkShapes)."""
    toArmBase = np.linalg.inv(armBaseInFootprint)
    return [TransformShape(shape, toArmBase) for shape in PlatformObstaclesFootprint() if shape['name'] not in armBaseAxisShapes]
