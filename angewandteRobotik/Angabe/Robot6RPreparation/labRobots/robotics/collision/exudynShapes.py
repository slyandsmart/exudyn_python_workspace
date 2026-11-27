# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: nativeCollision.py
#
# Details:  The shapes of collision.py (capsule, box) as shapes of the collision check of Exudyn
#           (exudyn.cUtilities.ConvexShape and HT, for CollisionModel.AddLinkShape and AddObstacle).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-09; 2026-11-12 moved from LabCourse/MobileRobot/tools/robotics; 2026-11-12 NativeCollisionChecker
#           removed (the robots check with Exudyn only)
# Notes:    Needs Exudyn >= 1.13.502 (CollisionModel).
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Shapes of collision.py for the collision check of Exudyn."""
import numpy as np
from labRobots.robotics.collision.shapes import shapeCapsule, CreateBox, CreateCapsule

baseLinkIndex = -1                  #link index of the robot base in exudyn.cUtilities.CollisionModel
minimumBoxEdge = 1e-6               #m, edge of a flat box in CollisionModel (larger by half of it: conservative)


def ExudynShape(shape, HT=None):
    """Return (ConvexShape, exudyn.HT) of a shape of collision.py (capsule or box), moved by the 4x4 matrix HT."""
    import exudyn as exu
    from exudyn.cUtilities import ConvexShape
    HT = np.eye(4) if HT is None else np.asarray(HT, dtype=float)
    if shape['type'] == shapeCapsule:
        start, end = shape['point0'], shape['point1']
        axis = end - start
        length = float(np.linalg.norm(axis))
        zAxis = axis/length if length > 0. else np.array([0., 0., 1.])
        xAxis = np.array([1., 0., 0.]) if abs(zAxis[0]) < 0.9 else np.array([0., 1., 0.])
        yAxis = np.cross(zAxis, xAxis)
        yAxis /= np.linalg.norm(yAxis)
        rotation, center = np.column_stack([np.cross(yAxis, zAxis), yAxis, zAxis]), 0.5*(start + end)
        convexShape = ConvexShape('Capsule', radius=shape['radius'], length=length)
    else:
        rotation, center = shape['rotation'], shape['center']
        #a flat box (an edge 0, e.g. fitted to a plate) gets minimumBoxEdge: ConvexShape needs positive edges
        convexShape = ConvexShape('Box', size=np.maximum(2.*shape['halfSize'], minimumBoxEdge).tolist())
    return convexShape, exu.HT(rotation=(HT[0:3, 0:3] @ rotation).tolist(), translation=(HT[0:3, 0:3] @ center + HT[0:3, 3]).tolist())


def RoboticsBoxShape(box):
    """Return a box of labRobots.robotics.collision.shapes as collision shape of exudyn.robotics (dict with type, size, HT);
    (moved from leobot/model.py 2026-11-13, used by all arms)."""
    import exudyn as exu
    return {'type': 'box', 'size': (2.*np.asarray(box['halfSize'])).tolist(), 'name': box['name'],
            'HT': exu.HT(rotation=np.asarray(box['rotation']).tolist(), translation=np.asarray(box['center']).tolist())}


def CollisionShapeOfRobotics(shape, name):
    """Return a collision shape of exudyn.robotics (dict: capsule with pAxis, vAxis, radius; box with size and HT) as
    shape of labRobots.robotics.collision.shapes named name (unless the box has a name) (PandaShape of
    leobot/model.py, moved 2026-11-13)."""
    if shape['type'] == 'box':
        HT = np.array(shape['HT'])
        return CreateBox(HT[0:3, 3], 0.5*np.asarray(shape['size']), HT[0:3, 0:3], name=shape.get('name', name))
    return CreateCapsule(shape['pAxis'], np.add(shape['pAxis'], shape['vAxis']), shape['radius'], name=name)
