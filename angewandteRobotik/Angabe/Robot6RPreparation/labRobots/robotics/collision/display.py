# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: collisionDisplay.py
#
# Details:  Display of the collision shapes of the arm planning of a lab robot in the native renderer: the shapes the
#           collision model of Exudyn checks (arm and gripper, platform, obstacles of the scene) at the present state,
#           translucent; the arm shapes red while the smallest distance (CollisionModel.MinimumDistance) is below the
#           safety distance. ObstacleGraphics draws shapes in world coordinates (e.g. the obstacles of a scene).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-12 (from LabCourse/MobileRobot/fahrzeug/collisionDisplay.py, 2026-11-07, which used the own
#           RobotCollisionChecker of the Kairos and colored the single pairs; Exudyn gives the smallest distance only)
# Notes:    The model gives CollisionShapes(configuration) -> (arm shapes, other shapes) as shapes of collision.py in
#           world coordinates and CollisionDistance(configuration) -> (smallest distance in m, safety distance in m)
#           (KairosModel, LeobotModel). The queries of CollisionModel are const: the renderer thread may call them while
#           the simulation plans. Not in VR.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Display of the collision shapes of the arm planning."""
import time
import numpy as np
import exudyn as exu
import exudyn.graphics as graphics
from exudyn.itemInterface import ObjectGround, VObjectGround
from labRobots.robotics.collision.shapes import shapeBox

#! Colors (RGBA, translucent): arm shapes, other shapes (platform, obstacles), arm closer than the safety distance.
armShapeColor, obstacleColor, collisionColor = [0.15, 0.45, 0.95, 0.35], [0.6, 0.6, 0.6, 0.3], [0.95, 0.1, 0.1, 0.6]
#! Tiles of the drawn cylinders and spheres.
shapeTiles = 12
#! Wall-clock period in s of recomputing the drawn shapes (the renderer gets the last ones in between).
refreshPeriod = 0.1
#! Shapes not drawn (they would cover the scene; still checked).
hiddenShapeNames = ('floor',)


def ShapeGraphics(shape, color):
    """Return the graphics list of a shape of collision.py (capsule or box) in its reference frame."""
    if shape['type'] == shapeBox:
        return [graphics.Move(graphics.Brick([0., 0., 0.], (2.*np.asarray(shape['halfSize'])).tolist(), color=color),
                              np.asarray(shape['center']).tolist(), np.asarray(shape['rotation']))]
    point0, point1 = np.asarray(shape['point0'], dtype=float), np.asarray(shape['point1'], dtype=float)
    result = [graphics.Sphere(point0.tolist(), shape['radius'], color=color, nTiles=shapeTiles)]
    if np.linalg.norm(point1 - point0) > 1e-9:
        result += [graphics.Cylinder(point0.tolist(), (point1 - point0).tolist(), shape['radius'], color=color, nTiles=shapeTiles),
                   graphics.Sphere(point1.tolist(), shape['radius'], color=color, nTiles=shapeTiles)]
    return result


def ObstacleGraphics(shapes, color=obstacleColor):
    """Return the graphics list of shapes of collision.py in world coordinates, e.g. the obstacles of a scene for
    mbs.CreateGround(graphicsDataList=ObstacleGraphics(scene['obstacles']))."""
    return [item for shape in shapes if shape['name'] not in hiddenShapeNames for item in ShapeGraphics(shape, color)]


def ShapesMesh(shapes, *extraGraphics):
    """one triangle mesh of shapes of collision.py and further graphics items, all in world coordinates (Claude Code,
    2026-11-18), e.g. for mbs.CreateDistanceSensorGeometry: the scene the camera and laser rays hit

    Args:
        shapes: list of shapes of collision.py (boxes, capsules)
        extraGraphics: further graphics items (e.g. a floor brick)

    Returns:
        (points [n x 3], triangles [m x 3]) as lists
    """
    points, triangles = [], []
    items = [item for shape in shapes for item in ShapeGraphics(shape, obstacleColor)] + list(extraGraphics)
    for item in items:
        itemPoints, itemTriangles = graphics.ToPointsAndTrigs(item)
        offset = len(points)
        points += [list(point) for point in itemPoints]
        triangles += [[index + offset for index in triangle] for triangle in itemTriangles]
    return points, triangles


class CollisionDisplay:
    """Translucent collision shapes of the arm planning of a robot model at the present state; create before
    mbs.Assemble()."""

    def __init__(self, mbs, model, shownInit=False):
        """initialize CollisionDisplay

        Args:
            mbs: Main system.
            model: KairosModel or LeobotModel (CollisionShapes, CollisionDistance).
            shownInit: Initial visibility.
        """
        self.mbs = mbs                      #! Main system.
        self.model = model                  #! Robot model.
        self.shown = bool(shownInit)        #! Visibility.
        self.graphicsCache = []             #! Last graphics of the shapes.
        self.graphicsCacheTime = -np.inf    #! Wall-clock time of the last recomputation.
        #! Ground object drawing the shapes.
        self.graphicsObject = mbs.AddObject(ObjectGround(visualization=VObjectGround(
            graphicsDataUserFunction=self.GraphicsUserFunction)))

    def SetShown(self, shown):
        """Show or hide the collision shapes."""
        self.shown = bool(shown)
        self.graphicsCacheTime = -np.inf

    def Graphics(self, configuration=exu.ConfigurationType.Current):
        """Return the graphics list of the shapes at the state of configuration (arm red below the safety distance)."""
        armShapes, otherShapes = self.model.CollisionShapes(configuration)
        distance, safetyDistance = self.model.CollisionDistance(configuration)
        armColor = collisionColor if distance < safetyDistance else armShapeColor
        return ObstacleGraphics(armShapes, armColor) + ObstacleGraphics(otherShapes, obstacleColor)

    def GraphicsUserFunction(self, mbs, objectNumber):
        """Return the graphics for the native renderer (empty while hidden); recomputed at most every refreshPeriod
        (wall clock), in between the last graphics."""
        if not self.shown:
            return []
        now = time.perf_counter()
        if now - self.graphicsCacheTime >= refreshPeriod:
            self.graphicsCacheTime = now
            self.graphicsCache = self.Graphics(exu.ConfigurationType.Visualization)
        return self.graphicsCache
