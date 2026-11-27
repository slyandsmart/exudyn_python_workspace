# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: collisionModel.py
# Details:  Kairos-specific collision geometry for arm path planning: UR5 link capsules
#           and WSG50 boxes fitted to the visual meshes, platform obstacles (chassis,
#           mast, Velodyne, SICK scanners, camera, UR5 base, wheels, floor) from the
#           URDF meshes, all in the arm base frame (summit_xl_ur5_base).
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-05: the parts of the robot are in labRobots/robots/kairos/collisionModel.py
# Notes:    Generic checking/planning is in labRobots/robotics; shapes are derived from the
#           same meshes as the graphics, so geometry has a single source.
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Collision shapes of the Kairos mobile manipulator."""
from labRobots.robotics.collision.shapes import CreateBox, TransformShape
from . import parameters as par


def TableObstacle(tablePosition, tableHeight=par.tableHeight):
    """Return the table as one box of its outer dimensions in world coordinates (the four
    parts of fahrzeug/table.py are drawn and used for the contact and the sensors; for the arm planning one solid box
    is cheaper - fewer pairs to certify - and conservative: the arm never reaches below the top plate). Claude Code,
    2026-11-12: in world coordinates, the obstacle of the scene for KairosModel.SetEnvironmentObstacles.

    @param tablePosition World x, y of the table center in m.
    @param tableHeight Height in m of the table top.
    """
    return CreateBox([tablePosition[0], tablePosition[1], 0.5*tableHeight],
                     [0.5*par.tableSize[0], 0.5*par.tableSize[1], 0.5*tableHeight], name='table')


def EnvironmentObstacles(tablePosition, worldToArmBase, tableHeight=par.tableHeight):
    """Return the table (TableObstacle) in the arm base frame.

    @param tablePosition World x, y of the table center in m.
    @param worldToArmBase 4x4 transformation from world to arm base coordinates.
    @param tableHeight Height in m of the table top.
    """
    return [TransformShape(TableObstacle(tablePosition, tableHeight), worldToArmBase)]
