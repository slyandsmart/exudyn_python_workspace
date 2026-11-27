# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: visualization.py
#
# Details:  Graphics of the lab of the 6R task for Exudyn: lab table, small
#           tables 1/2 with cup holders and the paper cup (Claude Code,
#           2026-11-12: the robot graphics are labRobots.robots.robot6R.visualization).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-03
# Notes:    Environment in base frame K0, cup in its own frame (origin at
#           bottom center).
#
# License:  BSD-3 license
#
# Installation:
#           pip install exudyn
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np

import exudyn.graphics as graphics

from labRobots.robots.robot6R.visualization import robotColorGroupColors
from . import parameters as par
from .labObjects import MeshFileName

#! RGBA colors of the color groups written by the CAD conversion: those of the robot and of the lab.
colorGroupColors = dict(robotColorGroupColors,
                        tablePlate=[0.52, 0.53, 0.52, 1.],   #stainless steel table plate
                        tableLegs=[0.12, 0.12, 0.12, 1.],
                        labObject=[0.30, 0.55, 0.85, 1.],    #printed small tables and cup holders
                        cup=[0.97, 0.97, 0.95, 1.])
cupContourTiles = 32    #resolution of the cup surface
cupHolderRadius = 0.041 #m, half width of becherhalter_V3; cup placed within this radius stands on the holder


def LoadMesh(fileName):
    """Load mesh (points, triangles, vertex normals) of the robot or the lab (written by the CAD conversion, removed 2026-11-20)."""
    data = np.load(MeshFileName(fileName))
    return data['points'].astype(float), data['triangles'].astype(int), data['normals'].astype(float)


def LoadMeshGraphics(fileName, color):
    """Load mesh file as Exudyn graphics data with smooth vertex normals."""
    points, triangles, normals = LoadMesh(fileName)
    return graphics.FromPointsAndTrigs(points, triangles, color=color, normals=normals)


def GetTablePoses():
    """Return list of (tablePosition, tableHeight) of the small tables 1 and 2 in K0."""
    return [(np.array(par.table1Position), par.table1Height), (np.array(par.table2Position), par.table2Height)]


def CreateEnvironmentGraphics(modelData, includeLabTables=True):
    """Return list of graphics data of the lab table (CAD) and, optionally, of the small tables 1/2 with cup holders (K0)."""
    environmentGraphics = [LoadMeshGraphics(entry['file'], colorGroupColors[entry['colorGroup']])
                           for entry in modelData['meshFiles'] if entry['link'] == -2]
    if includeLabTables:
        environmentGraphics += CreateLabTablesGraphics(modelData)
    return environmentGraphics


def CreateLabTablesGraphics(modelData, offset=(0., 0., 0.)):
    """Return list of graphics data of the small tables 1/2 with their cup holders in K0, moved by offset in m (e.g. at
    robots side by side)."""
    labObjects = modelData['labObjects']
    tablesGraphics = []
    for tableName, (tablePosition, tableHeight) in zip(['table1', 'table2'], GetTablePoses()):
        tableGraphics = LoadMeshGraphics(labObjects[tableName]['file'], colorGroupColors['labObject'])
        holderGraphics = LoadMeshGraphics(labObjects['cupHolder']['file'], colorGroupColors['labObject'])
        tablesGraphics.append(graphics.Move(tableGraphics, tablePosition + offset, np.eye(3)))
        tablesGraphics.append(graphics.Move(holderGraphics, tablePosition + offset + [0., 0., tableHeight], np.eye(3)))
    return tablesGraphics


def CreateCupGraphics():
    """Return graphics data of the paper cup in the cup frame (origin at bottom center, z = cup axis, open top)."""
    wallThickness = 0.001
    contour = [[0., 0.], [0., par.cupBottomRadius], [par.cupHeight, par.cupTopRadius],
               [par.cupHeight, par.cupTopRadius - wallThickness], [wallThickness, par.cupBottomRadius - wallThickness],
               [wallThickness, 0.]]
    return graphics.SolidOfRevolution(pAxis=[0., 0., 0.], vAxis=[0., 0., 1.], contour=contour,
                                      color=colorGroupColors['cup'], nTiles=cupContourTiles)


def GetInitialCupHT():
    """Return the initial pose of the cup: standing on the cup holder of table 1."""
    HT = np.eye(4)
    HT[0:3, 3] = np.array(par.table1Position) + [0., 0., par.table1Height + par.cupHolderHeight]
    return HT


def GetCupPlacementHeight(position):
    """Return height of the cup holder surface below a horizontal position (table 1 or 2), or 0 (lab table) elsewhere."""
    for tablePosition, tableHeight in GetTablePoses():
        if np.linalg.norm(np.array(position[0:2]) - tablePosition[0:2]) < cupHolderRadius:
            return tableHeight + par.cupHolderHeight
    return 0.


def RobotGraphicsAtJointAngles(linkGraphics, jawGraphics, q, offset=(0., 0., 0.), baseGraphics=None, jawOpeningPercent=0.):
    """Return graphics of the 6R robot drawn at joint angles q (rad), without simulating it (
    the solutions of the inverse kinematics, robots side by side).

    Args:
        linkGraphics (list): graphics of the links 1..6 in their joint frames, e.g. [link.visualization.graphicsData]
                             of labRobots.robots.robot6R.model.ManipulatorRobot6R
        jawGraphics (list): graphics of the jaws in the TCP frame (labRobots.robots.robot6R.visualization.CreateJawGraphics)
        q (array): joint angles in rad
        offset (array): position in m of the robot base K0 in the world (robots side by side)
        baseGraphics (list): graphics of the base in K0, None: no base
        jawOpeningPercent (float): gripper command of the jaws in percent (0: open)

    Returns:
        list: graphics data
    """
    from labRobots.robots.robot6R.model import ComputeJointAndLinkFrames, ForwardKinematics
    from labRobots.robots.robot6R import visualization as robotVisualization
    offset = np.asarray(offset, dtype=float)
    jointFrames, unusedLinkFrames = ComputeJointAndLinkFrames(np.asarray(q, dtype=float))
    graphicsList = [] if baseGraphics is None else [graphics.Move(graphicsData, offset, np.eye(3)) for graphicsData in baseGraphics]
    for frame, graphicsOfLink in zip(jointFrames, linkGraphics):
        graphicsList += [graphics.Move(graphicsData, frame[0:3, 3] + offset, frame[0:3, 0:3]) for graphicsData in graphicsOfLink]
    tcpPose = ForwardKinematics(np.asarray(q, dtype=float))
    tcpPose[0:3, 3] += offset
    graphicsList += robotVisualization.MoveJawGraphics(jawGraphics, tcpPose, jawOpeningPercent)
    return graphicsList
