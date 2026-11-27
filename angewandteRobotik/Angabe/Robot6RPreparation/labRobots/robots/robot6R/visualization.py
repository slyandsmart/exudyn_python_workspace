# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: visualization.py
#
# Details:  Graphics of the 6R robot for Exudyn: smooth meshes with vertex normals of base and links (from the CAD
#           conversion), gripper housing and jaws; jaw opening of the parallelogram gripper.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-03; 2026-11-12 moved from LabCourse/Robot6R/robot6R/visualization.py (lab tables and cup stay there)
# Notes:    Link meshes are given in the joint frames (KinematicTree link frames), base in base frame K0, jaws in
#           their jaw link frames (TCP frame, right jaw rotated by 180 deg about z6). modelData: LoadRobotModelData of
#           model.py (links, meshFiles, gripperJaws).
#
# License:  BSD-3 license
#
# Installation:
#           pip install exudyn
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Graphics of the 6R robot."""
import os
import numpy as np

import exudyn.graphics as graphics

from labRobots.labSetup.dataPaths import robot6RDescriptionFolder
from labRobots.robots.robot6R import parameters as par
from labRobots.robots.robot6R.gripper import JawOffset

#! Folder of the meshes of the robot (device data, written by the CAD conversion of the course).
robotMeshDirectory = os.path.join(str(robot6RDescriptionFolder), 'exudyn', 'meshes')
#! RGBA colors of the color groups of the robot written by the CAD conversion.
robotColorGroupColors = {'schunk': [0.45, 0.52, 0.62, 1.],        #anodized Schunk modules
                         'connectionCap': [0.80, 0.81, 0.83, 1.], #connection caps (ASK)
                         'printed': [0.60, 0.63, 0.68, 1.],       #PLA silver metallic
                         'gear': [0.93, 0.93, 0.90, 1.],          #POM bevel gears
                         'aluminum': [0.82, 0.83, 0.85, 1.],      #socket, adapter
                         'steel': [0.70, 0.70, 0.72, 1.],
                         'servo': [0.10, 0.10, 0.10, 1.]}


def RobotMeshFileName(fileName):
    """Return the path of a mesh file of the robot (LoadRobotModelData of model.py)."""
    return os.path.join(robotMeshDirectory, fileName)


def LoadRobotMesh(fileName):
    """Load a mesh (points, triangles, vertex normals) of the robot."""
    data = np.load(RobotMeshFileName(fileName))
    return data['points'].astype(float), data['triangles'].astype(int), data['normals'].astype(float)


def LoadRobotMeshGraphics(fileName, color):
    """Load a mesh of the robot as Exudyn graphics data with smooth vertex normals."""
    points, triangles, normals = LoadRobotMesh(fileName)
    return graphics.FromPointsAndTrigs(points, triangles, color=color, normals=normals)


def CreateLinkGraphics(modelData, linkIndex):
    """Return list of graphics data of link linkIndex (0..5) in its joint frame."""
    return [LoadRobotMeshGraphics(entry['file'], robotColorGroupColors[entry['colorGroup']])
            for entry in modelData['meshFiles'] if entry['link'] == linkIndex]


def CreateBaseGraphics(modelData):
    """Return list of graphics data of the robot base (socket, fixed flange of PW 90) in base frame K0."""
    return [LoadRobotMeshGraphics(entry['file'], robotColorGroupColors[entry['colorGroup']])
            for entry in modelData['meshFiles'] if entry['link'] == -1]


def CreateGripperHousingGraphics(modelData):
    """Return list of graphics data of the static gripper parts (adapter plate, servo housing, servos) in joint frame 6."""
    flangeZ = -par.flangeToTCPDistance
    housingTopZ = modelData['gripperJaws']['jawRight']['mountZ'] - par.gripperJawFlangeSize[2]
    housingBottomZ = flangeZ + par.gripperAdapterThickness
    housingHeight = housingTopZ - housingBottomZ
    servoSize = [par.gripperBodySize[0] + 0.004, 0.020, 0.8*housingHeight]
    gripperGraphicsTCP = [graphics.Cylinder([0., 0., flangeZ], [0., 0., par.gripperAdapterThickness],
                                            radius=par.gripperAdapterRadius, color=robotColorGroupColors['aluminum'], nTiles=32),
                          graphics.Brick([0., 0., housingBottomZ + 0.5*housingHeight],
                                         [par.gripperBodySize[0], par.gripperBodySize[1], housingHeight], robotColorGroupColors['printed'])]
    for side in [-1., 1.]:
        gripperGraphicsTCP.append(graphics.Brick([0., side*0.0115, housingBottomZ + 0.5*housingHeight], servoSize,
                                                 robotColorGroupColors['servo']))
    from labRobots.robots.robot6R.model import GetLinkLocalHT
    tcpToJoint = GetLinkLocalHT(5)  #TCP frame -> joint frame 6
    return [graphics.Move(graphicsData, tcpToJoint[0:3, 3], tcpToJoint[0:3, 0:3]) for graphicsData in gripperGraphicsTCP]


def GetJawLinkRotation(jawName):
    """Return rotation of the jaw link frame relative to the TCP frame: the right jaw (at -y6) is rotated by 180 deg
    about z6, so that the positive jaw coordinate opens both jaws."""
    if jawName == 'jawRight':
        return np.diag([-1., -1., 1.])
    return np.eye(3)


def CreateJawGraphics(modelData, jawName=None):
    """Return jaw graphics in the jaw link frame, or both jaws in the TCP frame if no name is given."""
    if jawName is None:
        return [graphics.Move(CreateJawGraphics(modelData, name), [0., 0., 0.], GetJawLinkRotation(name))
                for name in par.jawNames]
    mountZ = modelData['gripperJaws'][jawName]['mountZ']
    side = -1. if jawName == 'jawRight' else 1.
    jawGraphics = LoadRobotMeshGraphics(modelData['gripperJaws'][jawName]['file'], robotColorGroupColors['printed'])
    flangeCenter = [0., side*(0.5*par.gripperJawFlangeSize[1] + 0.002), mountZ - 0.5*par.gripperJawFlangeSize[2]]
    flangeGraphics = graphics.Brick(flangeCenter, par.gripperJawFlangeSize, robotColorGroupColors['printed'], addNormals=True)
    rotation = GetJawLinkRotation(jawName)
    return graphics.Move(graphics.MergeTriangleLists(jawGraphics, flangeGraphics), [0., 0., 0.], rotation.T)


def MoveJawGraphics(jawGraphics, HTtcp, openingPercent):
    """Move TCP-frame jaw graphics to their commanded positions in world coordinates."""
    offset = ComputeJawOffset(openingPercent)
    return [graphics.Move(jaw, HTtcp[0:3, 3] + HTtcp[0:3, 0:3] @ [0., side*offset, 0.], HTtcp[0:3, 0:3])
            for jaw, side in zip(jawGraphics, [-1., 1.])]


def ComputeJawOffset(openingPercent):
    """Compute the commanded jaw coordinate (m): displacement of each jaw along the closing direction relative to the
    position with the jaw arc center on the TCP (reached at gripperContactPercent).

    Args:
        openingPercent (float): gripper command in percent (0 = open, 70 = closed)

    Returns:
        float: positive = opened; negative for commands above gripperContactPercent (stronger squeezing)
    """
    return JawOffset(openingPercent)   #the gripper model of the robot (labRobots.robots.robot6R.gripper)
