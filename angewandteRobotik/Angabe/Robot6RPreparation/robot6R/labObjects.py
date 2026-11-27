# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: labObjects.py
#
# Details:  The lab objects of the 6R task (lab table, small tables, cup holders; labObjectData.json and meshes/ of
#           this package) merged with the robot data of labRobots.robots.robot6R.model (LoadModelData, MeshFileName), and the
#           paper cup as payload (GetCupInertia, GetPayloadInertia).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-03; 2026-11-12 the part of the course of robot6R/model.py (the robot is labRobots.robots.robot6R.model)
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Lab objects and cup of the 6R task."""
import os
import json

from exudyn.rigidBodyUtilities import RigidBodyInertia, InertiaCylinder

from labRobots.robots.robot6R.model import LoadRobotModelData, GetLinkLocalHT
from labRobots.robots.robot6R.visualization import RobotMeshFileName
from . import parameters as par

packageDirectory = os.path.dirname(os.path.abspath(__file__))
#! CAD based data of the lab objects of the course (lab table, small tables, cup holder).
labObjectDataFileName = os.path.join(packageDirectory, 'data', 'labObjectData.json')
meshDirectory = os.path.join(packageDirectory, 'meshes')


def LoadModelData():
    """Load the CAD based model data: the robot (LoadRobotModelData) merged with the lab objects of the course
    (labObjectData.json): 'meshFiles' holds both mesh lists, 'labObjects' the gripper jaws and the lab objects,
    'gripperJaws' the jaws (for the graphics of labRobots.robots.robot6R.visualization)."""
    modelData = LoadRobotModelData()
    if not os.path.isfile(labObjectDataFileName):
        raise FileNotFoundError('model data ' + labObjectDataFileName + ' not found (converted from CAD; the conversion tool is removed)')
    with open(labObjectDataFileName, 'r', encoding='utf-8') as fileHandle:
        labData = json.load(fileHandle)
    modelData['meshFiles'] = labData['meshFiles'] + modelData['meshFiles']
    modelData['labObjects'] = dict(modelData['gripperJaws'], **labData['labObjects'])
    return modelData


def MeshFileName(fileName):
    """Return the path of a mesh file of LoadModelData: robot meshes in the device data, lab objects in the course."""
    robotFileName = RobotMeshFileName(fileName)
    return robotFileName if os.path.isfile(robotFileName) else os.path.join(meshDirectory, fileName)


def GetCupInertia(mass):
    """Return RigidBodyInertia of the cup with liquid (cylinder approximation) in the cup frame (origin at bottom center).

    Args:
        mass (float): mass of cup and liquid in kg
    """
    meanRadius = 0.5*(par.cupBottomRadius + par.cupTopRadius)
    cylinder = InertiaCylinder(density=1., length=par.cupHeight, outerRadius=meanRadius, axis=2)
    return RigidBodyInertia(mass=mass, inertiaTensor=cylinder.InertiaCOM()*mass/cylinder.Mass(),
                            com=[0., 0., 0.], inertiaTensorAtCOM=True).Translated([0., 0., 0.5*par.cupHeight])


def GetPayloadInertia(payloadMass, payloadHTinTCP):
    """Return the cup inertia transformed from its frame into joint frame 6."""
    return GetCupInertia(payloadMass).Transformed(GetLinkLocalHT(5) @ payloadHTinTCP)
