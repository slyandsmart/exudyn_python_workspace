# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: planning.py
#
# Details:  Collision-free motions of the 6R robot with the collision check and the path planning of Exudyn only
#           (exudyn.cUtilities.CollisionModel, exudyn.robotics.navigation), as the planners of the Kairos and the
#           Leobot: capsules fitted to the CAD meshes of each link (with gripper housing and the open jaws on link 6)
#           and of the socket (exudyn.robotics.urdfImport.FitCapsulesToMesh, as tools/showCollisionShapes.py of the
#           course), stored once beside the robot data; the obstacles of the scene in the base frame K0; RRT-Connect
#           (PlanJointPath), shortcuts (ShortcutPath) and a blended trajectory (TrajectoryFromPath) of the shared
#           ArmPlanner (labRobots.arms.armPlanning: before a planner of its own).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-12
# Notes:    The fitted capsules of the compact modules overlap in every configuration for the link pairs socket - link 2,
#           link 2 - link 4 and link 4 - link 6 (wrist); CollisionModel excludes these whole link pairs
#           (ExcludeCollidingPairs with margin 0 at collisionReferenceAnglesDeg). Fitting the capsules takes about a
#           minute; the result is stored in collisionCapsulesFileName and fitted again when the robot data change.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Collision-free arm motions of the 6R robot by Exudyn."""
import hashlib
import json
import numpy as np
import exudyn.graphics as graphics
from exudyn.robotics import Robot, RobotLink, RobotBase, RobotTool
from exudyn.robotics.urdfImport import FitCapsulesToMesh
from labRobots.labSetup.dataPaths import robot6RDescriptionFolder
from labRobots.robots.robot6R import parameters as par
from labRobots.robots.robot6R.gripper import JawOffset
from labRobots.arms.armPlanning import ArmPlanner

#! File of the fitted collision capsules (beside robotModelData.json of the device data).
collisionCapsulesFileName = robot6RDescriptionFolder / 'exudyn' / 'collisionCapsules.json'
#! Version of the fitting (2026-11-12 b: the jaws in the DH frame of link 6, before they were 6 cm off).
capsulesVersion = 2
#! Capsules fitted per link and to the socket.
capsulesPerLink, capsulesOfBase = 2, 1
#! Joint coordinates at which the link pairs overlapping by construction are found (home position of the course).
collisionReferenceAnglesDeg = [0., -60., 90., 0., 30., 0.]
#! Distance in m which the shapes keep (as the planning of the course, tools/showCollisionShapes.py).
planningMargin = 0.005


def GraphicsPointsAndTriangles(graphicsList):
    """Return (points, triangles) of a list of triangle list graphics merged into one mesh."""
    pointList, triangleList, offset = [], [], 0
    for item in graphicsList:
        points, triangles = graphics.ToPointsAndTrigs(item)
        pointList.append(np.asarray(points, dtype=float))
        triangleList.append(np.asarray(triangles, dtype=int).reshape(-1, 3) + offset)
        offset += len(pointList[-1])
    return np.vstack(pointList), np.vstack(triangleList)


def CapsulesSourceKey():
    """Return the key of the capsules: the robot data, the open jaws and the numbers of capsules."""
    from labRobots.robots.robot6R.model import robotModelDataFileName
    with open(robotModelDataFileName, 'rb') as fileHandle:
        dataHash = hashlib.sha256(fileHandle.read()).hexdigest()
    return json.dumps({'version': capsulesVersion, 'robotModelData': dataHash, 'jawOffset': round(float(JawOffset(par.gripperInitialPercent)), 9),
                       'capsulesPerLink': capsulesPerLink, 'capsulesOfBase': capsulesOfBase})


def FitCollisionCapsules():
    """Return (capsules of the 6 links in their DH link frames, capsules of the socket in K0) fitted to the meshes of
    ManipulatorRobot6R; link 6 with the gripper housing and both jaws at the initial (open) command."""
    from labRobots.robots.robot6R.model import ManipulatorRobot6R, GetLinkLocalHT
    description = ManipulatorRobot6R(includeJaws=True)
    #the frames of the links at the open jaws (Robot.LinkHT): the jaw links hang at the DH frame of link 6 (the TCP)
    frames = [np.array(frame) for frame in description.LinkHT([0.]*par.numberOfJoints +
                                                              [JawOffset(par.gripperInitialPercent)]*len(par.jawNames))]
    linkCapsules = []
    for linkIndex in range(par.numberOfJoints):
        toLink = np.linalg.inv(np.asarray(GetLinkLocalHT(linkIndex)))     #the meshes of a link are in its joint frame
        parts = [(description.links[linkIndex].visualization.graphicsData, toLink)]
        if linkIndex == par.numberOfJoints - 1:      #the open jaws, from their link frames into the DH frame of link 6
            parts += [(jawLink.visualization.graphicsData, np.linalg.inv(frames[linkIndex]) @ frames[jawIndex])
                      for jawIndex, jawLink in enumerate(description.links) if jawIndex >= par.numberOfJoints]
        pointList, triangleList, offset = [], [], 0
        for graphicsList, HT in parts:
            points, triangles = GraphicsPointsAndTriangles(graphicsList)
            pointList.append(points @ HT[0:3, 0:3].T + HT[0:3, 3])
            triangleList.append(triangles + offset)
            offset += len(points)
        linkCapsules.append(FitCapsulesToMesh(np.vstack(pointList), np.vstack(triangleList), capsulesPerLink))
    points, triangles = GraphicsPointsAndTriangles(description.base.visualization.graphicsData)
    return linkCapsules, FitCapsulesToMesh(points, triangles, capsulesOfBase)


def CollisionCapsules():
    """Return (capsules of the links, capsules of the socket) as dicts of exudyn.robotics (type, pAxis, vAxis, radius):
    from collisionCapsulesFileName if they were fitted for the same robot data, else fitted and stored there."""
    key = CapsulesSourceKey()
    if collisionCapsulesFileName.is_file():
        with open(collisionCapsulesFileName, 'r', encoding='utf-8') as fileHandle:
            stored = json.load(fileHandle)
        if stored.get('sourceKey') == key:
            return stored['links'], stored['base']
    linkCapsules, baseCapsules = FitCollisionCapsules()
    Plain = lambda capsules: [{'type': 'capsule', 'pAxis': [float(value) for value in shape['pAxis']],
                               'vAxis': [float(value) for value in shape['vAxis']], 'radius': float(shape['radius'])}
                              for shape in capsules]
    stored = {'sourceKey': key, 'links': [Plain(capsules) for capsules in linkCapsules], 'base': Plain(baseCapsules)}
    with open(collisionCapsulesFileName, 'w', encoding='utf-8') as fileHandle:
        json.dump(stored, fileHandle, indent=1)
    return stored['links'], stored['base']


class Robot6RArmPlanner(ArmPlanner):
    """Plans collision-free motions of the 6 joints of the 6R (base frame K0) with the CollisionModel of Exudyn: the
    shared ArmPlanner with the DH kinematics and the fitted capsules of the 6R, the joint limits (within +-pi), speeds
    and accelerations of the lab software."""

    def __init__(self):
        """Build the DH kinematics of the 6R with the collision capsules of links and socket (no obstacles yet)."""
        from labRobots.robots.robot6R.model import GetLinkLocalHT
        linkCapsules, baseCapsules = CollisionCapsules()
        #the kinematics in K0 with the capsules (DH link frames, as its KinematicChain; tool = DH frame 6, the TCP)
        robot = Robot(gravity=par.gravity, base=RobotBase(), tool=RobotTool(), referenceConfiguration=[])
        for linkIndex in range(par.numberOfJoints):
            robot.AddLink(RobotLink(mass=0., COM=np.zeros(3), inertia=np.zeros((3, 3)),
                                    localHT=GetLinkLocalHT(linkIndex), collisionShapes=linkCapsules[linkIndex]))
        robot.base.collisionShapes = baseCapsules
        lower = np.maximum(np.radians(par.labJointLowerLimitsDeg), -np.pi)
        upper = np.minimum(np.radians(par.labJointUpperLimitsDeg), np.pi)
        ArmPlanner.__init__(self, robot, np.column_stack([lower, upper]), np.radians(par.labMaxVelocitiesDeg),
                            np.radians(par.labMaxAccelerationsDeg), np.radians(collisionReferenceAnglesDeg), planningMargin)
