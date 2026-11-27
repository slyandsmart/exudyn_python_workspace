# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: platform.py
# Details:  Floating-base vehicle tree with physical wheels, UR5 and rolling contacts.
# Authors:  Martin Sereinig; Peter Manzl; Michael Pieber (created with Claude Code)
# Date:     2026-11-05; 2026-11-12 moved into labRobots/robots/kairos (from LabCourse/MobileRobot/fahrzeug)
# Notes:    Preserves the original inertias, geometry and rolling-disc penalty law.
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Coupled mobile manipulator dynamics."""
import numpy as np
import exudyn as exu
import exudyn.graphics as graphics
from exudyn.itemInterface import (NodeGenericODE2, ObjectKinematicTree,
                                 VObjectKinematicTree, NodeGenericData, ObjectConnectorRollingDiscPenalty,
                                 VObjectConnectorRollingDiscPenalty)
from exudyn.rigidBodyUtilities import (InertiaCuboid, InertiaCylinder, RigidBodyInertia, HTtranslate, HTrotateZ)
from labRobots.robots.kairos import parameters as par
from labRobots.robots.kairos.visualization import WheelGraphics, PlatformGraphics
from labRobots.robots.kairos.treeBody import TreeBody


def CreateVehicleTree(mbs, initialPose, baseGraphics, armRobot, groundMarker, name=''):
    #! Build a six-DOF floating base, four wheel branches and a six-joint arm branch.
    #the wheels roll on groundMarker (rigid marker of the floor of the scene); floor and Vicon
    #origin are created by the scene; name: name of the KinematicTree object
    wheelInertia = InertiaCylinder(par.wheelDensity, par.wheelWidth, par.wheelRadius, axis=0)
    bodyMass = par.platformMass - par.numberOfWheels*wheelInertia.mass
    width, length, height = par.platformDimensions
    #side lengths in the order x, y, z: the vehicle is long in x (0.718 m) and wide in y (0.575 m)
    #as the chassis mesh and the datasheet; the inherited call passed [width, length, height] and swapped Ixx and Iyy
    platformInertia = InertiaCuboid(bodyMass/(length*width*height), [length, width, height])
    platformInertia = platformInertia.Translated(par.platformMass/bodyMass*np.asarray(par.platformCOM))
    jointTypes, parents, jointPoses, inertias, linkGraphics, proportionalGains, derivativeGains = [], [], [], [], [], [], []

    def AddLink(jointType, parent, jointPose, inertia, graphicsList, proportionalGain=0., derivativeGain=0.):
        #! Append one minimal coordinate and its physical inertia, geometry and controller.
        linkIndex = len(jointTypes)
        jointTypes.append(jointType)
        parents.append(parent)
        jointPoses.append(jointPose)
        inertias.append(inertia)
        linkGraphics.append(graphicsList)
        proportionalGains.append(proportionalGain)
        derivativeGains.append(derivativeGain)
        return linkIndex

    # Zero-mass intermediate joints introduce coordinates, not artificial physical bodies.
    baseJointTypes = (exu.JointType.PrismaticX, exu.JointType.PrismaticY, exu.JointType.PrismaticZ,
                      exu.JointType.RevoluteZ, exu.JointType.RevoluteY, exu.JointType.RevoluteX)
    for coordinateIndex, jointType in enumerate(baseJointTypes):
        isPlatform = coordinateIndex == par.numberOfBaseCoordinates - 1
        platformLink = AddLink(jointType, coordinateIndex - 1, np.eye(4),
                              platformInertia if isPlatform else RigidBodyInertia(),
                              PlatformGraphics() + baseGraphics if isPlatform else [])
    wheelLinks = []
    for wheelIndex in range(par.numberOfWheels):
        longitudinal = par.wheelBase/2.*(1. if wheelIndex < 2 else -1.)
        lateral = par.wheelTrack/2.*(1. if wheelIndex in (0, 2) else -1.)
        wheelPose = HTtranslate([longitudinal, lateral, par.wheelRadius - par.platformCenterHeight]) @ HTrotateZ(-np.pi/2.)
        # The visual wheel's +x axle points along platform -y; its coordinate is the negative motor angle.
        wheelLinks.append(AddLink(exu.JointType.RevoluteX, platformLink, wheelPose, wheelInertia,
                                  WheelGraphics(wheelIndex), par.wheelStiffness, par.wheelDamping))
    armLinks, armBodyPoses = [], []
    parent, jointPose = platformLink, HTtranslate(par.mountPosition) @ HTrotateZ(par.mountYaw)
    for armLink in armRobot.links:
        bodyPose = np.asarray(armLink.localHT)
        inertia = RigidBodyInertia()
        inertia.SetWithCOMinertia(armLink.mass, armLink.inertia, armLink.COM)
        inertia = inertia.Transformed(bodyPose)
        armGraphics = [graphics.Move(mesh, bodyPose[:3, 3], bodyPose[:3, :3]) for mesh in armLink.visualization.graphicsData]
        parent = AddLink(exu.JointType.RevoluteZ, parent, jointPose, inertia, armGraphics, *armLink.GetPDcontrol())
        armLinks.append(parent)
        armBodyPoses.append(bodyPose)
        jointPose = bodyPose
    numberOfCoordinates = len(jointTypes)
    initialCoordinates = np.zeros(numberOfCoordinates)
    initialCoordinates[:par.dimension3D] = [initialPose[0], initialPose[1],
                                           par.platformCenterHeight + 125.*9.81/par.contactStiffness]
    initialCoordinates[par.baseYawCoordinate] = initialPose[2]
    #start angles of the arm model (par.armInitialAngles or the live robot state)
    initialCoordinates[armLinks] = armRobot.referenceConfiguration
    node = mbs.AddNode(NodeGenericODE2(referenceCoordinates=np.zeros(numberOfCoordinates),
                                      initialCoordinates=initialCoordinates,
                                      initialCoordinates_t=np.zeros(numberOfCoordinates),
                                      numberOfODE2Coordinates=numberOfCoordinates))
    tree = mbs.AddObject(ObjectKinematicTree(
        name=name, nodeNumber=node, jointTypes=jointTypes, linkParents=parents,
        jointTransformations=exu.Matrix3DList([pose[:3, :3] for pose in jointPoses]),
        jointOffsets=exu.Vector3DList([pose[:3, 3] for pose in jointPoses]),
        linkInertiasCOM=exu.Matrix3DList([inertia.InertiaCOM() for inertia in inertias]),
        linkCOMs=exu.Vector3DList([inertia.COM() for inertia in inertias]),
        linkMasses=[inertia.Mass() for inertia in inertias], gravity=par.gravity,
        jointPControlVector=proportionalGains, jointDControlVector=derivativeGains,
        jointPositionOffsetVector=initialCoordinates, jointVelocityOffsetVector=np.zeros(numberOfCoordinates),
        jointForceVector=np.zeros(numberOfCoordinates),
        visualization=VObjectKinematicTree(showJoints=False, showLinks=False, graphicsDataList=linkGraphics)))
    wheels = [TreeBody(mbs, tree, link) for link in wheelLinks]
    contacts = []
    for wheelIndex, wheel in enumerate(wheels):
        contactNode = mbs.AddNode(NodeGenericData(numberOfDataCoordinates=par.dimension3D, initialCoordinates=[0., 0., 0.]))
        frictionAngle = np.pi/4.*(-1. if wheelIndex in (1, 2) else 1.)
        contacts.append(mbs.AddObject(ObjectConnectorRollingDiscPenalty(
            markerNumbers=[groundMarker, wheel.markerNumber], nodeNumber=contactNode, discRadius=par.wheelRadius,
            dryFriction=par.dryFriction, dryFrictionAngle=frictionAngle,
            dryFrictionProportionalZone=0.01, viscousFriction=par.wheelViscousFriction, useLinearProportionalZone=True,
            contactStiffness=par.contactStiffness, contactDamping=par.contactDamping,
            visualization=VObjectConnectorRollingDiscPenalty(show=False))))
    bodies = {'platform': TreeBody(mbs, tree, platformLink), 'wheels': wheels, 'contacts': contacts}
    armBodies = [TreeBody(mbs, tree, link, pose) for link, pose in zip(armLinks, armBodyPoses)]
    treeData = {'object': tree, 'node': node, 'wheelCoordinates': np.asarray(wheelLinks),
                'armCoordinates': np.asarray(armLinks), 'graphics': linkGraphics,
                'positionOffsets': initialCoordinates.copy(), 'velocityOffsets': np.zeros(numberOfCoordinates),
                'forces': np.zeros(numberOfCoordinates)}
    return bodies, armBodies, treeData
