# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: scene.py
#
# Details:  The lab scene of the Kairos task in the style of exudyn.robotics: CreateLabScene(mbs) builds the floor with
#           the Vicon origin, the GeneralContact of the scene, the lab table and the free cup with their contact into a
#           MainSystem and returns the dict of the created items, among them the marker of the floor and the contact for
#           the robots (CreateKairos(..., markerGround, generalContact)) and the table as obstacle of the arm planning.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-12 (before part of VehicleModel.CreateObject)
# Notes:    The table stands below the cup: its top at the cup bottom (with Vicon the cup height comes from the
#           measured marker; run 20261107-183943: the real table is 9 mm lower than par.tableHeight).
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Lab scene of the Kairos task."""
import numpy as np
import exudyn as exu
from exudyn.itemInterface import ObjectGround, VObjectGround
from exudyn.rigidBodyUtilities import InertiaCylinder
from . import parameters as par
from labRobots.robots.kairos.treeBody import AddRigidMarker
from .visualization import ObjectGraphics, FloorGraphics, ViconOriginGraphics
from .table import TableGraphics
from .collisionModel import TableObstacle
from .contact import CreateSceneContact, AddTableContact, AddObjectContact


def CreateLabScene(mbs, objectPosition=None):
    """Create floor, Vicon origin, contact, table and cup of the lab in a MainSystem, as mbs.CreateRigidBody.

    Args:
        mbs: the MainSystem, which will be extended
        objectPosition: center of the cup in world coordinates in m; the table stands below it (its top at the cup
            bottom); None: par.objectPosition

    Returns:
        dict with 'ground' (floor), 'markerGround' (rigid marker of the floor, for the wheels), 'viconOrigin' (drawn
        with the coordinate frames), 'generalContact' (floor, table, cup; the gripper of a robot is added by
        CreateKairos), 'cup' (rigid body), 'table' (ground object; its reference position shifts graphics and contact),
        'tablePosition' (world x, y in m of the table center), 'tableTopHeight' (m) and 'obstacles' (the table as one
        box in world coordinates, for KairosModel.SetEnvironmentObstacles)

    Example:
        scene = CreateLabScene(mbs)
        kairos = CreateKairos(mbs, markerGround=scene['markerGround'], generalContact=scene['generalContact'])
        kairos.SetEnvironmentObstacles(scene['obstacles'])
    """
    objectPosition = np.asarray(par.objectPosition if objectPosition is None else objectPosition, dtype=float)
    ground = mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsData=FloorGraphics())))
    markerGround = AddRigidMarker(mbs, ground)
    #own object, shown with the coordinate frames (button Koordinatensysteme anzeigen)
    viconOrigin = mbs.AddObject(ObjectGround(visualization=VObjectGround(show=False, graphicsData=ViconOriginGraphics())))
    contact = CreateSceneContact(mbs, ground)
    #the table/contact supports the cylinder under gravity.
    cup = mbs.CreateRigidBody(
        inertia=InertiaCylinder(par.objectDensity, par.objectHeight, par.objectRadius, axis=2),
        referencePosition=objectPosition, gravity=par.gravity, nodeType=exu.NodeType.RotationRotationVector,
        graphicsDataList=ObjectGraphics())   #shape of the real cup with Vicon sphere
    tablePosition = np.array(objectPosition[:2], dtype=float)
    tableTopHeight = float(objectPosition[2] - par.objectHeight/2.)
    #walls, shelf, top plate, built at par.tableHeight and shifted to the cup bottom; movable
    #table (mirror mode): the reference position of its ground object shifts the graphics and the contact triangles
    table = mbs.AddObject(ObjectGround(referencePosition=[0., 0., tableTopHeight - par.tableHeight],
                                       visualization=VObjectGround(graphicsData=TableGraphics(tablePosition))))
    AddTableContact(contact, mbs, table, tablePosition)
    AddObjectContact(contact, mbs, cup)
    return {'ground': ground, 'markerGround': markerGround, 'viconOrigin': viconOrigin, 'generalContact': contact,
            'cup': cup, 'table': table, 'tablePosition': tablePosition, 'tableTopHeight': tableTopHeight,
            'obstacles': [TableObstacle(tablePosition, tableTopHeight)]}
