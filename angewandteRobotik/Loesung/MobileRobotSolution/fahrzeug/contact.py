# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: contact.py
# Details:  Cylinder sphere contact against floor, table and moving gripper boxes.
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-05: the parts of the robot are in labRobots/robots/kairos/contact.py
# Notes:    Uses the sphere/triangle approach of robot6R.contact in SI units.
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Physical object contact."""
import numpy as np
import exudyn.graphics as graphics
from exudyn.itemInterface import MarkerBodyRigid
from . import parameters as par
from .table import TableGraphics
from labRobots.robots.kairos.contact import AddContactMesh


def CylinderContactPoints():
    #! Return local sphere centers covering the cylinder wall and both end faces.
    radius = par.objectContactSphereRadius
    halfLength = par.objectHeight/2. - radius
    points = [[0., 0., -halfLength], [0., 0., halfLength]]
    for height in np.linspace(-halfLength, halfLength, par.objectContactNumberOfRings):
        for angle in np.linspace(0., 2.*np.pi, par.objectContactSpheresPerRing, endpoint=False):
            points.append([(par.objectRadius - radius)*np.cos(angle),
                           (par.objectRadius - radius)*np.sin(angle), height])
    return np.asarray(points)


def CreateSceneContact(mbs, groundBody):
    """Create the GeneralContact of a scene with the floor (the scene owns the contact; table,
    objects and the gripper of the robots are added to it).

    Args:
        mbs: the MainSystem of the scene
        groundBody: ground object index of the floor

    Returns:
        the GeneralContact
    """
    contact = mbs.AddGeneralContact()
    contact.SetFrictionPairings(np.array([[par.objectContactFriction]]))
    contact.SetFrictionProportionalZone(par.objectContactFrictionZone)
    # Spheres on the same cylinder overlap intentionally; only sphere/triangle pairs are needed.
    contact.sphereSphereContact = False
    contact.SetSearchTreeCellSize(numberOfCells=par.objectContactSearchCells)
    groundMarker = mbs.AddMarker(MarkerBodyRigid(bodyNumber=groundBody))
    AddContactMesh(contact, groundMarker, graphics.Brick(
        [0., 0., -par.floorThickness/2.],
        [2.*par.floorHalfLength, 2.*par.floorHalfLength, par.floorThickness]), static=True)
    return contact


def AddTableContact(contact, mbs, tableBody, tablePosition):
    """Add walls, shelf and top plate of the lab table (not static: the table can be moved in
    the mirror mode).

    Args:
        contact: GeneralContact of the scene
        mbs: the MainSystem of the scene
        tableBody: ground object of the table (its reference position shifts the table)
        tablePosition: world x, y in m of the table center at which the parts are built
    """
    tableMarker = mbs.AddMarker(MarkerBodyRigid(bodyNumber=tableBody))
    for brick in TableGraphics(tablePosition):
        AddContactMesh(contact, tableMarker, brick)


def AddObjectContact(contact, mbs, objectBody):
    """Add the contact spheres of the cylindrical object (cup) on its wall and end faces.

    Args:
        contact: GeneralContact of the scene
        mbs: the MainSystem of the scene
        objectBody: rigid body of the object
    """
    for point in CylinderContactPoints():
        marker = mbs.AddMarker(MarkerBodyRigid(bodyNumber=objectBody, localPosition=point))
        contact.AddSphereWithMarker(markerIndex=marker, radius=par.objectContactSphereRadius,
                                    contactStiffness=par.objectContactStiffness,
                                    contactDamping=par.objectContactDamping, frictionMaterialIndex=0)
