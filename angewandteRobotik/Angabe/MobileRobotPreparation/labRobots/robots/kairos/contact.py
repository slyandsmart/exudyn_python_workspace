# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: contact.py
# Details:  Cylinder sphere contact against floor, table and moving gripper boxes.
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-05; 2026-11-12 moved into labRobots/robots/kairos (from LabCourse/MobileRobot/fahrzeug); the parts of the lab scene stay there
# Notes:    Uses the sphere/triangle approach of robot6R.contact in SI units.
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Physical object contact."""
import numpy as np
from exudyn.rigidBodyUtilities import RotationMatrixZ
from labRobots.robots.kairos import parameters as par
from labRobots.robots.kairos.visualization import GripperFingerGraphics, GripperHousingGraphics
from labRobots.robots.kairos.treeBody import AddRigidMarker
from labRobots.robotics.contact.generalContact import AddContactMesh as SharedContactMesh, AddContactBoxes


def AddContactMesh(contact, marker, mesh, static=False, rotation=None):
    """Register an outward oriented mesh with the contact values of the Kairos and its scene
    (labRobots.robotics.contact.generalContact.AddContactMesh: shared with the Leobot)."""
    return SharedContactMesh(contact, marker, mesh, par.objectContactStiffness, par.objectContactDamping, static, rotation)


def AddGripperContact(contact, mbs, flangeBody):
    """Add the WSG50 housing and the two fingers on the flange of the arm.

    Args:
        contact: GeneralContact of the scene
        mbs: the MainSystem of the robot
        flangeBody: body of the flange (TreeBody or body index)

    Returns:
        list of the flange-attached finger markers (moved with the prescribed finger opening)
    """
    flangeMarker = AddRigidMarker(mbs, flangeBody)
    AddContactBoxes(contact, flangeMarker, GripperHousingGraphics(), par.gripperContactStiffness, par.objectContactDamping)
    fingerMarkers = []
    fingerGraphics = GripperFingerGraphics()
    for fingerIndex in range(par.numberOfGripperFingers):
        rotation = RotationMatrixZ(np.pi*fingerIndex)
        offset = par.gripperOpenWidth/2. + par.gripperFingerFaceOffset
        marker = AddRigidMarker(mbs, flangeBody, rotation @ [-offset, 0., 0.])
        AddContactBoxes(contact, marker, fingerGraphics, par.gripperContactStiffness, par.objectContactDamping, rotation)
        fingerMarkers.append(marker)
    return fingerMarkers
