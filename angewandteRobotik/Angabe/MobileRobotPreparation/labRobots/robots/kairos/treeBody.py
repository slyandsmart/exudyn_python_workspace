# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: treeBody.py
# Details:  Logical rigid body frames on KinematicTree links and shared markers.
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-05; 2026-11-12 moved into labRobots/robots/kairos (from LabCourse/MobileRobot/fahrzeug)
# Notes:    Keeps DH link frames distinct from the tree's joint frames.
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Rigid frame access for tree links."""
import numpy as np
import exudyn as exu
from exudyn.itemInterface import MarkerBodyRigid, MarkerKinematicTreeRigid
from exudyn.rigidBodyUtilities import HTtranslate


class TreeBody:
    """A physical body frame attached to one joint frame of the vehicle tree."""

    def __init__(self, mbsInit, objectNumberInit, linkNumberInit, localPoseInit=None):
        #! Store the joint-to-body transformation and create a reusable output marker.
        self.objectNumber = objectNumberInit  #! KinematicTree object index.
        self.linkNumber = linkNumberInit  #! Tree joint/link index.
        self.localPose = np.eye(4) if localPoseInit is None else np.asarray(localPoseInit).copy()  #! Body frame in joint frame.
        self.markerNumber = mbsInit.AddMarker(MarkerKinematicTreeRigid(
            objectNumber=self.objectNumber, linkNumber=self.linkNumber, localHT=self.localPose))  #! Body-origin marker.


def AddRigidMarker(mbs, body, localPosition=None):
    #! Return a rigid marker on a tree body or an ordinary rigid/ground object.
    if isinstance(body, TreeBody):
        if localPosition is None:
            return body.markerNumber
        return mbs.AddMarker(MarkerKinematicTreeRigid(objectNumber=body.objectNumber, linkNumber=body.linkNumber,
                                                      localHT=body.localPose @ HTtranslate(localPosition)))
    return mbs.AddMarker(MarkerBodyRigid(bodyNumber=body,
                                         localPosition=[0., 0., 0.] if localPosition is None else localPosition))


def GetBodyOutput(mbs, body, outputType, configuration=exu.ConfigurationType.Current):
    #! Read physical body-frame outputs independently of the multibody formulation; configuration Visualization in
    #! graphics user functions.
    if isinstance(body, TreeBody):
        return mbs.GetMarkerOutput(body.markerNumber, outputType, configuration)
    return mbs.GetObjectOutputBody(body, outputType, [0., 0., 0.], configuration)


def SetRigidMarkerPosition(mbs, marker, body, localPosition):
    #! Move a marker in physical body coordinates while preserving its orientation.
    position = np.asarray(localPosition)
    if isinstance(body, TreeBody):
        position = body.localPose[:3, :3] @ position + body.localPose[:3, 3]
    mbs.SetMarkerParameter(marker, 'localPosition', position)
