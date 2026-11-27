# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: mounting.py
#
# Details:  The base of an exudyn.robotics.Robot built by CreateKinematicTree: a fixed base with a rotated base frame
#           (RotateFixedBaseIntoTree) and a floating base fixed to a rigid marker of another body (MountBase), for
#           all fixed base arms of the lab (6R, Staubli, Fanuc); HideHelperGraphics switches off the coordinate systems
#           of the joints of all kinematic trees and the loads (every robot model calls it in CreateMBS).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-13 (moved from labRobots/robots/robot6R/model.py, 2026-11-12/13, used by the industrial arms too)
# Notes:    Exudyn 1.13.599: CreateKinematicTree with a fixed base takes only the translation of base.HT.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Base of a Robot of CreateKinematicTree: rotated fixed base, mounted floating base."""
import numpy as np
import exudyn as exu
import exudyn.graphics as graphics
from exudyn.itemInterface import MarkerBodyRigid, ObjectJointGeneric


def HideHelperGraphics(mbs):
    """Switch off the helper graphics in the renderer of the SystemContainer of mbs: the coordinate systems of the joints
    of the kinematic trees (visualizationSettings.bodies.kinematicTree.showJointFrames) and the loads (gravity of the
    bodies, forces; visualizationSettings.loads.show); settings of the container, Exudyn has none per object. The robot
    models and scenes do not show them, also not in the images of their cameras (at the user's
    request: joint frames off for all models, forces off as the standard)."""
    settings = mbs.GetSystemContainer().visualizationSettings
    settings.bodies.kinematicTree.showJointFrames = False
    settings.loads.show = False


def RotateFixedBaseIntoTree(robot):
    """Move the rotation of the base of a Robot with a fixed base into the first link: Exudyn
    1.13.599 CreateKinematicTree with a fixed base takes only the translation of base.HT (baseOffset), the links stay
    unrotated and only the graphics of the socket turn. Afterwards base.HT is the translation, link 0 is turned by the
    rotation (preHT) and the socket graphics are turned. (A floating base mounted on the ground would take the rotation,
    but its rotation vector node fails with the numerical Jacobian of the scenes at large angles.)"""
    baseHT = np.array(robot.base.HT)
    rotation = baseHT[0:3, 0:3]
    if np.allclose(rotation, np.eye(3)):
        return
    robot.links[0].preHT = exu.HT(rotation=rotation.tolist(), translation=[0., 0., 0.]) * robot.links[0].preHT
    robot.base.HT = exu.HT(rotation=np.eye(3).tolist(), translation=baseHT[0:3, 3].tolist())
    robot.base.visualization.graphicsData = [graphics.Move(item, [0., 0., 0.], rotation)
                                             for item in robot.base.visualization.graphicsData]


def MountBase(mbs, robot, markerGround):
    """Fix the floating base of a Robot of CreateKinematicTree (items 'objectBase') to a rigid marker by a joint at the
    marker, keeping the reference pose of the base.

    Args:
        mbs: the MainSystem
        robot: the Robot with robot.items of CreateKinematicTree with a floating base
        markerGround: rigid marker of the body the base is fixed to

    Returns:
        the ObjectJointGeneric
    """
    reference = exu.ConfigurationType.Reference
    markerPosition = np.asarray(mbs.GetMarkerOutput(markerGround, exu.OutputVariableType.Position, reference), dtype=float)
    markerRotation = np.asarray(mbs.GetMarkerOutput(markerGround, exu.OutputVariableType.RotationMatrix,
                                                    reference), dtype=float).reshape(3, 3)
    baseHT = np.array(robot.base.HT)
    baseRotation = baseHT[0:3, 0:3]
    #the joint point at the marker, on the base body at the same place and in the same orientation (marker localHT)
    baseMarker = mbs.AddMarker(MarkerBodyRigid(bodyNumber=robot.items['objectBase'], localHT=exu.HT(
        rotation=(baseRotation.T @ markerRotation).tolist(),
        translation=(baseRotation.T @ (markerPosition - baseHT[0:3, 3])).tolist())))
    return mbs.AddObject(ObjectJointGeneric(markerNumbers=[markerGround, baseMarker], constrainedAxes=[1]*6))
