# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: preparation12ForwardKinematics.py
#
# Details:  Preparation task 1.2 (Vorbereitung): standard DH parameters of the 6R robot and their forward
#           kinematics (product of exudyn.robotics.StdDH2HT), validated with the three reference poses
#           (Val1)-(Val3) of the task sheet; afterwards the robot moves through the validation poses in the
#           Exudyn simulation. The DH table is built row by row: the newest row is animated in the renderer (rotation
#           about z by theta, translation along z by d, along x by a, rotation about x by alpha).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-03 (master file of the student template: 2026-11-08)
# Notes:    Run: python preparation12ForwardKinematics.py [--noGraphics]
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
import numpy as np
from exudyn.robotics import StdDH2HT

from robot6R import parameters as par
from robot6R.labTrajectory import CreateTrajectoryMatrix
from robot6R.simulation import SimulateBahnfahrt
from robot6R.dhAnimation import DHFrames, DHRowAnimation


showVisualization = '--noGraphics' not in sys.argv
solutionDirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'solution')
validationTolerance = 1e-3     #reference values are given with 4 digits
#! Subtasks, each runs alone (True: run): validateDH compares the own DH model with the reference poses (Val1)-(Val3)
#! (printed); showDHFrame shows in the renderer the reference poses Val1-Val3 and the TCP frame of the own DH model
#! (dark, it moves with the robot and must sit on the TCP of the gripper in every pose); showJointFrames draws all DH
#! frames K1..K6 of the own DH parameters instead of the TCP only (frame Ki: at the axis of joint i+1, z along it; K6
#! is the TCP), compare them with the frames drawn into the kinematic diagram.
#! showLabRobot: False shows only the own DH model (the frames K1..K6 joined as kinematic chain, a joint cylinder along
#! each z axis) moving through the validation poses: in (Val1)-(Val3) K6 must coincide with the reference frame;
#! True: also the lab robot (with its correct kinematics).
#! animateDH: before the motion the renderer animates one row of the DH table step by step (with showJointFrames):
#! animateRow = None the last row of the table (build the table row by row and see each new row), a row number, or
#! 'all' all rows one after the other.
validateDH = True
showDHFrame = True
showJointFrames = True
showLabRobot = False
animateDH = True
animateRow = None


def DHParameters():
    """Return the standard DH parameters of the 6R robot, one row per joint:
    [theta offset in rad, d in m, a in m, alpha in rad] (theta_i = q_i + theta offset_i)."""
    ## TODO 1: Determine the standard DH parameters of the 6R robot from the kinematic diagram and the
    ##         dimensions of the task sheet (draw the joint frames into the figure first).
    ##         One row per joint: [theta offset in rad, d in m, a in m, alpha in rad]; the joint angle q_i
    ##         is added to the theta offset. The zero configuration q = 0 must give pose (Val1).
    ##         Fill in the table below row by row (angles e.g. as 0.5*np.pi): first row 1, run the file and see
    ##         in the renderer how the row moves K0 to K1 (animateDH); then uncomment row 2, fill it in, run, ...
    #        theta offset  d in m  a in m  alpha
    return [[0.,           0.,     0.,     0.],      #joint 1
    #       [0.,           0.,     0.,     0.],      #joint 2
    #       [0.,           0.,     0.,     0.],      #joint 3
    #       [0.,           0.,     0.,     0.],      #joint 4
    #       [0.,           0.,     0.,     0.],      #joint 5
    #       [0.,           0.,     0.,     0.],      #joint 6
            ]


def ForwardKinematicsDH(q, dhParameters):
    """Return the 4x4 TCP pose in K0 for joint angles q in rad: product of the DH transformations."""
    pose = np.eye(4)
    for jointIndex, (thetaOffset, d, a, alpha) in enumerate(dhParameters):
        pose = pose @ StdDH2HT([q[jointIndex] + thetaOffset, d, a, alpha])
    return pose


def PrintDHTable(dhParameters):
    """Print the DH parameters (ASCII table, works in every Windows console)."""
    print('joint |  theta offset in deg |  d in m  |  a in m  | alpha in deg')
    for jointIndex, (thetaOffset, d, a, alpha) in enumerate(dhParameters):
        print('{:5d} | {:20.1f} | {:8.4f} | {:8.4f} | {:12.1f}'.format(
            jointIndex+1, thetaOffset*par.radToDeg, d, a, alpha*par.radToDeg))


def ValidateDHParameters(dhParameters):
    """Compare the forward kinematics of the DH parameters with the reference poses (Val1)-(Val3); return max. error."""
    PrintDHTable(dhParameters)
    maxError = 0.
    for poseIndex, (anglesDeg, referencePose) in enumerate(zip(par.validationJointAnglesDeg, par.validationPoses)):
        pose = ForwardKinematicsDH(np.array(anglesDeg)*par.degToRad, dhParameters)
        error = np.max(np.abs(pose - referencePose))
        maxError = max(maxError, error)
        print('validation pose (Val' + str(poseIndex+1) + '), q =', anglesDeg, 'deg')
        print(np.round(pose, 4))
        print('  max. deviation from reference: {:.2e}'.format(error))
    return maxError


if __name__ == '__main__':
    numberOfRows = len(DHParameters())
    if validateDH and numberOfRows < par.numberOfJoints:
        PrintDHTable(DHParameters())
        print('DH table: %d of %d rows, the validation needs all rows' % (numberOfRows, par.numberOfJoints))
    elif validateDH:
        maxError = ValidateDHParameters(DHParameters())
        print('DH parameters valid:', maxError < validationTolerance)

    #move through the validation poses and back to the zero configuration (Exudyn model of the lab robot); with
    #showDHFrame the reference poses and the TCP of the own DH model are drawn (showJointFrames: all DH frames)
    jointAngleList = [np.array(anglesDeg)*par.degToRad for anglesDeg in par.validationJointAnglesDeg] + [np.zeros(6)]
    matrixQ = CreateTrajectoryMatrix(jointAngleList, speedPercent=60., gripperCommand=par.gripperCommandNone)
    frames = [('Val%d' % (index + 1), pose) for index, pose in enumerate(par.validationPoses)] if showDHFrame else None
    animation = DHRowAnimation(DHParameters, animateRow)      #the frames of the table, one row animated
    if showJointFrames:
        toolFrameFunction = animation.Frames
    elif showDHFrame:
        toolFrameFunction = lambda q: ForwardKinematicsDH(q, DHParameters())
    else:
        toolFrameFunction = None
    SimulateBahnfahrt(matrixQ, 'rad', useRenderer=showVisualization, showSolution=showVisualization, includeCup=False, includeLabTables=False,
                      frames=frames, toolFrameFunction=toolFrameFunction, showRobot=showLabRobot or not showJointFrames,
                      rendererAnimation=animation.Run if animateDH and showJointFrames else None,
                      solutionFileName=os.path.join(solutionDirectory, 'preparation12'))
