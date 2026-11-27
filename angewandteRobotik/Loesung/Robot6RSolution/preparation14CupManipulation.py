# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: preparation14CupManipulation.py
#
# Details:  Preparation task 1.4 (Vorbereitung): grasp pose K1 of the cup on table 1 and placing pose K2 on table 2 with
#           their joint angles in the given configuration of the arm (armConfiguration, see the configurations of task
#           1.3) and the free parameter of the grasp (rotation about the vertical cup axis); then the PTP sequence to
#           move the cup from table 1 to table 2 (home, approach, grasp, lift, transport at constant height, place,
#           retreat, home), the speed and the gripper command of every row; the complete matrix Q (dataLab.txt, the
#           input of lab task 1) and the simulation of the whole manipulation with gripper and empty paper cup by
#           Exudyn (fully synchronous PTP as the controller of the lab robot).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-03; 2026-11-21 grasp poses and free parameter from preparation13GraspPoses.py (task 1.3 is now the
#           robot configurations, preparation13Configurations.py); 2026-11-21 configuration of the arm given (no TODO,
#           no display of all solutions): the main task is the sequence
# Notes:    Run: python preparation14CupManipulation.py [--noGraphics]
#           Output: solution/dataLab.txt, the matrix Q loaded in the lab
##BEGIN_MASTERONLY
#           Master file: autoGenerateTemplates.py writes the student template (isTemplate blocks -> ## TODO).
##END_MASTERONLY
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
import numpy as np
from exudyn.rigidBodyUtilities import HTrotateX, HTrotateY, HTrotateZ, HTtranslate

from robot6R import parameters as par
from labRobots.robots.robot6R.model import ForwardKinematics, CheckLabConfiguration, GetKinematicChain
from robot6R.labTrajectory import CreateTrajectoryMatrix
from robot6R.simulation import SimulateBahnfahrt
from robot6R.plotting import PlotSimulationResults, PrintSimulationSummary
from robot6R.visualization import cupHolderRadius

##BEGIN_MASTERONLY
isTemplate = False   #True: run as the student template would (without the solution)
##END_MASTERONLY
if not isTemplate:
    from robot6R.reference import labTasks

showVisualization = '--noGraphics' not in sys.argv
solutionDirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'solution')
dataFileName = os.path.join(solutionDirectory, 'dataLab.txt')
speedPercent = 50.      #speed of every PTP motion in percent of the maximum speed of the lab robot (5 to 100)
#! Tolerance in m of the height of the placed cup and largest tilt in deg of a cup standing upright (CupPlacedOnTable2).
cupPlacedTolerance, cupPlacedTiltDeg = 0.01, 5.
freeAnglesDeg = [-90., -60., -30., 0., 30., 60., 90.]  #rotation about the cup axis (free parameter)
#! Joint angles in rad near the cup task in front of the robot (q3 > 0, q5 < 0): start of the inverse kinematics.
initialGuess = [0., 0.8, 1.6, 0., -0.8, 0.]
inverseKinematicsTolerance = 1e-4   #max. pose error (norm of the HT difference) accepted as solution
#! Configuration [shoulder, elbow, wrist] of the arm for the whole cup task (signs of KinematicChain.ArmConfiguration):
#! shoulder front, elbow up (q3 > 0), wrist q5 < 0, the only one of the 8 solutions of K1 valid in the lab (see
#! task 1.3).
armConfiguration = [-1, 1, -1]
#! Subtasks, each runs alone (True: run): showGraspPoses draws K1 and K2 (TODO 1, 2) as frames in the renderer and
#! prints their joint angles in the configuration armConfiguration; studyFreeParameter checks the free parameter of the
#! grasp pose (needs TODO 1; question of the task sheet); showTaskPoses draws the poses of the cup manipulation
#! (TODO 3, labeled with their names; also those the robot cannot reach). Each TODO names the switches to set for it;
#! the template starts with TODO 1 and 2 only.
if not isTemplate:
    showGraspPoses = studyFreeParameter = showTaskPoses = True
else:
    showGraspPoses = True
    studyFreeParameter = False
    showTaskPoses = False


def GraspPoseTable1():
    """Return the grasp pose K1 of the cup on table 1: 4x4 TCP pose in K0."""
    if not isTemplate:
        #gripper axis z_G horizontal towards +x0 (rotation about y), then about z_G until x_G points up (cup axis)
        position = [par.table1Position[0], par.table1Position[1], labTasks.graspHeightTable1]
        return HTtranslate(position) @ HTrotateY(0.5*np.pi) @ HTrotateZ(np.pi)
    else:
        ## TODO 1: Return the 4x4 TCP pose K1 for grasping the cup on table 1 (position par.table1Position,
        ##         dimensions of table, cup holder and cup in robot6R/parameters.py and the task sheet).
        ##         K1 is the frame K1 drawn in the task sheet ("Abmessungen des Arbeitsbereiches"):
        ##         the TCP lies in the cup axis, the jaws close around the cup (not a pose in front of it).
        ##         The gripper approaches the cup horizontally; choose the orientation of the TCP frame
        ##         (gripper axis z_G) such that the cup axis is vertical.
        ##         Build the pose from elementary transformations instead of writing the matrix, e.g.
        ##         HTtranslate(position) @ HTrotateY(angle) @ HTrotateZ(angle): read from left to right, each further
        ##         rotation turns the gripper about its own (already turned) axes. First turn z_G horizontal towards
        ##         the cup, then about z_G until the jaws close horizontally; check in the renderer (frame K1,
        ##         showGraspPoses): blue z_G towards the cup, red x_G up along the cup axis.
        ##         Switches: showGraspPoses = True, the others False.
        return np.eye(4)


def GraspPoseTable2():
    """Return the placing pose K2 of the cup on table 2: 4x4 TCP pose in K0."""
    if not isTemplate:
        return labTasks.GetGraspPoseTable2()
    else:
        ## TODO 2: Return the 4x4 TCP pose K2 for placing the cup on table 2 (position par.table2Position):
        ##         the frame K2 drawn in the task sheet, the TCP in the axis of the placed cup.
        ##         Hint: rotate the orientation of K1 about the vertical cup axis so that the robot can reach
        ##         the pose (post-multiplication with a rotation, e.g. HTrotateX).
        ##         Switches: showGraspPoses = True, the others False.
        return np.eye(4)


def SolveGraspInverseKinematics(pose, guess=initialGuess):
    """Return (q in rad, pose error, valid in the lab, message) of a TCP pose in the configuration armConfiguration;
    q is the solution nearest to guess; (guess, inf, False, message) if there is none."""
    chain = GetKinematicChain()
    solutions = chain.InverseKinematicsAnalytic(np.array(pose, dtype=float), np.array(guess, dtype=float), list(armConfiguration))
    if len(solutions) == 0:
        return np.array(guess, dtype=float), np.inf, False, 'not reachable in the configuration %s' % list(armConfiguration)
    q = (np.array(solutions[0], dtype=float) + np.pi) % (2.*np.pi) - np.pi     #angles in [-pi, pi)
    poseError = np.linalg.norm(ForwardKinematics(q) - pose)
    isValid, message = CheckLabConfiguration(q)
    return q, poseError, isValid, message


def PrintGraspJointAngles():
    """Print the grasp poses K1, K2 and their joint angles in the configuration armConfiguration (or why they are not
    reached, e.g. TODO 1 or 2 not solved yet): the values of the report."""
    for name, pose in [('K1 (table 1)', GraspPoseTable1()), ('K2 (table 2)', GraspPoseTable2())]:
        q, poseError, isValid, message = SolveGraspInverseKinematics(pose)
        print('grasp pose', name, ':\n', np.round(pose, 4))
        if not poseError < inverseKinematicsTolerance:      #also inf or nan
            print('  pose', name, 'not reached:', message)
            continue
        print('  q =', np.round(np.asarray(q)*par.radToDeg, 3), 'deg, pose error', '{:.1e}'.format(poseError),
              ', lab check:', 'valid' if isValid else 'REJECTED', message)


def StudyFreeParameter():
    """The cup is rotationally symmetric: the rotation about its axis is free; check reachability for several angles."""
    print('free parameter: rotation of the gripper about the vertical cup axis (x_G) on table 1')
    for angleDeg in freeAnglesDeg:
        pose = GraspPoseTable1() @ HTrotateX(angleDeg*par.degToRad)
        q, poseError, isValid, message = SolveGraspInverseKinematics(pose)
        if poseError >= inverseKinematicsTolerance:
            print('  angle {:6.1f} deg: not reachable ({})'.format(angleDeg, message))
            continue
        print('  angle {:6.1f} deg: reachable, lab check {}, q = {} deg'.format(
            angleDeg, 'valid' if isValid else 'rejected (' + message + ')', np.round(q*par.radToDeg, 1)))


def CupTaskPoses():
    """Return [(name, 4x4 TCP pose in K0)] of the cup manipulation in the order the robot moves through them; 'home' is
    the zero configuration (its pose is not used)."""
    if not isTemplate:
        return labTasks.ComputeCupTaskPoses()
    else:
        ## TODO 3: Define the poses of the cup manipulation as list of (name, 4x4 pose), starting and ending
        ##         with ('home', None), the zero configuration. Use GraspPoseTable1() and GraspPoseTable2() (TODO 1,
        ##         2) for grasping and placing; which intermediate poses do you need to avoid collisions with the
        ##         tables and the cup holder, and how does the cup stay upright during the transport?
        ##         Pre-multiplication HTtranslate(p) @ T moves in K0, post-multiplication T @ HTtranslate(p) in
        ##         the gripper frame. The poses are drawn in the renderer (showTaskPoses); SolvePoseSequence
        ##         computes their joint angles in the configuration armConfiguration and reports a pose that cannot be
        ##         reached or is rejected.
        ##         Switches: showTaskPoses = True.
        return [('home', None), ('home', None)]


def SolvePoseSequence(poses):
    """Return (names, list of joint angle arrays in rad) of a sequence of poses [(name, 4x4 pose)]: 'home' is the zero
    configuration; each other pose is solved by the inverse kinematics in the configuration armConfiguration,
    starting from the solution of the pose before (the robot stays in its configuration, the motion is continuous). A
    pose that is not reached is replaced by the zero configuration; one rejected by the lab check is kept (the lab
    skips its row of Q)."""
    names, jointAngleList = [], []
    guess = np.array(initialGuess, dtype=float)
    for name, pose in poses:
        if name == 'home':
            q = np.zeros(par.numberOfJoints)
        else:
            q, poseError, isValid, message = SolveGraspInverseKinematics(pose, guess)
            if not poseError < inverseKinematicsTolerance:
                print('pose "%s" not reached (%s): the zero configuration instead' % (name, message))
                q = np.zeros(par.numberOfJoints)
            else:
                guess = np.array(q, dtype=float)
                if not isValid:
                    print('pose "%s" rejected by the lab check: %s' % (name, message))
        names.append(name)
        jointAngleList.append(np.array(q, dtype=float))
    return names, jointAngleList


def CupTaskMatrixQ(names, jointAngleList):
    """Return the matrix Q of the cup manipulation: per row the 6 joint angles in rad, the speed in percent and the
    gripper command executed after the row is reached (par.gripperCommandNone -1, par.gripperCommandOpen 0,
    par.gripperCommandClose 1)."""
    if not isTemplate:
        commands = [par.gripperCommandClose if name == 'grasp table 1' else
                    par.gripperCommandOpen if name == 'place table 2' else par.gripperCommandNone for name in names]
        return CreateTrajectoryMatrix(jointAngleList, speedPercent=speedPercent, gripperCommand=commands)
    else:
        ## TODO 4: Complete the matrix Q: the speed of every row (speedPercent, 5 to 100 %) and the gripper command
        ##         of every row (par.gripperCommandNone, par.gripperCommandOpen, par.gripperCommandClose): after
        ##         which pose must the gripper close, after which open? CreateTrajectoryMatrix(jointAngleList,
        ##         speedPercent, gripperCommand) takes one value for all rows or a list with one value per row.
        ##         Without gripper commands the robot moves, but the cup stays on table 1 (see the simulation).
        return CreateTrajectoryMatrix(jointAngleList, speedPercent=speedPercent, gripperCommand=par.gripperCommandNone)


def CupPlacedOnTable2(results):
    """Return True if the cup stands upright on the cup holder of table 2 at the end of the simulation: its bottom on
    the holder (within cupPlacedTolerance of the height of table 2 and within the radius of the holder from its axis),
    tilted by less than cupPlacedTiltDeg."""
    cupBottom = results['cupPosition'][-1]
    onHolder = np.linalg.norm(cupBottom[0:2] - np.array(par.table2Position[0:2])) < cupHolderRadius
    atHeight = abs(cupBottom[2] - par.table2Height) < cupPlacedTolerance
    return bool(onHolder and atHeight and results['cupTiltDeg'][-1] < cupPlacedTiltDeg)


def CreateAndSaveMatrixQ():
    """Compute the matrix Q of the cup manipulation (CupTaskPoses, SolvePoseSequence, CupTaskMatrixQ), print it and save it as
    dataFileName (the input of lab task 1); return (names of the poses, Q)."""
    names, jointAngleList = SolvePoseSequence(CupTaskPoses())
    matrixQ = CupTaskMatrixQ(names, jointAngleList)
    for name, row in zip(names, matrixQ):
        print('{:18s} q = {} deg, speed {:5.1f} %, gripper {:2.0f}'.format(name, np.round(row[0:6]*par.radToDeg, 3), row[6], row[7]))
    os.makedirs(solutionDirectory, exist_ok=True)
    np.savetxt(dataFileName, matrixQ)
    print('saved', dataFileName)
    return names, matrixQ


if __name__ == '__main__':
    if showGraspPoses:
        PrintGraspJointAngles()
    if studyFreeParameter:
        StudyFreeParameter()
    names, matrixQ = CreateAndSaveMatrixQ()
    frames = []
    if showTaskPoses:                   #the poses of TODO 3 (without home)
        frames += [(name, pose) for name, pose in CupTaskPoses() if name != 'home']
    if showGraspPoses:                  #the grasp poses of TODO 1, 2
        frames += [('K1', GraspPoseTable1()), ('K2', GraspPoseTable2())]
    results, simulation = SimulateBahnfahrt(matrixQ, 'rad', useRenderer=showVisualization, showSolution=showVisualization,
                                            frames=frames if frames else None,
                                            solutionFileName=os.path.join(solutionDirectory, 'preparation14'))
    PrintSimulationSummary(results)
    PlotSimulationResults(results, os.path.join(solutionDirectory, 'preparation14'), 'cup manipulation:')
    print('cup placed upright on table 2:', 'yes' if CupPlacedOnTable2(results) else 'no')
