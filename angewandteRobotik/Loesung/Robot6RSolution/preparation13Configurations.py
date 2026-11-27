# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: preparation13Configurations.py
#
# Details:  Preparation task 1.3 (Vorbereitung): robot configurations, a demonstration without TODO (as task 1.5). The
#           lab robot reaches one TCP pose T with several sets of joint angles (analytic inverse kinematics of Exudyn,
#           KinematicChain.InverseKinematicsAnalytic): axes 1-3 are the 3R robot of the homework with the solutions
#           elbow up and elbow down. Both are printed and shown side by side in the renderer, each robot moving from
#           the zero configuration into its configuration (shoulder front, hand not turned over, valid in the lab).
#           The motion zero -> elbow up -> zero -> elbow down -> zero is saved as matrix Q
#           (solution/dataConfigurations.txt), which the lab PC executes at the real robot (lab task 2).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-21 (lab task 2 labTask2Configurations.py of 2026-11-03 moved into the preparation; replaces the
#           grasp poses of preparation13GraspPoses.py, which are now part of task 1.4); 2026-11-21 demonstration of
#           elbow up and elbow down only (no TODO, robots side by side instead of all 8 solutions one after the other)
# Notes:    Run: python preparation13Configurations.py [--noGraphics]
#           Space in the renderer starts (and repeats) the motion, Q ends.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
import numpy as np
from exudyn.robotics.motion import Trajectory

from robot6R import parameters as par
from labRobots.robots.robot6R.model import GetKinematicChain, CheckLabConfiguration, ForwardKinematics
from robot6R.poseUtilities import OrthonormalPose
from robot6R.labTrajectory import CreateTrajectoryMatrix
from robot6R.ptpProfile import CreatePTPProfile
from robot6R.robotsSideBySide import RobotsSideBySide

showVisualization = '--noGraphics' not in sys.argv
solutionDirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'solution')
#! Matrix Q of elbow up and elbow down (q1..q6 in rad, speed, gripper), executed at the real robot in the lab.
dataFileName = os.path.join(solutionDirectory, 'dataConfigurations.txt')
#! Speed of the PTP motions of Q in percent of the maximum speed of the lab robot (5 to 100).
speedPercent = 80.
#! Signs of KinematicChain.ArmConfiguration [shoulder, elbow, wrist] shown: shoulder front (towards the target, as
#! the 3R robot of the homework) and the names of the two elbow solutions (elbow = sign of q3; Claude Code,
#! 2026-11-21: q3 > 0 is elbow up, the elbow lies above the line shoulder -> wrist point, q3 < 0 elbow down).
shoulderFront = -1
elbowNames = {1: 'elbow up', -1: 'elbow down'}
#! Largest pose error (norm of the HT difference) accepted as solution of the inverse kinematics.
inverseKinematicsTolerance = 1e-4
#! Subtasks (True: run): printConfigurations prints the joint angles of elbow up and elbow down and saves their matrix
#! Q; showConfigurations shows both robots side by side in the renderer.
printConfigurations = True
showConfigurations = True


def TaskPose():
    """Return the pose T of the task (4x4 TCP pose in K0, rotation orthonormalized: the task gives 4 digits)."""
    return OrthonormalPose(par.configurationTaskPose)


def ElbowConfigurations(pose):
    """Return [(name, q in rad)] of elbow up and elbow down of the pose: the solutions of the inverse kinematics with
    the shoulder in front, valid in the lab and reaching the pose; of the two wrist solutions the one with the smaller
    |q4| (hand not turned over)."""
    chain = GetKinematicChain()
    configurations = []
    for elbowSign, name in elbowNames.items():
        candidates = []
        for solution in chain.InverseKinematicsAnalytic(np.array(pose, dtype=float)):
            q = (np.array(solution, dtype=float) + np.pi) % (2.*np.pi) - np.pi     #angles in [-pi, pi)
            shoulder, elbow, unusedWrist = chain.ArmConfiguration(q)
            isValid, unusedMessage = CheckLabConfiguration(q)
            if (shoulder == shoulderFront and elbow == elbowSign and isValid
                    and np.linalg.norm(ForwardKinematics(q) - pose) < inverseKinematicsTolerance):
                candidates.append(q)
        if candidates:
            configurations.append((name, min(candidates, key=lambda q: abs(q[3]))))
    return configurations


def ConfigurationMatrixQ(configurations):
    """Return the matrix Q zero -> configuration -> zero for each configuration [(name, q)]."""
    zero = np.zeros(par.numberOfJoints)
    jointAngleList = []
    for unusedName, q in configurations:
        jointAngleList += [q, zero]
    return CreateTrajectoryMatrix(jointAngleList, speedPercent=speedPercent, gripperCommand=par.gripperCommandNone)


def ConfigurationMotions(configurations):
    """Return [(name, Trajectory)]: the fully synchronous PTP motion of the lab from the zero configuration into each
    configuration, at the maximum speed of the lab robot."""
    maxVelocities = np.array(par.labMaxVelocitiesDeg)*par.degToRad
    maxAccelerations = np.array(par.labMaxAccelerationsDeg)*par.degToRad
    motions = []
    for name, q in configurations:
        trajectory = Trajectory(initialCoordinates=np.zeros(par.numberOfJoints))
        trajectory.Add(CreatePTPProfile(q, maxVelocities, maxAccelerations, 'fullySynchronous', par.labMotionTimeFactor))
        motions.append((name, trajectory))
    return motions


if __name__ == '__main__':
    pose = TaskPose()
    configurations = ElbowConfigurations(pose)
    if printConfigurations:
        print('pose T:\n', np.round(pose, 4))
        for name, q in configurations:
            print('{:10s}: q = {} deg'.format(name, np.round(q*par.radToDeg, 2)))
        os.makedirs(solutionDirectory, exist_ok=True)
        np.savetxt(dataFileName, ConfigurationMatrixQ(configurations))
        print('saved', dataFileName, '(zero -> elbow up -> zero -> elbow down -> zero, for the lab)')
    if showConfigurations and showVisualization:
        RobotsSideBySide(ConfigurationMotions(configurations), np.array(par.labMaxVelocitiesDeg)*par.degToRad,
                         markerPoints=[pose[0:3, 3]], showScopesInit=False).Show()
