# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: labTasks.py
#
# Details:  Building blocks of the lab tasks (preparation 1.3-1.5 and lab tasks
#           1-3): grasp poses K1 (table 1) and K2 (table 2), approach/retreat
#           poses using pre- and post-multiplication, the analytic inverse
#           kinematics of Exudyn (KinematicChain.InverseKinematicsAnalytic) with
#           lab checks, and the joint angle sequence of the cup manipulation.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-03
# Notes:    Reference solution for the instructors; the pose definitions follow
#           VorbereitungLabor.py (pre-/post-multiplication, constant transport
#           height), with the approach along the gripper axis z_G.
#           analytic inverse kinematics of Exudyn instead of
#           ikine_LM of the robotics toolbox (same solutions, about 100 times faster).
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np

from .. import parameters as par
from labRobots.robots.robot6R.model import ForwardKinematics, CheckLabConfiguration, GetKinematicChain

#grasp poses of the solution of task 1.3 (moved from parameters.py)
graspHeightTable1 = 0.157           #TCP height for grasping on table 1 (K1)
graspHeightTable2 = 0.127           #TCP height for placing on table 2 (K2): table 0.086 + holder and grasp 0.041, as K1: was 0.1275
graspRotationTable1 = np.array([[0.,0.,1.],[0.,-1.,0.],[1.,0.,0.]])  #R of K1: z1 = x0 (approach), x1 = z0 (up)
graspRotationAngleTable2 = 45.*par.degToRad  #K2 = K1 rotated about x1 (vertical) by 45 deg
approachDistance = 0.1      #m, approach of table 1 along the gripper axis z_G
liftHeight = 0.1            #m, lift of the cup above table 1 (world z)
retreatDistance = 0.15      #m, retreat from table 2 along -z_G
inverseKinematicsTolerance = 1e-4   #max. pose error (norm of HT difference, approx. 0.1 mm) accepted as solution
initialGuessCupTask = [0., 0.8, 1.6, 0., -0.8, 0.]     #configuration of the cup task in front of the robot (q3 > 0, q5 < 0)


def TranslationHT(translation):
    """Return homogeneous transformation of a pure translation."""
    HT = np.eye(4)
    HT[0:3, 3] = translation
    return HT


def RotationXHT(angle):
    """Return homogeneous transformation of a rotation about the local x-axis."""
    HT = np.eye(4)
    HT[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    return HT


def GetGraspPoseTable1():
    """Return grasp pose K1 of the cup on table 1 (TCP frame, gripper axis z_G horizontal towards +x0, x_G vertical)."""
    HT = np.eye(4)
    HT[0:3, 0:3] = graspRotationTable1
    HT[0:3, 3] = [par.table1Position[0], par.table1Position[1], graspHeightTable1]
    return HT


def GetGraspPoseTable2():
    """Return placing pose K2 on table 2: K1 orientation rotated by 45 deg about the vertical axis x_G (post-multiplication)."""
    HT = GetGraspPoseTable1() @ RotationXHT(graspRotationAngleTable2)
    HT[0:3, 3] = [par.table2Position[0], par.table2Position[1], graspHeightTable2]
    return HT


def RotateAboutCupAxis(graspPose, angle):
    """Rotate a grasp pose about the vertical cup axis (x_G) by angle; this is the free parameter of the grasp pose."""
    return graspPose @ RotationXHT(angle)


def ComputeCupTaskPoses():
    """Return list of (name, HT) of the cup manipulation: home, approach, grasp, lift, transport, place, retreat, home.

    Pre-multiplication (TranslationHT @ T) moves in base coordinates, post-multiplication (T @ TranslationHT) in gripper coordinates.
    """
    graspPose1 = GetGraspPoseTable1()
    placePose2 = GetGraspPoseTable2()
    approachPose1 = graspPose1 @ TranslationHT([0., 0., -approachDistance])          #back along gripper axis
    liftPose1 = TranslationHT([0., 0., liftHeight]) @ graspPose1                     #lift in world z
    abovePose2 = TranslationHT([0., 0., liftPose1[2, 3] - placePose2[2, 3]]) @ placePose2  #same height as lift pose
    retreatPose2 = placePose2 @ TranslationHT([0., 0., -retreatDistance])            #back along gripper axis
    homePose = ForwardKinematics(np.zeros(par.numberOfJoints))
    return [('home', homePose), ('approach table 1', approachPose1), ('grasp table 1', graspPose1),
            ('lift table 1', liftPose1), ('above table 2', abovePose2), ('place table 2', placePose2),
            ('retreat table 2', retreatPose2), ('home', homePose)]


def WrapAngles(q):
    """Map joint angles to the range [-pi, pi) (equivalent configuration of revolute joints)."""
    return (np.asarray(q, dtype=float) + np.pi) % (2.*np.pi) - np.pi


def SolveInverseKinematics(HT, initialGuess, configuration=None):
    """Solve the inverse kinematics analytically (Exudyn KinematicChain.InverseKinematicsAnalytic) and check the result.

    Args:
        HT (np.array): desired TCP pose 4x4
        initialGuess (array): joint angles q0; the solution nearest to q0 is returned
        configuration (list): [shoulder, elbow, wrist] (each +1, -1 or 0 for any) of the wanted solution
                              (KinematicChain.ArmConfiguration); None: the configuration of initialGuess

    Returns:
        tuple: (q, poseError, isValidInLab, labMessage); if the pose cannot be reached: (initialGuess, inf, False, message)
    """
    chain = GetKinematicChain()
    initialGuess = np.array(initialGuess, dtype=float)
    if configuration is None:
        configuration = chain.ArmConfiguration(initialGuess)
    solutions = chain.InverseKinematicsAnalytic(np.array(HT, dtype=float), initialGuess, configuration)
    if len(solutions) == 0:
        return initialGuess, np.inf, False, 'pose not reachable in the configuration ' + str(list(configuration))
    q = WrapAngles(solutions[0])
    poseError = np.linalg.norm(ForwardKinematics(q) - HT)
    isValid, message = CheckLabConfiguration(q)
    return q, poseError, isValid, message


def ComputeCupTaskJointAngles(initialGuess=initialGuessCupTask, verbose=True):
    """Compute joint angles of all poses of the cup manipulation; each IK starts from the previous solution (continuity).
    The home pose uses the exact zero configuration.

    Returns:
        tuple: (names, list of joint angle arrays)
    """
    names = []
    jointAngleList = []
    guess = np.array(initialGuess, dtype=float)
    for name, HT in ComputeCupTaskPoses():
        if name == 'home':
            q = np.zeros(par.numberOfJoints)
        else:
            q, poseError, isValid, message = SolveInverseKinematics(HT, guess)
            if poseError > inverseKinematicsTolerance:
                raise ValueError('inverse kinematics failed for pose "' + name + '" (error ' + str(poseError) + ')')
            if not isValid:
                raise ValueError('pose "' + name + '" is rejected by the lab check: ' + message)
            guess = q
        if verbose:
            print('{:18s} q = {} deg'.format(name, np.round(q*par.radToDeg, 3)))
        names.append(name)
        jointAngleList.append(q)
    return names, jointAngleList
