# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: labTrajectory.py
#
# Details:  Interpretation of the PTP matrix Q of the lab (BahnfahrtPython(Q, str))
#           as joint trajectory with gripper actions, replicating the lab software:
#           row k of Q = [q1..q6, speed in %, gripper command]; the robot moves
#           from the previous position to row k with fully synchronous PTP
#           interpolation (Positionieren_T.m) and executes the gripper command
#           afterwards (-1 none, 0 open, 1 close 70 %, 2 close 75 %).
#           Rows rejected by check_input.m are skipped (with a warning) as in the
#           lab, the following rows start from the last valid position.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-03
# Notes:    In the physical lab mode the next motion starts immediately after a
#           gripper command; here a dwell of gripperActionTime (1.5 s, as in the
#           virtual mode of the lab software) is used so that the jaws are closed
#           before lifting. Set gripperActionTime=0 to reproduce the lab timing.
#           PTP profiles by CreatePTPProfile ('synchronous'
#           = Exudyn ProfilePTP), the gripper dwell by Exudyn ProfilePolynomial5.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
from exudyn.robotics.motion import Trajectory, ProfilePolynomial5

from . import parameters as par
from labRobots.robots.robot6R.model import CheckLabConfiguration
from .ptpProfile import CreatePTPProfile

numberOfMatrixColumns = 8   #q1..q6, speed, gripper
angleUnits = {'rad': 1., 'deg': par.degToRad}


def CreateTrajectoryMatrix(jointAngleList, speedPercent=50., gripperCommand=par.gripperCommandNone):
    """Create the 8 column matrix Q for BahnfahrtPython from a list of joint angle vectors
    (same as createTrajFromAngleList of RobotUtils.py).

    Args:
        jointAngleList (list): list of 6 joint angles per configuration (rad or deg)
        speedPercent (float or list): speed in percent of the lab max. speed, per row or for all rows
        gripperCommand (int or list): gripper command per row or for all rows

    Returns:
        np.array: matrix with rows [q1..q6, speed, gripper]
    """
    angles = np.array([np.array(q, dtype=float).reshape(-1) for q in jointAngleList])
    numberOfRows = len(angles)
    speeds = np.broadcast_to(np.array(speedPercent, dtype=float), (numberOfRows,))
    gripperCommands = np.broadcast_to(np.array(gripperCommand, dtype=float), (numberOfRows,))
    return np.column_stack([angles, speeds, gripperCommands])


class LabTrajectory:
    """Joint trajectory and gripper timeline of a lab PTP matrix Q (as executed by BahnfahrtPython)."""

    def __init__(self, matrixQ, angleUnit='rad', initialJointAngles=None, mode='fullySynchronous',
                 gripperActionTime=par.gripperActionTime, initialGripperPercent=par.gripperInitialPercent,
                 checkConfigurations=True, verbose=True):
        """Build trajectory from matrix Q.

        Args:
            matrixQ (array): n x 8 matrix [q1..q6, speed percent, gripper command]
            angleUnit (str): 'rad' or 'deg' (as second argument of BahnfahrtPython)
            initialJointAngles (array): start configuration in rad; default: zero configuration (lab start)
            mode (str): PTP interpolation, 'fullySynchronous' (lab), 'synchronous' or 'asynchronous'
            gripperActionTime (float): duration of a gripper action (dwell) in s
            initialGripperPercent (float): gripper opening command at start (0 = open)
            checkConfigurations (bool): apply the input check of the lab software to every row
            verbose (bool): print motions, warnings and skipped rows
        """
        matrixQ = np.atleast_2d(np.array(matrixQ, dtype=float))
        if matrixQ.shape[1] != numberOfMatrixColumns:
            raise ValueError('LabTrajectory: Q needs 8 columns [q1..q6, speed, gripper]; use CreateTrajectoryMatrix(...)')
        if angleUnit not in angleUnits:
            raise ValueError('LabTrajectory: angleUnit must be "rad" or "deg"')
        self.matrixQ = matrixQ
        self.angleUnit = angleUnit
        self.mode = mode
        self.verbose = verbose
        self.initialJointAngles = np.zeros(par.numberOfJoints) if initialJointAngles is None else np.array(initialJointAngles, dtype=float)
        self.initialGripperPercent = initialGripperPercent
        self.segments = []          #list of dicts: row, startTime, endTime, target, speedPercent
        self.gripperEvents = []     #list of dicts: row, startTime, endTime, startPercent, endPercent, command
        self.skippedRows = []       #list of (row, message)
        self.warnings = []          #list of (row, message)

        self.trajectory = Trajectory(initialCoordinates=self.initialJointAngles, initialTime=0.)
        currentAngles = self.initialJointAngles.copy()
        currentGripperPercent = initialGripperPercent
        maxVelocities = np.array(par.labMaxVelocitiesDeg)*par.degToRad
        maxAccelerations = np.array(par.labMaxAccelerationsDeg)*par.degToRad
        timeFactor = par.labMotionTimeFactor if mode == 'fullySynchronous' else 1.

        for row in range(matrixQ.shape[0]):
            target = matrixQ[row, 0:6]*angleUnits[angleUnit]
            speedPercent = float(np.clip(matrixQ[row, 6], par.labMinimumSpeedPercent, par.labMaximumSpeedPercent))
            if speedPercent != matrixQ[row, 6]:
                self.AddWarning(row, 'speed ' + str(matrixQ[row, 6]) + ' % clamped to ' + str(speedPercent) + ' %')
            isValid, message = CheckLabConfiguration(target) if checkConfigurations else (True, '')
            if not isValid:
                self.skippedRows.append((row, message))
                self.Print('row ' + str(row) + ' rejected (motion skipped as in the lab): ' + message)
                continue
            if message != '':
                self.AddWarning(row, message)

            if np.any(target != currentAngles):
                startTime = self.GetDuration()
                self.trajectory.Add(CreatePTPProfile(target, maxVelocities*speedPercent/100., maxAccelerations*speedPercent/100.,
                                                     mode=mode, timeFactor=timeFactor))
                self.segments.append({'row': row, 'startTime': startTime, 'endTime': self.GetDuration(),
                                      'target': target.copy(), 'speedPercent': speedPercent})
                self.Print('row ' + str(row) + ': PTP motion to ' + str(np.round(target*par.radToDeg, 3)) + ' deg, speed ' +
                           str(speedPercent) + ' %, duration ' + str(round(self.segments[-1]['endTime'] - startTime, 3)) + ' s')
                currentAngles = target.copy()

            command = int(matrixQ[row, 7])
            if command == par.gripperCommandNone:
                continue
            if command not in par.gripperCommandPercent:
                self.AddWarning(row, 'gripper command ' + str(command) + ' not defined')
                continue
            startTime = self.GetDuration()
            if gripperActionTime > 0.:
                self.trajectory.Add(ProfilePolynomial5(currentAngles, gripperActionTime))     #dwell: rest at currentAngles
            targetPercent = par.gripperCommandPercent[command]
            self.gripperEvents.append({'row': row, 'startTime': startTime, 'endTime': startTime + gripperActionTime,
                                       'startPercent': currentGripperPercent, 'endPercent': targetPercent, 'command': command})
            self.Print('row ' + str(row) + ': gripper ' + ('opens' if command == par.gripperCommandOpen else 'closes') +
                       ' to ' + str(targetPercent) + ' %')
            currentGripperPercent = targetPercent
        self.trajectory.Initialize()

    def Print(self, text):
        """Print text if verbose."""
        if self.verbose:
            print('LabTrajectory:', text)

    def AddWarning(self, row, message):
        """Store and print a warning for a row of Q."""
        self.warnings.append((row, message))
        self.Print('row ' + str(row) + ' warning: ' + message)

    def GetDuration(self):
        """Return total duration of the trajectory in s."""
        return self.trajectory.GetTimes()[-1]

    def Evaluate(self, time):
        """Return [q, q_t, q_tt] of the joint trajectory at time (rad, rad/s, rad/s^2)."""
        if len(self.trajectory) == 0:
            numberOfJoints = len(self.initialJointAngles)
            return [self.initialJointAngles.copy(), np.zeros(numberOfJoints), np.zeros(numberOfJoints)]
        return self.trajectory.Evaluate(time)

    def GetGripperPercent(self, time):
        """Return gripper opening command (percent) at time; linear transition during gripper actions."""
        percent = self.initialGripperPercent
        for event in self.gripperEvents:
            if time >= event['endTime']:
                percent = event['endPercent']
            elif time > event['startTime']:
                ratio = (time - event['startTime'])/(event['endTime'] - event['startTime'])
                return event['startPercent'] + ratio*(event['endPercent'] - event['startPercent'])
            else:
                break
        return percent

    def GetFinalJointAngles(self):
        """Return joint angles at the end of the trajectory."""
        return np.array(self.trajectory.GetFinalCoordinates(), dtype=float)
