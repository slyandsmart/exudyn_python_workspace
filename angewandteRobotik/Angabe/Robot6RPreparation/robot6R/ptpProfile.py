# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: ptpProfile.py
#
# Details:  Point-to-point (PTP) motion profiles with trapezoidal velocity per
#           axis for exudyn.robotics.motion.Trajectory, created by CreatePTPProfile:
#           'asynchronous'     - every axis moves with its own max. velocity and
#                                acceleration and stops on its own
#           'synchronous'      - all axes stop at the same time, each axis keeps
#                                its max. acceleration: Exudyn ProfilePTP
#           'fullySynchronous' - all axes stop at the same time and have the same
#                                acceleration and deceleration phases; this is the
#                                interpolation of the lab (Positionieren_T.m):
#                                lead axis time t_e = s/v + v/a, duration
#                                T = timeFactor*t_e, acceleration time t_b = v/a
#                                of the lead axis
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-03
# Notes:    Velocities and accelerations in rad/s and rad/s^2.
#           'synchronous' by Exudyn ProfilePTP (identical
#           timing), TrapezoidalPTPProfile keeps the two modes Exudyn does not have.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
from exudyn.robotics.motion import ProfilePTP

ptpModes = ('asynchronous', 'synchronous', 'fullySynchronous')
trapezoidalModes = ('asynchronous', 'fullySynchronous')     #modes of TrapezoidalPTPProfile; 'synchronous' is ProfilePTP


def CreatePTPProfile(finalCoordinates, maxVelocities, maxAccelerations, mode='fullySynchronous', timeFactor=1.):
    """Return the PTP profile of mode for exudyn.robotics.motion.Trajectory.Add(...).

    Args:
        finalCoordinates (array): target coordinates
        maxVelocities (array): max. velocity per coordinate (> 0)
        maxAccelerations (array): max. acceleration per coordinate (> 0)
        mode (str): 'asynchronous', 'synchronous' (Exudyn ProfilePTP) or 'fullySynchronous' (lab interpolation)
        timeFactor (float): duration of 'fullySynchronous' is multiplied by this factor (lab: 1.05); must be 1 otherwise

    Returns:
        TrapezoidalPTPProfile or exudyn.robotics.motion.ProfilePTP
    """
    if mode not in ptpModes:
        raise ValueError('CreatePTPProfile: invalid mode "' + str(mode) + '", use one of ' + str(ptpModes))
    if mode == 'synchronous':
        if timeFactor != 1.:
            raise ValueError('CreatePTPProfile: timeFactor is only available for mode "fullySynchronous"')
        return ProfilePTP(finalCoordinates, syncAccTimes=False, maxVelocities=maxVelocities, maxAccelerations=maxAccelerations)
    return TrapezoidalPTPProfile(finalCoordinates, maxVelocities, maxAccelerations, mode, timeFactor)


class TrapezoidalPTPProfile:
    """PTP profile with trapezoidal velocity per axis, modes 'asynchronous' and 'fullySynchronous' (not available in
    Exudyn); add it to exudyn.robotics.motion.Trajectory with Trajectory.Add(...), see CreatePTPProfile.

    Example:
        trajectory = Trajectory(initialCoordinates=[0.]*6)
        trajectory.Add(TrapezoidalPTPProfile([0.5]*6, maxVelocities=[0.4]*6, maxAccelerations=[1.6]*6))
        [q, q_t, q_tt] = trajectory.Evaluate(1.)
    """

    def __init__(self, finalCoordinates, maxVelocities, maxAccelerations, mode='fullySynchronous', timeFactor=1.):
        """Initialize profile.

        Args:
            finalCoordinates (array): target coordinates
            maxVelocities (array): max. velocity per coordinate (> 0)
            maxAccelerations (array): max. acceleration per coordinate (> 0)
            mode (str): 'asynchronous' or 'fullySynchronous'
            timeFactor (float): fully synchronized duration is multiplied by this factor (lab: 1.05)
        """
        if mode not in trapezoidalModes:
            raise ValueError('TrapezoidalPTPProfile: invalid mode "' + str(mode) + '", use one of ' + str(trapezoidalModes) +
                             '; "synchronous" is exudyn.robotics.motion.ProfilePTP, see CreatePTPProfile')
        if np.min(maxVelocities) <= 0 or np.min(maxAccelerations) <= 0:
            raise ValueError('TrapezoidalPTPProfile: maxVelocities and maxAccelerations must be > 0')
        if timeFactor < 1.:
            raise ValueError('TrapezoidalPTPProfile: timeFactor must be >= 1')
        self.finalCoordinates = np.array(finalCoordinates, dtype=float)
        self.maxVelocities = np.array(maxVelocities, dtype=float)
        self.maxAccelerations = np.array(maxAccelerations, dtype=float)
        self.mode = mode
        self.timeFactor = timeFactor
        #timing data, computed in GetBasicProfile
        self.initialTime = 0.
        self.finalTime = 0.
        self.initialCoordinates = np.zeros(len(self.finalCoordinates))
        self.signs = np.zeros(len(self.finalCoordinates))
        self.axisDurations = np.zeros(len(self.finalCoordinates))
        self.accelerationTimes = np.zeros(len(self.finalCoordinates))
        self.cruiseVelocities = np.zeros(len(self.finalCoordinates))
        self.accelerations = np.zeros(len(self.finalCoordinates))

    def GetBasicProfile(self, initialTime, initialCoordinates, globalMaxVelocities, globalMaxAccelerations):
        """Return a copy of the profile with timing computed for initial time and coordinates (interface of Trajectory);
        global max. velocities/accelerations of the Trajectory are used if smaller than the profile values."""
        maxVelocities = self.maxVelocities if len(globalMaxVelocities) == 0 else np.minimum(self.maxVelocities, globalMaxVelocities)
        maxAccelerations = self.maxAccelerations if len(globalMaxAccelerations) == 0 else np.minimum(self.maxAccelerations, globalMaxAccelerations)
        profile = TrapezoidalPTPProfile(self.finalCoordinates, maxVelocities, maxAccelerations, self.mode, self.timeFactor)
        profile.ComputeTiming(initialTime, initialCoordinates)
        return profile

    def ComputeTiming(self, initialTime, initialCoordinates):
        """Compute durations, acceleration times, cruise velocities and accelerations of all axes."""
        self.initialTime = initialTime
        self.initialCoordinates = np.array(initialCoordinates, dtype=float)
        if len(self.initialCoordinates) != len(self.finalCoordinates):
            raise ValueError('TrapezoidalPTPProfile: initial and final coordinates have different size')
        distances = self.finalCoordinates - self.initialCoordinates
        self.signs = np.sign(distances)
        absoluteDistances = np.abs(distances)
        isMoving = absoluteDistances > 0.
        vMax = self.maxVelocities
        aMax = self.maxAccelerations

        if self.mode == 'fullySynchronous':
            #lab formula (Positionieren_T.m): t_e = s/v + v/a, lead axis has max. t_e
            endTimes = np.where(isMoving, absoluteDistances/vMax + vMax/aMax, 0.)
            if np.any(isMoving):
                leadAxis = int(np.argmax(endTimes))
                duration = self.timeFactor*endTimes[leadAxis]
                accelerationTime = min(vMax[leadAxis]/aMax[leadAxis], 0.5*duration)
            else:
                duration = 0.
                accelerationTime = 0.
            self.axisDurations = np.where(isMoving, duration, 0.)
            self.accelerationTimes = np.where(isMoving, accelerationTime, 0.)
            self.cruiseVelocities = np.where(isMoving, absoluteDistances/np.maximum(duration - accelerationTime, 1e-15), 0.)
            self.accelerations = np.where(isMoving, self.cruiseVelocities/max(accelerationTime, 1e-15), 0.)
        else:
            #asynchronous: time optimal profile per axis (triangular if max. velocity is not reached)
            reachesMaxVelocity = absoluteDistances >= vMax**2/aMax
            ownAccelerationTimes = np.where(reachesMaxVelocity, vMax/aMax, np.sqrt(absoluteDistances/aMax))
            ownDurations = np.where(reachesMaxVelocity, absoluteDistances/vMax + vMax/aMax, 2.*ownAccelerationTimes)
            self.axisDurations = np.where(isMoving, ownDurations, 0.)
            self.accelerationTimes = np.where(isMoving, ownAccelerationTimes, 0.)
            self.accelerations = np.where(isMoving, aMax, 0.)
            self.cruiseVelocities = self.accelerations*self.accelerationTimes
            duration = np.max(self.axisDurations, initial=0.)
        self.finalTime = initialTime + duration

    def GetInitialCoordinates(self):
        """Return coordinates at the start of the profile."""
        return self.initialCoordinates

    def GetFinalCoordinates(self):
        """Return coordinates at the end of the profile."""
        return self.finalCoordinates

    def GetDuration(self):
        """Return duration of the profile."""
        return self.finalTime - self.initialTime

    def Evaluate(self, time):
        """Return [coordinates, velocities, accelerations] at given (absolute) time."""
        t = time - self.initialTime
        tAcc = self.accelerationTimes
        tEnd = self.axisDurations
        a = self.accelerations
        v = self.cruiseVelocities
        sign = self.signs

        inAcceleration = (t > 0.) & (t < tAcc)
        inCruise = (t >= tAcc) & (t < tEnd - tAcc)
        inDeceleration = (t >= np.maximum(tEnd - tAcc, tAcc)) & (t < tEnd)
        finished = t >= tEnd
        remaining = tEnd - t

        displacement = np.zeros(len(sign))
        velocity = np.zeros(len(sign))
        acceleration = np.zeros(len(sign))
        displacement = np.where(inAcceleration, 0.5*a*t**2, displacement)
        velocity = np.where(inAcceleration, a*t, velocity)
        acceleration = np.where(inAcceleration, a, acceleration)
        displacement = np.where(inCruise, 0.5*a*tAcc**2 + v*(t - tAcc), displacement)
        velocity = np.where(inCruise, v, velocity)
        totalDistance = np.abs(self.finalCoordinates - self.initialCoordinates)
        displacement = np.where(inDeceleration, totalDistance - 0.5*a*remaining**2, displacement)
        velocity = np.where(inDeceleration, a*remaining, velocity)
        acceleration = np.where(inDeceleration, -a, acceleration)
        displacement = np.where(finished, totalDistance, displacement)
        if t <= 0.:
            displacement[:] = 0.
            velocity[:] = 0.
            acceleration[:] = 0.
        return [self.initialCoordinates + sign*displacement, sign*velocity, sign*acceleration]

    def EvaluateCoordinate(self, time, coordinate):
        """Return [coordinate, velocity, acceleration] of a single coordinate at given time."""
        [q, q_t, q_tt] = self.Evaluate(time)
        return [q[coordinate], q_t[coordinate], q_tt[coordinate]]

    def __repr__(self):
        return ('TrapezoidalPTPProfile(mode=' + self.mode + ', initialTime=' + str(self.initialTime) +
                ', finalTime=' + str(self.finalTime) + ', finalCoordinates=' + str(self.finalCoordinates) + ')')
