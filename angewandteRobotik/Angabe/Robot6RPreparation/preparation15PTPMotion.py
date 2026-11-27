# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: preparation15PTPMotion.py
#
# Details:  Preparation task 1.5 (Vorbereitung): PTP motion modes and blending, shown on the lab robot. The timing of
#           one PTP motion (zero configuration -> K2) in the asynchronous, synchronous and fully synchronous mode
#           (trapezoidal velocity per joint, the formulas of exercise sheet 3, max. velocities and accelerations of the
#           lab robot) is printed; three robots side by side in the renderer move in the three modes at once, with the
#           traces of their TCP and their joint velocities in the scopes. A path over two via points above the tables
#           is then driven with the blend factors 0 (stop at each via point), 0.5 and 1 (Exudyn
#           Trajectory.FromWaypoints): duration and distance of the TCP path from the via points are printed, the three
#           paths are shown side by side. The results are interpreted in the report.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-16
# Notes:    Run: python preparation15PTPMotion.py [--noGraphics]
#           The lab robot moves fully synchronous and stops at every row of Q (no blending); the other modes and
#           blending are shown in the simulation only.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys
import numpy as np
from exudyn.robotics.motion import Trajectory

from robot6R import parameters as par
from robot6R.ptpProfile import CreatePTPProfile
from robot6R.robotsSideBySide import RobotsSideBySide
from labRobots.robots.robot6R.model import ForwardKinematics

showVisualization = '--noGraphics' not in sys.argv
#! The PTP motion of the comparison: from the zero configuration to a placing pose on table 2 (joint angles in deg).
startAnglesDeg = [0., 0., 0., 0., 0., 0.]
targetAnglesDeg = [38.9, 58.7, 74.6, -8.8, -43.6, 6.4]
#! The path of the blending: zero configuration, 15 cm above table 1, 15 cm above table 2, the placing pose (deg).
waypointsDeg = [[0., 0., 0., 0., 0., 0.], [0., 47.2, 62.1, 0., -19.3, 0.], [38.9, 41.5, 76.3, -12.9, -28.4, 11.4],
                targetAnglesDeg]
#! Blend factors compared (0: stop at each via point, 1: Exudyn's full blending).
blendFactors = [0., 0.5, 1.]
#! Subtasks, each runs alone (True: run): printTiming prints the timing of the three PTP modes; showPTPModes shows them
#! side by side (robots, TCP traces, joint velocities in percent of the maximum in the scopes); showBlending compares
#! the path over the via points for the blendFactors (duration, distance from the via points) and shows the three
#! paths side by side.
printTiming = True
showPTPModes = True
showBlending = True


def MaxVelocities():
    """Return the max. joint velocities in rad/s of the lab robot (100 % speed)."""
    return np.array(par.labMaxVelocitiesDeg)*par.degToRad


def MaxAccelerations():
    """Return the max. joint accelerations in rad/s^2 of the lab robot (100 % speed)."""
    return np.array(par.labMaxAccelerationsDeg)*par.degToRad


def AsynchronousDurations(distances, maxVelocities, maxAccelerations):
    """Return the duration in s of the time optimal motion of each joint (asynchronous PTP): trapezoidal velocity, or a
    triangle for a distance s < v_max^2/a_max that does not reach v_max. Distances |q_end - q_start| in rad, max.
    velocities in rad/s, max. accelerations in rad/s^2 (arrays, one value per joint)."""
    reachesMaxVelocity = distances >= maxVelocities**2/maxAccelerations
    return np.where(reachesMaxVelocity, distances/maxVelocities + maxVelocities/maxAccelerations,
                    2.*np.sqrt(distances/maxAccelerations))


def SynchronousVelocities(distances, maxVelocities, maxAccelerations, duration):
    """Return the cruise velocity in rad/s of each joint of the synchronous PTP: all joints end at duration (the
    longest joint), each accelerates and decelerates with its max. acceleration a: s = v (T - v/a), the smaller root
    v = (a T - sqrt(a^2 T^2 - 4 a s))/2."""
    product = maxAccelerations*duration
    return 0.5*(product - np.sqrt(product**2 - 4.*maxAccelerations*distances))


def FullySynchronousTiming(distances, maxVelocities, maxAccelerations, timeFactor):
    """Return (duration in s, acceleration time in s, cruise velocities in rad/s, accelerations in rad/s^2) of the fully
    synchronous PTP of the lab controller: all joints end at the same time and accelerate during the same time. The
    lead axis has the largest t_e = s/v_max + v_max/a_max; duration T = timeFactor t_e (the safety margin of the lab,
    1.05), acceleration time t_b = v_max/a_max of the lead axis; every joint moves with v = s/(T - t_b) and accelerates
    with a = v/t_b."""
    endTimes = distances/maxVelocities + maxVelocities/maxAccelerations
    leadAxis = int(np.argmax(endTimes))
    duration = timeFactor*endTimes[leadAxis]
    accelerationTime = maxVelocities[leadAxis]/maxAccelerations[leadAxis]
    velocities = distances/(duration - accelerationTime)
    return duration, accelerationTime, velocities, velocities/accelerationTime


def PrintTiming():
    """Print the timing of the PTP motion in the three modes: durations, cruise velocities and accelerations per joint."""
    distances = np.abs(np.radians(targetAnglesDeg) - np.radians(startAnglesDeg))
    maxVelocities, maxAccelerations = MaxVelocities(), MaxAccelerations()
    print('PTP motion from', startAnglesDeg, 'to', targetAnglesDeg, 'deg; max. velocities', par.labMaxVelocitiesDeg,
          'deg/s, max. accelerations', par.labMaxAccelerationsDeg, 'deg/s^2')
    durations = AsynchronousDurations(distances, maxVelocities, maxAccelerations)
    print('asynchronous:     duration %.3f s, durations of the joints %s s' % (np.max(durations), np.round(durations, 3)))
    velocities = SynchronousVelocities(distances, maxVelocities, maxAccelerations, np.max(durations))
    print('synchronous:      duration %.3f s, cruise velocities %s deg/s' % (np.max(durations), np.round(np.degrees(velocities), 3)))
    duration, accelerationTime, velocities, accelerations = FullySynchronousTiming(
        distances, maxVelocities, maxAccelerations, par.labMotionTimeFactor)
    print('fully synchronous: duration %.3f s (factor %g), acceleration time %.3f s, cruise velocities %s deg/s, '
          'accelerations %s deg/s^2' % (duration, par.labMotionTimeFactor, accelerationTime,
                                        np.round(np.degrees(velocities), 3), np.round(np.degrees(accelerations), 3)))


def PTPMotions():
    """Return [(mode, Trajectory)] of the PTP motion from startAnglesDeg to targetAnglesDeg in the three modes."""
    start, target = np.radians(startAnglesDeg), np.radians(targetAnglesDeg)
    motions = []
    for mode in ('asynchronous', 'synchronous', 'fullySynchronous'):
        timeFactor = par.labMotionTimeFactor if mode == 'fullySynchronous' else 1.
        trajectory = Trajectory(initialCoordinates=start)
        trajectory.Add(CreatePTPProfile(target, MaxVelocities(), MaxAccelerations(), mode, timeFactor))
        motions.append((mode, trajectory))
    return motions


def BlendedMotion(blendFactor):
    """Return the Trajectory of the path over the via points (waypointsDeg) blended with blendFactor (0 ... 1)."""
    if not 0. <= blendFactor <= 1.:
        raise ValueError('the blend factor must be in [0, 1], not %g' % blendFactor)
    waypoints = [np.radians(angles) for angles in waypointsDeg]
    return Trajectory.FromWaypoints(waypoints, MaxVelocities(), MaxAccelerations(), blendFactor=blendFactor)


def BlendedMotions():
    """Return [(label, Trajectory)] of the path over the via points for each of the blendFactors."""
    return [('blend factor %g' % blendFactor, BlendedMotion(blendFactor)) for blendFactor in blendFactors]


def ViaPoints():
    """Return the TCP positions in K0 of the via points (the corners of the path above the tables)."""
    return [ForwardKinematics(np.radians(angles))[0:3, 3] for angles in waypointsDeg[1:-1]]


def ViaPointDistances(trajectory, numberOfSamples=4001):
    """Return the smallest distance in m of the TCP path of a trajectory from each via point (the TCP path sampled at
    numberOfSamples times by the forward kinematics)."""
    times = np.linspace(0., trajectory.GetTimes()[-1], numberOfSamples)
    path = np.array([ForwardKinematics(trajectory.Evaluate(time)[0])[0:3, 3] for time in times])
    return np.array([np.min(np.linalg.norm(path - point, axis=1)) for point in ViaPoints()])


def BlendingComparison():
    """Return [(blend factor, duration in s, largest distance in m of the TCP path from the via points)] for the
    blendFactors."""
    comparison = []
    for blendFactor in blendFactors:
        trajectory = BlendedMotion(blendFactor)
        comparison.append((blendFactor, trajectory.GetTimes()[-1], float(np.max(ViaPointDistances(trajectory)))))
    return comparison


if __name__ == '__main__':
    if printTiming:
        PrintTiming()
    if showPTPModes and showVisualization:
        RobotsSideBySide(PTPMotions(), MaxVelocities()).Show()
    if showBlending:
        print('path over two via points (synchronous PTP of Exudyn Trajectory.FromWaypoints):')
        for blendFactor, duration, distance in BlendingComparison():
            print('  blend factor %.1f: duration %.3f s, largest distance from the via points %.1f mm'
                  % (blendFactor, duration, 1000.*distance))
        if showVisualization:
            RobotsSideBySide(BlendedMotions(), MaxVelocities(), markerPoints=ViaPoints()).Show()
