# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: motionLimits.py
#
# Details:  Limits of commanded motions, shared by the robot interface (last check before a robot) and the models of
#           the lab tasks (state machines of the students): platform velocity clamped to a speed and a turn rate.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-12
# Notes:    numpy only; part of the labRobots snapshot of the student packages (no connection, no control).
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Limits of commanded motions."""
import numpy as np

#! Relative tolerance of the limits: a velocity scaled to the limit itself (e.g. 0.3 m/s as
#! 0.30000000000000004) is not reported as clamped.
limitRelativeTolerance = 1e-9


def ClampPlatformVelocity(velocity, maximumSpeed, maximumTurnRate):
    """Return (velocity clamped to maximumSpeed (norm of vx, vy) and maximumTurnRate, True if it was clamped); the last
    check before a velocity leaves for a robot (a student state machine reached the real
    robot unclamped). A velocity within limitRelativeTolerance of a limit is clamped without report.

    Args:
        velocity: Local platform velocity [vx in m/s, vy in m/s, omega in rad/s].
        maximumSpeed: Largest translational speed in m/s.
        maximumTurnRate: Largest turn rate in rad/s.
    """
    velocity = np.array(velocity, dtype=float)
    speed = np.linalg.norm(velocity[:2])
    clamped = False
    if speed > maximumSpeed:
        velocity[:2] *= maximumSpeed/speed
        clamped = speed > maximumSpeed*(1. + limitRelativeTolerance)
    if abs(velocity[2]) > maximumTurnRate:
        clamped = clamped or abs(velocity[2]) > maximumTurnRate*(1. + limitRelativeTolerance)
        velocity[2] = np.sign(velocity[2])*maximumTurnRate
    return velocity, clamped
