# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: kinematics.py
# Details:  Mecanum wheel kinematics and world-pose feedback control.
# Authors:  Peter Manzl; Michael Pieber (created with Claude Code)
# Date:     2026-11-04; 2026-11-12 moved into labRobots/robots/kairos (from LabCourse/MobileRobot/fahrzeug)
# Notes:    Wheel order: front left, front right, rear left, rear right.
#           wheel kinematics by exudyn.robotics.mobile.MobileKinematics
#           (same matrix as before: half track and half base give (track + base)/2).
#           parameters of the robot's base controller (cmd_vel -> wheel speeds); the
#           simulated wheels move the platform with the physical geometry (wheelRadius, wheelBase, wheelTrack).
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Kinematics of the preparation model."""
import numpy as np
from exudyn.robotics.mobile import MobileKinematics
from labRobots.robots.kairos import parameters as par
from labRobots.interface.safety.motionLimits import ClampPlatformVelocity as SharedClampPlatformVelocity

#! Mecanum kinematics of the base controller (O-configuration); MobileKinematics expects half track and half base.
mecanumKinematics = MobileKinematics(par.controllerWheelRadius, par.controllerWheelTrack/2., par.controllerWheelBase/2.)


def WheelSpeeds(velocity):
    """Return the four wheel angular velocities for a local platform velocity [vx, vy, omega].

    Args:
        velocity: Local platform velocity in m/s, m/s, rad/s.

    Returns:
        Four wheel speeds in rad/s.
    """
    return mecanumKinematics.GetWheelVelocities(np.asarray(velocity, dtype=float))


def PlatformVelocity(wheelSpeeds):
    """Return local platform velocity [vx, vy, omega] from four wheel angular velocities."""
    return mecanumKinematics.GetCartesianVelocities(np.asarray(wheelSpeeds, dtype=float))


def ClampPlatformVelocity(velocity):
    """Return (velocity clamped to par.maximumDriveSpeed (norm of vx, vy) and par.maximumTurnRate, True if it was
    clamped): the velocity of a student state machine reached the real robot unclamped.
    The check itself is the one of the robot interface (labRobots.interface.safety.motionLimits, 2026-11-12)."""
    return SharedClampPlatformVelocity(velocity, par.maximumDriveSpeed, par.maximumTurnRate)


def WrapAngle(angle):
    """Return angle in radians wrapped to [-pi, pi)."""
    return (angle + np.pi) % (2.*np.pi) - np.pi


