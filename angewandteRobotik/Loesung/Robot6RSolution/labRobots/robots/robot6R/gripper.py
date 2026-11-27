# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: gripper.py
#
# Details:  Servo gripper of the 6R robot: jaw coordinate of a gripper command in percent (parallelogram of
#           Ableitung_Greifer), opening width between the jaw arcs and the inverse (percent of a width). 0 % is open,
#           gripperContactPercent puts the jaw arc centers on the TCP (opening = 2 arc radii), more squeezes.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-12 (formula of robot6R/visualization.py ComputeJawOffset, 2026-11-03)
# Notes:    numpy only; part of the labRobots snapshot of the student packages. The arc radius of the jaws comes from
#           the CAD data (robotModelData.json, gripperJaws, about 25.6 mm).
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Servo gripper of the 6R robot."""
import numpy as np
from .geometry import gripperServoAngleOffsetDeg, gripperServoAnglePerPercentDeg, gripperLinkLength, gripperContactPercent

#! Range of the servo angle in deg (Positionieren_Greifer.m).
gripperMinimumServoAngle, gripperMaximumServoAngle = -50., 21.


def ServoAngle(percent):
    """Return the servo angle in rad of a gripper command in percent."""
    return np.radians(gripperServoAngleOffsetDeg + gripperServoAnglePerPercentDeg*percent)


def JawOffset(percent):
    """Return the jaw coordinate in m: displacement of each jaw along the closing direction relative to the position
    with the jaw arc center on the TCP (reached at gripperContactPercent); positive = opened, negative = squeezing.

    Args:
        percent: Gripper command in percent (0 = open, 70 = closed on the cup).
    """
    return gripperLinkLength*(np.sin(ServoAngle(gripperContactPercent)) - np.sin(ServoAngle(percent)))


def WidthFromPercent(percent, arcRadius):
    """Return the opening width in m between the jaw arcs of a gripper command in percent.

    Args:
        percent: Gripper command in percent.
        arcRadius: Radius in m of the jaw arcs (CAD data).
    """
    return 2.*(arcRadius + JawOffset(percent))


def PercentFromWidth(width, arcRadius):
    """Return the gripper command in percent of an opening width in m between the jaw arcs; raise ValueError if the
    servo cannot reach it.

    Args:
        width: Opening width in m.
        arcRadius: Radius in m of the jaw arcs (CAD data).
    """
    sine = np.sin(ServoAngle(gripperContactPercent)) - (0.5*width - arcRadius)/gripperLinkLength
    servoAngle = np.degrees(np.arcsin(sine)) if abs(sine) <= 1. else np.nan
    if not gripperMinimumServoAngle - 1e-9 <= servoAngle <= gripperMaximumServoAngle + 1e-9:
        raise ValueError('Gripper width %.4f m is outside the range of the 6R gripper (%.4f .. %.4f m)'
                         % (width, WidthFromPercent(PercentOfServoAngle(gripperMaximumServoAngle), arcRadius),
                            WidthFromPercent(PercentOfServoAngle(gripperMinimumServoAngle), arcRadius)))
    return PercentOfServoAngle(servoAngle)


def PercentOfServoAngle(servoAngle):
    """Return the gripper command in percent of a servo angle in deg."""
    return (servoAngle - gripperServoAngleOffsetDeg)/gripperServoAnglePerPercentDeg
