# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: basePose.py
#
# Details:  Base pose of the mobile robots of the lab in the models of the style of exudyn.robotics: the planar pose
#           [x, y, yaw] of the platform reference point (center of the platform on the floor, x forward, z up) from the
#           HT of RobotBase, with the check that the base stands on the floor and rotates about z only.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-12 (models in the style of Exudyn)
# Notes:    numpy only; part of the labRobots snapshot of the student packages (no connection, no control).
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Planar base pose of the mobile robots."""
import numpy as np

#! Tolerance of the check that a base HT is planar (rotation about z, on the floor).
planarTolerance = 1e-9


def PlanarPoseFromHT(HT):
    """the planar pose of a mobile robot from the HT of its reference point

    Args:
        HT: exudyn.HT or 4x4 matrix: center of the platform on the floor, rotated about z only

    Returns:
        numpy array [x in m, y in m, yaw in rad]; ValueError if the HT is tilted or not on the floor (z = 0)
    """
    matrix = np.array(HT, dtype=float).reshape(4, 4)
    if abs(matrix[2, 2] - 1.) > planarTolerance or abs(matrix[2, 3]) > planarTolerance:
        raise ValueError('PlanarPoseFromHT: the base HT of a mobile robot must lie on the floor (z = 0) and rotate about z '
                         'only, got\n' + str(matrix))
    return np.array([matrix[0, 3], matrix[1, 3], np.arctan2(matrix[1, 0], matrix[0, 0])])
