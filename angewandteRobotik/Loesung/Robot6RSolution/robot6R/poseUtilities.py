# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: poseUtilities.py
#
# Details:  Helpers for TCP poses of the tasks: OrthonormalPose makes the rotation of a pose given with a few digits
#           (e.g. parameters.configurationTaskPose, the pose T of the preparation task 1.3) exactly orthonormal, so that
#           the analytic inverse kinematics and the forward kinematics agree to machine precision.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-21 (from robot6R/ikSolutionsAnimation.py, removed: the display of all solutions is no longer used)
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Helpers for TCP poses of the tasks."""
import numpy as np


def OrthonormalPose(pose):
    """Return the 4x4 pose with an exactly orthonormal rotation matrix (the nearest one, by the singular value
    decomposition); the position is kept.

    Args:
        pose (array): 4x4 homogeneous transformation, rotation possibly given with a few digits only.

    Returns:
        array: 4x4 homogeneous transformation with orthonormal rotation.
    """
    result = np.array(pose, dtype=float)
    u, unusedSingularValues, vt = np.linalg.svd(result[0:3, 0:3])
    result[0:3, 0:3] = u @ vt
    return result
