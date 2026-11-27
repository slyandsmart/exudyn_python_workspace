# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: jointLimits.py
#
# Details:  Joint check of the lab software of the 6R robot (first part of check_input.m): joint limits and the sum of
#           the wrist servos. Used by the CAN driver (every PTP target) and by the model of the lab task, which adds the
#           workspace check of the lab setup.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-12 (from robot6R/model.py CheckLabConfiguration, 2026-11-03)
# Notes:    numpy only; part of the labRobots snapshot of the student packages.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Joint check of the 6R robot."""
import numpy as np
from . import geometry as par


def CheckJointLimits(q):
    """Joint check of the lab software (first part of check_input.m) for joint angles q in rad: joint limits and the
    sum of the wrist servos. The workspace check of the lab setup is the lab task's (configurationCheck).

    Returns:
        tuple: (isValid, message)
    """
    qDeg = np.degrees(np.array(q, dtype=float))
    for jointIndex in range(par.numberOfJoints):
        if qDeg[jointIndex] < par.labJointLowerLimitsDeg[jointIndex] or qDeg[jointIndex] > par.labJointUpperLimitsDeg[jointIndex]:
            return False, 'joint ' + str(jointIndex+1) + ' angle ' + str(round(qDeg[jointIndex], 3)) + ' deg outside of [' + \
                str(par.labJointLowerLimitsDeg[jointIndex]) + ', ' + str(par.labJointUpperLimitsDeg[jointIndex]) + '] deg'
    if qDeg[4] + qDeg[5] > par.labWristServoSumLimitDeg:
        return False, 'q5 + q6 > ' + str(par.labWristServoSumLimitDeg) + ' deg (wrist servo deflection too large)'
    return True, ''
