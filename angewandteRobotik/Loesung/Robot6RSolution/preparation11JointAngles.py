# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: preparation11JointAngles.py
#
# Details:  Preparation task 1.1 (Vorbereitung): getting to know the simulation model of the lab robot - joint angles
#           given by hand are driven as PTP motions (fully synchronous, as the controller of the lab robot) from the
#           zero configuration and back; for every configuration the TCP pose (ForwardKinematics) and the check of the
#           lab (CheckLabConfiguration: joint limits, collisions with the tables) are printed, the TCP poses drawn as
#           frames P1, P2, ... in the renderer.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-15
# Notes:    Run: python preparation11JointAngles.py [--noGraphics]
##BEGIN_MASTERONLY
#           Master file: autoGenerateTemplates.py writes the student template (isTemplate blocks -> ## TODO).
##END_MASTERONLY
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
import numpy as np

from robot6R import parameters as par
from labRobots.robots.robot6R.model import ForwardKinematics, CheckLabConfiguration
from robot6R.labTrajectory import CreateTrajectoryMatrix
from robot6R.simulation import SimulateBahnfahrt
from robot6R.plotting import PrintSimulationSummary

##BEGIN_MASTERONLY
isTemplate = False   #True: run as the student template would (without the solution)
##END_MASTERONLY

showVisualization = '--noGraphics' not in sys.argv
solutionDirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'solution')
speedPercent = 40.      #speed of the PTP motions in percent of the maximum speed of the lab robot (5 to 100)
#! Subtasks, each runs alone (True: run): printPoses prints the TCP pose and the lab check of every configuration,
#! driveConfigurations drives them in the simulation (the TCP poses as frames P1, P2, ... in the renderer).
printPoses = True
driveConfigurations = True


def JointAnglesDeg():
    """Return the configurations to drive: list of 6 joint angles in deg (axes 1 to 6)."""
    if not isTemplate:
        return [[30., 0., 0., 0., 0., 0.],          #axis 1 turns the whole arm about the vertical axis
                [30., 30., 60., 0., 0., 0.],        #axes 2 and 3 move the arm forward and down
                [30., 30., 60., 0., -60., 0.],      #axis 5 tilts the gripper
                [0., 45., 90., 0., -45., 0.],       #gripper horizontal in front of the robot
                [0., 0., 0., 0., 0., 0.]]           #back to the zero configuration
    else:
        ## TODO 1: Give at least four configurations of the robot (6 joint angles in deg each, axis 1 to 6): move
        ##         the axes one after the other (e.g. axis 1 by 30 deg, then axes 2 and 3, then axis 5) and observe
        ##         in the renderer how the arm and the TCP frame move. Which configurations are rejected by the lab
        ##         check, and why? End with the zero configuration [0, 0, 0, 0, 0, 0].
        return [[30., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.]]


def PrintPoses(jointAnglesList):
    """Print the TCP pose and the lab check of every configuration (joint angles in rad)."""
    for index, q in enumerate(jointAnglesList):
        isValid, message = CheckLabConfiguration(q)
        pose = ForwardKinematics(q)
        print('P%d: q = %s deg, lab check: %s %s' % (index + 1, np.round(np.asarray(q)*par.radToDeg, 1),
                                                    'valid' if isValid else 'REJECTED', message))
        print('    TCP position %s m, gripper axis z_G %s' % (np.round(pose[0:3, 3], 4), np.round(pose[0:3, 2], 3)))


if __name__ == '__main__':
    jointAnglesList = [np.array(angles, dtype=float)*par.degToRad for angles in JointAnglesDeg()]
    if printPoses:
        PrintPoses(jointAnglesList)
    if driveConfigurations:
        frames = [('P%d' % (index + 1), ForwardKinematics(q)) for index, q in enumerate(jointAnglesList)]
        matrixQ = CreateTrajectoryMatrix(jointAnglesList, speedPercent=speedPercent, gripperCommand=par.gripperCommandNone)
        results, simulation = SimulateBahnfahrt(matrixQ, 'rad', useRenderer=showVisualization, showSolution=showVisualization, includeCup=False,
                                                frames=frames, solutionFileName=os.path.join(solutionDirectory, 'preparation11'))
        PrintSimulationSummary(results)
