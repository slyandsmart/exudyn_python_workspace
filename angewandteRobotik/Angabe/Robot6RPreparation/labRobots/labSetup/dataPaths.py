# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: dataPaths.py
#
# Details:  The one place that knows where the robot data (URDF packages, meshes) of the lab robots are:
#           data/Exudyn/robotics of mektcommon2 (workspace), or the snapshot labRobots/data/Exudyn/robotics that the
#           student packages carry beside their scripts.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-12
# Notes:    The snapshot folder wins when it exists (student package without mektcommon2).
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Paths of the robot data of the lab robots."""
from pathlib import Path

#! Robot data copied into a student package (labRobots/data/Exudyn/robotics) and in the workspace
#! (mektcommon2/data/Exudyn/robotics beside mektcommon2/lab/equipment/robots/labRobots); this file is in
#! labRobots/labSetup (restructure of labRobots: parents[1] is labRobots, parents[5] is mektcommon2).
#! the workspace folder only where the path has enough levels (a student package unpacked
#! into a short path such as C:\Temp\package has no parents[5]: IndexError before); None outside the workspace.
labRobotsFolder = Path(__file__).resolve().parents[1]
workspaceIndexAboveLabRobots = 3        #! labRobots.parents[3] is mektcommon2 (lab/equipment/robots/labRobots).
snapshotRoboticsFolder = labRobotsFolder / 'data' / 'Exudyn' / 'robotics'
workspaceRoboticsFolder = (labRobotsFolder.parents[workspaceIndexAboveLabRobots] / 'data' / 'Exudyn' / 'robotics'
                           if len(labRobotsFolder.parents) > workspaceIndexAboveLabRobots else None)
#! Folder data/Exudyn/robotics in use and its URDF packages.
if snapshotRoboticsFolder.is_dir():
    roboticsDataFolder = snapshotRoboticsFolder
elif workspaceRoboticsFolder is not None:
    roboticsDataFolder = workspaceRoboticsFolder
else:
    raise FileNotFoundError('labRobots: no robot data, neither the snapshot %s nor the folder data/Exudyn/robotics '
                            'of mektcommon2' % snapshotRoboticsFolder)
urdfFolder = roboticsDataFolder / 'URDF'
#! URDF packages and meshes of the robots of the lab.
kairosDescriptionFolder = urdfFolder / 'kairos_description'
leobotDescriptionFolder = urdfFolder / 'leobot_description'
robot6RDescriptionFolder = urdfFolder / 'robot6R_description'
#! Meshes of grippers and fingers usable on several robots (labRobots.arms.grippers).
grippersFolder = roboticsDataFolder / 'grippers'
