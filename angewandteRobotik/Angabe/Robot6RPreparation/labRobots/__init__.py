# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: __init__.py
#
# Details:  labRobots: the robots of the lab as equipment, shared by all of them (new layout):
#           robots/ (the lab robots kairos, leobot, robot6R, staubli, fanuc), arms/ (the shared base of the arms given
#           by a robot dictionary, grippers), interface/ (connection/: rosbridge, Vicon DataStream, ROS messages, laser
#           geometry; control/: one robot interface for simulation and real robot; safety/: limits of commanded
#           motions), labSetup/ (network, logging, data paths, scene settings), panels/ (ONE live data panel: the core
#           knows no robot; a RobotProfile per robot supplies topics, decoders, demo streams and the page layout) and
#           robotics/ (robot-independent collision checking, planning, sensors in the simulation).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-12
# Notes:    Staging copy in the workspace (mektcommon2/lab/equipment/robots); moves to the mektcommon
#           repository once Kairos, Leobot and the 6R robot run on it with all tests.
#
# License:  BSD-3 license
#
# Installation:
#           python -m pip install -e mektcommon2/lab/equipment/robots --config-settings editable_mode=compat
#           (pulls roslibpy, aiohttp; compat: a plain path entry, so a snapshot beside scripts wins); Vicon:
#           pip install "<Vicon DataStream SDK>\Win64\Python"
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Robots of the lab as equipment."""
