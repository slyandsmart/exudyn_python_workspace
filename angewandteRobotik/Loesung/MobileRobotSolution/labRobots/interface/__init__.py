# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: __init__.py
#
# Details:  The robot interface of the lab, the same for every robot: connection/ (rosbridge, Vicon DataStream,
#           ROS messages, laser geometry), control/ (one interface for simulation and real robot) and
#           safety/ (limits of commanded motions). This marker imports nothing: the student packages
#           carry interface/safety without connection and control.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-20
# Notes:    Restructure of labRobots (step 1: move only).
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Robot interface of the lab."""
