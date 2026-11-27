# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: topics.py
#
# Details:  Connection and panel values of the 6R robot: CAN channel and panel ports (labNetwork.json), the topics
#           under which the CAN states are served to the panel (local rosbridge of panelProfile.py), the joint names
#           of the four Schunk axes with state feedback, poll period and age limit of the states.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-12
# Notes:    The wrist servos (axes 5, 6) and the gripper have no feedback: the panel shows axes 1-4.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Connection and panel values of the 6R robot."""
from labRobots.labSetup.labNetwork import RobotNetwork

#! Addresses of the 6R in the lab network (labRobots/labSetup/labNetwork.json): PCAN channel and the ports of its panels.
robot6RNetwork = RobotNetwork('robot6R')
canChannel, dataPanelPort, demoPanelPort = robot6RNetwork['canChannel'], robot6RNetwork['panelPort'], robot6RNetwork['demoPanelPort']
#! Topics of the local rosbridge that serves the CAN states to the panel, and their message types.
jointStatesTopic, jointStatesMessageType = '/robot6R/joint_states', 'sensor_msgs/JointState'
moduleStatusTopic, moduleStatusMessageType = '/robot6R/modules', 'robot6R/ModuleStatus'
#! Names of the four Schunk axes with state feedback (axes 1-4).
jointNames = ('joint1', 'joint2', 'joint3', 'joint4')
#! Period in s of the GET_STATE queries (20 Hz for the four modules) and the age in s after which a state counts as lost.
pollPeriod, staleTimeout = 0.05, 0.5
