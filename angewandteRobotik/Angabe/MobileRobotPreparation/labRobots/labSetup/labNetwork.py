# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: labNetwork.py
#
# Details:  The addresses and ports of the lab (labNetwork.json): Vicon PC, lab PC, robot PCs with rosbridge, ROS
#           master, SSH user and the ports of their panels. The device data (topics.py of the robots) take their
#           addresses from here; the startLab.ps1 scripts read the same JSON file.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-12
# Notes:    Addresses belong to the lab, not to the device: a robot moved into another network changes only
#           labNetwork.json. Part of the labRobots snapshot of the student packages. No secrets in the file.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Addresses and ports of the lab."""
import json
from pathlib import Path

#! The JSON file of the lab network.
labNetworkFileName = Path(__file__).resolve().parent / 'labNetwork.json'
with open(labNetworkFileName, 'r', encoding='utf-8') as labNetworkFile:
    #! Content of labNetwork.json.
    labNetwork = json.load(labNetworkFile)

#! Vicon PC (DataStream server) of the lab and its port.
viconHost, viconDataStreamPort = labNetwork['viconPc']['host'], labNetwork['viconPc']['dataStreamPort']
#! Lab PC in the WLAN of the robots and the port of its ExuXR viewer.
labPcHost, exuxrPort = labNetwork['labPc']['host'], labNetwork['labPc']['exuxrPort']


def RobotNetwork(name):
    """Return the network entry of a robot (host, rosbridgePort, rosMasterPort, user, panelPort, demoPanelPort, ...).

    Args:
        name: Robot name of labNetwork.json ('kairos', 'leobot', 'robot6R').
    """
    if name not in labNetwork['robots']:
        raise ValueError('Robot %r not in %s (known: %s)' % (name, labNetworkFileName, ', '.join(sorted(labNetwork['robots']))))
    return dict(labNetwork['robots'][name])


def ViconHost(name=None):
    """Return the address of the Vicon PC as seen from the network of a robot: the entry
    'viconHost' of the robot (Leobot: second address of the Vicon PC in the WLAN Leobot_5G), else viconHost.

    Args:
        name: Robot name of labNetwork.json; None: the lab default viconHost.
    """
    if name is None:
        return viconHost
    return RobotNetwork(name).get('viconHost', viconHost)
