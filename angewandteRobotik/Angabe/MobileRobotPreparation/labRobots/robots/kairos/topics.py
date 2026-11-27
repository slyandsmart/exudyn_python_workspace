# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: topics.py
#
# Details:  Connection and command values of the real Kairos: rosbridge of the robot PC, joint states of the UR driver,
#           drive command topic with send period, watchdog and diagnostics topics, UR5 trajectory action with start
#           check, arrival and timeouts, WSG50 services, widths and status, speed limits of the lab. Moved from
#           LabCourse/MobileRobot/fahrzeug/parameters.py (2026-11-12); the topics of the panel follow
#           in step 5: Vicon object, gamepad, SSH user, panel port (fahrzeug/parameters.py) and the laser
#           mountings and topics (fahrzeug/sensors/lidar.py).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-12
# Notes:    Pure data file of the device (numbers, names, tuples; numpy only): the robot as built, the same for
#           every scene, simulation and the real robot. No scene values (tables, objects, start poses, named
#           configurations), no lab values (Vicon, calibrations), no model settings (contact, control, solver).
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Connection, command and panel values of the Kairos."""
import numpy as np
from labRobots.labSetup.labNetwork import RobotNetwork, viconHost, viconDataStreamPort as labViconDataStreamPort

#! Addresses of the Kairos in the lab network (labRobots/labSetup/labNetwork.json, the one place of the addresses).
kairosNetwork = RobotNetwork('kairos')

#! rosbridge websocket port of the robot PC (bringup_lecture.py).
rosbridgePort = kairosNetwork['rosbridgePort']
#! Robot PC of the Kairos in the lab WLAN (rosbridge, SSH user robotUser): default of the panel.
robotHost = kairosNetwork['host']
#live arm angles of the real robot via its rosbridge (labRobots/robots/kairos/robot.py RobotStateClient, --robot HOST).
#! Joint state topic and message type of the robot (ur_driver, namespace summit_xl).
robotStateTopic, robotStateMessageType = '/summit_xl/joint_states', 'sensor_msgs/JointState'
#! Minimum time between two joint state messages in ms (the robot publishes at about 170 Hz); 50 Hz keep the
#! start angles of sent arm trajectories current.
robotStateThrottleRate = 20
#! Connection/first data timeout in s; arm angles older than robotStateDataTimeout in s are reported as old.
robotStateConnectTimeout, robotStateDataTimeout = 10., 1.
#commands of the simulation sent to the real robot (labRobots/robots/kairos/robot.py KairosRobot, --send).
#! WSG50 gripper as in bringup_lecture.py: move service and type, closing and opening width in mm, speed in mm/s.
#! Open: 100 mm, clear of the end stop; 110 mm (the WSG50 maximum) ran into it and blocked the axis: every move
#! failed with error 255 "Axis is blocked in negative moving direction" (robot log 2026-11-06).
robotGripperService, robotGripperServiceType = '/summit_xl/wsg_50_driver/move', 'wsg_50_common/Move'
robotGripperCloseWidth, robotGripperOpenWidth, robotGripperSpeed = 80., 100., 35.
#! Time in s the gripper may take before the task stops with an error.
robotGripperTimeout = 10.
#homing at every start/reset clears a blocked axis (e.g. after running into the end stop)
#! WSG50 homing service and type, minimum duration in s before an unchanged width counts as finished, timeout in s.
robotGripperHomingService, robotGripperHomingServiceType = '/summit_xl/wsg_50_driver/homing', 'std_srvs/Empty'
robotGripperHomingMinimumTime, robotGripperHomingTimeout = 3., 20.
#! Gripper status (width in mm, force in N; at 1 Hz): the driver runs in polling mode on the robot, where the move
#! service may answer very late and the status speed is always 0. The gripper counts as finished when its width is
#! within robotGripperWidthTolerance in mm of the target, or when the width stays unchanged (within the tolerance)
#! over two status messages at least robotGripperBlockTime in s after sending (blocked at the object).
robotGripperStatusTopic, robotGripperStatusType = '/summit_xl/wsg_50_driver/status', 'wsg_50_common/Status'
robotGripperWidthTolerance, robotGripperBlockTime = 2., 2.
#! Closing with a force below this value in N is reported: nothing gripped (object narrower than the close width).
robotGripperMinimumGraspForce = 2.
#! Drive command topic and type (as TemplateLab.py / bringup_lecture.py); in the robot's twist_mux this input
#! (rc_sender, priority 20) overrides the gamepad (priority 10) until 0.5 s after the last command.
robotDriveTopic, robotDriveMessageType = '/summit_xl/cmd_vel', 'geometry_msgs/Twist'
#! Send period in s of wall-clock time (own thread; the base controller stops after cmd_vel_timeout 0.25 s).
robotDriveCommandPeriod = 0.05
#! Watchdog in s: without a new velocity of the simulation for this wall-clock time, zero is sent.
robotDriveWatchdogTime = 0.3
#! Drive diagnostics: twist_mux output and safety controller output (= input of the base controller), read only.
robotDriveMonitorTopics = ('/summit_xl/twist_mux/cmd_vel', '/summit_xl/robotnik_base_control/cmd_vel')
#! Gap in s that stops the base controller (cmd_vel_timeout of summit_xl_control); reported by the diagnostics.
robotBaseCommandTimeout = 0.25
#! Arm trajectory action of the UR driver: server name and action type.
robotArmActionServer, robotArmActionType = '/summit_xl/follow_joint_trajectory', 'control_msgs/FollowJointTrajectoryAction'
#! Sampling period in s of the simulated arm trajectory for the trajectory points sent to the robot.
robotArmSamplePeriod = 0.1
#! Largest difference in rad between the real arm angles and the start of a trajectory to send it; the UR driver
#! (ur_modern_driver) rejects a goal whose first point differs by more than 0.01 rad from the actual angles.
robotArmStartTolerance = 0.008
#! A sent arm motion is finished when the real arm stands still (all joint speeds below robotArmStationarySpeed in
#! rad/s) within robotArmArrivalTolerance in rad of the trajectory end (plausibility; the UR driver stops a few
#! thousandths of a rad before the end, the next motion is planned from the real angles anyway).
robotArmStationarySpeed, robotArmArrivalTolerance = 0.01, 0.05
#! Time in s the real arm may need beyond the duration of a sent trajectory before the task stops with an error.
robotArmFinishTimeout = 10.
#! Planning/sending an arm motion waits up to this time in s for joint angles not older than robotStateDataTimeout
#! (the simulation thread hardly lets the websocket thread run); then the task stops with an error.
robotStateStaleTimeout = 5.
#step-by-step execution (runSimulation.py): every drive segment is simulated, then replayed
#on the robot, then the simulated vehicle is set to the Vicon pose of the real robot.
#! Wall-clock sleep in s per control step while the simulation waits for the real robot (frees the websocket thread).
robotWaitSleepTime = 0.005
#the real base does not follow the replayed profile exactly; a drive action whose target is
#missed after the Vicon update is simulated and replayed again from the real pose (within the action tolerances).
#lab recordings of 2026-11-06 (drive.txt vs vehicle.txt): the base starts 0.23-0.29 s
#(rotation) and about 0.18 s (translation) after the first command, but stops at once; the lost start was the
#41 instead of 47 deg (not slip: the robot's odometry agrees with Vicon). The replay holds the first command longer.
#! Time in s by which the first command of a replayed drive profile is held longer (start lag of the real base).
robotDriveStartCompensation = 0.22
#! Speed limits of the lab in m/s and rad/s: commands to the real robot are clamped to them (also the limits of the
#! simulated pose controller).
maximumDriveSpeed, maximumTurnRate = 0.3, 0.4
#! Wheel odometry of the base controller (nav_msgs/Odometry, about 50 Hz): GetState()['odometry'] of the real robot, as
#! the Leobot (before only in panelProfile.py).
robotOdometryTopic, robotOdometryMessageType = '/summit_xl/robotnik_base_control/odom', 'nav_msgs/Odometry'

#Panel and lab connection of the Kairos
#live Vicon data (fahrzeug/lab/vicon.py, --vicon HOST): topics and message types as
#vicon_bridge and bringup_lecture.py.
viconRobotTopic, viconRobotMessageType = '/vicon/kairos/kairos', 'geometry_msgs/TransformStamped'
viconMarkersTopic, viconMarkersMessageType = '/vicon/markers', 'vicon_bridge/Markers'
#! Markers closer than this distance in m to the vehicle are not the object (bringup_lecture.py callbackMarker).
viconRobotRejectionRadius = 0.5
#Vicon comes directly from the Vicon PC with the Vicon DataStream SDK on Windows
#(fahrzeug/lab/viconDataStream.py, --vicon HOST); the same Vicon PC serves the Leobot (ExuXRAdvancedExamples/Leobot).
#! Vicon PC (DataStream server) of the lab and its port.
viconDataStreamHost, viconDataStreamPort = viconHost, labViconDataStreamPort
#markers of the vehicle object in VR/AR (check of a scene placement aligned by hand)
#! Vicon subject of the vehicle, radius in m and color of its markers in the viewer.
viconRobotSubject = 'kairos'
#gamepad of the robot (pad_teleop) in the data panel: sticks and buttons live, Bluetooth of the
#controller on the robot PC over SSH (labRobots/panels/gamepadManager.py)
#! Gamepad topic of the robot (sensor_msgs/Joy) and the labels of the axes and buttons of a PS4 controller (joy_node).
robotJoyTopic, robotJoyMessageType = '/summit_xl/joy', 'sensor_msgs/Joy'
joyAxisLabels = ['Links x', 'Links y', 'L2', 'Rechts x', 'Rechts y', 'R2', 'Kreuz x', 'Kreuz y']
joyButtonLabels = ['X', 'Kreis', 'Dreieck', 'Rechteck', 'L1', 'R1', 'L2', 'R2', 'Share', 'Options', 'PS', 'L3', 'R3']
#! SSH user of the robot PC (the key of the lab PC is installed: startLab.ps1), Bluetooth adapter for bluetoothctl (None:
#! the default adapter of the robot PC: CSR dongle 00:1A:7D:DA:71:03) and the address of the controller of the Kairos
#! (F4:93:9F:AB:9B:D3; the Leobot keeps 1C:A0:B8:FB:66:06, a PS4 controller pairs with one PC).
robotUser, robotBluetoothAdapter, robotGamepadAddress = kairosNetwork['user'], None, 'F4:93:9F:AB:9B:D3'
#live data panel (LabCourse/MobileRobot/labPanel/dataPanel.py); runLab.py --debug records
#through it when it runs. Ports 8050 (MAYA board) and 8444 (VR viewer) are taken.
#! HTTP port and URL of the live data panel on this computer.
dataPanelPort = kairosNetwork['panelPort']
dataPanelUrl = 'http://127.0.0.1:%d' % dataPanelPort

#Laser scanners of the Kairos
#! Scan topics of the two S300 safety scanners.
safetyScannerTopics = ('/summit_xl/front_laser/scan', '/summit_xl/rear_laser/scan')
#! Offsets in m of the laser links along the sensor z axis above the sensor base links (robotnik_sensors).
s300LaserLinkOffset, vlp16LaserLinkOffset = 0.116, 0.035
#! S300 mounting rotations (URDF roll, pitch, yaw in rad) and mounting points in base_link in m (robot description).
s300Rotations = (np.array([0., -np.pi, 0.75*np.pi]), np.array([0., -np.pi, -0.25*np.pi]))
s300MountingPoints = (np.array([0.2865, -0.20894, 0.2973]), np.array([-0.2865, 0.20894, 0.2973]))
#! VLP-16 laser link in base_link (no rotation) and the topics of robotSide/velodyneRelay.py (labRobots/robots/kairos): 2D scan of a height band
#! (angle 0 along the platform x axis) and voxel cloud (x, y, z in mm in base_link, flat Int16MultiArray).
velodynePosition = np.array([-0.34785, 0.304931, 0.78542 + vlp16LaserLinkOffset])
velodyneScanTopic, velodyneCloudTopic = '/summit_xl/velodyne_relay/scan', '/summit_xl/velodyne_relay/cloud'
#! The scan frames in base_link (laserMountings) are computed from these values in lasers.py.
#! Message type of the 2D scans.
laserScanMessageType = 'sensor_msgs/LaserScan'
#! Point cloud type (VLP-16): only listed, never subscribed by default (several MB/s as JSON).
pointCloudMessageType = 'sensor_msgs/PointCloud2'
