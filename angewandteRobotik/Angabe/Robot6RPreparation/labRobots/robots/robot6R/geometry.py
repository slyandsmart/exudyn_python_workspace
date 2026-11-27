# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: geometry.py
#
# Details:  Device numbers of the 6R robot of the lab: standard DH parameters, TCP, joint limits and motion
#           parameters of the lab control software, module directions, offsets and currents on the CAN bus, Schunk
#           PW 90 / PW 70 drive data and masses, gripper commands, servo mapping, geometry and masses of gripper and
#           jaws. Moved from LabCourse/Robot6R/robot6R/parameters.py (2026-11-12; sources [RU], [MAT],
#           [PW], [CAD], [EST] as listed there); meshes and mass data: mektcommon2/data/Exudyn/robotics/URDF/
#           robot6R_description/exudyn.
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
"""Device numbers of the 6R robot."""
import numpy as np

#kinematics: standard Denavit-Hartenberg parameters [RU]
#joint i: T_i = RotZ(q_i + jointAngleOffsets[i]) * TransZ(linkOffsetsD[i])
#               * TransX(linkLengthsA[i]) * RotX(linkTwistsAlpha[i])
numberOfJoints = 6
linkOffsetsD = [0.2245, 0., 0., 0.37, 0., 0.17]                        #d_i in m
linkLengthsA = [0., 0.301, 0., 0., 0., 0.]                             #a_i in m
linkTwistsAlpha = [-0.5*np.pi, 0., -0.5*np.pi, 0.5*np.pi, -0.5*np.pi, 0.] #alpha_i in rad
jointAngleOffsets = [0., -0.5*np.pi, -0.5*np.pi, 0., 0., 0.]           #theta offset in rad
toolLength = linkOffsetsD[5]        #distance wrist center -> TCP (included in d_6)
wristToFlangeDistance = 0.065       #wrist center -> wrist flange [CAD], Ableitung_roboter
#lab control software (Matlab, Positionieren_T.m, check_input.m) [MAT]
#joint limits checked before every PTP motion (degree); q5+q6 is limited by the wrist servos
labJointLowerLimitsDeg = [-35., -106., -121., -190., -90., -np.inf]
labJointUpperLimitsDeg = [121., 106., 121., 190., 90., np.inf]
labWristServoSumLimitDeg = 180.     #check: q5 + q6 <= 180 deg
#100% motion parameters per axis (Modul_move_param.mat) = 10% of Schunk maxima for axes 1-4
labMaxVelocitiesDeg = [24., 15., 24., 36., 321., 321.]          #deg/s
labMaxAccelerationsDeg = [96., 60., 96., 144., 2295., 2295.]    #deg/s^2
labMotionTimeFactor = 1.05          #T = 1.05*t_lead ("Sicherheitspuffer" in Positionieren_T.m)
labMinimumSpeedPercent = 5.         #speed is clamped to [5, 100] percent
labMaximumSpeedPercent = 100.
#module coordinates sent on the CAN bus (Positionieren_T.m): moduleAngle = R_dir*q + offset (degree)
#added 2026-11-04 by Michael Pieber (Claude Code) for the real robot measurements (robot6R.lab package)
labModuleDirections = [1., 1., -1., 1., -1., 1.]                   #R_dir in Modul_move_param.mat
labModuleOffsetsDeg = [-90., 0., 0., -92.43, -3.1563, -4.8313]      #offset in Modul_move_param.mat
labModuleCurrentsAmpere = [4., 4.4, 4., 2.1]                       #current limit sent with every motion (axes 1-4)
#Schunk PW 90 (axes 1, 2) and PW 70 (axes 3, 4) drive data [PW]
#robot axis 1 = PW 90 rotation, axis 2 = PW 90 tilt, axis 3 = PW 70 tilt, axis 4 = PW 70 rotation
schunkNominalTorques = [12., 23., 12., 2.]          #Nm
schunkPeakTorques = [24., 46., 24., 4.]             #Nm
schunkMaxVelocitiesDeg = [240., 150., 240., 360.]   #deg/s
schunkMaxAccelerationsDeg = [960., 600., 960., 1440.] #deg/s^2
massPW90 = 3.4                      #kg, module without connection cap
massPW70 = 1.8                      #kg, module without connection cap
#gripper (BahnfahrtPythonInterface.m, Positionieren_Greifer.m) [MAT]
gripperCommandNone = -1             #code in column 8 of Q: no gripper action
gripperCommandOpen = 0              #open gripper (0 %)
gripperCommandClose = 1             #close gripper (70 %)
gripperCommandCloseStrong = 2       #close gripper (75 %)
gripperCommandPercent = {gripperCommandOpen: 0., gripperCommandClose: 70., gripperCommandCloseStrong: 75.}
gripperServoAngleOffsetDeg = -50.   #servo angle at 0 % (Deg_Greifer = -50 + Pos*0.71)
gripperServoAnglePerPercentDeg = 0.71
gripperLinkLength = 0.020           #parallelogram link length (Ableitung_Greifer)
flangeToTCPDistance = toolLength - wristToFlangeDistance   #0.105 m
#gripper geometry in TCP frame (frame 6; z6 = approach direction, y6 = closing direction) [EST from Ableitung_Greifer]
gripperAdapterRadius = 0.030        #round adapter plate at the wrist flange
gripperAdapterThickness = 0.006
gripperBodySize = [0.036, 0.0775, 0.030]   #x6, y6, z6 extent of the servo housing
gripperJawFlangeSize = [0.021, 0.036, 0.010] #jaw flange (Flansch Greiferbacken); jaws: see robotModelData.json
gripperMass = 0.11                  #kg: housing and 2 Futaba S3777SV servos, without jaws [EST]
gripperCOM = [0., 0., -0.085]       #m, in frame 6 [EST]
#gripper jaws: prismatic links of the kinematic tree, position controlled by the gripper servos [EST]
jawNames = ['jawRight', 'jawLeft']  #link 7 (at -y6) and link 8 (at +y6)
jawMass = 0.025                     #kg per jaw incl. flange and linkage share
jawSize = [0.02, 0.015, 0.075]      #m, cuboid for the jaw inertia
jawCOM = [0., 0.033, -0.02]         #m, in the jaw link frame
connectionCapMass = 0.25            #kg per Schunk connection cap (ASK) incl. cable gland [EST]
wristServoMass = 0.068              #kg, Hitec HSB-9380TH (datasheet)
#gripper timing and contact of the lab software [MAT]
#the cup (radius 27 mm at grasp height) is larger than the jaw arc -> grip force
gripperContactPercent = 70.         #closing command at which the jaw arc center lies on the TCP (jaw coordinate 0)
gripperActionTime = 1.5             #s, dwell for gripper motion (pause(1.5) in virtual mode) [MAT]
