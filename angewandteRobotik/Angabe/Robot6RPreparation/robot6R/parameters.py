# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: parameters.py
#
# Details:  Central parameter set of the 6R robot of the applied robotics lab
#           (PR Angewandte Robotik, Teil I): kinematics (DH parameters), joint
#           limits and motion parameters of the lab control software, Schunk
#           PW 70 / PW 90 drive data, gripper, lab layout (tables, cup), mass
#           model and CAD conversion settings. Every value states its source.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-03
# Notes:    All quantities in SI units (m, kg, s, rad) unless the name ends
#           with Deg, Percent or Millimeter. Sources:
#           [RU]  RobotUtils.py (getRobot) of the former lab software of the robot (thesis Manzl 2018)
#           [MAT] Matlab control code of the former lab software (check_input.m, Positionieren_T.m,
#                 Positionieren_Greifer.m, BahnfahrtPythonInterface.m, Modul_move_param.mat)
#           [PW]  Schunk PW 70-90 manual IM0007122 / thesis Manzl 2018, table A.1
#           [LAB] task sheet of the preparation, figure of the lab layout (Abb. Labor_Aufbau)
#           [CAD] CAD assembly of the robot (aufbau_6r_gesamt_asm.stp, bachelor thesis on the 6R robot)
#           [EST] engineering estimate, not measured; adapt if better data is available
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
from labRobots.robots.robot6R.geometry import (gripperContactPercent, gripperActionTime)
from labRobots.robots.robot6R.geometry import (numberOfJoints, linkOffsetsD, linkLengthsA, linkTwistsAlpha, jointAngleOffsets,
                                        toolLength, wristToFlangeDistance, flangeToTCPDistance, labJointLowerLimitsDeg,
                                        labJointUpperLimitsDeg, labWristServoSumLimitDeg, labMaxVelocitiesDeg,
                                        labMaxAccelerationsDeg, labMotionTimeFactor, labMinimumSpeedPercent,
                                        labMaximumSpeedPercent, labModuleDirections, labModuleOffsetsDeg,
                                        labModuleCurrentsAmpere, schunkNominalTorques, schunkPeakTorques,
                                        schunkMaxVelocitiesDeg, schunkMaxAccelerationsDeg, massPW90, massPW70,
                                        gripperCommandNone, gripperCommandOpen, gripperCommandClose,
                                        gripperCommandCloseStrong, gripperCommandPercent, gripperServoAngleOffsetDeg,
                                        gripperServoAnglePerPercentDeg, gripperLinkLength, gripperAdapterRadius,
                                        gripperAdapterThickness, gripperBodySize, gripperJawFlangeSize, gripperMass,
                                        gripperCOM, jawNames, jawMass, jawSize, jawCOM, connectionCapMass,
                                        wristServoMass)
#the parameters of the robot model (gravity, joint and jaw control, initial gripper opening,
#workspace check of the lab software) are in labRobots/robots/robot6R/parameters.py
from labRobots.robots.robot6R.parameters import (gravity, controlBandwidthHz, controlDampingRatio, gripperInitialPercent,
                                          jawStiffness, jawDampingRatio, labMinimumX, labMinimumY, labMinimumZ,
                                          labWarningZ)

degToRad = np.pi/180.   #conversion factor degree -> radian
radToDeg = 180./np.pi   #conversion factor radian -> degree

#the device numbers (DH parameters, joint limits and motion parameters of the lab
#control software, CAN module data, Schunk drive data, gripper and jaw geometry) are in labRobots/robots/robot6R/geometry.py;
#the values below are those of the model, the lab layout and the simulation.

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#kinematic model
#joint limits of the robotics toolbox model [RU] (rad)
jointLimitsModel = [[-np.pi, np.pi],
                    [-106.*degToRad, 106.*degToRad],
                    [-121.*degToRad, 121.*degToRad],
                    [-np.pi, np.pi],
                    [-90.*degToRad, 90.*degToRad],
                    [-np.pi, np.pi]]

#validation poses of the preparation task 1.2 [LAB], eq. (Val1)-(Val3)
validationJointAnglesDeg = [[0., 0., 0., 0., 0., 0.],
                            [0., 60., 30., 0., 0., 0.],
                            [31.218, 29.953, 110.499, 38.167, -57.007, -23.171]]
validationPoses = [np.array([[-1.,0.,0.,0.],[0.,-1.,0.,0.],[0.,0.,1.,1.0655],[0.,0.,0.,1.]]),
                   np.array([[0.,0.,1.,0.8007],[0.,-1.,0.,0.],[1.,0.,0.,0.375],[0.,0.,0.,1.]]),
                   np.array([[0.,0.,1.,0.5],[0.,-1.,0.,0.2],[1.,0.,0.,0.2],[0.,0.,0.,1.]])]
#pose T of the preparation task 1.3 (robot configurations) [LAB], given with 4 digits (orthonormalize with
#robot6R.poseUtilities.OrthonormalPose); 8 solutions of the inverse kinematics, 4 of them valid in the lab
#(before lab task 2, labTask2Configurations.py and vr/motionModes.py)
configurationTaskPose = np.array([[0., -0.7071, 0.7071, 0.4406],
                                  [1., 0., 0., 0.],
                                  [0., 0.7071, 0.7071, 0.8307],
                                  [0., 0., 0., 1.]])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#gripper model (simulation) [MAT], [EST]
gripperGraspPercent = 50.           #legacy kinematic grasp threshold
gripperCaptureDistance = 0.012      #m, legacy kinematic grasp distance to cup axis

gripperHoldStiffness = 300.         #N/m, compliant grasp assistance (estimate, not measured)
gripperHoldDamping = 2.             #Ns/m, damping of grasp assistance

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#lab layout in base frame K0 (origin on table top, on axis 1) [LAB]
table1Position = [0.740, 0., 0.]    #center of small table 1 (cup start)
table2Position = [0.530, 0.451, 0.] #center of small table 2 (cup target)
table1Height = 0.116                #Tisch_gross.stl
table2Height = 0.086                #Tisch_klein.stl
tableContactSizes = {'table1': [0.14, 0.14, table1Height], 'table2': [0.08, 0.08, table2Height]}  #m [STL]
tableContactPlateThickness = {'table1': 0.01, 'table2': 0.008}  #m [STL]
tableContactLegWidth = {'table1': 0.01, 'table2': 0.008}      #m [STL], approximate rectangular feet
cupHolderHeight = 0.004             #becherhalter_V3.stl

#paper cup [EST]; the R26 jaw profile and grasp heights (37 mm above cup bottom) fit these values
cupBottomRadius = 0.024
cupTopRadius = 0.030
cupHeight = 0.075
cupMass = 0.01                      #kg, empty paper cup; add liquid mass via simulation settings
cupPlacementTolerance = 0.01        #m, max. height error of the placed cup above the support
cupMaxPlacementTiltDeg = 10.        #deg, max. tilt of the placed cup
cupLiftHeight = 0.005               #m, cup regarded as lifted (grasped) above this height over its start position

#contact (Exudyn GeneralContact, same approach as ExuXRAdvancedExamples/RobotAI_LNF_NEW/pandaRobot.py) [EST]
contactFrictionCoefficient = 0.5    #PLA jaws on paper cup, cup on PLA holder
contactStiffness = 1000.            #N/m per contact sphere
contactDampingFactor = 0.002        #contact damping = factor * stiffness
contactFrictionProportionalZone = 0.01  #m/s, regularized friction below this sliding velocity
contactSphereRadius = 0.003         #m, spheres on cup surface and jaw arcs
cupSpheresPerRing = 16
cupSphereRingHeights = [0.003, 0.02, 0.037, 0.054, 0.071]  #m above the cup bottom
jawSphereAnglesDeg = [-75., -58.6, -42.1, -25.7, -9.3, 7.1, 23.6, 40.]  #along the R26 arc, measured from the closing axis
jawSphereLayers = [-0.006, 0., 0.006]   #m, positions across the jaw thickness (x6)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#mass model of the CAD parts (densities in kg/m^3) [EST], all printed parts are PLA (thesis Manzl 2018)
densityPLA = 1240.
densitySteel = 7850.
densityAluminum = 2700.
densityPOM = 1410.                  #bevel gears of the wrist
densityRubber = 1200.               #toothed belts
densityElectronics = 1850.

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#CAD conversion [CAD]: base frame K0 expressed in CAD coordinates (millimeter)
#K0: origin on table top on axis 1, x0 = +y_CAD (towards table 1), y0 = -x_CAD (towards table 2)
cadLengthUnit = 1e-3                #CAD unit millimeter -> m
cadBaseOriginMillimeter = [-300., -440., 10.]
cadBaseRotation = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])  #columns: x0, y0, z0 in CAD coordinates
#initial guess for the joint angles of the CAD pose; q4 = pi follows from the photo of the zero pose
#(single cover of the forearm faces +x0 for q4 = 0)
cadJointAnglesInitialGuess = [0.33, 0.95, 1.57, np.pi, -0.6, 0.]
#rendering time grows with the number of triangles (approx. 0.35 us per triangle): 120k triangles -> 20 frames/s;
#robot parts are meshed with second order netgen elements (settings of the CAD conversion, removed 2026-11-20);
jawMeshCellSizeMillimeter = 1.0     #cell size for the gripper jaws
labObjectMeshCellSizeMillimeter = 3.    #cell size for small tables and cup holders (simple shapes, fine print models)
cadMaxAxisResidualMillimeter = 6.   #accepted deviation between CAD joint axes and DH model

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#simulation defaults
simulationStepSize = 2e-3           #s, implicit integration of the current kinematic cup model
physicalContactStepSize = 2e-4      #s, explicit integration of the free cup and contact
#four workers for offline and VR simulation, requested by the user.
contactNumberOfThreads = 4
simulationWritePeriod = 0.02        #s, solution and sensor output

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#interactive (VR) simulation, robot6R.vr [EST]; Michael Pieber 2026-11-03
cupDragStiffness = 200.             #N/m, virtual rigid body spring between hand and grabbed cup point (translation)
cupDragDampingRatio = 1.            #damping ratio of the virtual spring with respect to the cup (and liquid) mass/inertia
                                    #rotational stiffness: same natural frequency as the translation (see interactiveSimulation)
teleopUpdatePeriod = 0.01           #s, update period of the differential inverse kinematics of the TCP teleoperation
teleopSpeedFactor = 1.              #joint speed limit of the teleoperation relative to labMaxVelocitiesDeg (100 % lab speed)
teleopDampingFactor = 0.05          #damping of the damped least squares inverse kinematics (singularity robustness)
teleopRotationWeight = 0.2          #m, weight of the orientation error relative to the position error
#joint grab handles (hand guiding of single axes): (frame type, frame index, point in that frame in m) of
#model.ComputeJointAndLinkFrames; 'joint' frames rotate with their joint, 'link' frames are the DH frames after the joint;
#each handle is rigid with the body moved by its joint and has a lever arm to the joint axis
jointHandleDefinitions = [('joint', 0, [0.12, 0., 0.15]),   #axis 1: beside the base column
                          ('link', 1, [0., 0., 0.]),        #axis 2: elbow
                          ('link', 2, [0., 0., 0.22]),      #axis 3: middle of the forearm
                          ('joint', 3, [0.07, 0., 0.30]),   #axis 4: beside the outer forearm
                          ('link', 4, [0., 0., 0.08]),      #axis 5: in front of the wrist center
                          ('joint', 5, [0.06, 0., 0.12])]   #axis 6: beside the flange
jointGrabMinimumLever = 0.03        #m, hand closer to the joint axis: the joint follows the hand rotation about the axis
teleopNullspaceDamping = 0.05       #damping of the grabbed joints in the null space of the held TCP position
gripperOpeningRate = 70./gripperActionTime  #percent/s, interactive gripper command (open <-> close in gripperActionTime)
cupLostHeight = -0.5                #m, cup below this height (fell off the lab table) is reported once
