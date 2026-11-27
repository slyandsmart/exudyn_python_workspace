# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: parameters.py
#
# Details:  Parameters of the Exudyn model of the Kairos (model.py, platform.py, contact.py, collisionModel.py,
#           visualization.py, kinematics.py): the device numbers of geometry.py and topics.py the model uses (as par.X),
#           the floating base, wheel contact and motors, arm control, planning and collision checking, manual arm,
#           gripper motion and the colors of the robot graphics.
#
# Authors:  Peter Manzl; Martin Sereinig; Michael Pieber (created with Claude Code)
# Date:     2026-11-04; 2026-11-12 moved from LabCourse/MobileRobot/fahrzeug/parameters.py (the model is equipment)
# Notes:    SI units. The values of the lab task (table, cup, Vicon, tolerances of the task, simulation, video) stay
#           in fahrzeug/parameters.py, which imports these.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Parameters of the model of the Kairos."""
import numpy as np
from labRobots.labSetup.dataPaths import (  # noqa: F401
    kairosDescriptionFolder)
from labRobots.robots.kairos.geometry import (  # noqa: F401
    armBaseFootprintPosition, armBaseHousingRadius, armCOM, armDH, armInertias, armJointHousingRadii,
    armJointLowerLimits, armJointUpperLimits, armLinkNames, armLinkShapeCounts, armMasses, baseLinkHeight,
    chassisHousingHeight, chassisSideColor, chassisTopColor, chassisTopHeight, chassisTopPlateTolerance, controllerWheelBase,
    controllerWheelRadius, controllerWheelTrack, footprintOffset, frontCameraBaseline, frontCameraColor, frontCameraLensColor,
    frontCameraLensRadius, frontCameraPosition, frontCameraSize, gripperCarriageLimits, gripperFingerFaceOffset, kairosParts,
    lidarHousingColor, lidarRadius, lidarWindowColor, lidarWindowHeights, mountPosition, mountYaw,
    moveItDisabledPairs, numberOfArmJoints, numberOfGripperFingers, numberOfWheels, platformCOM, platformCenterHeight,
    platformDimensions, platformMass, toolBoxSize, toolCOM, toolLength, toolMass,
    ur5CapColor, ur5HousingColor, ur5TubeColor, wheelBase, wheelRadius, wheelTrack,
    wheelWidth, wsg50CenterOffset)
from labRobots.robots.kairos.topics import (  # noqa: F401
    maximumDriveSpeed, maximumTurnRate)

#! Meshes of the Kairos (URDF meshes of the real robot description, UR5 visual meshes).
kairosMeshDirectory = kairosDescriptionFolder / 'meshes'
ur5VisualMeshFile = kairosDescriptionFolder / 'ur5_visual_assets' / 'ur5_visual_meshes.npz'
#! Number of spatial dimensions.
dimension3D = 3
#! Floating platform: world x/y/z translation followed by yaw/pitch/roll.
numberOfBaseCoordinates = 6
#! Floating-base rotation coordinate indices (ZYX order).
baseYawCoordinate, basePitchCoordinate, baseRollCoordinate = 3, 4, 5
wheelDensity = 1600.
#! Rolling disc contact and motor parameters of the preparation model.
contactStiffness = 1e6
contactDamping = 50.*np.sqrt(1e5)
#10 times stiffer (was 50, 2.5): the soft motors let the vehicle sway on them for about 2 s
#with 0.8 s period after a stop (up to 0.14 m/s), the real wheels follow their speeds closely (identified drive
#response = wheel motion); with the drive response, drive actions now end within 2 mm instead of 16 mm.
#! Wheel motor (PD on the wheel angle) stiffness in Nm/rad and damping in Nms/rad.
wheelStiffness, wheelDamping = 500., 25.
dryFriction = [1., 0.]
#contact study (long drive 20261107-152443): the
#viscous friction along the free roller direction (was 0.1) let the model slip sideways (1.18 m sideways 7.5 cm short);
#0.02: end errors sideways 0.7/1.3 cm, diagonal 4.6/0.7 cm (RMS 2.4 cm instead of 4.7 cm), forward unchanged.
#! Viscous friction of the wheel contact: across the rollers (traction) and along the free roller direction.
wheelViscousFriction = [0.1, 0.02]
gravity = [0., 0., -9.81]
armProportionalGains = [4000., 4000., 4000., 100., 100., 10.]
armDerivativeGains = [40., 40., 100., 1., 1., 0.1]
#reference configuration of the collision model (SRDF home, all joints 0): pairs touching
#there are not checked (MoveIt default collisions)
collisionReferenceAngles = np.zeros(numberOfArmJoints)
armMaximumSpeed, armMaximumAcceleration = np.pi*0.55, 3.
#park pose of the real Kairos as joint target (photo Kairos2.jpg: upper arm 40 deg up to the
#rear, forearm 42 deg up to the front, gripper horizontal pointing forward with horizontal jaws, joint 2 axis to the
#right in driving direction). Estimated from the photo (about +-5..10 deg); the grasp poses are reached without
#wrist flip. Replace by the values read on the robot (rostopic echo -n1 /summit_xl/joint_states).
#! Arm park joint angles in rad (ur5.move_joint_pose on the robot).
armHomeAngles = np.radians([45., -40., -98., 138., -90., 0.])
#the real UR5 starts where it was parked (absolute encoders, no homing);
#the MoveIt simulation starts at 0 only because joint_state_publisher has no robot data.
#! Arm joint angles in rad at the simulation start.
armInitialAngles = armHomeAngles.copy()
#sphere/triangle object contact, separate from the wheel contact tuning.
#! Cylinder density in kg/m^3, contact stiffness in N/m and damping in Ns/m (per contact primitive).
#tuned for the 50 g cylinder and the 2 ms implicit step;
#overly stiff/damped contact caused repeated step reductions even while resting on the table.
objectContactStiffness, objectContactDamping = 2e3, 2.
#! Contact stiffness in N/m of the boxes of gripper flange and fingers (as the Leobot): with 2e3 the
#! fingers pressed a held cup so weakly that the regularized friction let it creep out downwards by 2.3 mm/s (it fell
#! after 20 s of driving); with 2e4 about 0.25 mm/s.
gripperContactStiffness = 2e4
#collision checking and RRT-Connect path planning of the arm (labRobots/robotics).
#! Clearance in m kept between arm shapes and obstacles/other links.
collisionSafetyDistance = 0.01
#! Planner: tree step and validation resolution in rad, iterations, shortcut iterations, random seed.
plannerMaximumStep = 0.3
#the collision checks of Exudyn (IsEdgeFree, PlanJointPath, TrajectoryFromPath) at joint steps
#of at most collisionEdgeStep: a joint step moves a point of the arm at most by its distance from the joint axes, summed
#over the joints about 3.2 m for the UR5 with gripper, so 0.003 rad move it at most about 10 mm (the safety distance)
#! Largest joint step in rad between two collision checks.
collisionEdgeStep = 0.003
#! Platform motion (change of the arm base pose matrix) after which the scene obstacles are transformed again.
obstaclePoseTolerance = 1e-6
plannerMaximumIterations, plannerShortcutIterations, plannerSeed = 5000, 100, 0
#! Largest tracking error in rad for which planning starts from the commanded instead of the measured angles.
plannerStartTolerance = 0.05
#! Weights of the joint motion when choosing among inverse kinematics solutions (base, shoulder, elbow, wrists):
#! moving the large links counts more.
armJointWeights = np.array([2., 2., 1.5, 1., 1., 1.])
#! Floor obstacle: half length and thickness in m.
floorHalfLength, floorThickness = 5., 0.1
#! Spacing in m of the surface samples to which the link shapes are fitted.
collisionSampleSpacing = 0.02
#WSG50 gap between the finger inner faces, as commanded on the real gripper; closed on the
#object diameter. Prescribed opening with contact. open 100 mm = robotGripperOpenWidth (the
#real gripper opens to 100 mm, log 2026-11-06: 109.5 mm; 80 mm put the collision shapes of the fingers 10 mm per
#side too far inside).
#! Finger gap open/grasp in m and motion duration in s.
gripperOpenWidth, gripperMotionDuration = 0.100, 0.5
#! WSG50 meshes in mm are scaled to m.
millimeterScale = 1e-3
#! Normal z above which a chassis face counts as top face; lidar window height range and housing radius in m.
upwardNormalThreshold = 0.7
ur5BlueThreshold, ur5BlackThreshold, ur5TubeThreshold = 0.1, 0.08, 0.4
accentColor = [0.08, 0.33, 0.52, 1.]
rubberColor = [0.17, 0.18, 0.20, 1.]   #lighter, shading stays visible
metalColor = [0.48, 0.53, 0.59, 1.]
armPositionTolerance, armVelocityTolerance = 0.03, 0.06
#! Static gravity compensation update period in s, as in the 6R model.
gravityCompensationPeriod = 0.01
armHoldDuration = 0.1
#manual arm (VR teach mode): TCP handle and stick jogging with collision checking.
#! Stick jogging: TCP speed at full deflection in m/s; update period of the manual arm steps in s (one differential
#! inverse kinematics step and one collision check each; joint speed limited by armMaximumSpeed).
jogTranslationSpeed, jogUpdatePeriod = 0.25, 0.02
#! Differential inverse kinematics: orientation weight in m, damping, remaining TCP error (m, rad)
#! that counts as reached when the arm cannot move further.
manualRotationWeight, manualDamping, manualTargetTolerance = 0.2, 0.05, 0.005
#! Joint step in rad below which the arm counts as standing (target reached or not reachable).
manualMinimumJointStep = 1e-5
#! Coordinate system arrows in meters, and axis colors.
frameLength, frameRadius = 0.15, 0.0025
axisNames = ('x', 'y', 'z')
axisColors = ([1., 0., 0., 1.], [0., 1., 0., 1.], [0., 0., 1., 1.])
