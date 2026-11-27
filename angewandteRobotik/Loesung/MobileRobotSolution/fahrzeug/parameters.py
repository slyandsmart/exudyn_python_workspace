# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: parameters.py
# Details:  Geometry, dynamics, control limits and task tolerances in SI units.
# Authors:  Peter Manzl; Martin Sereinig; Michael Pieber (created with Claude Code)
# Date:     2026-11-04 (MoveIt geometry: 2026-11-05): robot values in labRobots/robots/kairos/parameters.py
# Notes:    Physical values retained from ExudynModelV3 and MobileRobot_V2.
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Shared model and control parameters."""
from pathlib import Path
import numpy as np
from labRobots.robots.kairos.topics import (viconRobotTopic, viconRobotMessageType, viconMarkersTopic,
                                     viconMarkersMessageType, viconRobotRejectionRadius, viconRobotSubject,
                                     viconDataStreamHost, viconDataStreamPort, robotJoyTopic, robotJoyMessageType,
                                     joyAxisLabels, joyButtonLabels, robotUser, robotBluetoothAdapter,
                                     robotGamepadAddress, dataPanelPort, dataPanelUrl)
from labRobots.robots.kairos.topics import (rosbridgePort, robotHost, robotStateTopic, robotStateMessageType,
                                     robotStateThrottleRate, robotStateConnectTimeout, robotStateDataTimeout,
                                     robotStateStaleTimeout, robotWaitSleepTime, robotGripperService,
                                     robotGripperServiceType, robotGripperCloseWidth, robotGripperOpenWidth,
                                     robotGripperSpeed, robotGripperTimeout, robotGripperHomingService,
                                     robotGripperHomingServiceType, robotGripperHomingMinimumTime,
                                     robotGripperHomingTimeout, robotGripperStatusTopic, robotGripperStatusType,
                                     robotGripperWidthTolerance, robotGripperBlockTime, robotGripperMinimumGraspForce,
                                     robotDriveTopic, robotDriveMessageType, robotDriveCommandPeriod,
                                     robotDriveWatchdogTime, robotDriveMonitorTopics, robotBaseCommandTimeout,
                                     robotArmActionServer, robotArmActionType, robotArmSamplePeriod,
                                     robotArmStartTolerance, robotArmStationarySpeed, robotArmArrivalTolerance,
                                     robotArmFinishTimeout, robotDriveStartCompensation, maximumDriveSpeed, maximumTurnRate)
from labRobots.labSetup.dataPaths import kairosDescriptionFolder
from labRobots.robots.kairos.geometry import (numberOfWheels, numberOfArmJoints, platformDimensions, platformMass,
                                       platformCenterHeight, platformCOM, wheelRadius, wheelWidth, wheelBase,
                                       wheelTrack, controllerWheelRadius, controllerWheelBase, controllerWheelTrack,
                                       footprintOffset, baseLinkHeight, armBaseFootprintPosition, mountPosition,
                                       mountYaw, kairosParts, frontCameraSize, frontCameraBaseline,
                                       frontCameraLensRadius, frontCameraImageSize, frontCameraFocalLength,
                                       frontCameraColor, frontCameraLensColor, chassisHousingHeight, chassisTopHeight,
                                       chassisTopPlateTolerance, frontCameraPosition, armDH, armMasses, armCOM,
                                       armInertias, armJointLowerLimits, armJointUpperLimits, armTorqueLimits,
                                       armJointNames, armLinkNames, armLinkShapeCounts, armJointHousingRadii,
                                       armBaseHousingRadius, moveItDisabledPairs, toolMass, toolCOM, toolBoxSize,
                                       wsg50CenterOffset, labGripperOffset, toolLength, gripperFingerFaceOffset,
                                       gripperClosedWidth, numberOfGripperFingers, gripperCarriageLimits,
                                       chassisTopColor, chassisSideColor, lidarHousingColor, lidarWindowColor,
                                       lidarWindowHeights, lidarRadius, ur5CapColor, ur5HousingColor, ur5TubeColor)
#the parameters of the robot model are in labRobots/robots/kairos/parameters.py
from labRobots.robots.kairos.parameters import (  # noqa: F401
    accentColor, armDerivativeGains, armHoldDuration, armHomeAngles, armInitialAngles, armJointWeights,
    armMaximumAcceleration, armMaximumSpeed, armPositionTolerance, armProportionalGains, armVelocityTolerance, axisColors,
    axisNames, basePitchCoordinate, baseRollCoordinate, baseYawCoordinate, collisionEdgeStep, collisionReferenceAngles,
    collisionSafetyDistance, collisionSampleSpacing, contactDamping, contactStiffness, dimension3D, dryFriction,
    floorHalfLength, floorThickness, frameLength, frameRadius, gravity, gravityCompensationPeriod,
    gripperMotionDuration, gripperOpenWidth, jogTranslationSpeed, jogUpdatePeriod, kairosMeshDirectory, manualDamping,
    manualMinimumJointStep, manualRotationWeight, manualTargetTolerance, metalColor, millimeterScale, numberOfBaseCoordinates,
    objectContactDamping, objectContactStiffness, obstaclePoseTolerance, plannerMaximumIterations, plannerMaximumStep, plannerSeed,
    plannerShortcutIterations, plannerStartTolerance, rubberColor, upwardNormalThreshold, ur5BlackThreshold, ur5BlueThreshold,
    ur5TubeThreshold, ur5VisualMeshFile, wheelDamping, wheelDensity, wheelStiffness, wheelViscousFriction)

#! Folder of the lab task (LabCourse/MobileRobot); independent of the current working directory.
courseDirectory = Path(__file__).resolve().parents[1]
#! Target/table geometry in meters, and compliant grasp parameters.
#the real grasp object (lab): a cup 80 mm high, 55 mm diameter, widening over the top 10 mm
#to 60 mm; the Vicon marker is a sphere of 30 mm diameter in its opening (center at the rim height). Contact, inertia
#and grasp width use the 55 mm body.
#table height from Vicon: cup marker at 0.7165 m (runs 20261107-182633/-183943) minus the cup
#height 0.08 m (sphere center at the rim) gives 0.636 m (was 0.645 m: the grasp and the arm clearance were 9 mm off).
objectHeight, objectRadius, tableHeight = 0.08, 0.0275, 0.636
#! Default object center [x, y, z] in m (standing on the table).
objectPosition = [1.5, 0., tableHeight + objectHeight/2.]
#! Radius in m at the top rim and height in m of the widening below the rim; radius in m of the Vicon sphere.
objectRimRadius, objectRimHeight, objectMarkerSphereRadius = 0.03, 0.01, 0.015
#! Wall thickness and depth of the opening in m (graphics only).
objectWallThickness, objectOpeningDepth = 0.003, 0.012
#! RGBA color of the grasp object (silver, as the real cup).
objectColor = [0.75, 0.75, 0.78, 1.]
#lab table (photos): the vehicle faces the long closed side wall; two walls perpendicular to
#x, open in y; walls and plates 20 mm, middle shelf 0.28-0.30 m above the floor, top plate below tableHeight
#(fahrzeug/table.py).
#! Outer footprint in m: x (depth across the walls, approach direction), y (wall length).
tableSize = [0.3, 0.6]
#! Wall and plate thickness in m, height in m of the shelf top above the floor.
tableWallThickness, tablePlateThickness, tableShelfHeight = 0.02, 0.02, 0.30
#! Cartesian approach distance of the gripper to the object in m (the distance of the
#! approach pose of the platform from the object is part of the preparation, fahrzeug.reference.taskSequence).
objectApproachDistance = 0.1
objectRetreatOffset = [-0.15, 0., 0.15]
#! Clearance above the table before releasing the cylinder in m.
objectPlacementClearance = 0.003
#! Conservative horizontal radius of the parked vehicle/arm and drive clearance in m.
tableDriveRadius, tableDriveMargin = 0.55, 0.05
objectDensity = 500.
#! Inscribed sphere radius in m; five rings cover both ends and the cylindrical wall.
objectContactSphereRadius = 0.008
objectContactNumberOfRings, objectContactSpheresPerRing = 5, 8
#! Dry friction coefficient and regularization velocity in m/s.
objectContactFriction, objectContactFrictionZone = 0.5, 0.01
#! Broad-phase grid; bounds follow the geometry automatically, including a displaced table.
objectContactSearchCells = [16, 16, 8]
#! Half edge length in m of the drawn floor plate (also the walls of the simulated lab room of the laser sensors).
floorDrawHalfLength = 3.
graspDistance = 0.06
#! PosObj of the real run is a single Vicon marker on top of the object: height above the object center in m
#! (center of the Vicon sphere at the rim).
viconObjectMarkerHeight = objectHeight/2.
#the origin of the Vicon object kairos was set in Tracker to the center of the top plate surface,
#x forward along the chassis, z up (as the Leobot); the former yaw offset of 0.69 deg (object against chassis) is gone.
#Recordings before 2026-11-18 have the object yaw 0.69 deg above the chassis yaw.
#the renderer warned 'timeout for user function drawing' (collision shapes about 20 ms, first
#call 140 ms building the checker; sensor dashes 17 ms per frame).
#! Wall-clock period in s of recomputing the overlays of graphics user functions (collision shapes, sensor dashes).
overlayRefreshPeriod = 0.1
#! Minimum time between two received messages per topic in ms (rosbridge throttle_rate, 50 Hz).
viconThrottleRate = 20
#! Connection/first data timeout in s; object data older than viconDataTimeout in s count as lost.
viconConnectTimeout, viconDataTimeout = 10., 0.5
#drive measurements run without table and object (no object marker in Vicon)
#! Waiting time in s for the object marker after the vehicle pose; table and object position x, y in m without marker
#! (far outside the lab, nothing to collide with).
viconObjectWaitTime, viconMissingObjectPosition = 2., [30., 30.]
#! Live object: spring-damper between object center and the Vicon target, in N/m and Ns/m; roll/pitch keep the
#! cylinder upright, the yaw of the single marker is unknown and free.
objectTrackingStiffness, objectTrackingDamping = 1000., 10.
objectTrackingRotationalStiffness = objectTrackingStiffness*objectRadius**2
objectTrackingRotationalDamping = objectTrackingDamping*objectRadius**2
#! Largest speed in m/s of the tracking target (smooth start after release, no jumps at marker glitches).
objectTrackingMaximumSpeed = 0.5
#! Raw Vicon object marker in the viewer: radius in m and color.
viconMarkerRadius, viconMarkerColor = 0.012, '#d62dd6'
viconRobotMarkerRadius, viconRobotMarkerColor = 0.008, '#22d3ee'
#mirror mode (button Spiegeln): the vehicle follows the Vicon pose, the arm the real angles
#! Period in s of taking over the real pose and arm angles in the mirror mode (Vicon at 50 Hz).
mirrorUpdatePeriod = 0.02
#! The table follows the object marker while the marker is within this height difference in m of its start height
#! (object standing on the table; a lifted object leaves the table in place).
mirrorTableHeightTolerance = 0.05
#! Text file of the drive diagnostics (sent and received commands with wall-clock time).
robotDriveLogFileName = courseDirectory / 'solution' / 'robotDriveLog.txt'
#! Text file of all commands to the robot and their results (unix time in s, text), written during the run.
robotCommandLogFileName = courseDirectory / 'solution' / 'robotCommandLog.txt'
#! Command log of robotGripper.py (separate, so it does not overwrite the log of the last simulation run).
robotGripperLogFileName = courseDirectory / 'solution' / 'robotGripperLog.txt'
#! Largest deviation of the Vicon pose from the simulated pose after a replayed drive: position in m, yaw in rad
#! (plausibility only: larger means a Vicon or robot fault; smaller deviations are corrected by drive corrections).
poseUpdateMaximumDistance, poseUpdateMaximumAngle = 0.5, np.radians(20.)
#measurement drive: the real base rolls on for about 0.3 s after the last command; the pose was
#taken while it still moved (distances seemed 25 % short). The pose update waits until Vicon shows the base at rest:
#! moved less than viconStandstillDistance in m and viconStandstillAngle in rad within viconStandstillWindow in s, at
#! most viconStandstillTimeout in s (then the pose is taken anyway and the waiting is logged).
viconStandstillDistance, viconStandstillAngle = 0.002, np.radians(0.2)
viconStandstillWindow, viconStandstillTimeout = 0.2, 2.
#! Largest number of correction drives per drive action.
#only with the setting 'driveCorrectionsEnabled' (runLab.py --correctDrives); by default
#none: the Vicon pose is taken over and the arm (world frame targets) compensates the remaining error.
maximumDriveCorrections = 3
#! Remaining error after a replayed drive that triggers a correction: position in m, yaw in rad (coarser than the
#! simulated drive tolerance: the real base hardly executes very small corrections, each costs about 5 s).
driveCorrectionDistance, driveCorrectionAngle = 0.03, np.radians(2.)
#! Grasp point below the Vicon marker in m (on the 55 mm body below the widening; gripper body clear of the table).
graspDistanceBelowMarker = 0.02
#! Grasp point above the object center in m.
graspHeightAboveCenter = viconObjectMarkerHeight - graspDistanceBelowMarker
graspStiffness, graspDamping = 1000., 100.
#! Rotational grasp gains in Nm/rad and Nms/rad, scaled by the contact lever arm.
graspRotationalStiffness = graspStiffness*objectRadius**2
graspRotationalDamping = graspDamping*objectRadius**2
gripperGraspWidth = 2.*objectRadius
#! Muted materials keep the platform, arm, rubber wheels and floor visually distinct.
#light floor (the dark floor hid the black chassis sides)
floorColor = [0.70, 0.71, 0.72, 1.]
floorGridColor = [0.60, 0.62, 0.64, 1.]
#! World pose controller and completion conditions.
driveGain, turnGain = 1.2, 1.5
positionTolerance, angleTolerance = 0.015, 0.02
#require actual settling before a drive action releases its pose feedback.
linearVelocityTolerance, angularVelocityTolerance = 0.005, 0.01
actionTimeout = 60.
#! Longest duration in s of a velocity action (steps of the identification program).
velocityActionMaximumDuration = 5.
#! Simulation/stream timing in seconds and independent vehicle viewer port.
#! Show the floor in the ExuXR browser/VR/AR scene (native renderer and SolutionViewer always show it).
showFloorInViewer = False
stepSize, commandPollPeriod, maximumSimulationTime = 0.002, 0.01, 86400.
#! Explicit contact integration step in s.
explicitStepSize = 0.0002
#with the default 0.9 the wheel contacts could settle after a hard stop into a numerical
#oscillation with a period of 3 steps (contact force +213/-46 N, the friction zone switched every step) that ratcheted the
#standing vehicle sideways and around (several cm and degrees per second); 0.6 damps only these step frequencies.
#! Spectral radius of the implicit GeneralizedAlpha integration (numerical damping of the highest frequencies).
implicitSpectralRadius = 0.6
#! Exudyn workers for desktop and VR, requested by the user.
numberOfThreads = 4
#! Task/drive feedback period in s, independent of the smaller explicit physics step.
controlUpdatePeriod = 0.002
#solution file for the SolutionViewer (runSimulation.py --solutionViewer).
#! Write period in s (25 frames per second) and default file.
solutionWritePeriod = 0.04
solutionFileName = courseDirectory / 'solution' / 'vehicleSolution.txt'
#demonstration (runLab.py --demo --video).
#! Demo start pose: world x, y in m and yaw in rad, farther away and turned, so that the mecanum platform
#! drives diagonally while turning in front of the object.
demoInitialPose = [-0.9, 1.1, np.radians(50.)]
#! Video: frame period in simulated s (25 frames/s in real time), window size in pixels, frame directory.
videoFramePeriod = 0.04
videoWindowSize = [1280, 720]
#! H.264 quality of the demo video (ffmpeg crf: 18 very good, 23 default, 28 small file).
videoQuality = 20
#! Video only (CPU shadows slow down the renderer about 20 times): shadow intensity of light0, its direction
#! (directional light from the upper front left, 4th value 0) and diffuse share, multisampling of edges.
videoShadow = 0.3
videoLightPosition = [1., -1.5, 2.5, 0.]
videoLightDiffuse = 0.6
videoMultiSampling = 2
videoDirectory = courseDirectory / 'solution' / 'video'
#! Renderer view: zoom (half view height in m) per m of the distance between start pose and object, minimum zoom.
viewZoomPerDistance, minimumViewZoom = 0.75, 1.2
#! Oblique renderer view: elevation and azimuth of the viewing direction in rad (camera south-west of the scene).
viewElevation, viewAzimuth = np.radians(30.), np.radians(-25.)
#! Grabbable TCP handle in the viewer: radius in m and color.
tcpHandleRadius, tcpHandleColor = 0.035, [0.95, 0.45, 0.1, 1.]
#! Simulated time in s after the task end before an offline run (runSimulation.py) stops.
finishedSettlingTime = 1.
defaultPort = 8444
