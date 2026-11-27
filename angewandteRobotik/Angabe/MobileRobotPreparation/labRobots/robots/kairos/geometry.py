# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: geometry.py
#
# Details:  Device numbers of the Kairos (Robotnik Summit-XL Steel, mecanum wheels, UR5 with WSG50-110 gripper,
#           2 SICK S300, Velodyne VLP-16, Bumblebee2 camera): platform and wheels, mounting of arm and sensors, UR5
#           (standard DH, masses, joint limits of the robot, link collision shape layout, ROS joint names), tool
#           (gripper, payload, TCP), colors of the real robot. Moved from LabCourse/MobileRobot/fahrzeug/parameters.py
#           (2026-11-12); the meshes are in mektcommon2/data/Exudyn/robotics/URDF/kairos_description.
#
# Authors:  Peter Manzl; Martin Sereinig; Michael Pieber (created with Claude Code)
# Date:     2026-11-12
# Notes:    Pure data file of the device (numbers, names, tuples; numpy only): the robot as built, the same for
#           every scene, simulation and the real robot. No scene values (tables, objects, start poses, named
#           configurations), no lab values (Vicon, calibrations), no model settings (contact, control, solver).
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Device numbers of the Kairos."""
import numpy as np

#! Number of wheels and arm joints.
numberOfWheels, numberOfArmJoints = 4, 6
#! Platform geometry in meters and total platform/wheel mass in kg.
platformDimensions = [0.575, 0.718, 0.495]
platformMass = 125. - 18.4
platformCenterHeight = 0.375
platformCOM = [0., 0., -0.1]
#radius and width equal the URDF visual omni wheel mesh (0.254 m diameter, 0.099 m width).
#physical wheel positions of the datasheets RobotnikDATASHEETSUMMITXLSTEEL.pdf and
#Robotnik_DATASHEET_RB-KAIROS_EN.pdf (robotnik/mobrobdrivestation/kairos_handbooks): wheel base 430 mm (dimensioned
#from wheel center to wheel center), track about 459 mm (measured in the front view, +-5 mm; 578 mm is the width of
#the housing, not of the wheels). The URDF summit_xls (0.445 x 0.409 m) placed the wheels 25 mm too far inside (AR).
#! Physical wheel radius, width, wheel base (front-rear axle distance) and track (left-right wheel center distance) in m.
wheelRadius, wheelWidth, wheelBase, wheelTrack = 0.127, 0.1, 0.430, 0.459
#the base controller of the robot converts cmd_vel to wheel speeds with its own parameters
#(summit_xl_steel_omni_control.yaml: wheel_diameter 0.25, wheel_base 0.445, track_width 0.415); its odometry agrees
#with Vicon within 3 % in rotation in the lab recordings of 2026-11-06, so the simulated controller uses them as well.
#! Wheel radius, wheel base and track in m of the robot's base controller (kinematics cmd_vel -> wheel speeds).
controllerWheelRadius, controllerWheelBase, controllerWheelTrack = 0.125, 0.445, 0.415
#geometry of the MoveIt/URDF description summit_xls_171220A.urdf.xacro.
#! URDF footprint frame (summit_xl_base_footprint) = platform frame shifted down by platformCenterHeight.
footprintOffset = np.array([0., 0., -platformCenterHeight])
#! Height of summit_xl_base_link (origin of the chassis mesh) above the footprint in m.
baseLinkHeight = 0.125
#! UR5 base frame summit_xl_ur5_base (= standard DH base) in footprint coordinates and its yaw.
armBaseFootprintPosition = [0.1813, 0., 0.50712]
mountPosition = (np.asarray(armBaseFootprintPosition) + footprintOffset).tolist()
mountYaw = -np.pi/4.
#! Meshes rigidly mounted on the platform: file, footprint position in m, roll/pitch/yaw in rad (URDF rpy), color.
kairosParts = [
    {'name': 'chassis', 'file': 'summit_xls_chassis.stl', 'position': [0., 0., baseLinkHeight], 'rpy': [0., 0., 0.]},
    {'name': 'rearVelodyneStructure', 'file': 'structure_rear_velodyne.stl',
     'position': [-0.35785, 0.29431, 0.455], 'rpy': [0., 0., 0.]},
    #yaw pi: on the real robot the lidar cable leaves to the front (URDF: to the rear)
    {'name': 'velodyne', 'file': 'velodyne_vlp16.stl', 'position': [-0.34785, 0.304931, 0.91042], 'rpy': [0., 0., np.pi]},
    {'name': 'frontLaser', 'file': 'sick_s300.stl', 'position': [0.2865, -0.20894, 0.4223], 'rpy': [np.pi, 0., -np.pi/4.]},
    {'name': 'rearLaser', 'file': 'sick_s300.stl', 'position': [-0.2865, 0.20894, 0.4223], 'rpy': [np.pi, 0., 3.*np.pi/4.]},
]
#the real robot carries a Point Grey Bumblebee2 stereo camera (robot log; photo Kairos.jpg) instead
#of the Orbbec Astra of the URDF: beige housing in the middle of the front face of the upper housing, lenses forward.
#! Bumblebee2 (position: frontCameraPosition below chassisTopHeight): housing size [depth x, width y, height z] in m
#! (datasheet 157 x 36 x 47.4 mm), lens baseline and lens radius in m, colors.
frontCameraSize = [0.036, 0.157, 0.0474]
frontCameraBaseline, frontCameraLensRadius = 0.12, 0.008
#measured on the robot (left camera_info, laser distance to a box ahead): the images have
#640 x 480 pixels; mean focal length fx, fy = 553.7, 555.5 px of the left camera (about 60 deg horizontal); the
#disparity of 97 px at 0.644 m confirms the baseline of 0.12 m (the stereo calibration of the robot states 0.193 m).
#! Front camera image size in pixels (width, height) and focal length in pixels of the undistorted left image.
frontCameraImageSize, frontCameraFocalLength = (640, 480), 554.6
#! Cameras by name with the keys of the Leobot's geometry.cameraModels (the same access for both
#! mobile robots): model, housing [depth x, width y, height z] in m, colorOffset in m of the lens of the used image beside
#! the housing middle (left lens: half the baseline), colorImage [width, height] in px and colorVerticalFieldOfView in
#! deg of the undistorted pinhole image; 'base' is the Bumblebee2 at the front of the platform (no camera on the arm).
cameraModels = {'base': {'model': 'Bumblebee2', 'housing': list(frontCameraSize), 'colorOffset': 0.5*frontCameraBaseline,
                         'colorImage': list(frontCameraImageSize),
                         'colorVerticalFieldOfView': float(np.degrees(2.*np.arctan(0.5*frontCameraImageSize[1]/
                                                                                   frontCameraFocalLength)))}}
frontCameraColor, frontCameraLensColor = [0.76, 0.66, 0.46, 1.], [0.12, 0.12, 0.14, 1.]
#! Chassis mesh: points above this height over summit_xl_base_link belong to the upper housing (color split).
chassisHousingHeight = 0.2
#! Standard DH [theta, d, a, alpha], masses, COM and diagonal inertias of the existing UR5 model;
#! d1 = 0.089159 m as in ur_description (with it, DH frame 6 equals URDF tool0 exactly).
armDH = [[0., 0.089159, 0., np.pi/2.], [0., 0., -0.425, 0.], [0., 0., -0.39225, 0.],
         [0., 0.10915, 0., np.pi/2.], [0., 0.09465, 0., -np.pi/2.], [0., 0.0823, 0., 0.]]
armMasses = [3.7, 8.393, 2.33, 1.219, 1.219, 0.1897]
#(parameter review): the forearm COM lies 0.12 m from the elbow (its motor sits in the forearm;
#ur_description and the UR controller parameters agree), i.e. 0.272 m from the wrist along x of DH frame 3; the old
#value 0.150 (robotics toolbox) put it 0.24 m from the elbow. The upper arm COM omits the UR offset of 0.113 m along
#the joint 2 axis (no change of the gravity torques). The upper arm inertia 0.0078/0.0042 kg m^2 of the toolbox was
#30 times too small for 8.4 kg on 0.425 m (a homogeneous rod alone gives 0.126); now ur_description, axial x.
armCOM = [[0., -0.02561, 0.00193], [0.2125, 0., 0.], [0.272, 0., 0.043],
          [0., -0.00180, 0.016340], [0., -0.00180, 0.016340], [0., 0., -0.001159]]
armInertias = np.array([[84., 64., 84.], [151., 2269., 2269.], [16., 462., 462.],
                       [16., 16., 9.], [16., 16., 9.], [1., 1., 1.]])*1e-4
#! Payload at the flange (tool0): WSG50-110 gripper with adapter and fingers; mass in kg as the payload setting of
#! the UR5 of the lab (3.2 kg), COM in m in the flange frame (estimate: center of the gripper housing, not measured)
#! and the size in m of the box of the inertia (the gripper had no mass before).
toolMass, toolCOM, toolBoxSize = 3.2, [0., 0., 0.07], [0.146, 0.05, 0.08]
#TCP as in the real run: MoveIt end effector summit_xl_wsg50_center is 0.023 m
#beyond the flange (tool0), and bringup_lecture.py maps the student's TArm by gripperOffset = -0.120 m onto it.
#! Flange (tool0) -> summit_xl_wsg50_center, and summit_xl_wsg50_center -> TCP of TArm, in m along tool z.
wsg50CenterOffset, labGripperOffset = 0.023, 0.120
toolLength = wsg50CenterOffset + labGripperOffset
#! UR5 joint limits in rad of the MoveIt robot description (summit_xls_171220A.urdf.xacro): pan, lift, elbow,
#! wrist 1..3; the elbow limit +-pi excludes paths through the folded elbow.
armJointLowerLimits = np.array([-4., -4., -np.pi, 0., -3.5, -4.])
armJointUpperLimits = np.array([4., 0., np.pi, 4., 3.5, 4.])
#largest joint torques in N m of the UR5 (datasheet of Universal Robots: base, shoulder and
#elbow 150 N m, wrists 28 N m), for the display of the joint load in VR
armTorqueLimits = np.array([150., 150., 150., 28., 28., 28.])
#! Names of the UR5 link shapes (DH links 1..6) and gripper shapes for collision reports.
armLinkNames = ('shoulder', 'upperArm', 'forearm', 'wrist1', 'wrist2', 'wrist3')
#! Number of shapes (capsule or box per slice along the principal axis) fitted to each UR5 link mesh.
armLinkShapeCounts = (1, 3, 3, 1, 1, 1)
#! Radius in m of the cylindrical UR5 joint housings around the joint axes 1..6 (measured on the meshes).
armJointHousingRadii = (0.0605, 0.0605, 0.0605, 0.0405, 0.0405, 0.0405)
#! Radius in m of the UR5 base cylinder around joint axis 1 (base diameter 149 mm).
armBaseHousingRadius = 0.076
#! Shape name prefixes of pairs that MoveIt never checks (robotnik_ur5_moveit_config/config/summit_xl.srdf,
#! disable_collisions reason "Never": wrist_1 - wrist_3, wrist_1/wrist_2 - gripper and fingers).
moveItDisabledPairs = [('wrist1', 'wrist3'), ('wrist1', 'gripper'), ('wrist1', 'finger'),
                       ('wrist2', 'gripper'), ('wrist2', 'finger')]
#! ROS names of the UR5 joints in the order of the model's joint angles (= URDF/MoveIt order and angles).
armJointNames = ('summit_xl_ur5_shoulder_pan_joint', 'summit_xl_ur5_shoulder_lift_joint', 'summit_xl_ur5_elbow_joint',
                 'summit_xl_ur5_wrist_1_joint', 'summit_xl_ur5_wrist_2_joint', 'summit_xl_ur5_wrist_3_joint')
#! WSG50 finger carriage: inner finger face in carriage x (WSG-FMF.stl) and URDF prismatic joint limits in m.
gripperFingerFaceOffset = 0.003
#! Finger gap after closing without an object in m.
gripperClosedWidth = 0.
numberOfGripperFingers = 2
gripperCarriageLimits = (0.0027, 0.055)
#! Real Kairos: silver top plate, black sides and mast; Velodyne with silver rings and dark blue window.
chassisTopColor = [0.80, 0.81, 0.83, 1.]
chassisSideColor = [0.25, 0.26, 0.28, 1.]   #anthracite: pure black hides the shading
lidarHousingColor = [0.78, 0.79, 0.80, 1.]
lidarWindowColor = [0.07, 0.13, 0.22, 1.]
#! Height of the chassis top plate above summit_xl_base_link (URDF chassis mesh) and tolerance for its faces in m.
chassisTopHeight, chassisTopPlateTolerance = 0.3916, 0.02
#! Bumblebee2 housing center in footprint coordinates in m: center of the front face of the upper housing (y = 0, half
#! height between chassisHousingHeight and chassisTopHeight above base_link); Michael Pieber 2026-11-07.
frontCameraPosition = [0.3775, 0., baseLinkHeight + 0.5*(chassisHousingHeight + chassisTopHeight)]
#steps of velodyne_vlp16.stl (bottom ring 0-18.7 mm, window 19.3-57.4 mm, top ring from 58 mm);
#with 12 mm the long triangles of the bottom ring were colored alternately (sawtooth)
lidarWindowHeights = (0.019, 0.0577)
lidarRadius = 0.053
#simulated laser beams (KairosModel.AddLaserRays) with the names and keys of the Leobot's
#geometry; values of the scans of the real robot (before in panelDemo.py).
#! Scan angles (first, last beam) in rad of the scanners: 'laser_front', 'laser_back' the SICK S300 (270 deg, 0.5 deg
#! steps), 'velodyne' the 2D scan of the VLP-16 (360 deg in 720 bins as robotSide/velodyneRelay.py).
laserScanAngles = {'laser_front': (-0.75*np.pi, 0.75*np.pi), 'laser_back': (-0.75*np.pi, 0.75*np.pi),
                   'velodyne': (-np.pi, np.pi - 2.*np.pi/720.)}
#! Number of beams of the scans (per ring for the 3D scan of the VLP-16: velodyneAzimuthBeams).
laserScanBeams = {'laser_front': 541, 'laser_back': 541, 'velodyne': 720}
#! Range limits in m of the S300 scans (range_min, range_max of the messages).
laserRangeMinimum, laserRangeMaximum = 0.1, 30.
#! Range limits in m of the VLP-16 scans (2D scan of the relay; the sensor itself reaches 100 m).
velodyneRangeMinimum, velodyneRangeMaximum = 0.4, 30.
#! VLP-16: elevation angles in rad of the 16 rings (-15 to 15 deg in 2 deg steps) and beams per ring over 360 deg (0.4 deg).
velodyneElevations = np.radians(np.arange(-15., 16., 2.))
velodyneAzimuthBeams = 900
#! UR5 (CB3) colors of the lab robot: light blue caps, light grey joint housings, aluminium tubes and base;
#! classification of the mesh colors: blue if b - r is larger, black below, tube/base from this grey value on.
ur5CapColor = [0.62, 0.80, 0.90, 1.]
ur5HousingColor = [0.58, 0.59, 0.61, 1.]
ur5TubeColor = [0.78, 0.79, 0.80, 1.]
