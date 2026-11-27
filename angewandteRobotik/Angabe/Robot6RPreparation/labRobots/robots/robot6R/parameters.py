# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: parameters.py
#
# Details:  Parameters of the Exudyn model of the 6R robot (model.py, visualization.py): the device numbers of
#           geometry.py the model uses, the gravity, the PD joint control, the jaw servo control, the initial gripper
#           opening and the workspace check of the lab software (check_input.m).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-03; 2026-11-12 moved from LabCourse/Robot6R/robot6R/parameters.py (the model is equipment)
# Notes:    SI units (m, kg, s, rad) unless the name ends with Deg or Percent. Sources as in geometry.py:
#           [MAT] lab control software (check_input.m), [EST] engineering estimate.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Parameters of the model of the 6R robot."""
#the device numbers the model uses, as par.X of model.py and visualization.py
from labRobots.robots.robot6R.geometry import (  # noqa: F401
                                        numberOfJoints, linkOffsetsD, linkLengthsA, linkTwistsAlpha, jointAngleOffsets,
                                        flangeToTCPDistance, massPW90, massPW70, gripperAdapterRadius,
                                        gripperAdapterThickness, gripperBodySize, gripperJawFlangeSize, gripperMass,
                                        gripperCOM, jawNames, jawMass, jawSize, jawCOM, labJointLowerLimitsDeg,
                                        labJointUpperLimitsDeg, labMaxVelocitiesDeg, labMaxAccelerationsDeg)

#! Gravity in m/s^2.
gravity = [0., 0., -9.81]
#! PD joint control of the arm: natural frequency in Hz and damping ratio (gains from the mass matrix).
controlBandwidthHz = 8.
controlDampingRatio = 1.
#! Gripper command in percent at the start: open [MAT].
gripperInitialPercent = 0.
#! Servo position control of each jaw: stiffness in N/m (grip force approx. 7 N per jaw) and damping ratio [EST].
jawStiffness = 5000.
jawDampingRatio = 1.

#lab control software (Matlab, check_input.m) [MAT]: workspace of the lab setup; Cartesian checks of TCP (frame 6),
#wrist center (frame 5) and axis 3/4 (frame 3) in base frame K0
labMinimumX = -0.35                 #! x >= -0.35 m (wall behind the robot)
labMinimumY = -0.33                 #! y >= -0.33 m (cable side)
labMinimumZ = 0.06                  #! z >= 0.06 m (table)
labWarningZ = 0.1                   #! TCP z < 0.1 m only gives a warning (and ends the check, as in check_input.m)

#the 6R mounted on a body of a scene (CreateRobot6R with markerGround): the base becomes a free
#rigid body (Robot.CreateKinematicTree, floatingBase) fixed to the marker by a joint
#! Mass in kg, radius and height in m of the base (socket with the fixed flange of the PW 90) as a cylinder [EST].
robotBaseMass, robotBaseRadius, robotBaseHeight = 2., 0.06, 0.12
#! Contact of gripper housing and jaws with the objects of the scene (GeneralContact), as the grippers of the Kairos
#! and the Leobot: stiffness in N/m and damping in N s/m per contact primitive.
objectContactStiffness, objectContactDamping = 2e3, 2.
#end stops of the joints at the limits of the lab software (labJointLower/UpperLimitsDeg of
#geometry.py; joint 6 has none); Robot(enableEndStops=True) of Exudyn, a CoordinateSpringDamperExt per joint
#! Stiffness of an end stop relative to the P gain of its joint (damping with the same damping ratio as the control).
endStopStiffnessFactor = 10.
