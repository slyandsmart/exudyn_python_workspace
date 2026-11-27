# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: model.py
#
# Details:  The Kairos (Robotnik Kairos platform, UR5, WSG50 gripper) as model in the style of exudyn.robotics: the
#           description KairosModel(base=RobotBase(HT=...), q0=...), CreateMBS(mbs, name, markerGround, generalContact)
#           builds it into the MainSystem of a scene (coupled floating-base KinematicTree with rolling wheels), SetControl
#           and PreStepControl drive it; arm planning with collision checking against the platform and the obstacles of
#           the scene. Floor, table and objects are the scene's (lab task: model.VehicleModel).
#
# Authors:  Peter Manzl; Michael Pieber (created with Claude Code)
# Date:     2026-11-05 (floating-base KinematicTree); 2026-11-12 robot without scene, Exudyn conventions; 2026-11-12 moved into labRobots/robots/kairos (from LabCourse/MobileRobot/fahrzeug)
# Notes:    No renderer, global system or solver calls. UR5 kinematics, inverse kinematics and gravity torques by
#           Exudyn's KinematicChain (C++).
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Kairos model in the style of exudyn.robotics."""
import numpy as np
import exudyn as exu
from exudyn.itemInterface import ObjectGround, VObjectGround
from exudyn.rigidBodyUtilities import HTtranslate, HTrotateZ, InertiaCuboid, RigidBodyInertia, RotationMatrix2RotationVector
from exudyn.robotics import Robot, RobotBase, RobotLink, RobotTool, VRobotLink, StdDH2HT
from exudyn.robotics.motion import Trajectory, ProfileConstantAcceleration
from labRobots.robots.kairos import parameters as par
from labRobots.robots.kairos.platform import CreateVehicleTree
from labRobots.robotics.modeling.mounting import HideHelperGraphics
from labRobots.robots.kairos.treeBody import AddRigidMarker, GetBodyOutput, SetRigidMarkerPosition
from labRobots.robots.kairos.visualization import LoadArmGraphics, GripperHousingGraphics, GripperFingerGraphics
from labRobots.robots.kairos.kinematics import WheelSpeeds, WrapAngle
from labRobots.robots.kairos.collisionModel import ArmLinkShapes, PlatformObstacles
from labRobots.robots.kairos.contact import AddGripperContact
from labRobots.robots.kairos.planning import KairosArmPlanner
from labRobots.robotics.collision.shapes import TransformShape
from labRobots.robotics.motion.trajectories import CartesianLinearTrajectory, DifferentialInverseKinematicsStep, LinearStepTrajectory
from labRobots.robotics.modeling.basePose import PlanarPoseFromHT

tcpInFlange = [0., 0., par.toolLength]  #TCP position in the flange frame (DH frame 6) in m
#kinds of arm commands, reported to the real robot by the simulation (robotCommand.py)
#! Hold the present reference, planned path (joint or Cartesian), manual teach step.
armCommandHold, armCommandPath, armCommandStep = 'hold', 'path', 'step'
#! First duration in s of a straight TCP motion, prolonged by CartesianLinearTrajectory to the joint limits.
linearFirstDuration = 0.1


def CreateArmKinematicRobot():
    """Return the exudyn Robot of the UR5 kinematics in its base frame with the TCP as tool; its KinematicChain gives
    ToolHT (TCP pose), Jacobian, InverseKinematicsAnalytic (UR geometry, up to 8 solutions) and ArmConfiguration
    [shoulder, elbow, wrist]; the Robot itself serves the Cartesian profiles (ProfileCartesianLinear)."""
    robot = Robot(gravity=par.gravity, base=RobotBase(), tool=RobotTool(HT=HTtranslate(tcpInFlange)), referenceConfiguration=[])
    for values in par.armDH:
        #no inertia: kinematics only (the dynamic arm is CreateArm)
        robot.AddLink(RobotLink(mass=0., COM=np.zeros(par.dimension3D), inertia=np.zeros((par.dimension3D, par.dimension3D)),
                                localHT=StdDH2HT(values)))
    return robot


def NearestJointEquivalent(angles, referenceAngles):
    """Return, per joint, the 2 pi equivalent of angles within the UR5 joint limits that is closest to
    referenceAngles, or None if a joint has no equivalent within its limits."""
    shifts = 2.*np.pi*np.arange(-2, 3)[:, None]
    equivalents = np.asarray(angles, dtype=float)[None, :] + shifts
    inside = (equivalents >= par.armJointLowerLimits) & (equivalents <= par.armJointUpperLimits)
    if not np.all(np.any(inside, axis=0)):
        return None
    distance = np.where(inside, np.abs(equivalents - np.asarray(referenceAngles, dtype=float)), np.inf)
    return equivalents[np.argmin(distance, axis=0), np.arange(equivalents.shape[1])]

def FingerLocalPose(fingerIndex, carriageOffset):
    """the pose of a WSG50 finger carriage in the flange frame (DH frame 6 = URDF tool0)

    Args:
        fingerIndex: 0 = left (carriage at -x of tool0), 1 = right (rotated by pi about tool z)
        carriageOffset: offset in m of the carriage from the flange axis (KairosModel.CarriageOffset)

    Returns:
        4x4 homogeneous transformation
    """
    return HTrotateZ(np.pi*fingerIndex) @ HTtranslate([-carriageOffset, 0., 0.])


def PlatformStandstill(t):
    """the platform velocity of a standing platform, the default of KairosModel.SetControl

    Args:
        t: the time

    Returns:
        zero local velocity [vx, vy, omega]
    """
    return np.zeros(par.dimension3D)


class KairosModel:
    """The Kairos (platform, UR5, WSG50) as exudyn.robotics.Robot: description in the constructor, CreateMBS builds
    it into the MainSystem of a scene; arm planning with collision checking, control (SetControl, PreStepControl)
    and state, from VehicleModel (the scene parts stay there)."""

    def __init__(self, base=None, q0=None):
        """the description of the Kairos (Robotnik Kairos platform, UR5, WSG50 gripper), as exudyn.robotics.Robot: no
        system yet; CreateMBS builds it into a MainSystem

        Args:
            base: RobotBase with HT, the pose of the platform reference point (center of the platform on the floor, x
                forward, z up) in the world (Vicon) frame; only x, y and the rotation about z (ValueError else); None:
                the origin
            q0: UR5 joint angles in rad at the start (e.g. of the real robot); None: par.armInitialAngles

        Note:
            the drive is ideal: the platform velocity commands act at once on the wheels of radius par.wheelRadius
            (the identified drive model of the real Kairos is removed, only simulations)
        """
        self.base = RobotBase() if base is None else base  #! Base of the platform reference point (RobotBase.HT).
        self.initialPose = PlanarPoseFromHT(self.base.HT)  #! Start pose [x in m, y in m, yaw in rad] of the platform.
        #start arm angles selectable (live robot state, --robot HOST).
        #! UR5 joint angles in rad at the start.
        self.armInitialAngles = np.array(par.armInitialAngles if q0 is None else q0, dtype=float)
        self.baseGraphics, linkGraphics = LoadArmGraphics()  #! Graphics of the UR5 base (on the platform link).
        self.robot = self.CreateArm(linkGraphics)  #! Robotics representation matching the physical DH frames.
        self.mbs = None  #! MainSystem of the model created by CreateMBS, None before.
        self.items = None  #! Dict of the items created by CreateMBS, None before.
        self.armKinematicRobot = CreateArmKinematicRobot()  #! UR5 kinematics Robot in the arm base frame, TCP as tool.
        self.armChain = self.armKinematicRobot.GetKinematicChain()  #! Its KinematicChain (Exudyn, C++).
        self.armTrajectory = Trajectory(initialCoordinates=self.armInitialAngles.tolist(), initialTime=0.)
        self.armTrajectory.Add(ProfileConstantAcceleration(self.armInitialAngles.tolist(), par.armHoldDuration))
        self.armTarget = self.armInitialAngles.copy()  #! Last commanded arm angles.
        self.armPath = [self.armTarget.copy()]  #! Waypoints of the last planned arm motion.
        #! Number and kind (armCommandHold/Path/Step) of the arm trajectories set so far.
        self.armCommand = {'count': 0, 'kind': armCommandHold}
        #arm motions sent to the real robot start at its measured angles, not at the
        #simulated arm (which sags under gravity by a few degrees)
        #! Optional function returning the measured arm angles in rad at which planned motions start; None: simulated arm.
        self.armStartAnglesFunction = None
        #! Real arm angles in rad the arm follows in the mirror mode instead of its trajectory, or None (Claude Code,
        #! 2026-11-07).
        self.armMirrorAngles = None
        self.platformObstacles = PlatformObstacles()  #! Platform shapes in the arm base frame.
        #! Obstacles of the scene (shapes in world coordinates, SetEnvironmentObstacles).
        self.environmentObstacles = []
        #collision check and planning by Exudyn only (CollisionModel, navigation)
        #! Collision-free arm motions (KairosArmPlanner: arm, gripper, platform and the obstacles of the scene).
        self.armPlanner = KairosArmPlanner(self.armChain, ArmLinkShapes(), self.platformObstacles)
        #! World pose (4x4) of the arm base at the last UpdateCollisionObstacles (the environment obstacles of the
        #! planner are in this arm base frame), None before, for the collision display.
        self.collisionObstaclesArmBasePose = None
        self.wheelAngles = np.zeros(par.numberOfWheels)  #! Integrated wheel motor reference angles.
        self.lastWheelSpeeds = np.zeros(par.numberOfWheels)  #! Speeds used to undo a rejected physics step.
        self.lastControlTime = 0.  #! Last actuator update time in seconds.
        self.nextGravityUpdate = 0.  #! Next feed-forward update.
        #! Gripper state: grasp active (connector of a scene object), the animated opening of the fingers and whether the
        #! drawn fingers follow the flange (on by default, the lab simulation switches it off
        #! without renderer and solution file).
        self.grasp = {'active': False, 'openingStart': par.gripperOpenWidth, 'openingTarget': par.gripperOpenWidth,
                      'openingTime': 0., 'nativeGraphics': True}
        #! Function time -> local platform velocity [vx, vy, omega] used by PreStepControl (SetControl).
        self.platformVelocity = PlatformStandstill
        #! Simulated camera rays by camera name (AddCameraRays, as LeobotModel).
        self.cameraRays = {}
        #! Simulated laser beams by scanner name (AddLaserRays, as LeobotModel).
        self.laserRays = {}
        #! GeneralContact of floor and obstacles the rays hit if the scene gives none (RayContact), None before.
        self.rayContact = None

    def CreateMBS(self, mbs, name='kairos', markerGround=None, generalContact=None):
        """build the Kairos into an existing MainSystem: one floating-base KinematicTree with
        platform, four wheels (rolling disc contact on markerGround) and UR5 with WSG50; floor, table and objects are
        the scene's

        Args:
            mbs: the MainSystem, which will be extended
            name: name of the KinematicTree object
            markerGround: rigid body marker of the floor on which the wheels roll; None: a ground object at the origin
                without graphics
            generalContact: GeneralContact of the scene to which the gripper housing and fingers are added (grasping of
                scene objects); None: no gripper contact

        Returns:
            dict with 'objectKinematicTree', 'nodeGeneric', 'markerGround', 'platformBody', 'wheelBodies', 'armBodies',
            'flangeBody' (TreeBody objects for the bodies of the tree), 'wheelContacts', 'toolMarker' (TCP on the
            flange), 'fingerObjects' (ground objects of the drawn fingers) and 'fingerContactMarkers'

        Note:
            a KairosModel builds one model; its functions (GetState, PreStepControl, ...) act on it
        """
        if self.mbs is not None:
            raise ValueError('KairosModel.CreateMBS: the model is already built; create a KairosModel per robot')
        self.mbs = mbs
        if markerGround is None:
            markerGround = AddRigidMarker(mbs, mbs.AddObject(ObjectGround()))
        #one floating-base tree preserves all wheel/arm/platform reactions.
        self.bodies, armBodies, self.tree = CreateVehicleTree(mbs, self.initialPose, self.baseGraphics, self.robot,
                                                              markerGround, name)
        self.arm = {'bodyList': armBodies}  #! Physical DH frames of the tree's six UR5 links.
        #! Tree object/node, coordinate indices and reusable joint control vectors are in self.tree.
        self.toolMarker = AddRigidMarker(mbs, armBodies[-1], [0., 0., par.toolLength])  #! TCP marker on the flange.
        #massless visual fingers follow the TCP in both native and WebXR renderers.
        #placed at the start pose of the flange (drawn before the first step)
        flangePose = np.array(self.robot.LinkHT(self.armInitialAngles)[-1])
        fingerPoses = [flangePose @ FingerLocalPose(fingerIndex, self.CarriageOffset())
                       for fingerIndex in range(par.numberOfGripperFingers)]
        self.bodies['fingers'] = [mbs.AddObject(ObjectGround(
            referencePosition=pose[:3, 3].tolist(), referenceRotation=pose[:3, :3].tolist(),
            visualization=VObjectGround(graphicsData=GripperFingerGraphics()))) for pose in fingerPoses]
        #moving gripper contact acts on the physical flange, also in headless/VR runs.
        #! Flange-attached markers of the prescribed finger opening (in the contact of the scene).
        self.fingerContactMarkers = [] if generalContact is None else AddGripperContact(generalContact, mbs, armBodies[-1])
        self.SetArmOffsets(self.armTarget, np.zeros(par.numberOfArmJoints))
        #! Dict of the created items (also returned).
        HideHelperGraphics(mbs)
        self.items = {'objectKinematicTree': self.tree['object'], 'nodeGeneric': self.tree['node'], 'markerGround': markerGround,
                      'platformBody': self.bodies['platform'], 'wheelBodies': self.bodies['wheels'], 'armBodies': armBodies,
                      'flangeBody': armBodies[-1], 'wheelContacts': self.bodies['contacts'], 'toolMarker': self.toolMarker,
                      'fingerObjects': self.bodies['fingers'], 'fingerContactMarkers': self.fingerContactMarkers}
        return self.items

    def Initialize(self):
        """Initialize continuous minimal-coordinate references after assembly."""
        #tree coordinates do not need angle-history data nodes.
        self.armTarget = self.GetJointAngles()
        trajectory = Trajectory(initialCoordinates=self.armTarget.tolist(), initialTime=0.)
        trajectory.Add(ProfileConstantAcceleration(self.armTarget.tolist(), par.armHoldDuration))
        self.SetArmTrajectory(trajectory, armCommandHold)
        self.SetArmOffsets(self.armTarget, np.zeros(par.numberOfArmJoints))
        self.UpdateGripper()

    def CreateArm(self, linkGraphics):
        """Create UR5 robotics data with the moving mount orientation and the existing inertias."""
        platformPose = HTtranslate([self.initialPose[0], self.initialPose[1],
                                    par.platformCenterHeight + 125.*9.81/par.contactStiffness]) @ HTrotateZ(self.initialPose[2])
        basePose = platformPose @ HTtranslate(par.mountPosition) @ HTrotateZ(par.mountYaw)
        robot = Robot(gravity=par.gravity, base=RobotBase(HT=basePose),
                      tool=RobotTool(HT=HTtranslate([0., 0., par.toolLength])),
                      referenceConfiguration=self.armInitialAngles)
        for jointIndex in range(par.numberOfArmJoints):
            linkInertia = RigidBodyInertia()
            linkInertia.SetWithCOMinertia(par.armMasses[jointIndex], np.diag(par.armInertias[jointIndex]), par.armCOM[jointIndex])
            if jointIndex == par.numberOfArmJoints - 1:
                #payload of the gripper at the flange (DH frame 6 = tool0), as the payload
                #setting of the real UR5; before, the gripper had no mass
                toolInertia = InertiaCuboid(par.toolMass/np.prod(par.toolBoxSize), par.toolBoxSize).Translated(par.toolCOM)
                linkInertia = linkInertia + toolInertia
            robot.AddLink(RobotLink(mass=linkInertia.Mass(), COM=linkInertia.COM(),
                                   inertia=linkInertia.InertiaCOM(), localHT=StdDH2HT(par.armDH[jointIndex]),
                                   PDcontrol=(par.armProportionalGains[jointIndex], par.armDerivativeGains[jointIndex]),
                                   visualization=VRobotLink(
                                       graphicsData=linkGraphics[jointIndex] +
                                       (GripperHousingGraphics() if jointIndex == par.numberOfArmJoints - 1 else []),
                                       showMBSjoint=False, showCOM=False)))
        return robot

    def SetPlatformPose(self, pose):
        """Move the standing vehicle to a world pose [x in m, y in m, yaw in rad] during a simulation (e.g. the Vicon
        pose of the real robot after a replayed drive); arm and wheel angles are kept, the platform velocities set to
        zero (a grasped object of the scene: VehicleModel). Sets the current and start-of-step configuration (both solvers).
        """
        #step-by-step execution on the real robot
        pose = np.asarray(pose, dtype=float)
        treeIndex = self.mbs.GetNodeODE2Index(self.tree['node'])
        for configuration in (exu.ConfigurationType.Current, exu.ConfigurationType.StartOfStep):
            coordinates = np.array(self.mbs.systemData.GetODE2Coordinates(configuration=configuration))
            velocities = np.array(self.mbs.systemData.GetODE2Coordinates_t(configuration=configuration))
            coordinates[treeIndex + 0], coordinates[treeIndex + 1] = pose[0], pose[1]
            coordinates[treeIndex + par.baseYawCoordinate] = pose[2]
            velocities[treeIndex:treeIndex + par.numberOfBaseCoordinates] = 0.
            self.mbs.systemData.SetODE2Coordinates(coordinates.tolist(), configuration=configuration)
            self.mbs.systemData.SetODE2Coordinates_t(velocities.tolist(), configuration=configuration)
        self.robot.base.HT = self.ArmBasePose()

    def PlatformPose(self):
        """Return the planar pose [x in m, y in m, yaw in rad] of the platform in world coordinates, as
        LeobotModel.PlatformPose (the same name for both mobile robots)."""
        return self.GetState()['pose']

    def BodyPose(self, body, configuration=exu.ConfigurationType.Current):
        """Return a rigid body's world homogeneous transformation (configuration Visualization in graphics user
        functions)."""
        pose = np.eye(4)
        pose[:3, 3] = GetBodyOutput(self.mbs, body, exu.OutputVariableType.Position, configuration)
        pose[:3, :3] = np.asarray(GetBodyOutput(self.mbs, body, exu.OutputVariableType.RotationMatrix, configuration)).reshape(
            par.dimension3D, par.dimension3D)
        return pose

    def ArmBasePose(self, configuration=exu.ConfigurationType.Current):
        """Return the actual moving arm base in world coordinates."""
        return self.BodyPose(self.bodies['platform'], configuration) @ HTtranslate(par.mountPosition) @ HTrotateZ(par.mountYaw)

    def FlangePose(self):
        """Return the actual UR5 flange (DH frame 6 = URDF tool0) in world coordinates."""
        return self.BodyPose(self.arm['bodyList'][-1])

    def ToolPose(self, armAngles=None):
        """Return the TCP (pose of the student's TArm) in world coordinates: the actual one, or at the arm angles
        armAngles in rad at the present platform pose (as LeobotModel.ToolPose; the kinematic
        chain of the planner includes the tool)."""
        if armAngles is not None:
            return self.ArmBasePose() @ np.array(self.armPlanner.chain.ToolHT(np.asarray(armAngles, dtype=float)))
        return self.FlangePose() @ HTtranslate([0., 0., par.toolLength])

    def GetJointAngles(self, configuration=exu.ConfigurationType.Current):
        """Return the six continuous UR5 coordinates directly from the vehicle tree."""
        return np.asarray(self.mbs.GetNodeOutput(self.tree['node'], exu.OutputVariableType.Coordinates, configuration))[
            self.tree['armCoordinates']]

    def JointMonitorValues(self):
        """Return (arm angles in rad, arm velocities in rad/s, arm torques in N m, wheel speeds in rad/s) for the
        display in VR: the arm torques are those of the tree (PD control to the offsets
        plus the gravity compensation), the wheel speeds those of the motors (sign of WheelSpeeds)."""
        coordinates = np.asarray(self.mbs.GetNodeOutput(self.tree['node'], exu.OutputVariableType.Coordinates))
        velocities = np.asarray(self.mbs.GetNodeOutput(self.tree['node'], exu.OutputVariableType.Coordinates_t))
        tree, arm = self.tree['object'], self.tree['armCoordinates']
        proportional = np.asarray(self.mbs.GetObjectParameter(tree, 'jointPControlVector'))[arm]
        derivative = np.asarray(self.mbs.GetObjectParameter(tree, 'jointDControlVector'))[arm]
        torques = (proportional*(self.tree['positionOffsets'][arm] - coordinates[arm]) +
                   derivative*(self.tree['velocityOffsets'][arm] - velocities[arm]) + self.tree['forces'][arm])
        return coordinates[arm], velocities[arm], torques, -velocities[self.tree['wheelCoordinates']]

    def GetState(self):
        """Return actual platform pose, velocity and arm/gripper readiness."""
        #reuse the floating-base coordinates instead of building several pose matrices.
        coordinates = np.asarray(self.mbs.GetNodeOutput(self.tree['node'], exu.OutputVariableType.Coordinates))
        velocities = np.asarray(self.mbs.GetNodeOutput(self.tree['node'], exu.OutputVariableType.Coordinates_t))
        jointAngles, jointVelocity = coordinates[self.tree['armCoordinates']], velocities[self.tree['armCoordinates']]
        # For world Rz(yaw)*Ry(pitch)*Rx(roll), angularVelocityWorldZ = yaw_t - sin(pitch)*roll_t.
        turnRate = velocities[par.baseYawCoordinate] - np.sin(coordinates[par.basePitchCoordinate])*velocities[par.baseRollCoordinate]
        return {'pose': np.array([coordinates[0], coordinates[1], WrapAngle(coordinates[par.baseYawCoordinate])]),
                'velocity': np.array([velocities[0], velocities[1], turnRate]),
                'armReady': bool(self.lastControlTime >= self.armTrajectory.GetTimes()[-1] and
                                 np.max(np.abs(jointAngles - self.armTarget)) < par.armPositionTolerance and
                                 np.max(np.abs(jointVelocity)) < par.armVelocityTolerance),
                'grasped': self.grasp['active'],
                'gripperClosed': bool(self.grasp['openingTarget'] < par.gripperOpenWidth),  #with or without object
                #gripper motion finished (the WSG50 move service blocks until then)
                'gripperReady': bool(self.lastControlTime - self.grasp['openingTime'] >= par.gripperMotionDuration)}

    def SetEnvironmentObstacles(self, obstacles):
        """set the obstacles of the scene for the arm collision checker

        Args:
            obstacles: list of shapes (labRobots.robotics.collision.shapes) in world coordinates; UpdateCollisionObstacles
                transforms them into the arm base frame of the present platform pose
        """
        self.environmentObstacles = [dict(shape) for shape in obstacles]
        self.collisionObstaclesArmBasePose = None     #set at the next UpdateCollisionObstacles

    def UpdateCollisionObstacles(self):
        """Set the obstacles of the scene of the planner for the present platform pose (arm base frame); only if there
        are obstacles and the platform moved since the last update or the obstacles changed."""
        armBasePose = self.ArmBasePose()
        if (self.collisionObstaclesArmBasePose is not None and
                np.max(np.abs(armBasePose - self.collisionObstaclesArmBasePose)) <= par.obstaclePoseTolerance):
            return
        worldToArmBase = np.linalg.inv(armBasePose)
        self.collisionObstaclesArmBasePose = armBasePose
        self.armPlanner.SetObstacles([TransformShape(shape, worldToArmBase) for shape in self.environmentObstacles])

    def CollisionShapes(self, configuration=exu.ConfigurationType.Current):
        """Return (arm shapes, other shapes) of the collision model in world coordinates at the state of configuration
        (labRobots.robotics.collision.display): arm and gripper at the joint angles, the platform at the arm base, the
        obstacles of the scene where the planner has them (arm base pose of their last update)."""
        armShapes, otherShapes = self.RobotShapeGroups(configuration)
        if self.collisionObstaclesArmBasePose is not None:
            otherShapes += [TransformShape(shape, self.collisionObstaclesArmBasePose) for shape in self.armPlanner.obstacles]
        return armShapes, otherShapes

    def RobotShapeGroups(self, configuration=exu.ConfigurationType.Current):
        """Return (arm shapes, platform shapes) of the robot itself in world coordinates at the state of configuration:
        arm and gripper at the joint angles, the platform at the arm base (from
        CollisionShapes, without the obstacles of the scene)."""
        armBase = self.ArmBasePose(configuration)
        frames = [np.array(frame) for frame in self.armChain.LinkHT(self.GetJointAngles(configuration))]
        armShapes = [TransformShape(shape, armBase @ frames[linkIndex]) for linkIndex, shape in self.armPlanner.linkShapes]
        platformShapes = [TransformShape(shape, armBase) for shape in self.armPlanner.platformShapes]
        return armShapes, platformShapes

    def RobotShapes(self, configuration=exu.ConfigurationType.Current):
        """Return the shapes of the robot itself (arm and platform, without the obstacles of the scene) in world
        coordinates at the state of configuration: the hull other robots' sensor rays hit
        (labRobots.robotics.sensors.movingShapes.ConnectRays)."""
        armShapes, platformShapes = self.RobotShapeGroups(configuration)
        return armShapes + platformShapes

    def CollisionDistance(self, configuration=exu.ConfigurationType.Current):
        """Return (smallest distance in m of the collision model at the joint angles of configuration, safety distance
        in m)."""
        return self.armPlanner.MinimumDistance(self.GetJointAngles(configuration)), self.armPlanner.margin

    def ArmForwardKinematics(self, angles):
        """Return the 4x4 TCP pose in the arm base frame for the joint angles in rad (KinematicChain.ToolHT)."""
        return np.array(self.armChain.ToolHT(angles))

    def ArmJacobian(self, angles):
        """Return the 6 x 6 geometric Jacobian [v; omega] of the TCP in the arm base frame (KinematicChain.Jacobian)."""
        return np.array(self.armChain.Jacobian(angles, -1, tcpInFlange))

    def InverseKinematics(self, targetPose, seedAngles):
        """Return the inverse kinematics solution in the arm configuration of seedAngles that is nearest to them
        (analytic, KinematicChain.InverseKinematicsAnalytic; nearest 2 pi equivalent within the joint limits), or
        None; used for Cartesian paths (seed = previous waypoint)."""
        solutions = self.armChain.InverseKinematicsAnalytic(np.asarray(targetPose, dtype=float), seedAngles,
                                                            self.armChain.ArmConfiguration(seedAngles))
        return NearestJointEquivalent(solutions[0], seedAngles) if len(solutions) > 0 else None

    def CollisionFreeInverseKinematics(self, targetPose, angles):
        """Return the collision-free inverse kinematics solution with the smallest weighted joint motion
        (par.armJointWeights) from the present angles; all solutions of the analytic inverse kinematics (up to 8
        UR5 branches: shoulder, elbow, wrist flip) are compared. This avoids large wrist flips (MoveIt picks a
        random branch).

        Args:
            targetPose: 4x4 TCP pose in the arm base frame.
            angles: Present joint angles in rad.
        """
        #nearest branch instead of the first one found; 2026-11-06: all analytic branches
        solutions = self.armChain.InverseKinematicsAnalytic(np.asarray(targetPose, dtype=float), angles)
        candidates, distances, numberOutsideLimits = [], [], 0
        for solution in solutions:
            candidate = NearestJointEquivalent(solution, angles)
            if candidate is None:
                numberOutsideLimits += 1
            elif self.armPlanner.IsFree(candidate):
                candidates.append(candidate)
            else:
                distances.append(self.armPlanner.MinimumDistance(candidate))
        if not candidates:
            #report all reasons
            raise ValueError('Arm target has no collision-free inverse kinematics solution (%d solutions, %d outside the '
                             'joint limits; smallest distances of the others in mm: %s)'
                             % (len(solutions), numberOutsideLimits, [round(1e3*distance, 1) for distance in distances]))
        return min(candidates, key=lambda candidate: np.linalg.norm(par.armJointWeights*(candidate - angles)))

    def JointAnglesAtToolPose(self, toolPose, startAngles):
        """Return the collision-free joint angles in rad at which the tool center point is at a 4x4 pose in world
        coordinates, as KairosModel, LeobotModel and Robot6RModel: the pose in the arm base
        frame (ArmBasePose), the analytic inverse kinematics with the smallest weighted joint motion
        (CollisionFreeInverseKinematics); raises ValueError if there is none. Its result is a goal of
        armPlanner.Plan or MoveArmToJoints.

        Args:
            toolPose: 4x4 pose of the tool center point in world coordinates
            startAngles: joint angles in rad from which the arm moves (the nearest solution is taken)

        Returns:
            joint angles in rad
        """
        self.UpdateCollisionObstacles()
        armPose = np.linalg.inv(self.ArmBasePose()) @ np.asarray(toolPose, dtype=float)
        return self.CollisionFreeInverseKinematics(armPose, np.asarray(startAngles, dtype=float))

    def PlanningStart(self):
        """Update the obstacles and return (measured angles, start angles of the planning).

        The measured arm sags under gravity and platform acceleration; if this tracking error alone violates the safety
        distance, planning starts from the last commanded (free) angles, as MoveIt's FixStartStateCollision, and the
        motion goes straight from the measured angles to them first. Otherwise a start closer than the safety distance
        is refused (ValueError). With self.armStartAnglesFunction (real robot), its angles are the measured angles.
        """
        angles = self.GetJointAngles() if self.armStartAnglesFunction is None else \
            np.asarray(self.armStartAnglesFunction(), dtype=float)
        self.UpdateCollisionObstacles()
        planner = self.armPlanner
        if planner.IsFree(angles):
            return angles, angles
        if planner.IsFree(self.armTarget) and np.max(np.abs(angles - self.armTarget)) <= par.plannerStartTolerance:
            return angles, self.armTarget.copy()
        raise ValueError('Arm start closer than the safety distance (smallest distance %.1f mm)'
                         % (1e3*planner.MinimumDistance(angles)))

    def MoveArm(self, targetPose, time):
        """Plan a collision-free arm motion to a TCP pose in the arm base frame (RRT-Connect in joint space,
        as OMPL in MoveIt); reject unreachable or colliding targets."""
        angles, planningStart = self.PlanningStart()
        self.PlanArmMotion(angles, planningStart, self.CollisionFreeInverseKinematics(targetPose, planningStart), time)

    def MoveArmLinear(self, targetPose, time):
        """Move the TCP on a straight line (position linear, orientation slerp) to a pose in the arm base frame without
        stops, as moveEEmethod / computeCartesianPath on the robot: Exudyn's ProfileCartesianLinear (smooth time law,
        analytic inverse kinematics), checked by the planner along its samples; raises ValueError if the line leaves
        the workspace, passes a singularity or collides."""
        #Cartesian approach and retreat; 2026-11-12: Exudyn only
        angles, planningStart = self.PlanningStart()
        if planningStart is not angles:
            raise ValueError('Linear arm motion failed: the measured arm is closer than the safety distance')
        try:
            trajectory = CartesianLinearTrajectory(self.armKinematicRobot, angles, targetPose, time, linearFirstDuration,
                                                   par.armMaximumSpeed, par.armMaximumAcceleration)
        except ValueError as error:
            raise ValueError('Linear arm motion failed: ' + str(error)) from error
        if not self.armPlanner.TrajectoryIsFree(trajectory):
            raise ValueError('Linear arm motion failed: the straight motion is not collision-free')
        self.armPath = [angles, np.asarray(trajectory.GetFinalCoordinates(), dtype=float)]
        self.SetArmTrajectory(trajectory, armCommandPath)
        self.armTarget = self.armPath[-1]

    def ManualArmStep(self, targetPose, time):
        """One manual (teleoperation) step of the arm towards a TCP pose in the arm base frame: damped least
        squares differential inverse kinematics from the present reference, joint speed limited by
        par.armMaximumSpeed, clipped to the joint limits; the step is executed only if it is collision free.
        Called every par.jogUpdatePeriod, the reference moves with constant velocity between the steps.

        Returns:
            '' if the arm moves or the target is reached, otherwise the reason why it stays (German, for the panel).
        """
        #VR teach mode (TCP handle and sticks), method of the 6R robot teleoperation
        reference = np.asarray(self.armTrajectory.Evaluate(time)[0])
        pose = self.ArmForwardKinematics(reference)
        step = DifferentialInverseKinematicsStep(pose, targetPose, self.ArmJacobian(reference),
                                                 par.manualRotationWeight, par.manualDamping)
        step /= max(1., np.max(np.abs(step))/(par.armMaximumSpeed*par.jogUpdatePeriod))
        candidate = np.clip(reference + step, par.armJointLowerLimits, par.armJointUpperLimits)
        if np.max(np.abs(candidate - reference)) < par.manualMinimumJointStep:
            remaining = max(np.linalg.norm(targetPose[:3, 3] - pose[:3, 3]),
                            np.linalg.norm(RotationMatrix2RotationVector(pose[:3, :3].T @ targetPose[:3, :3])))
            return '' if remaining < par.manualTargetTolerance else 'Ziel nicht erreichbar (Arbeitsraum oder Gelenkgrenze)'
        self.UpdateCollisionObstacles()
        if not self.armPlanner.IsFree(candidate):
            return 'Kollision: Arm bleibt stehen (Abstand %.0f mm)' % (1e3*self.armPlanner.MinimumDistance(candidate))
        self.SetArmTrajectory(LinearStepTrajectory(reference, candidate, time, par.jogUpdatePeriod), armCommandStep)
        self.armTarget = candidate
        return ''

    def CommandedToolPose(self):
        """Return the TCP pose of the last commanded arm angles in the arm base frame."""
        return self.ArmForwardKinematics(self.armTarget)

    def MoveArmToJoints(self, targetAngles, time, speedScale=1.):
        """Plan a collision-free arm motion to joint angles in rad (reproducible, unlike pose targets).

        Args:
            targetAngles: 6 arm joint angles in rad.
            time: Simulation time in s at which the motion starts.
            speedScale: Factor in (0, 1] of the joint speed and acceleration limits.
        """
        #joint target, as ur5.move_joint_pose on the robot
        angles, planningStart = self.PlanningStart()
        self.PlanArmMotion(angles, planningStart, np.asarray(targetAngles, dtype=float), time, speedScale)

    def PlanArmMotion(self, angles, planningStart, targetAngles, time, speedScale=1.):
        """Plan the joint path from planningStart to targetAngles (Exudyn: RRT-Connect, shortcuts, blended trajectory)
        and set the arm trajectory from the measured angles; raises ValueError if the target collides or no path is
        found."""
        try:
            trajectory, waypoints = self.armPlanner.Plan(planningStart, targetAngles, time, speedScale,
                                                         None if planningStart is angles else angles)
        except ValueError as error:
            raise ValueError('Arm motion planning failed: ' + str(error)) from error
        self.armPath = waypoints
        self.SetArmTrajectory(trajectory, armCommandPath)
        self.armTarget = np.asarray(targetAngles, dtype=float)

    def HoldArm(self, angles, time):
        """Hold the arm at joint angles in rad from time on (end of the mirror mode)."""
        self.armTarget = np.asarray(angles, dtype=float).copy()
        trajectory = Trajectory(initialCoordinates=list(self.armTarget), initialTime=time)
        trajectory.Add(ProfileConstantAcceleration(list(self.armTarget), par.armHoldDuration))
        self.SetArmTrajectory(trajectory, armCommandHold)

    def StopArm(self, time):
        """Cancel the arm trajectory and hold its present reference without resetting physical coordinates."""
        reference, unusedVelocity, unusedAcceleration = self.armTrajectory.Evaluate(time)
        self.armTarget = np.asarray(reference)
        trajectory = Trajectory(initialCoordinates=list(reference), initialTime=time)
        trajectory.Add(ProfileConstantAcceleration(list(reference), par.armHoldDuration))
        self.SetArmTrajectory(trajectory, armCommandHold)

    def SetArmTrajectory(self, trajectory, kind):
        """Set the arm reference trajectory and count it as a new arm command.

        Args:
            trajectory: Exudyn Trajectory (Evaluate(time) -> angles, velocities, accelerations, GetTimes()).
            kind: armCommandHold, armCommandPath or armCommandStep.
        """
        #arm commands can be sent to the real robot (robotCommand.py)
        self.armTrajectory = trajectory
        self.armCommand = {'count': self.armCommand['count'] + 1, 'kind': kind}

    def SetGripperWidth(self, width):
        """start the finger motion of the gripper to an opening (grasping a scene object:
        VehicleModel.SetGrasp)

        Args:
            width: gap between the finger inner faces in m
        """
        self.grasp['openingStart'] = self.GripperOpening()
        self.grasp['openingTarget'] = width
        self.grasp['openingTime'] = self.lastControlTime

    def GripperOpening(self):
        """Return the smoothly animated visual gap between the finger inner faces in meters."""
        #most physics steps have a constant opening; avoid scalar NumPy allocations.
        fraction = (self.lastControlTime - self.grasp['openingTime'])/par.gripperMotionDuration
        if fraction <= 0.:
            return self.grasp['openingStart']
        if fraction >= 1.:
            return self.grasp['openingTarget']
        fraction = 0.5 - 0.5*np.cos(np.pi*fraction)
        return self.grasp['openingStart'] + fraction*(self.grasp['openingTarget'] - self.grasp['openingStart'])

    def FingerPose(self, fingerIndex):
        """Return the world pose of a WSG50 finger carriage (URDF summit_xl_wsg50_gripper_left/right).

        Args:
            fingerIndex: 0 = left (carriage at -x of tool0), 1 = right (rotated by pi about tool z).
        """
        #URDF prismatic finger joints; carriage offset = half gap + inner face offset.
        return self.FlangePose() @ FingerLocalPose(fingerIndex, self.CarriageOffset())

    def CarriageOffset(self):
        """Return the offset in m of each finger carriage from the flange axis at the present opening (URDF prismatic
        finger joints: half gap + inner face offset, within the carriage limits)."""
        return min(max(self.GripperOpening()/2. + par.gripperFingerFaceOffset,
                       par.gripperCarriageLimits[0]), par.gripperCarriageLimits[1])

    def UpdateGripper(self):
        """Update prescribed finger contact markers and optional native renderer geometry."""
        #update flange-local markers only while the opening changes.
        carriageOffset = self.CarriageOffset()
        if carriageOffset != self.grasp.get('contactOffset'):
            for fingerIndex, marker in enumerate(self.fingerContactMarkers):
                direction = -1. if fingerIndex == 0 else 1.
                SetRigidMarkerPosition(self.mbs, marker, self.arm['bodyList'][-1], [direction*carriageOffset, 0., 0.])
            self.grasp['contactOffset'] = carriageOffset
        if self.grasp.get('nativeGraphics', False):
            self.PlaceFingerGraphics(carriageOffset)

    def PlaceFingerGraphics(self, carriageOffset, configuration=exu.ConfigurationType.Current):
        """Place the drawn fingers (ground objects) at the flange of a configuration with a carriage offset in m
        (CarriageOffset): also for videos rendered from recorded coordinates (configuration
        Visualization), where the fingers do not follow the flange by themselves."""
        flangePose = self.BodyPose(self.arm['bodyList'][-1], configuration)
        for fingerIndex, body in enumerate(self.bodies['fingers']):
            pose = flangePose @ FingerLocalPose(fingerIndex, carriageOffset)
            self.mbs.SetObjectParameter(body, 'referencePosition', pose[:3, 3])
            self.mbs.SetObjectParameter(body, 'referenceRotation', pose[:3, :3])

    def SetArmOffsets(self, angles, velocities):
        """Apply arm references together with the wheel references to the tree's built-in PD controllers."""
        self.tree['positionOffsets'][self.tree['armCoordinates']] = angles
        self.tree['velocityOffsets'][self.tree['armCoordinates']] = velocities
        self.mbs.SetObjectParameter(self.tree['object'], 'jointPositionOffsetVector', self.tree['positionOffsets'])
        self.mbs.SetObjectParameter(self.tree['object'], 'jointVelocityOffsetVector', self.tree['velocityOffsets'])

    def Update(self, time, velocity):
        """Update wheel/arm actuators; wheel angles integrate local body velocity commands."""
        duration = time - self.lastControlTime
        speeds = WheelSpeeds(velocity)
        #undo reference integration when an adaptive solver rejects a step.
        self.wheelAngles += duration*(self.lastWheelSpeeds if duration < 0. else speeds)
        self.lastWheelSpeeds = speeds
        #wheel tree axes follow the meshes (-y), motor references use +y.
        self.tree['positionOffsets'][self.tree['wheelCoordinates']] = -self.wheelAngles
        self.tree['velocityOffsets'][self.tree['wheelCoordinates']] = -speeds
        angles, velocities, unusedAcceleration = self.armTrajectory.Evaluate(time)
        if self.armMirrorAngles is not None:   #mirror mode: the real arm angles
            angles, velocities = self.armMirrorAngles, np.zeros(par.numberOfArmJoints)
        self.SetArmOffsets(angles, velocities)
        #Update the base pose for gravity compensation on a moving platform.
        #reuse the 6R's 100 Hz gravity feed forward, including after a rejected step.
        if time >= self.nextGravityUpdate or time < self.lastControlTime:
            self.robot.base.HT = self.ArmBasePose()
            # StaticTorques returns J^T*m*g; tree jointForceVector is on the RHS and must oppose gravity.
            #KinematicChain (C++) of the arm on the present platform pose
            self.tree['forces'][self.tree['armCoordinates']] = -self.robot.GetKinematicChain().StaticTorques(self.GetJointAngles())
            self.mbs.SetObjectParameter(self.tree['object'], 'jointForceVector', self.tree['forces'])
            self.nextGravityUpdate = time + par.gravityCompensationPeriod
        self.lastControlTime = time
        self.UpdateGripper()
        for rays in list(self.cameraRays.values()) + list(self.laserRays.values()):   #Add...Rays
            rays.Update(time)

    def CameraPose(self, name, configuration=exu.ConfigurationType.Current):
        """Return the 4x4 world pose of the optical frame of a camera ('base': left lens of the Bumblebee2 at the front of
        the platform) at the state of configuration: z the viewing direction, x right, y down (ROS optical frame, as
        LeobotModel.CameraPose). Raises ValueError for other names (no camera on the arm)."""
        #the sensor modules are imported here: the student packages carry the model without them
        from labRobots.robots.kairos.cameras import OpticalHT
        return self.BodyPose(self.bodies['platform'], configuration) @ OpticalHT(name)

    def RayContact(self):
        """Return the GeneralContact the camera rays hit if the scene gives none, created at the first call: the floor
        at z = 0 and the obstacles of SetEnvironmentObstacles set before (labRobots.robotics.sensors.rayContact, as
        LeobotModel.RayContact)."""
        from labRobots.robotics.sensors.rayContact import CreateRayContact
        if self.rayContact is None:
            self.rayContact = CreateRayContact(self.mbs, self.environmentObstacles)
        return self.rayContact

    def AddCameraRays(self, name, generalContact=None, shown=True, raysPerEdge=None):
        """Show what a camera would see: rays along the edge of its image to where they hit the objects of a
        GeneralContact, and the outline of the image there (labRobots.robotics.sensors.cameraRays, as
        LeobotModel.AddCameraRays); call after CreateMBS, before mbs.Assemble(). Update
        (PreStepControl) casts them; without a running simulation call cameraRays[name].Update().

        Args:
            name: camera name of cameras.cameraNames ('base'; the Kairos has no camera on the arm: ValueError)
            generalContact: GeneralContact the rays hit (its triangles not static); None: the floor at z = 0 and the
                obstacles of SetEnvironmentObstacles (RayContact; set the obstacles before)
            shown: initially shown (CameraRays.SetShown switches)
            raysPerEdge: rays per image edge; None: cameraRays.defaultRaysPerEdge

        Returns:
            CameraRays (also in self.cameraRays[name])
        """
        from labRobots.robots.kairos.cameras import CheckCameraName, CameraImageSize, VerticalFieldOfView
        from labRobots.robotics.sensors.cameraRays import CameraRays
        CheckCameraName(name)
        if self.mbs is None:
            raise ValueError('AddCameraRays needs the model built into a MainSystem (CreateMBS) first')
        options = {} if raysPerEdge is None else {'raysPerEdge': raysPerEdge}
        contact = self.RayContact() if generalContact is None else generalContact
        self.cameraRays[name] = CameraRays(self.mbs, contact, lambda configuration: self.CameraPose(name, configuration),
                                           CameraImageSize(name), VerticalFieldOfView(name), shown=shown, **options)
        return self.cameraRays[name]

    def LaserPose(self, name, configuration=exu.ConfigurationType.Current):
        """Return the 4x4 world pose of the scan frame of a laser scanner ('laser_front', 'laser_back': the SICK S300,
        upside down as mounted; 'velodyne': the VLP-16 with the axes of base_link) at the state of configuration (as
        LeobotModel.LaserPose)."""
        from labRobots.robots.kairos.lasers import LaserHT
        return self.BodyPose(self.bodies['platform'], configuration) @ LaserHT(name)

    def AddLaserRays(self, name, generalContact=None, shown=True, numberOfBeams=None, drawnBeams=None, rings=None):
        """Show the beams of a laser scanner to where they hit the objects of a GeneralContact (labRobots.robotics.sensors.
        laserRays, as LeobotModel.AddLaserRays); call after CreateMBS, before mbs.Assemble().
        Update (PreStepControl) casts the drawn beams while shown; laserRays[name].Scan() gives the simulated 2D scan as
        LaserScan message dict with the angles, beams and ranges of the real scans (geometry.laserScanAngles,
        laserScanBeams), laserRays[name].Points() the hit points in the scan frame (3D cloud of the VLP-16 with rings).

        Args:
            name: 'laser_front', 'laser_back' (SICK S300, 270 deg) or 'velodyne' (VLP-16)
            generalContact: GeneralContact the beams hit; None: floor and the obstacles of SetEnvironmentObstacles
                (RayContact; set the obstacles before)
            shown: initially shown (LaserRays.SetShown switches)
            numberOfBeams: measured beams (per ring); None: those of the real scans (S300: 541; VLP-16: 720 of the 2D
                scan, 900 per ring with rings)
            drawnBeams: number of drawn beams (per ring), evenly spread; None: laserRays.defaultDrawnBeams
            rings: only 'velodyne': None: the 2D scan in the horizontal plane of the sensor; 'all': the 16 rings of the
                VLP-16 (geometry.velodyneElevations, -15 to 15 deg); or a list of ring indices 0..15 (3D scan)

        Returns:
            LaserRays (also in self.laserRays[name])
        """
        from labRobots.robotics.sensors.laserRays import LaserRays, defaultDrawnBeams
        from labRobots.robots.kairos.lasers import CheckLaserName
        from labRobots.robots.kairos.geometry import (laserScanAngles, laserScanBeams, laserRangeMinimum, laserRangeMaximum,
                                                      velodyneRangeMinimum, velodyneRangeMaximum, velodyneElevations,
                                                      velodyneAzimuthBeams)
        CheckLaserName(name)
        if self.mbs is None:
            raise ValueError('AddLaserRays needs the model built into a MainSystem (CreateMBS) first')
        elevations, angles, beams = None, laserScanAngles[name], laserScanBeams[name]
        rangeMinimum, rangeMaximum = laserRangeMinimum, laserRangeMaximum
        if name == 'velodyne':
            rangeMinimum, rangeMaximum = velodyneRangeMinimum, velodyneRangeMaximum
            if rings is not None:
                elevations = velodyneElevations if isinstance(rings, str) and rings == 'all' else \
                    velodyneElevations[np.asarray(rings, dtype=int)]
                beams = velodyneAzimuthBeams
                angles = (-np.pi, np.pi - 2.*np.pi/beams)
        elif rings is not None:
            raise ValueError('rings: only the VLP-16 (velodyne) has rings, the S300 %r scans in one plane' % name)
        contact = self.RayContact() if generalContact is None else generalContact
        self.laserRays[name] = LaserRays(self.mbs, contact, lambda configuration: self.LaserPose(name, configuration),
                                         angles[0], angles[1], rangeMaximum, shown=shown,
                                         numberOfBeams=beams if numberOfBeams is None else numberOfBeams,
                                         drawnBeams=defaultDrawnBeams if drawnBeams is None else drawnBeams,
                                         rangeMinimum=rangeMinimum, elevations=elevations)
        return self.laserRays[name]

    def SetControl(self, trajectory=None, platformVelocity=None):
        """the control of the model for PreStepControl, as Robot.SetControl

        Args:
            trajectory: arm trajectory, an exudyn.robotics.motion.Trajectory (or an object with Evaluate(t) returning
                [q, q_t, q_tt] and GetTimes()); None: keep the present one
            platformVelocity: function t -> local platform velocity [vx in m/s, vy in m/s, omega in rad/s]; None: keep
                the present one (at the start: standstill)
        """
        if trajectory is not None:
            self.SetArmTrajectory(trajectory, armCommandPath)
        if platformVelocity is not None:
            self.platformVelocity = platformVelocity

    def PreStepControl(self, mbs, t):
        """the PreStepUserFunction of the control of SetControl, as Robot.PreStepControl: pass it directly,
        mbs.SetPreStepUserFunction(model.PreStepControl), or call it in the PreStepUserFunction of the scene for several
        robots

        Args:
            mbs: the MainSystem of the model
            t: the time

        Returns:
            True
        """
        self.Update(t, self.platformVelocity(t))
        return True


def CreateKairos(mbs, name='kairos', referenceHT=None, q0=None, markerGround=None, generalContact=None):
    """Create the Kairos in a MainSystem in one call, as mbs.CreateRigidBody: the description
    KairosModel, built by CreateMBS.

    Args:
        mbs: the MainSystem, which will be extended
        name: name of the KinematicTree object
        referenceHT: HT of the platform reference point (center of the platform on the floor, x forward, z up) in the
            world (Vicon) frame, an exudyn.HT or 4x4 matrix; only x, y and the rotation about z; None: the origin
        q0: UR5 joint angles in rad at the start; None: par.armInitialAngles
        markerGround: rigid body marker of the floor on which the wheels roll (scene); None: a ground object at the
            origin without graphics
        generalContact: GeneralContact of the scene for the gripper (grasping scene objects); None: no gripper contact

    Returns:
        KairosModel: the built Kairos with the dict of the created items in kairos.items (see CreateMBS); its SetControl,
        PreStepControl, planning and state functions act on it

    Example:
        kairos = CreateKairos(mbs, referenceHT=HTtranslate([1., 0., 0.]))
        mbs.SetPreStepUserFunction(kairos.PreStepControl)
    """
    kairos = KairosModel(base=RobotBase(HT=referenceHT), q0=q0)
    kairos.CreateMBS(mbs, name=name, markerGround=markerGround, generalContact=generalContact)
    return kairos
