# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: armModel.py
#
# Details:  An arm of the lab as Exudyn model: ArmModel is the base of all arm models (the 6R, the Staubli TX2-90L,
#           the Fanuc LR Mate 200iD/7L): its description (an exudyn.robotics.Robot of the subclass) built by
#           CreateKinematicTree into the MainSystem of a scene, the base fixed in the world or mounted on any rigid marker
#           (the base pose follows it); PD control with gravity or inverse dynamics compensation (Robot.SetControl), the
#           collision-free motions of its ArmPlanner with the obstacles of the scene, collision display.
#           IndustrialArmModel gives the description by a robot dictionary of exudyn.robotics.models with the meshes of
#           its URDF (AddURDFgraphics) and an optional parallel gripper (ArmWithTool); Robot6RModel
#           (labRobots/robots/robot6R/model.py) by the DH model and the CAD data of the 6R.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-13; 2026-11-20 ArmModel as the base of IndustrialArmModel and Robot6RModel (was industrialArm/model.py)
# Notes:    A parallel gripper adds its jaws as prismatic links of the kinematic tree after the
#           arm links, as the jaws of the 6R; its jaw coordinate is the travel of each jaw from the closed width, moved
#           smoothly to the commanded width (SetGripperWidth); the arm planner keeps the arm links only (the collision
#           shapes of the jaws stand on the last link at the open width). Mounting needs the inertia of the base
#           (FloatingBaseInertia of the subclass).
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Arms of the lab as Exudyn models."""
import copy
import numpy as np
import exudyn as exu
import exudyn.graphics as graphics
from exudyn.robotics import RobotBase, RobotLink, VRobotLink, RobotTool
from exudyn.robotics.models import LinkDict2Robot
from exudyn.rigidBodyUtilities import ComputeOrthonormalBasisVectors
from exudyn.robotics.motion import Trajectory, ProfileConstantAcceleration
from exudyn.robotics.urdfImport import AddURDFgraphics
from labRobots.labSetup.dataPaths import urdfFolder
from labRobots.robotics.collision.shapes import TransformShape
from labRobots.robotics.collision.exudynShapes import CollisionShapeOfRobotics
from labRobots.robotics.modeling.mounting import RotateFixedBaseIntoTree, MountBase, HideHelperGraphics
from labRobots.arms.armPlanning import IndustrialArmPlanner, InitialConfiguration

#! Duration in s of the profile holding the arm; joint distance in rad below which a target is the present angles.
holdDuration, holdTolerance = 0.1, 1e-9
#! Base motion (change of the base pose matrix) after which the obstacles are transformed into the base frame again.
obstaclePoseTolerance = 1e-6
#! Value of compensation in SetControl: the compensation of the model class (ArmModel.defaultCompensation).
modelCompensation = 'model'
#! Gripper jaw link: mass in kg, principal moment of inertia in kg m^2 (small parts of
#! aluminium; estimates), PD control stiffness in N/m and damping ratio of the jaw coordinate.
jawMass, jawInertia, jawStiffness, jawDampingRatio = 0.05, 1e-5, 2000., 0.7


class ArmModel:
    """An arm as KairosModel and LeobotModel (the base of IndustrialArmModel and Robot6RModel):
    the description (CreateDescription of the subclass, an exudyn.robotics.Robot in self.robot), CreateMBS builds it
    into the MainSystem of a scene (base fixed in the world or mounted on markerGround); collision-free arm motions with
    the obstacles of the scene (self.armPlanner, an ArmPlanner), control (SetControl, PreStepControl), collision display."""

    #! Compensation of Robot.SetControl ('static', 'dynamic' or None) if SetControl is called without one.
    defaultCompensation = 'dynamic'
    #! Name of the collision shapes of the base in CollisionShapes.
    baseShapeName = 'base'
    #! Name of the KinematicTree object of CreateMBS.
    defaultTreeName = 'arm'

    def __init__(self, armPlannerInit, numberOfJointsInit, baseInit, q0Init):
        """the description of the arm

        Args:
            armPlannerInit: ArmPlanner of the arm (kinematics and collision shapes in the base frame).
            numberOfJointsInit: Number of arm joints (without the jaws of a gripper).
            baseInit: RobotBase with HT, the pose of the base frame in the world; None: base frame = world frame.
            q0Init: Arm joint angles in rad at the start.
        """
        self.base = RobotBase() if baseInit is None else baseInit   #! Base frame of the arm (RobotBase.HT).
        self.q0 = np.asarray(q0Init, dtype=float)   #! Arm joint angles in rad at the start.
        self.numberOfJoints = numberOfJointsInit    #! Number of arm joints.
        self.robot = None   #! exudyn.robotics.Robot built by CreateMBS, None before.
        self.mbs = None     #! MainSystem of the model created by CreateMBS, None before.
        self.items = None   #! Dict of the created items, None before.
        self.armPlanner = armPlannerInit        #! Collision-free arm motions (collision model of Exudyn).
        self.environmentObstacles = []          #! Obstacles of the scene (shapes of collision.py) in world coordinates.
        #! World pose (4x4) of the base frame at the last UpdateCollisionObstacles (the obstacles of the planner are in
        #! this frame; at the start the reference pose, the planner has no obstacles), None: to be transformed.
        self.collisionObstaclesBasePose = np.array(self.base.HT)
        self.armTarget = self.q0.copy()         #! Last commanded arm angles in rad.

    def CreateDescription(self):
        """Return the exudyn.robotics.Robot of the arm for CreateMBS (base: a copy of self.base, q0 of all links);
        given by the subclass."""
        raise NotImplementedError('%s: CreateDescription is given by the subclass of ArmModel' % type(self).__name__)

    def FloatingBaseInertia(self):
        """Return the RigidBodyInertia of the base for a mounted arm (floatingBase of CreateKinematicTree); the subclass
        gives it if its arm can be mounted, here: ValueError."""
        raise ValueError('%s: the arm cannot be mounted (no inertia of its base); build it with markerGround=None'
                         % type(self).__name__)

    def ArmTrajectory(self, armTrajectory):
        """Return the control trajectory of all links of the tree for a trajectory of the arm joints: the trajectory
        itself (a subclass with jaws adds their coordinates)."""
        return armTrajectory

    def CreateMBS(self, mbs, name=None, markerGround=None):
        """build the arm into an existing MainSystem: Robot.CreateKinematicTree of the description, the arm holding q0

        Args:
            mbs: the MainSystem, which will be extended
            name: name of the KinematicTree object; None: defaultTreeName
            markerGround: rigid marker (of any body: ground, rigid or flexible body, link of another robot) to which the
                base is fixed in its pose; the base becomes a free rigid body (floatingBase of CreateKinematicTree) held
                by a joint at the marker; None: the base fixed in the world

        Returns:
            dict with 'nodeGeneric', 'objectKinematicTree', 'nodeBase', 'objectBase' (of CreateKinematicTree) and
            'jointBase' (of the mount, or None)
        """
        if self.mbs is not None:
            raise ValueError('%s.CreateMBS: the model is already built; create a model per robot' % type(self).__name__)
        name = self.defaultTreeName if name is None else name
        floatingBase = None if markerGround is None else self.FloatingBaseInertia()
        self.mbs = mbs
        #CreateDescription takes a copy of the base: RotateFixedBaseIntoTree changes it, self.base keeps the pose of the
        #base frame (ArmBasePose, planning)
        self.robot = self.CreateDescription()
        if markerGround is None:
            RotateFixedBaseIntoTree(self.robot)
        items = self.robot.CreateKinematicTree(mbs, name=name, floatingBase=floatingBase)
        HideHelperGraphics(mbs)
        #Exudyn 1.13.530: CreateKinematicTree does not pass name to the object (its docstring says it does)
        mbs.SetObjectParameter(items['objectKinematicTree'], 'name', name)
        self.robot.items = items
        items['jointBase'] = None if markerGround is None else MountBase(mbs, self.robot, markerGround)
        self.items = items
        self.HoldArm(self.q0, 0.)
        return self.items

    def SetControl(self, trajectory, compensation=modelCompensation):
        """the control of the model for PreStepControl, as Robot.SetControl of Exudyn

        Args:
            trajectory: trajectory with Evaluate(t) returning [q, q_t, q_tt] of all links of the tree (arm joints and
                the jaws of a gripper; ArmTrajectory)
            compensation: compensation of Robot.SetControl ('static', 'dynamic' or None); modelCompensation: the
                defaultCompensation of the model class
        """
        self.robot.SetControl(trajectory, compensation=self.defaultCompensation if compensation == modelCompensation
                              else compensation)

    def FollowArmTrajectory(self, armTrajectory):
        """Let the arm follow a trajectory of the arm joints (with the jaws of ArmTrajectory), the compensation of the
        model class; used by HoldArm and MoveArmToJoints."""
        ArmModel.SetControl(self, self.ArmTrajectory(armTrajectory))   #not a SetControl of the subclass: no second ArmTrajectory

    def PreStepControl(self, mbs, t):
        """the PreStepUserFunction of the control, as Robot.PreStepControl; returns True"""
        return self.robot.PreStepControl(mbs, t)

    def HoldArm(self, angles, time):
        """Hold the arm at joint angles in rad from time on."""
        self.armTarget = np.asarray(angles, dtype=float).copy()
        hold = Trajectory(initialCoordinates=list(self.armTarget), initialTime=time)
        hold.Add(ProfileConstantAcceleration(list(self.armTarget), holdDuration))
        self.FollowArmTrajectory(hold)

    def MoveArmToJoints(self, targetAngles, time, speedScale=1.):
        """Plan a collision-free motion from the present angles to joint angles in rad (Exudyn: RRT-Connect, shortcuts,
        blended trajectory, within the speed limits of the planner) and let the arm follow it; raises ValueError if the
        target collides or no path is found."""
        self.UpdateCollisionObstacles()
        angles, targetAngles = self.JointAngles(), np.asarray(targetAngles, dtype=float)
        if np.allclose(angles, targetAngles, rtol=0., atol=holdTolerance):
            self.HoldArm(targetAngles, time)
            return
        trajectory, unusedPath = self.armPlanner.Plan(angles, targetAngles, time, speedScale)
        self.FollowArmTrajectory(trajectory)
        self.armTarget = targetAngles

    def JointAngles(self, configuration=exu.ConfigurationType.Current):
        """Return the arm joint angles in rad (without the jaw coordinates of a gripper)."""
        return np.asarray(self.mbs.GetNodeOutput(self.items['nodeGeneric'], exu.OutputVariableType.Coordinates,
                                                 configuration=configuration), dtype=float)[0:self.numberOfJoints]

    def BasePose(self, configuration=exu.ConfigurationType.Current):
        """Return the world pose (4x4) of the base frame (moving with the body it is mounted on)."""
        if self.items is None or self.items.get('objectBase') is None or not self.mbs.systemIsConsistent:
            return np.array(self.base.HT)       #fixed base, or before mbs.Assemble: the reference pose
        pose = np.eye(4)
        pose[0:3, 3] = self.mbs.GetObjectOutputBody(self.items['objectBase'], exu.OutputVariableType.Position, [0., 0., 0.],
                                                    configuration)
        pose[0:3, 0:3] = np.asarray(self.mbs.GetObjectOutputBody(self.items['objectBase'], exu.OutputVariableType.RotationMatrix,
                                                                 [0., 0., 0.], configuration)).reshape(3, 3)
        return pose

    def ArmBasePose(self, configuration=exu.ConfigurationType.Current):
        """Return the world pose (4x4) of the arm base frame = BasePose, the frame of armPlanner, as
        KairosModel.ArmBasePose and LeobotModel.ArmBasePose."""
        return self.BasePose(configuration)

    def ToolPose(self, armAngles=None):
        """Return the world pose (4x4) of the tool frame of armPlanner at arm angles in rad (None: the present angles)."""
        angles = self.JointAngles() if armAngles is None else np.asarray(armAngles, dtype=float)
        return self.ArmBasePose() @ np.array(self.armPlanner.robot.GetKinematicChain().ToolHT(angles))

    def JointAnglesAtToolPose(self, toolPose, startAngles):
        """Return the collision-free joint angles in rad at which the tool frame is at a 4x4 pose in world coordinates,
        as KairosModel and LeobotModel: the pose in the arm base frame (ArmBasePose), the free solution of the inverse
        kinematics nearest to startAngles (armPlanner.JointAnglesAtPose); raises ValueError if there is none. Its result
        is a goal of armPlanner.Plan or MoveArmToJoints.

        Args:
            toolPose: 4x4 pose of the tool frame in world coordinates
            startAngles: joint angles in rad from which the arm moves (the nearest solution is taken)

        Returns:
            joint angles in rad
        """
        self.UpdateCollisionObstacles()
        armPose = np.linalg.inv(self.ArmBasePose()) @ np.asarray(toolPose, dtype=float)
        return self.armPlanner.JointAnglesAtPose(armPose, startAngles)

    def ToolPoseConfigurations(self, toolPose, referenceAngles):
        """Return all free joint angle configurations in rad (armPlanner.JointAnglesConfigurations) at which the tool
        frame is at a 4x4 pose in world coordinates, nearest to referenceAngles first."""
        self.UpdateCollisionObstacles()
        armPose = np.linalg.inv(self.ArmBasePose()) @ np.asarray(toolPose, dtype=float)
        return self.armPlanner.JointAnglesConfigurations(armPose, referenceAngles)

    def SetEnvironmentObstacles(self, obstacles):
        """set the obstacles of the scene for the collision-free arm motions, as KairosModel.SetEnvironmentObstacles

        Args:
            obstacles: list of shapes of labRobots.robotics.collision.shapes in world coordinates (e.g. scene['obstacles']);
                transformed into the base frame at the present base pose, again by UpdateCollisionObstacles when the
                base moved
        """
        self.environmentObstacles = [dict(shape) for shape in obstacles]
        self.collisionObstaclesBasePose = None
        self.UpdateCollisionObstacles()

    def UpdateCollisionObstacles(self):
        """Set the obstacles of the planner for the present base pose; only if the base moved or the obstacles changed
        since the last update."""
        basePose = self.BasePose() if self.mbs is not None and self.mbs.systemIsConsistent else np.array(self.base.HT)
        if (self.collisionObstaclesBasePose is not None and
                np.max(np.abs(basePose - self.collisionObstaclesBasePose)) <= obstaclePoseTolerance):
            return
        worldToBase = np.linalg.inv(basePose)
        self.armPlanner.SetObstacles([TransformShape(shape, worldToBase) for shape in self.environmentObstacles])
        self.collisionObstaclesBasePose = basePose

    def CollisionShapes(self, configuration=exu.ConfigurationType.Current, angles=None):
        """Return (arm shapes, other shapes) of the collision model in world coordinates at the state of configuration
        (labRobots.robotics.collision.display): the shapes of the links (with the tool), of the base (named
        baseShapeName) and the obstacles of the scene where the planner has them; angles: joint angles in rad instead of
        those of configuration (the arm as obstacle of another robot, also before mbs.Assemble)."""
        basePose = self.BasePose(configuration)
        angles = self.JointAngles(configuration) if angles is None else np.asarray(angles, dtype=float)
        frames = [np.array(frame) for frame in self.armPlanner.robot.GetKinematicChain().LinkHT(angles)]
        armShapes = [TransformShape(CollisionShapeOfRobotics(shape, 'link%d' % (linkIndex + 1)), basePose @ frames[linkIndex])
                     for linkIndex, link in enumerate(self.armPlanner.robot.links) for shape in link.collisionShapes]
        otherShapes = [TransformShape(CollisionShapeOfRobotics(shape, self.baseShapeName), basePose)
                       for shape in self.armPlanner.robot.base.collisionShapes]
        if self.collisionObstaclesBasePose is not None:
            otherShapes += [TransformShape(shape, self.collisionObstaclesBasePose) for shape in self.armPlanner.obstacles]
        return armShapes, otherShapes

    def CollisionDistance(self, configuration=exu.ConfigurationType.Current):
        """Return (smallest distance in m of the collision model at the arm angles of configuration, its margin in m)."""
        return self.armPlanner.MinimumDistance(self.JointAngles(configuration)), self.armPlanner.margin


class ArmAndJawsTrajectory:
    """The control trajectory of an arm with gripper jaws (2026-11-20 also for the 6R): the arm
    trajectory followed by the jaw coordinates of a jaw reference; interface of exudyn.robotics.motion.Trajectory
    (Evaluate, GetTimes)."""

    def __init__(self, armTrajectoryInit, JawReferenceInit, numberOfJawsInit):
        """initialize ArmAndJawsTrajectory

        Args:
            armTrajectoryInit: Trajectory with Evaluate(t) returning [q, q_t, q_tt] of the arm joints.
            JawReferenceInit: Function of the simulation time returning (jaw coordinate in m, its velocity in m/s, its
                acceleration in m/s^2), the same for every jaw.
            numberOfJawsInit: Number of jaw links.
        """
        self.armTrajectory = armTrajectoryInit      #! Trajectory of the arm joints.
        self.JawReference = JawReferenceInit        #! Jaw coordinate, velocity and acceleration at a time.
        self.numberOfJaws = numberOfJawsInit        #! Number of jaw links.

    def Evaluate(self, time):
        """Return [q, q_t, q_tt] of arm and jaws at a simulation time in s."""
        arm = [np.asarray(value, dtype=float) for value in self.armTrajectory.Evaluate(time)]
        return [np.r_[armValue, np.full(self.numberOfJaws, jawValue)] for armValue, jawValue in zip(arm, self.JawReference(time))]

    def GetTimes(self):
        """Return the times of the arm trajectory."""
        return self.armTrajectory.GetTimes()


def CreateArmDescription(robotDict, base=None, q0=None):
    """Return the exudyn.robotics.Robot of a robot dictionary with the meshes of its URDF.

    Args:
        robotDict: robot dictionary of exudyn.robotics.models with 'urdfGraphics' ('file' relative to
            labRobots.labSetup.dataPaths.urdfFolder, 'baseLink', 'links', optional 'linkHTs': the URDF link frame in each
            robot link frame where they differ, optional 'color': one RGBA color for all meshes)
        base: RobotBase with HT, the pose of the base frame in the world; None: base frame = world frame
        q0: joint angles in rad at the start; None: the initial configuration of the dictionary

    Returns:
        the Robot (PD gains, joint limits, collision shapes, tool of the dictionary; graphics of the URDF; the jaws of a
        'gripper' of ArmWithTool as prismatic links after the arm links, starting at the open width)
    """
    robot = LinkDict2Robot(robotDict, initialConfiguration=InitialConfiguration(robotDict) if q0 is None else list(q0))
    if base is not None:
        robot.base.HT = base.HT
    description = robotDict['urdfGraphics']
    AddURDFgraphics(robot, str(urdfFolder / description['file']), description['links'], baseLinkName=description['baseLink'],
                    color=description.get('color'))         #None: the colors of the URDF materials
    for link, HT in zip(robot.links, description.get('linkHTs', [])):
        HT = np.array(HT)
        link.visualization.graphicsData = [graphics.Move(item, HT[0:3, 3].tolist(), HT[0:3, 0:3].tolist())
                                           for item in link.visualization.graphicsData]
    robot.links[-1].visualization.graphicsData = list(robot.links[-1].visualization.graphicsData) + list(robotDict.get('toolGraphics', []))
    gripper = robotDict.get('gripper')
    if gripper is not None:         #the jaws on the last arm link
        lastArmLink = len(robot.links) - 1
        #Exudyn allows a tool only on a serial robot: the tool frame lives in armPlanner.robot (ToolPose, inverse
        #kinematics), the tree has the jaws instead
        robot.tool = RobotTool()
        jawDamping = 2.*jawDampingRatio*np.sqrt(jawStiffness*jawMass)
        for jaw in gripper['jaws']:
            robot.AddLink(RobotLink(mass=jawMass, COM=[0., 0., 0.], inertia=jawInertia*np.eye(3), jointType='Px',
                                    parent=lastArmLink, preHT=jaw['preHT'], PDcontrol=(jawStiffness, jawDamping),
                                    visualization=VRobotLink(showMBSjoint=False, showCOM=False, graphicsData=jaw['graphics'])))
        robot.q0 = list(robot.q0) + [JawCoordinate(gripper, gripper['openWidth'])]*len(gripper['jaws'])
    return robot


def JawCoordinate(gripper, width):
    """Return the jaw coordinate in m (travel of each jaw from the closed width) of a gripper width in m."""
    return 0.5*(width - gripper['closedWidth'])


def JawFrame(direction):
    """Return the 4x4 rotation of a jaw link frame whose x axis is the opening direction of the jaw (a unit vector)."""
    pose = np.eye(4)
    pose[0:3, 0:3] = np.column_stack(ComputeOrthonormalBasisVectors(np.asarray(direction, dtype=float)))
    return pose


def ArmWithTool(robotDict, toolGraphics, toolShapes, toolHT, gripper=None):
    """Return a copy of a robot dictionary with a tool (e.g. a gripper) on its flange: the graphics and the collision
    capsules of the tool on the last link, the tool frame moved to the tool center point;
    optionally the moving jaws of a parallel gripper.

    Args:
        robotDict: robot dictionary of exudyn.robotics.models; its 'tool' (or the last link frame) is the flange
        toolGraphics: Exudyn graphics of the tool in the flange frame
        toolShapes: collision shapes of the tool in the flange frame, as exudyn.robotics: capsules ({'type': 'capsule',
            'pAxis', 'vAxis', 'radius'}, e.g. of FitCapsulesToMesh) or boxes ({'type': 'box', 'size', 'HT'}, e.g.
            nativeCollision.RoboticsBoxShape of collision.BoxFromPoints)
        toolHT: 4x4 pose of the tool center point in the flange frame
        gripper: None (the tool does not move) or a parallel gripper: dict with 'jaws' (list of {'graphics': Exudyn
            graphics of the jaw in the flange frame at the closed width, 'direction': unit vector of its opening in the
            flange frame}; not in toolGraphics), 'closedWidth' and 'openWidth' (m, between the jaws), 'motionTime' (s of
            a full stroke); toolShapes cover the jaws at the open width

    Returns:
        the dictionary with 'toolGraphics' (in the last link frame, added by CreateArmDescription), the shapes in
        'collisionShapes' of the last link, 'tool' at the tool center point and, with a gripper, 'gripper' (its jaws as
        'preHT' of the jaw link in the last link frame and 'graphics' in the jaw link frame)
    """
    flange = np.array(robotDict['tool']['HT']) if 'tool' in robotDict else np.eye(4)
    result = dict(robotDict)
    result['links'] = [dict(link) for link in robotDict['links']]
    lastLink = result['links'][-1]
    lastLink['collisionShapes'] = list(lastLink.get('collisionShapes', [])) + [ShapeInFrame(shape, flange) for shape in toolShapes]
    result['toolGraphics'] = list(robotDict.get('toolGraphics', [])) + [
        graphics.Move(item, flange[0:3, 3].tolist(), flange[0:3, 0:3].tolist()) for item in toolGraphics]
    tool = flange @ np.asarray(toolHT, dtype=float)
    result['tool'] = {'HT': exu.HT(rotation=tool[0:3, 0:3].tolist(), translation=tool[0:3, 3].tolist())}
    if gripper is not None:
        jaws = []
        for jaw in gripper['jaws']:
            jawPose = flange @ JawFrame(jaw['direction'])
            jaws.append({'preHT': jawPose, 'graphics': [graphics.Move(item, [0., 0., 0.], JawFrame(jaw['direction'])[0:3, 0:3].T.tolist())
                                                        for item in jaw['graphics']]})
        result['gripper'] = dict(gripper, jaws=jaws)
    return result


def ShapeInFrame(shape, HT):
    """Return a collision shape of exudyn.robotics (capsule or box dict) given in a frame, expressed in the parent frame of
    the 4x4 pose HT of that frame."""
    HT = np.asarray(HT, dtype=float)
    if shape['type'] == 'box':
        pose = HT @ np.array(shape.get('HT', np.eye(4)))
        return dict(shape, HT=exu.HT(rotation=pose[0:3, 0:3].tolist(), translation=pose[0:3, 3].tolist()))
    return {'type': 'capsule', 'pAxis': (HT[0:3, 0:3] @ np.asarray(shape['pAxis'], dtype=float) + HT[0:3, 3]).tolist(),
            'vAxis': (HT[0:3, 0:3] @ np.asarray(shape['vAxis'], dtype=float)).tolist(), 'radius': float(shape['radius'])}


class IndustrialArmModel(ArmModel):
    """An arm given by a robot dictionary of exudyn.robotics.models (Staubli, Fanuc), with an optional parallel gripper
    (ArmWithTool); the base is fixed in the world (no inertia of the base in the dictionary)."""

    defaultTreeName = 'industrialArm'

    def __init__(self, robotDict, base=None, q0=None):
        """the description of the arm

        Args:
            robotDict: robot dictionary of exudyn.robotics.models (see CreateArmDescription)
            base: RobotBase with HT, the pose of the base frame in the world; None: base frame = world frame
            q0: joint angles in rad at the start; None: the initial configuration of the dictionary
        """
        ArmModel.__init__(self, IndustrialArmPlanner(robotDict), len(robotDict['links']), base,
                          InitialConfiguration(robotDict) if q0 is None else q0)
        self.robotDict = robotDict                 #! Robot dictionary of the arm.
        self.gripper = robotDict.get('gripper')   #! Parallel gripper of ArmWithTool, None: no moving jaws.
        openWidth = None if self.gripper is None else self.gripper['openWidth']
        #! Commanded gripper width in m, the width before that command and the simulation time of the command.
        self.gripperWidth, self.previousGripperWidth, self.gripperCommandTime = openWidth, openWidth, 0.

    def CreateDescription(self):
        """Return the Robot of the robot dictionary (CreateArmDescription) at a copy of the base, holding q0."""
        return CreateArmDescription(self.robotDict, base=copy.deepcopy(self.base), q0=self.q0)

    def ArmTrajectory(self, armTrajectory):
        """Return the control trajectory of the tree: with a gripper the jaws follow the commanded width
        (ArmAndJawsTrajectory), without the arm trajectory itself."""
        if self.gripper is None:
            return armTrajectory
        return ArmAndJawsTrajectory(armTrajectory, self.JawReference, len(self.gripper['jaws']))

    def SetControl(self, trajectory, compensation=modelCompensation):
        """the control of the model for PreStepControl, as Robot.SetControl of Exudyn

        Args:
            trajectory: trajectory with Evaluate(t) returning [q, q_t, q_tt] of the arm joints (with a gripper, the
                jaws follow the commanded width, ArmTrajectory)
            compensation: compensation of Robot.SetControl ('static', 'dynamic' or None); modelCompensation: 'dynamic'
        """
        ArmModel.SetControl(self, self.ArmTrajectory(trajectory), compensation)

    def SetGripperWidth(self, width, time):
        """Command the gripper to a width in m between the jaws from a simulation time in s on; the jaws move there
        smoothly within the gripper's motionTime (for a full stroke); raises ValueError without gripper or outside the
        stroke."""
        if self.gripper is None:
            raise ValueError('IndustrialArmModel.SetGripperWidth: the arm has no gripper')
        if not self.gripper['closedWidth'] - holdTolerance <= width <= self.gripper['openWidth'] + holdTolerance:
            raise ValueError('IndustrialArmModel.SetGripperWidth: width %g m outside the stroke %g .. %g m'
                             % (width, self.gripper['closedWidth'], self.gripper['openWidth']))
        self.previousGripperWidth = self.CommandedGripperWidth(time)[0]     #from where the jaws are heading now
        self.gripperWidth, self.gripperCommandTime = float(width), time

    def CommandedGripperWidth(self, time):
        """Return (width in m, its velocity in m/s, its acceleration in m/s^2) of the reference of the jaws at a simulation
        time: a cosine motion from the previous to the commanded width, its duration the share of the stroke in
        motionTime."""
        stroke = self.gripper['openWidth'] - self.gripper['closedWidth']
        change = self.gripperWidth - self.previousGripperWidth
        duration = self.gripper['motionTime']*abs(change)/stroke
        if duration <= 0. or time >= self.gripperCommandTime + duration:
            return self.gripperWidth, 0., 0.
        phase = np.pi*max(time - self.gripperCommandTime, 0.)/duration
        return (self.previousGripperWidth + 0.5*change*(1. - np.cos(phase)), 0.5*change*np.pi/duration*np.sin(phase),
                0.5*change*(np.pi/duration)**2*np.cos(phase))

    def JawReference(self, time):
        """Return (jaw coordinate in m, velocity, acceleration) of the commanded gripper width at a simulation time
        (the jaw reference of ArmAndJawsTrajectory)."""
        width, velocity, acceleration = self.CommandedGripperWidth(time)
        return JawCoordinate(self.gripper, width), 0.5*velocity, 0.5*acceleration

    def GripperWidth(self, configuration=exu.ConfigurationType.Current):
        """Return the width in m between the jaws of the model (None without gripper)."""
        if self.gripper is None:
            return None
        coordinates = np.asarray(self.mbs.GetNodeOutput(self.items['nodeGeneric'], exu.OutputVariableType.Coordinates,
                                                        configuration=configuration), dtype=float)
        return self.gripper['closedWidth'] + 2.*float(np.mean(coordinates[self.numberOfJoints:]))
