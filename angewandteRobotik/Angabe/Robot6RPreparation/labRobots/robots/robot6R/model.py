# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: model.py
#
# Details:  Kinematic and dynamic model of the 6R robot (Schunk PW 90 for axes
#           1/2, Schunk PW 70 for axes 3/4, servo driven differential wrist for
#           axes 5/6): the kinematics of the DH model by Exudyn's KinematicChain
#           (forward kinematics, frames, geometric Jacobian, analytic inverse
#           kinematics), the joint and Cartesian checks of the lab software
#           (check_input.m), inertia data from the CAD model, the mass matrix by
#           the inverse dynamics and the creation of an exudyn.robotics.Robot
#           including mesh graphics and the two gripper jaws as prismatic links
#           (kinematic tree).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-03; 2026-11-12 moved from LabCourse/Robot6R/robot6R/model.py (the lab objects and the
#           cup stay in the course: robot6R/labObjects.py)
# Notes:    Joint frames J_i are the frames after the joint rotation q_i (used
#           by Exudyn's KinematicTree for graphics and inertia); link frames
#           are the standard DH frames T_i (T_6 = TCP frame of the gripper).
#           The CAD based data (robotModelData.json and meshes of the device data) was created by
#           tools/convertCadToMeshes.py of the course (removed 2026-11-20, the data are converted).
#           kinematics and mass matrix by Exudyn 1.13.402
#           (KinematicChain), the robotics toolbox is no longer used.
#           Robot6RModel is an ArmModel (labRobots.arms.armModel, shared with
#           the Staubli and the Fanuc): the 6R gives its description, the inertia of its base and
#           the gripper contact; building, mounting, control, planning and collision display are shared.
#
# License:  BSD-3 license
#
# Installation:
#           pip install exudyn
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import copy
import json
from functools import lru_cache
import numpy as np

import exudyn.graphics as graphics
from exudyn.rigidBodyUtilities import RigidBodyInertia, InertiaCuboid, InertiaCylinder, InverseHT
from exudyn.robotics import Robot, RobotLink, VRobotLink, RobotBase, VRobotBase, RobotTool, StdDH2HT
from exudyn.itemInterface import MarkerKinematicTreeRigid

from labRobots.labSetup.dataPaths import robot6RDescriptionFolder
from labRobots.robots.robot6R.jointLimits import CheckJointLimits
from labRobots.robots.robot6R import parameters as par
from labRobots.robots.robot6R.gripper import JawOffset
from labRobots.robotics.contact.generalContact import AddContactBoxes, AddContactMesh
from labRobots.arms.armModel import ArmModel, ArmAndJawsTrajectory

#! CAD based data of the robot (links, robot meshes, gripper jaws; mektcommon2/data/Exudyn/robotics).
robotModelDataFileName = os.path.join(str(robot6RDescriptionFolder), 'exudyn', 'robotModelData.json')

#configurations used to estimate the maximum diagonal entries of the mass matrix for control gains
gainDesignConfigurations = [[0., 0., 0., 0., 0., 0.],
                            [0., 0.5*np.pi, 0., 0., 0., 0.],
                            [0., 0.25*np.pi, 0.5*np.pi, 0., 0.5*np.pi, 0.],
                            [0., 0.5*np.pi, -0.5*np.pi, 0.5*np.pi, 0.5*np.pi, 0.]]


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#kinematics by Exudyn's KinematicChain (C++) of the DH model
def GetLinkLocalHT(linkIndex):
    """Return the standard DH transformation (exudyn.HT) of link linkIndex (0..5) without joint rotation q.

    Args:
        linkIndex (int): link index 0..5

    Returns:
        exudyn.HT: RotZ(theta offset) * TransZ(d) * TransX(a) * RotX(alpha)
    """
    return StdDH2HT([par.jointAngleOffsets[linkIndex], par.linkOffsetsD[linkIndex],
                     par.linkLengthsA[linkIndex], par.linkTwistsAlpha[linkIndex]])


@lru_cache(maxsize=None)
def GetKinematicRobot():
    """Return the exudyn.robotics.Robot of the DH kinematics only (6 revolute links, base K0, tool = DH frame 6);
    used for the kinematics and the Cartesian profiles of exudyn.robotics.motion. Created once, do not change it."""
    robot = Robot(gravity=par.gravity, base=RobotBase(), tool=RobotTool(), referenceConfiguration=[])
    for linkIndex in range(par.numberOfJoints):
        #no inertia: kinematics only (the dynamic model is ManipulatorRobot6R)
        robot.AddLink(RobotLink(mass=0., COM=np.zeros(3), inertia=np.zeros((3, 3)), localHT=GetLinkLocalHT(linkIndex)))
    return robot


@lru_cache(maxsize=None)
def GetKinematicChain():
    """Return the exudyn.cUtilities.KinematicChain of GetKinematicRobot(): ToolHT, JointHT, LinkHT, Jacobian,
    InverseKinematicsAnalytic (spherical wrist: up to 8 solutions) and ArmConfiguration [shoulder, elbow, wrist];
    for this robot elbow = sign(q3) (-1: elbow up as in lab task 2) and wrist = sign(q5)."""
    return GetKinematicRobot().GetKinematicChain()


def ComputeJointAndLinkFrames(q):
    """Compute joint frames (after joint rotation) and DH link frames in base frame K0.

    Args:
        q (array): 6 joint angles in rad

    Returns:
        tuple: (jointFrames, linkFrames), lists of six 4x4 homogeneous transformations (np.array);
               linkFrames[5] is the TCP frame
    """
    chain = GetKinematicChain()
    jointFrames = [np.array(frame) for frame in chain.JointHT(q)]
    linkFrames = [np.array(frame) for frame in chain.LinkHT(q)]
    return jointFrames, linkFrames


def ForwardKinematics(q):
    """Return 4x4 homogeneous transformation (np.array) of the TCP (gripper frame, frame 6) in base frame K0 for joint
    angles q (rad)."""
    return np.array(GetKinematicChain().ToolHT(q))


def GeometricJacobian(q):
    """Compute geometric Jacobian at the TCP in base frame coordinates (KinematicChain.Jacobian).

    Args:
        q (array): 6 joint angles in rad

    Returns:
        np.array: 6x6 matrix [v; omega] = J @ q_t
    """
    return np.array(GetKinematicChain().Jacobian(q))


def CheckLabConfiguration(q):
    """Replicate the input check of the lab software (check_input.m) for joint angles q (rad).

    Args:
        q (array): 6 joint angles in rad

    Returns:
        tuple: (isValid, message); the motion is rejected in the lab if isValid is False;
               a non-empty message with isValid True is a warning
    """
    isValid, message = CheckJointLimits(q)   #joint limits of the robot (labRobots.robots.robot6R.jointLimits)
    if not isValid:
        return isValid, message

    linkFrames = ComputeJointAndLinkFrames(q)[1]
    checkedPoints = [('TCP', linkFrames[5][0:3, 3]), ('wrist center', linkFrames[4][0:3, 3]),
                     ('axis 3/4', linkFrames[2][0:3, 3])]
    for pointIndex, (pointName, position) in enumerate(checkedPoints):
        if position[0] < par.labMinimumX:
            return False, pointName + ' x-coordinate < ' + str(par.labMinimumX) + ' m'
        if position[1] < par.labMinimumY:
            return False, pointName + ' y-coordinate < ' + str(par.labMinimumY) + ' m'
        if position[2] < par.labMinimumZ:
            return False, pointName + ' z-coordinate < ' + str(par.labMinimumZ) + ' m'
        if pointIndex == 0 and position[2] < par.labWarningZ:
            #check_input.m returns here with success (wrist and axis 3/4 are not checked any more)
            return True, 'TCP z-coordinate < ' + str(par.labWarningZ) + ' m, tool close to table'
    return True, ''


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#inertia data
def LoadRobotModelData():
    """Load the CAD based data of the robot (robotModelData.json of the device data): 'links' (mass properties),
    'meshFiles' (meshes of base and links) and 'gripperJaws' (jaw meshes and geometry)."""
    if not os.path.isfile(robotModelDataFileName):
        raise FileNotFoundError('model data ' + robotModelDataFileName + ' not found (converted from CAD into '
                                'mektcommon2/data/Exudyn/robotics; the conversion tool of the course is removed)')
    with open(robotModelDataFileName, 'r', encoding='utf-8') as fileHandle:
        return json.load(fileHandle)


def GetCadLinkInertias(modelData):
    """Return list of RigidBodyInertia of links 1..6 in joint frames (reference point = joint frame origin)."""
    linkInertias = []
    for linkData in modelData['links']:
        inertia = RigidBodyInertia()
        inertia.SetWithCOMinertia(linkData['mass'], np.array(linkData['inertiaCOM']), np.array(linkData['COM']))
        linkInertias.append(inertia)
    return linkInertias


def GetSimpleLinkInertias():
    """Return list of approximate RigidBodyInertia of links 1..6 in joint frames (cylinders along the links);
    used if no CAD data is available (e.g. during CAD conversion)."""
    linkMasses = [par.massPW90*0.6, par.massPW90*0.4 + 0.6, par.massPW70*0.5, par.massPW70*0.5 + 0.8, 0.05, 0.05]
    linkInertias = []
    for linkIndex in range(par.numberOfJoints):
        jointToLink = GetLinkLocalHT(linkIndex)
        linkVector = jointToLink[0:3, 3]
        length = max(np.linalg.norm(linkVector), 0.05)
        cylinder = InertiaCylinder(density=1., length=length, outerRadius=0.04, axis=2)
        inertia = RigidBodyInertia(mass=linkMasses[linkIndex], inertiaTensor=cylinder.InertiaCOM()*linkMasses[linkIndex]/cylinder.Mass(),
                                   com=[0., 0., 0.], inertiaTensorAtCOM=True)
        linkInertias.append(inertia.Translated(0.5*linkVector))
    return linkInertias


def GetGripperInertia():
    """Return RigidBodyInertia of the gripper (estimate, see parameters) in joint frame 6."""
    #housing height from the parameters (was a hard coded 0.05 m)
    sideLengths = list(par.gripperBodySize)
    cuboid = InertiaCuboid(density=1., sideLengths=sideLengths)
    gripper = RigidBodyInertia(mass=par.gripperMass, inertiaTensor=cuboid.InertiaCOM()*par.gripperMass/cuboid.Mass(),
                               com=[0., 0., 0.], inertiaTensorAtCOM=True).Translated(par.gripperCOM)
    return gripper.Transformed(GetLinkLocalHT(5))


def GetJawInertia():
    """Return RigidBodyInertia of one gripper jaw (cuboid estimate) in its jaw link frame."""
    cuboid = InertiaCuboid(density=1., sideLengths=par.jawSize)
    return RigidBodyInertia(mass=par.jawMass, inertiaTensor=cuboid.InertiaCOM()*par.jawMass/cuboid.Mass(),
                            com=[0., 0., 0.], inertiaTensorAtCOM=True).Translated(par.jawCOM)


def ComputeMassMatrix(chain, q):
    """Compute the joint space mass matrix of a KinematicChain (also a tree with prismatic jaw links) for the joint
    coordinates q by its inverse dynamics: column i is InverseDynamics(q, 0, e_i) - InverseDynamics(q, 0, 0).

    Args:
        chain (exudyn.cUtilities.KinematicChain): chain with inertia data, Robot.GetKinematicChain()
        q (array): joint coordinates of all links

    Returns:
        np.array: n x n mass matrix
    """
    numberOfLinks = len(q)
    zero = np.zeros(numberOfLinks)
    gravityForces = np.array(chain.InverseDynamics(q, zero, zero))
    unitAccelerations = np.eye(numberOfLinks)
    return np.column_stack([np.array(chain.InverseDynamics(q, zero, unitAccelerations[index])) - gravityForces
                            for index in range(numberOfLinks)])


def ComputeControlGains(robot, bandwidthHz=par.controlBandwidthHz, dampingRatio=par.controlDampingRatio):
    """Compute arm PD gains from maximum diagonal mass matrix entries over typical configurations.

    For a robot with jaw links, the mass matrix of the tree includes both jaws at their initial opening.

    Args:
        robot (Robot): exudyn robot with inertia data
        bandwidthHz (float): natural frequency of each joint control loop
        dampingRatio (float): damping ratio of each joint control loop

    Returns:
        tuple: (P, D) arrays with proportional (Nm/rad) and derivative (Nms/rad) gains
    """
    from labRobots.robots.robot6R.visualization import ComputeJawOffset
    chain = robot.GetKinematicChain()
    jawCoordinates = [ComputeJawOffset(par.gripperInitialPercent)]*(len(robot.links) - par.numberOfJoints)
    diagonalMax = np.zeros(par.numberOfJoints)
    for q in gainDesignConfigurations:
        massMatrix = ComputeMassMatrix(chain, list(q) + jawCoordinates)
        diagonalMax = np.maximum(diagonalMax, np.diag(massMatrix)[:par.numberOfJoints])
    omega = 2.*np.pi*bandwidthHz
    return omega**2*diagonalMax, 2.*dampingRatio*omega*diagonalMax



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#robot creation
def ManipulatorRobot6R(useCadData=True, includeGripper=True, includeJaws=False, bandwidthHz=par.controlBandwidthHz,
                  dampingRatio=par.controlDampingRatio, base=None, q0=None):
    """Create exudyn.robotics.Robot of the 6R robot with PD joint control: the description, as ManipulatorUR5 of
    exudyn.robotics.models (renamed from CreateRobot6R; CreateRobot6R builds it into a system).

    Args:
        useCadData (bool): if True, inertia and mesh graphics from the CAD conversion are used; otherwise simple
                           cylinder inertias and default Exudyn link graphics
        includeGripper (bool): if True, gripper housing mass is added to link 6 and its graphics are drawn
        includeJaws (bool): if True, the two jaws are added as prismatic links 7 (right) and 8 (left) with parent link 6;
                            jaw coordinate = opening displacement along the closing direction (0: arc center on TCP)
        bandwidthHz (float): natural frequency of joint control (gains from mass matrix)
        dampingRatio (float): damping ratio of joint control
        base (RobotBase): base frame K0 in the world (HT, e.g. the calibrated pose in the Vicon frame); without graphics
                          of its own it gets the base graphics of the 6R; None: K0 = world frame
        q0: initial joint coordinates in rad of the models of CreateKinematicTree (Robot.q0, also held by the control
            until SetControl), one per link (with jaws 8); None: the default of Robot

    Returns:
        Robot: robot with base frame K0 = base.HT, TCP = DH frame 6 (tool HT = identity)
    """
    from labRobots.robots.robot6R import visualization

    if useCadData:
        modelData = LoadRobotModelData()
        linkInertias = GetCadLinkInertias(modelData)
        linkGraphics = [visualization.CreateLinkGraphics(modelData, linkIndex) for linkIndex in range(par.numberOfJoints)]
        baseGraphics = visualization.CreateBaseGraphics(modelData)
    else:
        linkInertias = GetSimpleLinkInertias()
        linkGraphics = [[] for linkIndex in range(par.numberOfJoints)]
        baseGraphics = [graphics.Cylinder([0., 0., 0.], [0., 0., 0.12], radius=0.06, color=graphics.color.grey)]

    if includeGripper:
        linkInertias[5] = linkInertias[5] + GetGripperInertia()
        if not includeJaws:
            jawOffset = visualization.ComputeJawOffset(par.gripperInitialPercent)
            for jawName in par.jawNames:
                jawHT = np.eye(4)
                jawHT[0:3, 0:3] = visualization.GetJawLinkRotation(jawName)
                jawHT[0:3, 3] = jawHT[0:3, 0:3] @ [0., jawOffset, 0.]
                linkInertias[5] = linkInertias[5] + GetJawInertia().Transformed(GetLinkLocalHT(5) @ jawHT)
        if useCadData:
            linkGraphics[5] = linkGraphics[5] + visualization.CreateGripperHousingGraphics(modelData)

    if base is None:
        base = RobotBase()
    if len(base.visualization.graphicsData) == 0:
        base.visualization = VRobotBase(graphicsData=baseGraphics)
    #end stops at the joint limits of the lab software (SetJointLimits)
    robot = Robot(gravity=par.gravity, base=base, tool=RobotTool(), referenceConfiguration=[],
                  q0=q0, enableEndStops=True)
    for linkIndex in range(par.numberOfJoints):
        localHT = GetLinkLocalHT(linkIndex)
        inertiaLink = linkInertias[linkIndex].Transformed(InverseHT(localHT))  #joint frame -> DH link frame
        robot.AddLink(RobotLink(mass=inertiaLink.Mass(), COM=inertiaLink.COM(), inertia=inertiaLink.InertiaCOM(),
                                localHT=localHT,
                                visualization=VRobotLink(showMBSjoint=not useCadData, showCOM=False, jointRadius=0.04,
                                                         jointWidth=0.08, linkWidth=0.06,
                                                         linkColor=graphics.colorList[linkIndex],
                                                         graphicsData=linkGraphics[linkIndex])))

    if includeJaws:
        jawInertia = GetJawInertia()
        jawDamping = 2.*par.jawDampingRatio*np.sqrt(par.jawStiffness*par.jawMass)
        for jawName in par.jawNames:
            preHT = np.eye(4)
            preHT[0:3, 0:3] = visualization.GetJawLinkRotation(jawName)
            jawGraphics = [visualization.CreateJawGraphics(modelData, jawName)] if useCadData else []
            robot.AddLink(RobotLink(mass=jawInertia.Mass(), COM=jawInertia.COM(), inertia=jawInertia.InertiaCOM(),
                                    jointType='Py', parent=par.numberOfJoints-1, preHT=preHT,
                                    PDcontrol=(par.jawStiffness, jawDamping),
                                    visualization=VRobotLink(showMBSjoint=False, showCOM=False, graphicsData=jawGraphics)))
    pGains, dGains = ComputeControlGains(robot, bandwidthHz, dampingRatio)
    for linkIndex in range(par.numberOfJoints):
        robot.links[linkIndex].SetPDcontrol(pGains[linkIndex], dGains[linkIndex])
    SetJointLimits(robot, pGains, dGains)
    return robot


def SetJointLimits(robot, pGains, dGains):
    """Set the joint limits of the lab software (par.labJointLowerLimitsDeg, par.labJointUpperLimitsDeg; none for an
    infinite limit) and the stiffness and damping of the end stops there: par.endStopStiffnessFactor times the P gain of
    the joint, the damping with the same damping ratio as its control.

    Args:
        robot: the Robot of ManipulatorRobot6R (Robot(enableEndStops=True))
        pGains, dGains: the PD gains of the 6 joints (ComputeControlGains)
    """
    for linkIndex in range(par.numberOfJoints):
        lower, upper = np.radians(par.labJointLowerLimitsDeg[linkIndex]), np.radians(par.labJointUpperLimitsDeg[linkIndex])
        if np.isfinite(lower) and np.isfinite(upper):
            link = robot.links[linkIndex]
            link.jointLimits = [float(lower), float(upper)]
            link.endStopStiffness = par.endStopStiffnessFactor*pGains[linkIndex]
            link.endStopDamping = np.sqrt(par.endStopStiffnessFactor)*dGains[linkIndex]


class Robot6RModel(ArmModel):
    """The 6R robot as KairosModel and LeobotModel (2026-11-20 an ArmModel): the description
    (ManipulatorRobot6R with the jaws, an exudyn.robotics.Robot in self.robot), CreateMBS builds it into the MainSystem of
    a scene (base fixed in the world or mounted on markerGround, gripper contact with generalContact); collision-free arm
    motions with the obstacles of the scene (planning.Robot6RArmPlanner, Exudyn only), control (SetControl with the
    trajectory of all 8 links, PreStepControl), collision display."""

    #! Gravity compensation of the control (as the lab task of the 6R).
    defaultCompensation = 'static'
    #! Name of the collision shapes of the base (the socket of the 6R).
    baseShapeName = 'socket'
    #! Name of the KinematicTree object of CreateMBS.
    defaultTreeName = 'robot6R'

    def __init__(self, base=None, q0=None):
        """the description of the 6R

        Args:
            base: RobotBase with HT, the pose of the base frame K0 (axis 1 on the table top) in the world (Vicon) frame;
                None: K0 = world frame
            q0: arm joint angles in rad at the start (6 values); None: zero position
        """
        ArmModel.__init__(self, Robot6RArmPlanner(), par.numberOfJoints, base,
                          np.zeros(par.numberOfJoints) if q0 is None else q0)
        #the jaws are always links (drawn and moved by the gripper command, as the fingers of the
        #Leobot); generalContact only adds their contact
        self.numberOfJaws = len(par.jawNames)   #! Jaw links of the tree.

    def CreateDescription(self):
        """Return ManipulatorRobot6R with the jaws at a copy of the base, holding q0 and the open jaws."""
        q0 = list(self.q0) + [JawOffset(par.gripperInitialPercent)]*self.numberOfJaws
        return ManipulatorRobot6R(includeJaws=True, base=copy.deepcopy(self.base), q0=q0)

    def FloatingBaseInertia(self):
        """Return the RigidBodyInertia of the base of the 6R (a cylinder of the parameters) for a mounted 6R."""
        floatingBase = RigidBodyInertia()
        cylinder = InertiaCylinder(density=1., length=par.robotBaseHeight, outerRadius=par.robotBaseRadius, axis=2)
        floatingBase.SetWithCOMinertia(par.robotBaseMass, cylinder.InertiaCOM()*par.robotBaseMass/cylinder.Mass(),
                                       [0., 0., 0.5*par.robotBaseHeight])
        return floatingBase

    def CreateMBS(self, mbs, name='robot6R', markerGround=None, generalContact=None):
        """build the 6R into an existing MainSystem: Robot.CreateKinematicTree of ManipulatorRobot6R, the arm holding q0

        Args:
            mbs: the MainSystem, which will be extended
            name: name of the KinematicTree object
            markerGround: rigid marker (of any body: ground, rigid or flexible body, link of another robot) to which the
                base is fixed in its pose; the base becomes a free rigid body (floatingBase of CreateKinematicTree) held
                by a joint at the marker; None: the base fixed in the world
            generalContact: GeneralContact of the scene to which gripper housing and jaws are added; None: no gripper
                contact (the jaws are prismatic links 7 and 8 in both cases, PD control towards the jaw coordinate of the
                gripper command)

        Returns:
            dict with 'nodeGeneric', 'objectKinematicTree', 'nodeBase', 'objectBase' (of CreateKinematicTree),
            'jointBase' (of the mount, or None) and 'gripperContactMarkers'
        """
        items = ArmModel.CreateMBS(self, mbs, name=name, markerGround=markerGround)
        items['gripperContactMarkers'] = [] if generalContact is None else AddGripperContact(mbs, self.robot, generalContact)
        return items

    def ArmTrajectory(self, armTrajectory):
        """Return the trajectory of all links (arm and jaws) of an arm trajectory, the jaws held open."""
        jawReference = (JawOffset(par.gripperInitialPercent), 0., 0.)
        return ArmAndJawsTrajectory(armTrajectory, lambda time: jawReference, self.numberOfJaws)


def CreateRobot6R(mbs, name='robot6R', referenceHT=None, q0=None, markerGround=None, generalContact=None):
    """Create the 6R in a MainSystem in one call, as mbs.CreateRigidBody (as CreateKairos and
    CreateLeobot): the description Robot6RModel, built by CreateMBS.

    Args:
        mbs: the MainSystem, which will be extended
        name: name of the KinematicTree object
        referenceHT: HT of the base frame K0 in the world (e.g. the calibrated pose in the Vicon frame), an exudyn.HT or
                     4x4 matrix; None: K0 = world frame
        q0: initial joint angles in rad (6 values); None: zero position
        markerGround: rigid marker of the body the base is fixed to (see Robot6RModel.CreateMBS); None: fixed in the world
        generalContact: GeneralContact of the scene for the gripper (grasping scene objects); None: no gripper contact

    Returns:
        Robot6RModel: the built 6R with the dict of the created items in robot6R.items and its exudyn.robotics.Robot in
        robot6R.robot; its SetControl, PreStepControl, MoveArmToJoints, SetEnvironmentObstacles act on it

    Example:
        robot6R = CreateRobot6R(mbs, referenceHT=HTtranslate([1., 0., 0.8]))
        robot6R.SetEnvironmentObstacles(scene['obstacles'])
        mbs.SetPreStepUserFunction(robot6R.PreStepControl)
    """
    robot6R = Robot6RModel(base=RobotBase(HT=referenceHT), q0=q0)
    robot6R.CreateMBS(mbs, name=name, markerGround=markerGround, generalContact=generalContact)
    return robot6R


def AddGripperContact(mbs, robot, generalContact):
    """Add the gripper housing (on link 6) and the jaws (on their prismatic links) of a Robot of CreateKinematicTree to
    the GeneralContact of a scene: one box per graphics part (as the Kairos and the Leobot).

    Args:
        mbs: the MainSystem
        robot: the Robot of ManipulatorRobot6R with jaws, robot.items of CreateKinematicTree
        generalContact: GeneralContact of the scene

    Returns:
        list of the markers of housing and jaws
    """
    from labRobots.robots.robot6R import visualization
    tree = robot.items['objectKinematicTree']
    markers = []
    housing = mbs.AddMarker(MarkerKinematicTreeRigid(objectNumber=tree, linkNumber=par.numberOfJoints - 1, localPosition=[0., 0., 0.]))
    AddContactBoxes(generalContact, housing, visualization.CreateGripperHousingGraphics(LoadRobotModelData()),
                    par.objectContactStiffness, par.objectContactDamping)
    markers.append(housing)
    #the jaws by their meshes, not by boxes: the jaws are arcs around the cup, a box around a
    #jaw mesh (42 mm thick with the linkage) reaches inside the arc and pushed a grasped cup 14 mm deep (it was thrown off)
    for link in range(par.numberOfJoints, len(robot.links)):
        marker = mbs.AddMarker(MarkerKinematicTreeRigid(objectNumber=tree, linkNumber=link, localPosition=[0., 0., 0.]))
        for mesh in robot.links[link].visualization.graphicsData:
            AddContactMesh(generalContact, marker, mesh, par.objectContactStiffness, par.objectContactDamping)
        markers.append(marker)
    return markers


from labRobots.robots.robot6R.planning import Robot6RArmPlanner   # noqa: E402  (planning imports this module)
