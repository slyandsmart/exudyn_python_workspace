# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: armPlanning.py
#
# Details:  Collision-free motions of an arm of the lab with Exudyn only: ArmPlanner plans with the CollisionModel of an
#           exudyn.robotics.Robot of the kinematics with collision shapes (Robot.GetCollisionModel) and the obstacles of
#           the scene in the base frame, RRT-Connect (PlanJointPath), shortcuts (ShortcutPath), a blended trajectory
#           (TrajectoryFromPath) and the analytic inverse kinematics of the spherical wrist
#           (KinematicChain.InverseKinematicsAnalytic). IndustrialArmPlanner builds it from a robot dictionary of
#           exudyn.robotics.models (Staubli TX2-90L, Fanuc LR Mate 200iD/7L); Robot6RArmPlanner
#           (labRobots/robots/robot6R/planning.py) from the DH model of the 6R; LeobotArmPlanner
#           (labRobots/robots/leobot/planning.py) from the URDF tree of the Leobot with its platform (Claude Code,
#           2026-11-20).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-13; 2026-11-20 ArmPlanner as the base of IndustrialArmPlanner and Robot6RArmPlanner (was
#           industrialArm/planning.py)
# Notes:    The link pairs overlapping by construction are those overlapping in the free configuration of the planner
#           (ExcludeCollidingPairs with margin 0). The speeds of planned motions of IndustrialArmPlanner are a fraction
#           of the maximum joint speeds of the dictionary (the lab works at reduced speed), the accelerations reach them
#           in accelerationTime.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Collision-free motions of the arms of the lab by Exudyn."""
import itertools
import numpy as np
from exudyn.robotics.models import LinkDict2Robot
from exudyn.robotics.navigation import PlanJointPath, ShortcutPath, TrajectoryFromPath
from labRobots.robotics.collision.exudynShapes import ExudynShape
from labRobots.robotics.motion.trajectories import ShiftedTrajectory

#! Distance in m which the shapes keep (capsules contain the meshes); default of IndustrialArmPlanner.
planningMargin = 0.01
#! Largest joint step in rad between two collision checks.
collisionEdgeStep = 0.005
#! Speeds of planned motions as fraction of the maximum joint speeds of the robot (reduced speed in the lab).
speedFraction = 0.25
#! Time in s in which a planned motion reaches the speed limit (sets the acceleration limits).
accelerationTime = 0.5
#! Random numbers of RRT-Connect and the shortcuts: repeatable motions.
planningSeed = 1
#! Angle in rad of a full turn of a joint (FullTurnVariants).
fullTurn = 2.*np.pi
#! Joint distance in rad below which two configurations are the same (JointAnglesConfigurations).
sameConfigurationTolerance = 1e-9


def InitialConfiguration(robotDict):
    """Return the joint angles in rad of the initial configuration named in a robot dictionary ('initialConfiguration'
    of its 'configurations', e.g. 'qr'), the reference configuration if it names none."""
    name = robotDict.get('initialConfiguration')
    if name is None:
        return np.asarray(robotDict['referenceConfiguration'], dtype=float)
    return np.asarray(robotDict['configurations'][name], dtype=float)


class ArmPlanner:
    """Plans collision-free motions of the joints of an arm (base frame of the arm) with the CollisionModel of Exudyn;
    the base of IndustrialArmPlanner and Robot6RArmPlanner."""

    def __init__(self, robotInit, jointLimitsInit, speedLimitsInit, accelerationLimitsInit, freeConfigurationInit,
                 marginInit=planningMargin, edgeStepInit=collisionEdgeStep, baseShapesInit=None, excludeMarginInit=0.):
        """Set the kinematics with the collision shapes and the limits of planned motions (no obstacles yet).

        Args:
            robotInit: exudyn.robotics.Robot of the kinematics in the base frame with the collision shapes of its links
                (RobotLink.collisionShapes) and base (RobotBase.collisionShapes); its tool frame is the tool of
                JointAnglesAtPose.
            jointLimitsInit: Joint limits in rad of planned motions, [n x 2] (lower, upper).
            speedLimitsInit: Joint speed limits in rad/s of planned motions (n values).
            accelerationLimitsInit: Joint acceleration limits in rad/s^2 of planned motions (n values).
            freeConfigurationInit: Joint angles in rad at which the link pairs overlapping by construction are found
                (these pairs are not checked).
            marginInit: Distance in m which the shapes keep.
            edgeStepInit: Largest joint step in rad between two collision checks of a planned motion.
            baseShapesInit: Further shapes on the base (list of (exudyn.cUtilities.ConvexShape, exudyn.HT) in the base
                frame, e.g. the platform of a mobile robot; baseShapes of Robot.GetCollisionModel); None: none.
            excludeMarginInit: Margin in m of the link pairs overlapping in freeConfigurationInit (excludeMargin of
                Robot.GetCollisionModel).
        """
        self.robot = robotInit                  #! exudyn.robotics.Robot of the kinematics with the collision shapes.
        self.margin = marginInit                #! Distance in m which the shapes keep.
        self.edgeStep = edgeStepInit            #! Largest joint step in rad between two checks.
        self.baseShapes = baseShapesInit        #! Further shapes on the base (ConvexShape, HT), None: none.
        self.excludeMargin = excludeMarginInit  #! Margin in m of the link pairs overlapping by construction.
        self.jointLimits = np.asarray(jointLimitsInit, dtype=float)    #! Joint limits in rad, [n x 2].
        #! Joint speed and acceleration limits in rad/s and rad/s^2 of planned motions.
        self.speedLimits = np.asarray(speedLimitsInit, dtype=float)
        self.accelerationLimits = np.asarray(accelerationLimitsInit, dtype=float)
        #! Free configuration of the robot: link pairs overlapping there by construction are not checked.
        self.freeConfiguration = np.asarray(freeConfigurationInit, dtype=float)
        self.obstacles = []                     #! Obstacles of the scene (shapes of collision.py) in the base frame.
        self.collisionModel = None              #! exudyn.cUtilities.CollisionModel of arm, base and scene.
        self.SetObstacles([])

    def SetObstacles(self, obstacles):
        """Build the collision model with the obstacles of the scene (shapes of labRobots.robotics.collision.shapes in the base
        frame of the robot; the format of ConvexObstacles)."""
        self.obstacles = list(obstacles)
        self.collisionModel = self.robot.GetCollisionModel(obstacles=self.ConvexObstacles(), baseShapes=self.baseShapes,
                                                           excludeCollidingAt=self.freeConfiguration,
                                                           excludeMargin=self.excludeMargin)

    def ConvexObstacles(self):
        """Return the obstacles of the collision model as (exudyn.cUtilities.ConvexShape, exudyn.HT): self.obstacles,
        shapes of labRobots.robotics.collision.shapes (a subclass may keep them in another format)."""
        return [ExudynShape(shape) for shape in self.obstacles]

    def IsFree(self, angles, margin=None):
        """True if the arm at angles keeps the margin (default: self.margin) to itself, the base and the scene."""
        return self.collisionModel.IsFree(np.asarray(angles, dtype=float), self.margin if margin is None else margin)

    def MinimumDistance(self, angles):
        """Smallest distance in m of the checked pairs at the arm angles (negative: penetration)."""
        return self.collisionModel.MinimumDistance(np.asarray(angles, dtype=float))

    def IsWithinLimits(self, angles):
        """True if the joint angles in rad are within the joint limits."""
        angles = np.asarray(angles, dtype=float)
        return bool(np.all(angles >= self.jointLimits[:, 0]) and np.all(angles <= self.jointLimits[:, 1]))

    def Plan(self, start, goal, initialTime=0., speedScale=1.):
        """Return (trajectory from initialTime, waypoints) of a collision-free joint motion from start to a different
        goal within the speed and acceleration limits; raises ValueError if the goal is not free or no path is found.

        Args:
            speedScale: Factor in (0, 1] of the joint speed and acceleration limits.
        """
        path = PlanJointPath(self.collisionModel, np.asarray(start, dtype=float), np.asarray(goal, dtype=float),
                             self.jointLimits, self.margin, edgeStep=self.edgeStep, seed=planningSeed)
        path = ShortcutPath(self.collisionModel, path, self.margin, self.edgeStep, seed=planningSeed)
        trajectory = TrajectoryFromPath(self.collisionModel, path, speedScale*self.speedLimits,
                                        speedScale*self.accelerationLimits, self.margin, self.edgeStep)
        return ShiftedTrajectory(trajectory, initialTime), [np.asarray(angles, dtype=float) for angles in path]

    def JointAnglesAtPose(self, pose, startAngles):
        """Return the joint angles in rad at which the tool is at a 4x4 pose in the base frame: the solution of the
        analytic inverse kinematics nearest to startAngles which is free and within the joint limits; raises ValueError if
        there is none. startAngles is only the reference of the nearest solution and may be in collision (Claude Code,
        2026-11-13: e.g. the grasp angles, from which the approach pose is found on the same branch while the cup is an
        obstacle; not PlanToPose of exudyn.robotics.navigation, which also plans a path from startAngles)."""
        start = np.asarray(startAngles, dtype=float)
        solutions = [np.asarray(angles, dtype=float)
                     for angles in self.robot.GetKinematicChain().InverseKinematicsAnalytic(np.asarray(pose, dtype=float), start)]
        candidates = [angles for angles in solutions if self.IsWithinLimits(angles) and self.IsFree(angles)]
        if not candidates:
            #then the solutions with joints turned by full turns within their limits (the inverse
            #kinematics wraps the angles near startAngles: with the wrist of the Staubli turned by 2.2 rad a pose came
            #out beyond the limit of axis 6, the same pose a turn back was not tried); only as fallback: preferred
            #always, they moved the start of throws of Aschenputtel near the wrist limits
            candidates = [angles for solution in solutions for angles in self.FullTurnVariants(solution)
                          if self.IsFree(angles)]
        if not candidates:
            raise ValueError('No free solution of the inverse kinematics within the joint limits (%d solutions)' % len(solutions))
        return min(candidates, key=lambda angles: np.linalg.norm(angles - start))

    def JointAnglesConfigurations(self, pose, referenceAngles):
        """Return all joint angle configurations in rad at which the tool is at a 4x4 pose in the base frame: every
        solution of the analytic inverse kinematics with its joints also turned by full turns, within the joint limits
        and free, nearest to referenceAngles first (e.g. to choose the configuration of a grasp
        from which a following motion works; JointAnglesAtPose gives only the nearest)."""
        reference = np.asarray(referenceAngles, dtype=float)
        solutions = self.robot.GetKinematicChain().InverseKinematicsAnalytic(np.asarray(pose, dtype=float), reference)
        configurations = []
        for solution in solutions:
            for angles in self.FullTurnVariants(np.asarray(solution, dtype=float)):
                if self.IsFree(angles) and all(np.max(np.abs(angles - other)) > sameConfigurationTolerance
                                               for other in configurations):
                    configurations.append(angles)
        return sorted(configurations, key=lambda angles: np.linalg.norm(angles - reference))

    def FullTurnVariants(self, angles):
        """Return the joint angles equivalent to angles in rad (joints turned by multiples of 2 pi) within the joint
        limits (none if a joint has no equivalent angle within its limits)."""
        perJoint = []
        for angle, (lower, upper) in zip(np.asarray(angles, dtype=float), self.jointLimits):
            first = angle + fullTurn*np.ceil((lower - angle)/fullTurn)        #the smallest equivalent angle >= lower
            perJoint.append(list(np.arange(first, upper + 1e-12, fullTurn)))
        return [np.array(combination) for combination in itertools.product(*perJoint)]


class IndustrialArmPlanner(ArmPlanner):
    """ArmPlanner of an arm given by a robot dictionary of exudyn.robotics.models (Staubli, Fanuc)."""

    def __init__(self, robotDictInit, marginInit=planningMargin, speedFractionInit=speedFraction):
        """Build the kinematics of the robot dictionary with its collision capsules (no obstacles yet).

        Args:
            robotDictInit: Robot dictionary of exudyn.robotics.models (dhMode 'jointFrames', links with 'jointLimits',
                'maxVelocity' and 'collisionShapes', 'baseCollisionShapes', optional 'tool').
            marginInit: Distance in m which the shapes keep.
            speedFractionInit: Speeds of planned motions as fraction of the maximum joint speeds.
        """
        links = robotDictInit['links']
        speedLimits = speedFractionInit*np.array([link['maxVelocity'] for link in links], dtype=float)
        ArmPlanner.__init__(self, LinkDict2Robot(robotDictInit), [link['jointLimits'] for link in links], speedLimits,
                            speedLimits/accelerationTime, InitialConfiguration(robotDictInit), marginInit)
