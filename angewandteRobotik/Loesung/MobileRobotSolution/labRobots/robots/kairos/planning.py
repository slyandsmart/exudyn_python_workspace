# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: planning.py
#
# Details:  Collision-free motions of the UR5 with WSG50 on the Kairos with the collision check and the path planning
#           of Exudyn only (exudyn.cUtilities.CollisionModel, exudyn.robotics.navigation), as the ArmPlanner of the
#           Leobot: the link shapes of the UR5 and the gripper (collisionModel.ArmLinkShapes) on the links of the
#           KinematicChain of the arm, the platform shapes on its base, the obstacles of the scene (table) as obstacles,
#           all in the arm base frame; RRT-Connect (PlanJointPath), shortcuts (ShortcutPath) and a blended trajectory
#           (TrajectoryFromPath); straight TCP motions (ProfileCartesianLinear) checked along their samples (IsEdgeFree).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-12 (replaces RobotCollisionChecker, NativeCollisionChecker and CertifyTrajectory of the Kairos)
# Notes:    The checks are made at joint steps of at most par.collisionEdgeStep; between two checks no point of the
#           arm moves farther than about par.armReachSum*par.collisionEdgeStep (below the safety distance), but this
#           is not a proof of a gap-free motion. CollisionModel excludes whole link pairs: neighboring links, the
#           first link against the base, the MoveIt pairs par.moveItDisabledPairs and the link pairs overlapping in
#           par.collisionReferenceAngles (ExcludeCollidingPairs with margin 0: platform - upper arm, wrist 1 - wrist 3).
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Collision-free arm motions of the Kairos by Exudyn."""
import numpy as np
from exudyn.cUtilities import CollisionModel
from exudyn.robotics.navigation import PlanJointPath, ShortcutPath, TrajectoryFromPath
from exudyn.robotics.motion import Trajectory, ProfileConstantAcceleration
from labRobots.robots.kairos import parameters as par
from labRobots.robotics.collision.exudynShapes import ExudynShape, baseLinkIndex
from labRobots.robotics.motion.trajectories import ShiftedTrajectory


#! Joint distance in rad below which start and goal of a motion are the same (the arm holds the goal).
goalTolerance = 1e-9


class KairosArmPlanner:
    """Plans collision-free motions of the 6 UR5 joints (arm base frame) with the CollisionModel of Exudyn."""

    def __init__(self, kinematicChainInit, linkShapesInit, platformShapesInit):
        """Build the collision model of arm, gripper and platform (no scene obstacles yet, SetObstacles).

        Args:
            kinematicChainInit: exudyn.cUtilities.KinematicChain of the UR5 (DH frames = frames of the link shapes).
            linkShapesInit: [(link index, shape of labRobots.robotics.collision.shapes)] of arm and gripper (ArmLinkShapes).
            platformShapesInit: Shapes of the platform in the arm base frame (PlatformObstacles), on the base.
        """
        self.chain = kinematicChainInit                 #! KinematicChain of the arm.
        self.linkShapes = list(linkShapesInit)          #! (link index, shape) of arm and gripper in their DH frames.
        self.platformShapes = list(platformShapesInit)  #! Platform shapes in the arm base frame.
        self.obstacles = []                             #! Obstacles of the scene in the arm base frame (SetObstacles).
        self.margin = par.collisionSafetyDistance       #! Distance in m which the shapes keep.
        self.edgeStep = par.collisionEdgeStep           #! Largest joint step in rad between two checks.
        self.jointLimits = np.column_stack([par.armJointLowerLimits, par.armJointUpperLimits])
        #! Joint speed and acceleration limits in rad/s and rad/s^2 of planned motions (as the
        #! Leobot planner, for trajectories built outside Plan, e.g. Cartesian paths).
        self.speedLimits = np.full(par.numberOfArmJoints, par.armMaximumSpeed)
        self.accelerationLimits = np.full(par.numberOfArmJoints, par.armMaximumAcceleration)
        self.collisionModel = None                      #! exudyn.cUtilities.CollisionModel of arm, platform and scene.
        self.SetObstacles([])

    def SetObstacles(self, obstacles):
        """Build the collision model with the obstacles of the scene.

        Args:
            obstacles: Shapes of labRobots.robotics.collision.shapes in the arm base frame.
        """
        self.obstacles = list(obstacles)
        model = CollisionModel(self.chain)
        for linkIndex, shape in self.linkShapes:
            model.AddLinkShape(linkIndex, *ExudynShape(shape))
        for shape in self.platformShapes:
            model.AddLinkShape(baseLinkIndex, *ExudynShape(shape))
        for shape in self.obstacles:
            model.AddObstacle(*ExudynShape(shape))
        for linkA, linkB in self.DisabledLinkPairs():
            model.ExcludeLinkPair(linkA, linkB)
        #only link pairs overlapping in the reference configuration (by construction, e.g. the upper arm on the platform
        #base); with the safety distance as margin also forearm - wrist 2 (3 mm apart there) would never be checked
        model.ExcludeCollidingPairs(np.asarray(par.collisionReferenceAngles, dtype=float), 0.)
        self.collisionModel = model

    def DisabledLinkPairs(self):
        """Return the link pairs (link indices) of the MoveIt pairs par.moveItDisabledPairs (shape name prefixes), without
        neighboring links (never checked by CollisionModel)."""
        pairs = set()
        for prefixA, prefixB in par.moveItDisabledPairs:
            linksA = {link for link, shape in self.linkShapes if shape['name'].startswith(prefixA)}
            linksB = {link for link, shape in self.linkShapes if shape['name'].startswith(prefixB)}
            pairs |= {tuple(sorted((linkA, linkB))) for linkA in linksA for linkB in linksB if abs(linkA - linkB) >= 2}
        return sorted(pairs)

    def IsFree(self, angles, margin=None):
        """True if the arm at angles keeps the margin (default: the safety distance) to itself, platform and scene."""
        return self.collisionModel.IsFree(np.asarray(angles, dtype=float), self.margin if margin is None else margin)

    def MinimumDistance(self, angles):
        """Smallest distance in m of the checked pairs at the arm angles (negative: penetration)."""
        return self.collisionModel.MinimumDistance(np.asarray(angles, dtype=float))

    def IsEdgeFree(self, anglesA, anglesB, margin=None):
        """True if the straight joint motion from anglesA to anglesB keeps the margin (default: the safety distance)."""
        return self.collisionModel.IsEdgeFree(np.asarray(anglesA, dtype=float), np.asarray(anglesB, dtype=float),
                                              self.margin if margin is None else margin, self.edgeStep)

    def Plan(self, start, goal, initialTime, speedScale=1., firstAngles=None):
        """Return (trajectory from initialTime, waypoints) of a collision-free joint motion from start to goal:
        PlanJointPath, ShortcutPath and TrajectoryFromPath of Exudyn; raises ValueError if the goal is not free or no
        path is found.

        Args:
            speedScale: Factor in (0, 1] of the joint speed and acceleration limits.
            firstAngles: Angles before start (the measured arm when the planning starts at the commanded angles): the
                motion first goes straight from them to start (checked without the safety distance, they violate it).
        """
        if firstAngles is not None and not self.IsEdgeFree(firstAngles, start, 0.):
            raise ValueError('the motion from the measured arm to the planning start is not free')
        if firstAngles is None and np.allclose(start, goal, rtol=0., atol=goalTolerance):
            if not self.IsFree(goal):
                raise ValueError('the goal is not free')
            hold = Trajectory(initialCoordinates=list(np.asarray(goal, dtype=float)), initialTime=initialTime)
            hold.Add(ProfileConstantAcceleration(list(np.asarray(goal, dtype=float)), par.armHoldDuration))
            return hold, [np.asarray(goal, dtype=float)]
        path = PlanJointPath(self.collisionModel, np.asarray(start, dtype=float), np.asarray(goal, dtype=float),
                             self.jointLimits, self.margin, par.plannerMaximumStep, self.edgeStep,
                             par.plannerMaximumIterations, seed=par.plannerSeed)
        path = ShortcutPath(self.collisionModel, path, self.margin, self.edgeStep, par.plannerShortcutIterations,
                            seed=par.plannerSeed)
        if firstAngles is not None:
            path = np.vstack([np.asarray(firstAngles, dtype=float), path])
        trajectory = TrajectoryFromPath(self.collisionModel, path, [speedScale*par.armMaximumSpeed]*par.numberOfArmJoints,
                                        [speedScale*par.armMaximumAcceleration]*par.numberOfArmJoints, self.margin,
                                        self.edgeStep)
        return ShiftedTrajectory(trajectory, initialTime), [np.asarray(angles, dtype=float) for angles in path]

    def TrajectoryIsFree(self, trajectory):
        """True if a trajectory within the joint speed limit par.armMaximumSpeed keeps the safety distance: its samples
        at most par.collisionEdgeStep apart per joint, the straight motions between them checked by IsEdgeFree."""
        times = trajectory.GetTimes()
        numberOfSamples = int(np.ceil((times[-1] - times[0])*par.armMaximumSpeed/self.edgeStep)) + 1
        samples = [np.asarray(trajectory.Evaluate(time)[0], dtype=float)
                   for time in np.linspace(times[0], times[-1], max(numberOfSamples, 2))]
        return all(self.IsEdgeFree(anglesA, anglesB) for anglesA, anglesB in zip(samples[:-1], samples[1:]))
