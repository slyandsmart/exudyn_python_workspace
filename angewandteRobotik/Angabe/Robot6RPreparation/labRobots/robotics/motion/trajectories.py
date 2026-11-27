# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: planning.py
#
# Details:  Trajectory helpers of the arm motions of the lab robots beside the planning of Exudyn
#           (exudyn.robotics.navigation): a straight TCP motion by exudyn.robotics.motion.ProfileCartesianLinear within
#           the joint limits (CartesianLinearTrajectory), one damped least squares step of the differential inverse
#           kinematics and the linear joint motion of a manual step (teleoperation), and a trajectory of
#           TrajectoryFromPath shifted to a later start time (ShiftedTrajectory).
#           Cartesian paths of any arm (also the 7 joint Panda without analytic inverse
#           kinematics): the tool poses sampled densely and solved by the numerical or analytic inverse kinematics of
#           the model (ToolPathAngles), the joint path driven time-optimally by ProfileJointPath (JointPathTrajectory),
#           and the collision check of a trajectory along its samples (TrajectoryIsFree).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-05; 2026-11-12 moved from LabCourse/MobileRobot/tools/robotics; 2026-11-12 the own path planning
#           (RRT-Connect, shortcuts, smoothing, blending, certification) removed: the robots plan with Exudyn only
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Trajectory helpers of the arm motions."""
import numpy as np
import exudyn as exu
from exudyn.rigidBodyUtilities import RotationMatrix2RotationVector, RotationVector2RotationMatrix
from exudyn.robotics.motion import Trajectory, ProfileCartesianLinear, ProfileJointPath

#unused constants of the removed own planning (tree extension, blending per corner) deleted;
#the blended trajectory is exudyn.robotics.navigation.TrajectoryFromPath
#! Largest step in m of the tool position between two samples of a Cartesian path, and in rad of its rotation.
toolPathPositionStep, toolPathRotationStep = 0.005, 0.02
#! Largest joint step in rad between two solved samples of a Cartesian path (a larger one: a jump of the inverse
#! kinematics, e.g. near a singularity), and between two checked samples of a trajectory.
toolPathJointJump, trajectoryCheckStep = 0.15, 0.01
#! Joint distance in rad below which two successive waypoints count as one.
duplicateWaypointTolerance = 1e-5


def CartesianLinearTrajectory(robot, startAngles, targetPose, initialTime, duration, maximumVelocity, maximumAcceleration,
                              speedMargin=1.25, numberOfSamples=200):
    """Return an Exudyn Trajectory moving the TCP of robot on a straight line (rotation about one axis) from the pose at
    startAngles to targetPose by ProfileCartesianLinear (polynomial time law, analytic inverse kinematics in the arm
    configuration of the start), its duration prolonged so that the sampled joint velocities and accelerations stay
    within maximumVelocity/speedMargin and maximumAcceleration/speedMargin; raises ValueError if the line leaves the
    workspace or passes a singularity.

    Args:
        robot: exudyn.robotics.Robot of the kinematics (base frame = frame of targetPose).
        startAngles: Joint angles at the start.
        targetPose: 4x4 TCP pose at the end.
        initialTime: Start time in s.
        duration: First duration in s (e.g. of the former straight-line path).
        maximumVelocity: Joint speed limits in rad/s (array).
        maximumAcceleration: Joint acceleration limits in rad/s^2 (array).
        speedMargin: Factor between the limits and the largest sampled joint velocity/acceleration: the samples miss the
            peaks between them.
        numberOfSamples: Samples of the trajectory for the check of the limits.
    """
    def Build(profileDuration):
        trajectory = Trajectory(initialCoordinates=list(startAngles), initialTime=initialTime)
        trajectory.Add(ProfileCartesianLinear(robot, exu.HT(np.asarray(targetPose, dtype=float)), profileDuration))
        return trajectory
    trajectory = Build(duration)
    samples = [trajectory.Evaluate(time) for time in np.linspace(initialTime, initialTime + duration, numberOfSamples)]
    peakVelocity = max(np.max(np.abs(velocities)) for unusedAngles, velocities, unusedAccelerations in samples)
    peakAcceleration = max(np.max(np.abs(accelerations)) for unusedAngles, unusedVelocities, accelerations in samples)
    scale = max(1., peakVelocity*speedMargin/maximumVelocity, np.sqrt(peakAcceleration*speedMargin/maximumAcceleration))
    return Build(duration*scale) if scale > 1. else trajectory


def DifferentialInverseKinematicsStep(pose, targetPose, jacobian, rotationWeight=0.2, damping=0.05):
    """One damped least squares step of the differential inverse kinematics (robust at singularities).

    Args:
        pose: Present 4x4 TCP pose (same frame as targetPose and the jacobian).
        targetPose: Desired 4x4 TCP pose.
        jacobian: Geometric 6 x n Jacobian [linear; angular] of the TCP at the present joint angles.
        rotationWeight: Length in m that weights the orientation error relative to the position error.
        damping: Damping factor of the least squares solution.

    Returns:
        Unlimited joint step (n) towards targetPose.
    """
    #same method as the 6R robot teleoperation (LabCourse/Robot6R)
    pose, targetPose = np.asarray(pose, dtype=float), np.asarray(targetPose, dtype=float)
    positionError = targetPose[:3, 3] - pose[:3, 3]
    rotationError = 0.5*sum(np.cross(pose[:3, axis], targetPose[:3, axis]) for axis in range(3))
    weighted = np.array(jacobian, dtype=float)
    weighted[3:] *= rotationWeight
    error = np.r_[positionError, rotationWeight*rotationError]
    return weighted.T @ np.linalg.solve(weighted @ weighted.T + damping**2*np.eye(len(error)), error)


class LinearStepTrajectory:
    """Joint motion with constant velocity from qStart to qEnd within duration, holding qEnd afterwards; used for
    teleoperation with one step per update period (velocity feedforward, no stop between steps).
    Interface of exudyn.robotics.motion.Trajectory (Evaluate, GetTimes) and ShiftedTrajectory (endTime)."""

    def __init__(self, qStartInit, qEndInit, initialTimeInit, durationInit):
        """initialize LinearStepTrajectory

        Args:
            qStartInit, qEndInit: Joint angles at the start and end of the step.
            initialTimeInit: Start time in s.
            durationInit: Duration of the step in s (> 0).
        """
        self.qStart = np.asarray(qStartInit, dtype=float)  #! Joint angles at the start.
        self.qEnd = np.asarray(qEndInit, dtype=float)  #! Joint angles at the end.
        self.initialTime = float(initialTimeInit)  #! Start time in s.
        self.duration = float(durationInit)  #! Duration in s.
        self.endTime = self.initialTime + self.duration  #! End time in s.

    def GetTimes(self):
        """Return start and end time in s."""
        return [self.initialTime, self.initialTime + self.duration]

    def Evaluate(self, time):
        """Return joint angles, velocities and accelerations at time."""
        fraction = (time - self.initialTime)/self.duration
        if fraction >= 1.:
            return self.qEnd.tolist(), [0.]*len(self.qEnd), [0.]*len(self.qEnd)
        fraction = max(fraction, 0.)
        velocity = (self.qEnd - self.qStart)/self.duration
        return (self.qStart + fraction*(self.qEnd - self.qStart)).tolist(), velocity.tolist(), [0.]*len(self.qEnd)


class ShiftedTrajectory:
    """A trajectory starting at time 0 (e.g. exudyn.robotics.navigation.TrajectoryFromPath) evaluated from a later start
    time; same interface as exudyn.robotics.motion.Trajectory (Evaluate, GetTimes, GetFinalCoordinates)."""

    def __init__(self, trajectoryInit, startTimeInit):
        """initialize ShiftedTrajectory

        Args:
            trajectoryInit: Trajectory starting at time 0.
            startTimeInit: Time in s of its start.
        """
        self.trajectory = trajectoryInit    #! The trajectory from time 0.
        self.startTime = startTimeInit      #! Time in s of its start.
        self.endTime = startTimeInit + trajectoryInit.GetTimes()[-1]   #! Time in s of its end.

    def Evaluate(self, time):
        """[angles, velocities, accelerations] at time (before the start: the start, after the end: the goal)."""
        return self.trajectory.Evaluate(time - self.startTime)

    def GetTimes(self):
        """Return the times in s of the start and the ends of its profiles."""
        return [self.startTime + time for time in self.trajectory.GetTimes()]

    def GetFinalCoordinates(self):
        """Return the joint coordinates at the end."""
        return self.trajectory.GetFinalCoordinates()


def InterpolatedToolPoses(poseA, poseB):
    """Return the 4x4 poses from poseA (excluded) to poseB (included): the position on the straight line, the rotation
    about one axis (the relative rotation vector scaled), at most toolPathPositionStep and toolPathRotationStep apart."""
    poseA, poseB = np.asarray(poseA, dtype=float), np.asarray(poseB, dtype=float)
    rotationVector = np.asarray(RotationMatrix2RotationVector(poseA[0:3, 0:3].T @ poseB[0:3, 0:3]), dtype=float)
    distance = np.linalg.norm(poseB[0:3, 3] - poseA[0:3, 3])
    numberOfSteps = max(1, int(np.ceil(max(distance/toolPathPositionStep, np.linalg.norm(rotationVector)/toolPathRotationStep))))
    poses = []
    for fraction in np.linspace(0., 1., numberOfSteps + 1)[1:]:
        pose = np.eye(4)
        pose[0:3, 0:3] = poseA[0:3, 0:3] @ np.asarray(RotationVector2RotationMatrix(fraction*rotationVector), dtype=float)
        pose[0:3, 3] = (1. - fraction)*poseA[0:3, 3] + fraction*poseB[0:3, 3]
        poses.append(pose)
    return poses


def ToolPathAngles(jointAnglesAtToolPose, startAngles, toolPoses):
    """Return the joint angles [n x joints] along a Cartesian path: from the tool pose at startAngles through the
    tool poses (straight lines between them, sampled by InterpolatedToolPoses), each sample solved by the inverse
    kinematics from the solution before (a circle is a list of poses on it).

    Args:
        jointAnglesAtToolPose: Callable(toolPose, startAngles) -> joint angles of the model (e.g.
            model.JointAnglesAtToolPose: collision-free, within the joint limits, nearest to startAngles).
        startAngles: Joint angles at the start of the path.
        toolPoses: 4x4 tool poses of the path; the first one is the tool pose at startAngles.

    Raises:
        ValueError: If a sample has no solution or the joints jump by more than toolPathJointJump (singularity).
    """
    angles = [np.asarray(startAngles, dtype=float)]
    for poseA, poseB in zip(toolPoses[:-1], toolPoses[1:]):
        for sample in InterpolatedToolPoses(poseA, poseB):
            solution = np.asarray(jointAnglesAtToolPose(sample, angles[-1]), dtype=float)
            jump = np.max(np.abs(solution - angles[-1]))
            if jump > toolPathJointJump:
                raise ValueError('the joints jump by %.2f rad on the Cartesian path (near a singularity or the workspace '
                                 'border)' % jump)
            angles.append(solution)
    return np.array(angles)


def JointPathTrajectory(path, maximumVelocities, maximumAccelerations, blendRadius=1., initialTime=0.):
    """Return the exudyn.robotics.motion.Trajectory along joint waypoints without stops (ProfileJointPath: corners
    rounded by blendRadius, time-optimal within the joint limits); for a densely sampled Cartesian path the tool
    follows the path within a fraction of a millimeter."""
    points = [np.asarray(path[0], dtype=float)]
    for angles in path[1:]:      #nearly equal neighbors (e.g. the first sample of a path) make ProfileJointPath divide by ~0
        if np.max(np.abs(np.asarray(angles, dtype=float) - points[-1])) > duplicateWaypointTolerance:
            points.append(np.asarray(angles, dtype=float))
    trajectory = Trajectory(list(points[0]), initialTime, list(maximumVelocities), list(maximumAccelerations))
    trajectory.Add(ProfileJointPath(np.array(points), blendRadius=blendRadius))
    return trajectory


def TrajectoryIsFree(isFree, trajectory, maximumJointStep=trajectoryCheckStep):
    """True if all samples of a trajectory, at most about maximumJointStep apart in each joint, are free.

    Args:
        isFree: Callable(angles) -> bool of the planner (e.g. model.armPlanner.IsFree).
        trajectory: Trajectory with GetTimes and Evaluate.
        maximumJointStep: Largest joint step in rad between two checked samples.
    """
    times = trajectory.GetTimes()
    coarse = [np.asarray(trajectory.Evaluate(time)[0], dtype=float) for time in np.linspace(times[0], times[-1], 50)]
    largestStep = max(np.max(np.abs(b - a)) for a, b in zip(coarse[:-1], coarse[1:]))
    numberOfSamples = max(2, int(np.ceil(49*largestStep/maximumJointStep*1.5)) + 1)
    return all(isFree(np.asarray(trajectory.Evaluate(time)[0], dtype=float))
               for time in np.linspace(times[0], times[-1], numberOfSamples))

