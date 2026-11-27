# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: preparationTests.py
#
# Details:  Subtasks of the preparation of the mobile robot run alone (variable runTask of the task file): driving to
#           the approach pose (ApproachPose, DriveCommand), grasping from there (ArmBasePose, GraspPoses), driving
#           back (DriveCommand), or the whole mission (functionStateMachine); the results drawn as frames in the
#           renderer (switches showApproachPose, showArmBase, showGraspPoses of the task file), also before the other
#           subtasks are solved.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-15
# Notes:    The task file defines the helper functions; a test runs as a state machine with the six arguments of the
#           preparation (fahrzeug.stateMachine.PreparationController).
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Subtasks of the preparation run alone and their frames."""
import numpy as np
from . import parameters as par

#! Values of runTask: the whole mission (functionStateMachine; the only one meant for the lab), drive to the approach
#! pose, grasp from the approach pose, drive back from the approach pose to returnTarget.
taskStateMachine, taskDrive, taskGrasp, taskReturn = 'stateMachine', 'drive', 'grasp', 'return'
testTasks = (taskStateMachine, taskDrive, taskGrasp, taskReturn)
#! Target [x, y, yaw] of the test 'return': the start pose of the mission without --initialPose.
returnTarget = [0., 0., 0.]
#! Order of the arm motions of the test 'grasp' (keys of GraspPoses), the gripper closes after 'grasp'.
graspSequence = ('pregrasp', 'grasp', 'lift')
#! Errors of unsolved or unfinished student functions: the test reports them, the frames leave them out.
studentErrors = (ValueError, TypeError, IndexError, KeyError, ArithmeticError, np.linalg.LinAlgError)
#! Number of values of a platform pose [x, y, yaw].
dimensionPose = 3


def TaskOf(module):
    """Return runTask of a task file ('stateMachine' if it has none); raise ValueError for an unknown value."""
    task = getattr(module, 'runTask', taskStateMachine)
    if task not in testTasks:
        raise ValueError('runTask must be one of %s, not %r' % (', '.join(testTasks), task))
    return task


def ArmTarget(module, posPlatform, ThetaPlatform, worldPose):
    """Return the TCP target relative to the arm base (ArmBasePose of the task file) for a pose in K0."""
    return np.linalg.inv(np.asarray(module.ArmBasePose(posPlatform, ThetaPlatform), dtype=float)) @ np.asarray(worldPose, dtype=float)


def TestCallback(module, task):
    """Return a state machine (six arguments of the preparation) running one subtask with the helper functions of the
    task file module: 'drive' to ApproachPose, 'grasp' (the arm through GraspPoses, closing after 'grasp'), 'return'
    to returnTarget."""

    def Drive(t, posPlatform, ThetaPlatform, PosObj, myState, myDict):
        if 'target' not in myDict:
            myDict['target'] = list(module.ApproachPose(PosObj)) if task == taskDrive else list(returnTarget)
        v, reached = module.DriveCommand(t, posPlatform, ThetaPlatform, myDict['target'], myDict.setdefault('drive', {}))
        if reached:
            print('drive: target %s reached at t = %.2f s' % (np.round(myDict['target'], 3), t))
            return [0., 0., 0.], None, None, -1, myDict
        return v, None, None, myState, myDict

    def Grasp(t, posPlatform, ThetaPlatform, PosObj, myState, myDict):
        #states 0, 1: arm to 'pregrasp', 'grasp'; 2: close the gripper; 3: arm to 'lift'; then the end
        if 'poses' not in myDict:
            myDict['poses'] = module.GraspPoses(PosObj)
        if myState in (0, 1, 3):
            name = graspSequence[min(myState, 2)]
            return None, ArmTarget(module, posPlatform, ThetaPlatform, myDict['poses'][name]), None, myState + 1, myDict
        if myState == 2:
            return None, None, True, myState + 1, myDict
        return None, None, None, -1, myDict

    callback = Grasp if task == taskGrasp else Drive
    callback.preparationModule = module
    return callback


def ConfiguredStartPose(module):
    """Return testStartPose of the task file as [x in m, y in m, yaw in deg], None if not set; raise ValueError for
    another form than three numbers (e.g. [-0.5, -0, 5, 0] with a comma for the point)."""
    pose = getattr(module, 'testStartPose', None)
    if pose is None:
        return None
    try:
        values = np.asarray(pose, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        values = np.zeros(0)
    if values.shape != (dimensionPose,) or not np.all(np.isfinite(values)):
        raise ValueError('testStartPose must be [x in m, y in m, yaw in deg] (three numbers), not %r' % (pose,))
    return [float(value) for value in values]


def TestStartPose(module, task, objectPosition=None):
    """Return the start pose [x, y, yaw in deg] of the simulation of a task file: testStartPose of the file for every
    task (also 'drive' and 'stateMachine'); without it ApproachPose of the object marker for
    the tests that start at the table ('grasp', 'return'), None (the default start) for the other tasks."""
    pose = ConfiguredStartPose(module)
    if pose is not None:
        return pose
    if task not in (taskGrasp, taskReturn):
        return None
    objectPosition = par.objectPosition if objectPosition is None else objectPosition
    marker = np.asarray(objectPosition, dtype=float) + [0., 0., par.viconObjectMarkerHeight]
    x, y, yaw = module.ApproachPose(marker)
    return [float(x), float(y), float(np.degrees(yaw))]


def PreparationFrames(module, posPlatform, ThetaPlatform, PosObj):
    """Return [(label, 4x4 pose in K0)] of the results the task file shows (showApproachPose: ApproachPose at the
    floor; showArmBase: ArmBasePose of the present platform pose; showGraspPoses: GraspPoses); a function that is
    not solved yet (raising an error, returning a wrong shape) is left out."""
    frames = []

    def Add(label, pose):
        pose = np.asarray(pose, dtype=float)
        if pose.shape == (4, 4) and np.all(np.isfinite(pose)):
            frames.append((label, pose))

    if getattr(module, 'showApproachPose', False) and hasattr(module, 'ApproachPose'):
        try:
            x, y, yaw = module.ApproachPose(PosObj)
            pose = np.eye(4)
            pose[0:2, 0:2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
            pose[0:2, 3] = [x, y]
            Add('approach', pose)
        except studentErrors:
            pass
    if getattr(module, 'showArmBase', False) and hasattr(module, 'ArmBasePose'):
        try:
            Add('arm base', module.ArmBasePose(posPlatform, ThetaPlatform))
        except studentErrors:
            pass
    if getattr(module, 'showGraspPoses', False) and hasattr(module, 'GraspPoses'):
        try:
            for name, pose in module.GraspPoses(PosObj).items():
                Add(name, pose)
        except studentErrors:
            pass
    return frames
