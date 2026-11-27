# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: taskSequence.py
# Details:  Validated action lists, their execution with the drive controller (feedback completion) and the
#           missions of the lab (sample return, placing, drive demonstration); reference solution and
#           lab drives, not part of the student package (moved from fahrzeug/).
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-06
# Notes:    Drive targets: world [x,y,yaw]. Arm targets: 4x4 pose in specified frame.
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Sequential lab tasks."""
from copy import deepcopy
import numpy as np
from exudyn.rigidBodyUtilities import HTtranslate, HTrotateY, HTrotateZ
from .. import parameters as par
from labRobots.robots.kairos.kinematics import WrapAngle
from labRobots.robotics.collision.shapes import SegmentDistance, PointBoxDistance

#! Distance in m along x of the approach pose of the platform from the object (solution of the preparation, TODO 1;
#! was par.tableApproachDistance: not in the student package).
tableApproachDistance = 0.878

#! Supported action and frame identifiers.
actionDrive, actionArm, actionGrasp, actionWait = 'drive', 'arm', 'grasp', 'wait'
#! Arm motion to joint angles in rad (reproducible, e.g. the home pose).
actionJoints = 'joints'
#! Platform velocity [vx in m/s, vy in m/s, omega in rad/s] in the vehicle frame for a duration, then stop (open loop,
#! e.g. steps of the identification program; not corrected after the replay on the real robot).
actionVelocity = 'velocity'
frameWorld, frameArm = 'world', 'arm'
#! Arm path: planned joint-space motion (default) or straight TCP line.
pathJoint, pathLinear = 'joint', 'linear'
actionTypes = (actionDrive, actionArm, actionGrasp, actionWait, actionJoints, actionVelocity)


def DriveToPose(pose, targetPose):
    """Return bounded local velocity and world pose error for [x, y, yaw] targets in SI units."""
    error = np.asarray(targetPose, dtype=float) - np.asarray(pose, dtype=float)
    error[2] = WrapAngle(error[2])
    worldVelocity = par.driveGain*error[:2]
    speed = np.linalg.norm(worldVelocity)
    if speed > par.maximumDriveSpeed:
        worldVelocity *= par.maximumDriveSpeed/speed
    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
    localVelocity = np.array([[cosine, sine], [-sine, cosine]]) @ worldVelocity
    return np.r_[localVelocity, np.clip(par.turnGain*error[2], -par.maximumTurnRate, par.maximumTurnRate)], error


#moved from collisionModel.py (only the action lists use it)
def TableApproachTargets(startPose, tablePosition):
    #! Return world drive targets to the table's approach pose, detouring around its footprint if needed.
    #the parked vehicle uses a circular envelope, independent of its yaw.
    start = np.array([[startPose[0], startPose[1], 0.]])
    target = np.array([[tablePosition[0] - tableApproachDistance, tablePosition[1], 0.]])
    center = np.array([[tablePosition[0], tablePosition[1], 0.]])
    halfSize = np.array([[par.tableSize[0]/2., par.tableSize[1]/2., par.tableHeight/2.]])
    rotation = np.eye(par.dimension3D)[None, :, :]

    def DistanceToTable(points):
        #! Evaluate the existing convex box distance for the swept horizontal drive segment.
        return PointBoxDistance(points, center, halfSize, rotation)

    clearance = par.tableDriveRadius + par.tableDriveMargin
    finalPose = np.array([target[0, 0], target[0, 1], 0.])
    if SegmentDistance(start, target, DistanceToTable)[0] >= clearance:
        return [finalPose]
    side = 1. if startPose[1] >= tablePosition[1] else -1.
    sidePosition = tablePosition[1] + side*(par.tableSize[1]/2. + clearance)
    return [np.array([startPose[0], sidePosition, startPose[2]]),
            np.array([finalPose[0], sidePosition, 0.]), finalPose]


def ValidateActions(actions):
    """Return an independent validated action list; reject malformed commands before starting motion."""
    result = deepcopy(list(actions))
    if not result:
        raise ValueError('An action sequence must contain at least one action')
    for index, action in enumerate(result):
        if action.get('type') not in actionTypes or not isinstance(action.get('label'), str):
            raise ValueError('Every action needs a supported type and a label')
        kind = action['type']
        if kind in (actionDrive, actionArm):
            target = np.asarray(action.get('target'), dtype=float)
            expectedShape = (par.dimension3D,) if kind == actionDrive else (4, 4)
            if target.shape != expectedShape or not np.all(np.isfinite(target)):
                raise ValueError('Invalid target for action: ' + action['label'])
            if kind == actionArm:
                rotation = target[:3, :3]
                if not (np.allclose(target[3], [0., 0., 0., 1.]) and
                        np.allclose(rotation.T @ rotation, np.eye(par.dimension3D)) and np.isclose(np.linalg.det(rotation), 1.)):
                    raise ValueError('Arm target must be a homogeneous rigid transformation')
                if action.get('frame', frameArm) not in (frameWorld, frameArm):
                    raise ValueError('Arm frame must be world or arm')
                if action.get('path', pathJoint) not in (pathJoint, pathLinear):
                    raise ValueError('Arm path must be joint or linear')
            action['target'] = target
        elif kind == actionJoints:
            target = np.asarray(action.get('target'), dtype=float)
            if (target.shape != (par.numberOfArmJoints,) or not np.all(np.isfinite(target)) or
                    np.any(target < par.armJointLowerLimits) or np.any(target > par.armJointUpperLimits)):
                raise ValueError('Joint target needs %d finite angles in rad within the joint limits: %s'
                                 % (par.numberOfArmJoints, action['label']))
            #optional 'speedScale' in (0, 1] scales the joint speed and acceleration limits
            speedScale = action.get('speedScale', 1.)
            if not (isinstance(speedScale, (int, float)) and 0. < speedScale <= 1.):
                raise ValueError('speedScale of a joint action must be in (0, 1]: ' + action['label'])
            action['target'] = target
        elif kind == actionGrasp and (not isinstance(action.get('enabled'), bool)
                                      or not isinstance(action.get('requireObject', True), bool)):
            raise ValueError('A grasp action needs a boolean enabled value (and requireObject if given)')
        elif kind == actionWait and (not np.isfinite(action.get('duration', np.nan)) or action['duration'] < 0.):
            raise ValueError('A wait action needs a nonnegative duration in seconds')
        elif kind == actionVelocity:
            velocity = np.asarray(action.get('velocity'), dtype=float)
            duration = float(action.get('duration', np.nan))
            if velocity.shape != (par.dimension3D,) or not np.all(np.isfinite(velocity)):
                raise ValueError('A velocity action needs [vx, vy, omega]: ' + action['label'])
            if np.linalg.norm(velocity[:2]) > par.maximumDriveSpeed + 1e-12 or abs(velocity[2]) > par.maximumTurnRate + 1e-12:
                raise ValueError('Velocity action above the limits %.2f m/s, %.2f rad/s: %s'
                                 % (par.maximumDriveSpeed, par.maximumTurnRate, action['label']))
            if not 0. < duration <= par.velocityActionMaximumDuration:
                raise ValueError('A velocity action needs a duration in (0, %.0f] s: %s'
                                 % (par.velocityActionMaximumDuration, action['label']))
            #'continue': True passes on to the next velocity action without stopping
            if not isinstance(action.get('continue', False), bool):
                raise ValueError('continue of a velocity action must be True or False: ' + action['label'])
            if action.get('continue', False) and (index + 1 == len(result) or result[index + 1].get('type') != actionVelocity):
                raise ValueError('A continuing velocity action must be followed by a velocity action: ' + action['label'])
            action['velocity'] = velocity
    return result


def SampleReturnActions(model):
    """Return the preparation task: approach, pregrasp, grasp, retreat, and drive home carrying the object."""
    state = model.GetState()
    position = state['objectPosition']
    #remember the actual pose when the mission is requested, including after manual driving.
    returnPose = state['pose'].copy()
    #grasp above the object center, so that the gripper body stays clear of the table
    objectPose = HTtranslate(position + [0., 0., par.graspHeightAboveCenter]) @ HTrotateY(np.pi/2.) @ HTrotateZ(np.pi/2.)
    #drive only with the arm in the park pose (joint target par.armHomeAngles)
    return [
        {'type': actionJoints, 'label': 'Arm in Parkstellung', 'target': par.armHomeAngles},
        {'type': actionDrive, 'label': 'Vorposition anfahren', 'target': [position[0] - tableApproachDistance, position[1], 0.]},
        {'type': actionArm, 'label': 'Arm in Vorgriffposition',
         'target': objectPose @ HTtranslate([0., 0., -par.objectApproachDistance]), 'frame': frameWorld},
        {'type': actionArm, 'label': 'Arm zum Objekt', 'target': objectPose, 'frame': frameWorld, 'path': pathLinear},
        {'type': actionGrasp, 'label': 'Objekt greifen', 'enabled': True},
        {'type': actionArm, 'label': 'Arm zurueckziehen', 'target': HTtranslate(par.objectRetreatOffset) @ objectPose, 'frame': frameWorld,
         'path': pathLinear},
        {'type': actionJoints, 'label': 'Arm mit Objekt in Parkstellung', 'target': par.armHomeAngles},
        {'type': actionDrive, 'label': 'Zur Ausgangspose fahren', 'target': returnPose},
    ]


def PlaceObjectActions(model):
    #! Return a held cylinder upright to its original table from the current vehicle/arm pose.
    #preserve the measured grip offset instead of assuming a centered grasp.
    state = model.GetState()
    if not state['grasped']:
        raise ValueError('Kein Objekt im Greifer: zuerst ein Objekt greifen')
    toolInObject = np.linalg.inv(model.BodyPose(model.bodies['object'])) @ model.ToolPose()
    tablePosition = model.tablePosition   #present table position (movable table)
    objectTarget = HTtranslate([tablePosition[0], tablePosition[1],
                                model.TableTopHeight() + par.objectHeight/2. + par.objectPlacementClearance])
    # A cylinder has no preferred yaw: turn its free yaw so the gripper approaches along world +x.
    approachDirection = toolInObject[:2, 2]
    if np.linalg.norm(approachDirection) > np.sqrt(np.finfo(float).eps):
        objectTarget = objectTarget @ HTrotateZ(-np.arctan2(approachDirection[1], approachDirection[0]))
    placeTarget = objectTarget @ toolInObject
    aboveTarget = HTtranslate([0., 0., par.objectApproachDistance]) @ placeTarget
    driveActions = [{'type': actionDrive, 'label': 'Zum Stehtisch fahren', 'target': target}
                    for target in TableApproachTargets(state['pose'], tablePosition)]
    return ([{'type': actionJoints, 'label': 'Objekt in Transportstellung bringen', 'target': par.armHomeAngles}] +
            driveActions + [
                {'type': actionArm, 'label': 'Objekt ueber dem Stehtisch positionieren',
                 'target': aboveTarget, 'frame': frameWorld},
                {'type': actionArm, 'label': 'Objekt auf den Stehtisch absenken',
                 'target': placeTarget, 'frame': frameWorld, 'path': pathLinear},
                {'type': actionGrasp, 'label': 'Objekt abstellen und Greifer oeffnen', 'enabled': False},
                {'type': actionArm, 'label': 'Greifer vom Objekt zurueckziehen',
                 'target': aboveTarget, 'frame': frameWorld, 'path': pathLinear},
                {'type': actionJoints, 'label': 'Arm in Parkstellung', 'target': par.armHomeAngles},
            ])


def DriveDemoActions(model):
    """Return forward, sideways, turn and return actions relative to the present world pose."""
    pose = model.GetState()['pose']
    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
    forward = pose + [0.5*cosine, 0.5*sine, 0.]
    sideways = forward + [-0.4*sine, 0.4*cosine, 0.]
    return [{'type': actionDrive, 'label': label, 'target': target}
            for label, target in [('Vorwaerts', forward), ('Seitwaerts', sideways),
                                   ('Drehen', sideways + [0., 0., np.pi/2.]), ('Zurueck', pose)]]


class TaskSequence:
    """Execute a list using actual model feedback, with optional single-action operation."""

    def __init__(self, actionsInit):
        """Validate and store the sequence; no action starts until Start or Next is called."""
        self.actions = ValidateActions(actionsInit)  #! Independent sequence in execution order.
        self.index = 0  #! Index of current or next action.
        self.running = False  #! True while executing an action.
        self.singleAction = False  #! Stop at the next action boundary.
        self.started = False  #! Current action entry has been executed.
        self.actionStart = 0.  #! Simulation time at current action entry.
        self.targetError = None  #! Latest world drive error [m,m,rad].
        self.message = 'Bereit'  #! Human-readable task status.

    def Start(self, singleAction=False):
        """Start/resume execution, or replay a completed sequence from its first action."""
        if self.index >= len(self.actions):
            self.index = 0
            self.started = False
        self.singleAction = bool(singleAction)
        self.running = True
        self.message = ''

    def Stop(self):
        """Cancel execution and allow the current action to be planned anew on the next start."""
        self.running = False
        self.started = False
        self.message = 'Gestoppt'

    def Update(self, model, time):
        """Execute at most one action boundary; return local platform velocity in SI units."""
        velocity = np.zeros(par.dimension3D)
        if not self.running:
            return velocity
        action = self.actions[self.index]
        kind = action['type']
        if not self.started:
            self.actionStart = time
            self.targetError = None
            if kind == actionArm:
                target = action['target']
                if action.get('frame', frameArm) == frameWorld:
                    target = np.linalg.inv(model.ArmBasePose()) @ target
                if action.get('path', pathJoint) == pathLinear:
                    model.MoveArmLinear(target, time)
                else:
                    model.MoveArm(target, time)
            elif kind == actionJoints:
                model.MoveArmToJoints(action['target'], time, action.get('speedScale', 1.))
            elif kind == actionGrasp:
                #'requireObject': False closes the fingers on air (gripper measurement)
                model.SetGrasp(action['enabled'], action.get('requireObject', True))
            self.started = True
        state = model.GetState()
        complete = False
        if kind == actionDrive:
            velocity, self.targetError = DriveToPose(state['pose'], action['target'])
            inside = bool(np.linalg.norm(self.targetError[:2]) <= par.positionTolerance and
                          abs(self.targetError[2]) <= par.angleTolerance)
            standing = bool(np.linalg.norm(state['velocity'][:2]) <= par.linearVelocityTolerance and
                            abs(state['velocity'][2]) <= par.angularVelocityTolerance)
            complete = inside and standing
        elif kind in (actionArm, actionJoints):
            complete = state['armReady']
        elif kind == actionGrasp:
            #without required object (gripper measurement) only the finger motion counts
            complete = ((state['grasped'] == action['enabled'] or not action.get('requireObject', True))
                        and state['gripperReady'])
        elif kind == actionWait:
            complete = time - self.actionStart >= action['duration']
        elif kind == actionVelocity:
            #the commanded velocity for the duration, then zero until the vehicle stands
            if time - self.actionStart < action['duration']:
                velocity = np.array(action['velocity'], dtype=float)
            elif action.get('continue', False):   #the next velocity action takes over without a stop
                velocity = np.array(action['velocity'], dtype=float)
                complete = True
            else:
                complete = bool(np.linalg.norm(state['velocity'][:2]) <= par.linearVelocityTolerance and
                                abs(state['velocity'][2]) <= par.angularVelocityTolerance)
        if time - self.actionStart > par.actionTimeout and kind != actionWait:
            raise TimeoutError('Action did not reach its target: ' + action['label'])
        if complete:
            self.CompleteAction()
            if not (kind == actionVelocity and action.get('continue', False)):
                velocity[:] = 0.
        return velocity

    def CompleteAction(self):
        """Mark the current action complete and go on to the next one (also used by the step-by-step execution on the
        real robot: a drive action is complete when the Vicon pose was taken over)."""
        self.index += 1
        self.started = False
        self.running = not self.singleAction and self.index < len(self.actions)
        self.message = 'Fertig' if self.index == len(self.actions) else ('Wartet auf naechste Aktion' if self.singleAction else '')

    def Description(self):
        """Return the current action number, label and status for the control panel."""
        if self.index == len(self.actions):
            return 'Ablauf abgeschlossen (%d/%d)' % (self.index, len(self.actions))
        return 'Aktion %d/%d: %s%s' % (self.index + 1, len(self.actions), self.actions[self.index]['label'],
                                      ' | ' + self.message if self.message else '')
