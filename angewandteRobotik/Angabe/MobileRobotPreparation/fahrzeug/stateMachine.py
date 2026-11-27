# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: stateMachine.py
# Details:  Interface of the student state machine: functionStateMachine of the preparation (6 arguments) or
#           of the real run bringup_lecture.py (7 arguments, with armStatus) controls the model.
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-05 (moved from taskSequence.py: 2026-11-08)
# Notes:    Base of the student package; action lists are in fahrzeug/reference/taskSequence.py.
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Student state machine controller (from taskSequence.py)."""
import inspect
import numpy as np
from . import parameters as par
from labRobots.robots.kairos.kinematics import ClampPlatformVelocity

#! functionStateMachine argument counts: preparation, and real run with armStatus (bringup_lecture.py).
preparationArgumentCount, labArgumentCount = 6, 7
#! armStatus of bringup_lecture.py: 1 = arm motion finished (stateArm).
labArmStatusReady = 1


class PreparationController:
    """Run an unchanged preparation or real-run (bringup_lecture.py) functionStateMachine against the new model."""

    def __init__(self, callbackInit):
        """Store a callable with five return values and either the preparation's six arguments
        (t, posPlatform, ThetaPlatform, PosObj, myState, myDict) or the seven arguments of the real run,
        which add armStatus before myState."""
        if not callable(callbackInit):
            raise TypeError('Preparation callback must be callable')
        #accept the bringup_lecture.py signature so one file runs in Exudyn and on the robot.
        try:
            argumentCount = len(inspect.signature(callbackInit).parameters)
        except (TypeError, ValueError) as error:
            raise TypeError('functionStateMachine signature cannot be inspected: ' + str(error)) from error
        if argumentCount not in (preparationArgumentCount, labArgumentCount):
            raise TypeError('functionStateMachine needs %d arguments (preparation) or %d arguments with armStatus '
                            '(bringup_lecture.py), not %d' % (preparationArgumentCount, labArgumentCount, argumentCount))
        self.passArmStatus = argumentCount == labArgumentCount  #! True for the real-run signature.
        self.callback = callbackInit  #! User state-machine implementation.
        self.state = 0  #! Persistent preparation state.
        self.data = {}  #! Persistent preparation dictionary.
        self.startTime = None  #! Callback time starts at zero on first execution.
        self.finished = False  #! State -1 ends the callback task.
        self.warnings = []  #! (time, message) of failed arm/gripper commands; the real run continues after them.
        self.clampedStates = set()  #! States whose velocity exceeded the limits (warned once per state).
        #the arm is ready once it settled after its last motion and stays ready until the next
        #arm target (as armStatus of the real arm: motion finished); before, the swing of the wrist on the accelerating
        #platform (up to 1 rad/s) made the model not ready at every drive step, so the platform drove in jerks
        self.armSettled = False  #! The arm settled after its last motion (no new arm target since).

    def AddWarning(self, time, message):
        """Keep a warning of the state machine (unreachable arm target, clamped velocity) and print it at once
        (before, the warnings were collected but never shown; the run continues)."""
        self.warnings.append((time, message))
        print('Warning at t = %.2f s: %s' % (time, message))

    def Update(self, model, time):
        """Call the preparation controller when the arm is ready, retaining its original local velocity convention."""
        state = model.GetState()
        if self.startTime is None:
            self.startTime = time
        if state['armReady']:
            self.armSettled = True
        if self.finished or not self.armSettled or not state['gripperReady']:
            return np.zeros(par.dimension3D)
        pose = state['pose']
        position = np.r_[pose[:2], model.ArmBasePose()[2, 3]]
        #PosObj is the single Vicon marker on top of the object (bringup_lecture.py callbackMarker)
        objectMarker = state['objectPosition'] + [0., 0., par.viconObjectMarkerHeight]
        arguments = [time - self.startTime, position, pose[2], objectMarker]
        if self.passArmStatus:
            #as in bringup_lecture.py: the callback is only evaluated while the arm is not moving
            arguments.append(labArmStatusReady)
        velocity, armTarget, grasp, self.state, self.data = self.callback(*arguments, self.state, self.data)
        #as in bringup_lecture.py, v = None (or None entries) means no platform motion.
        if velocity is None or any(value is None for value in velocity):
            velocity = np.zeros(par.dimension3D)
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (par.dimension3D,) or not np.all(np.isfinite(velocity)):
            raise ValueError('Preparation callback must return a finite [vx,vy,omega] vector')
        #the limits of the lab (0.3 m/s, 0.4 rad/s) apply to the state machine as to every
        #other drive command; before, a too fast velocity went unclamped to the simulation and the real robot
        velocity, clamped = ClampPlatformVelocity(velocity)
        if clamped and self.state not in self.clampedStates:
            self.clampedStates.add(self.state)
            self.AddWarning(time, 'state %s: velocity clamped to %.2f m/s, %.2f rad/s'
                            % (self.state, par.maximumDriveSpeed, par.maximumTurnRate))
        #as bringup_lecture.py: a failed plan (go() returns False) or closing without object
        #does not stop the run; the state machine continues and the failure is reported as warning
        if armTarget is not None:
            try:
                model.MoveArm(np.asarray(getattr(armTarget, 'A', armTarget)), time)
                self.armSettled = False         #wait for the end of this motion
            except ValueError as error:
                self.AddWarning(time, 'state %s: %s' % (self.state, error))
            velocity[:] = 0.
        if grasp is not None:
            model.SetGrasp(bool(grasp), requireObject=False)
        self.finished = self.state == -1
        return np.zeros(par.dimension3D) if self.finished else velocity
