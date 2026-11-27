# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: preparationStateMachine.py
#
# Details:  Preparation of the mobile robot lab (Vorbereitung): state machine of the sample return mission.
#           The robot drives to the object on the table, grasps it with the UR5 and brings it back to its start
#           pose. functionStateMachine is called by the simulation (and in the lab on the real Robotnik Kairos)
#           whenever arm and gripper are at rest. The subtasks are helper functions (approach pose, drive command,
#           arm base, grasp poses), each run alone with runTask and seen in the renderer (switches show...).
#
# Authors:  state machine of the former preparation by Peter Manzl; Michael Pieber (created with Claude Code)
# Date:     2026-11-08
# Notes:    Run: python preparationStateMachine.py [--noGraphics]   (or F5; same as
#           python runSimulation.py --taskFile preparationStateMachine.py); the renderer shows the simulation live,
#           afterwards the SolutionViewer replays it
##BEGIN_MASTERONLY
#           Master file: autoGenerateTemplates.py writes the student template (isTemplate blocks -> ## TODO).
##END_MASTERONLY
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import sys
import numpy as np
from exudyn.rigidBodyUtilities import HTtranslate, HTrotateY, HTrotateZ
from fahrzeug import parameters as par

##BEGIN_MASTERONLY
isTemplate = False   #True: run as the student template would (without the solution)
##END_MASTERONLY

#! Subtask the simulation runs, each one alone, in the order of the TODOs: 'drive' to the approach pose (TODO 1, 2),
#! 'return' from the approach pose to the start pose [0, 0, 0] (TODO 2), 'grasp' from the approach pose (TODO 3, 4),
#! 'stateMachine' the whole mission (functionStateMachine, TODO 5). In the lab: always 'stateMachine'.
if not isTemplate:
    runTask = 'stateMachine'
else:
    runTask = 'drive'
#! Start pose [x in m, y in m, yaw in deg] of the simulation for every runTask, e.g. testStartPose = [-0.5, 0.5, 0.]
#! to test 'drive' or 'stateMachine' from other places. None: the default start, [0, 0, 0] for 'drive' and
#! 'stateMachine', ApproachPose (TODO 1) for 'grasp' and 'return'. Yaw in deg here (ApproachPose: rad). In the lab
#! the robot starts where it stands.
testStartPose = None
#! Frames in the renderer (True: show), also before the other subtasks are solved: the approach pose (TODO 1), the
#! arm base of ArmBasePose (TODO 3: it moves with the platform and must sit on the base of the UR5), the grasp
#! poses (TODO 4) at the object. The template starts with TODO 1 only.
if not isTemplate:
    showApproachPose = showArmBase = showGraspPoses = True
else:
    showApproachPose = True
    showArmBase = False
    showGraspPoses = False

if not isTemplate:
    from fahrzeug.reference.taskSequence import tableApproachDistance
    #! States of the mission; -1 ends the task.
    stateApproach, statePregrasp, stateGrasp, stateClose, stateLift, stateReturn, stateFinished = 0, 1, 2, 3, 4, 5, -1
    #! State after the drive to the approach pose: in the lab called with the Vicon pose after the replayed drive.
    stateCheckApproach = 6
    #! Correction drives to the approach pose at most (the measured pose missed it).
    maximumApproachCorrections = 3
    #! Gripper axis horizontal towards the object, fingers closing horizontally (as the real grasp).
    graspRotation = HTrotateY(np.pi/2.) @ HTrotateZ(np.pi/2.)
else:
    ## TODO 5: The states of your state machine: one name per step of the mission, each with its own number
    ##         (names and number of steps are yours; -1 ends the task). Every state you use in
    ##         functionStateMachine needs its name here, e.g.
    ##         stateDriveToTable, statePregrasp, stateGrasp, ..., stateDone = 0, 1, 2, ..., -1
    stateDriveToTable, stateDone = 0, -1


def ApproachPose(PosObj):
    """Pose of the platform in front of the table from which the arm reaches the object (TODO 1).

    Input:
        PosObj        position [x, y, z] in m of the object marker (center of the object top) in K0;
                      numpy array of shape (3,)
    Output:
        pose          [x in m, y in m, yaw in rad] of the platform in K0 (list or array of 3 floats)
    """
    if not isTemplate:
        return [PosObj[0] - tableApproachDistance, PosObj[1], 0.]
    else:
        ## TODO 1: Where must the platform stand to grasp the object? Return [x, y, yaw] in front of the table
        ##         (the platform facing +x of K0, the object on the table in front of it), using PosObj and the
        ##         data of fahrzeug/parameters.py: tableSize, tableHeight (table centered under the object),
        ##         mountPosition, mountYaw (arm base on the platform), platformDimensions; the reach of
        ##         the UR5 (850 mm) is given in the task sheet.
        ##         Check in the renderer: frame 'approach' on the floor (showApproachPose); runTask = 'drive' drives
        ##         there with DriveCommand (TODO 2).
        return [0., 0., 0.]


def DriveCommand(t, posPlatform, ThetaPlatform, target, driveData):
    """Drive command of one control step towards a target pose (TODO 2).

    Input:
        t             time in s (float), as in functionStateMachine
        posPlatform   position [x, y, z] in m of the platform in K0 (numpy array of shape (3,)); only x, y are used
        ThetaPlatform yaw in rad of the platform (float, in [-pi, pi))
        target        target pose [x in m, y in m, yaw in rad] in K0 (e.g. of ApproachPose)
        driveData     your own memory of this drive (dict), kept between the calls (DriveCommand runs every 2 ms,
                      local variables are lost): empty at the first call of a drive, fill it there with what you need
                      later, e.g. the start time and start pose of a ramp profile. Keys and content are free, only
                      your DriveCommand reads it; a P control needs nothing. Changed in place, not returned; one dict
                      per drive
    Output:
        v             velocity [vx in m/s, vy in m/s, omega in rad/s] in the local frame of the platform (x forward,
                      y left); list or array of 3 floats
        reached       True if the target is reached within par.positionTolerance and par.angleTolerance (bool)
    """
    if not isTemplate:
        errorWorld = np.array(target[:2]) - np.array(posPlatform[:2])
        errorYaw = (target[2] - ThetaPlatform + np.pi) % (2.*np.pi) - np.pi
        cosine, sine = np.cos(ThetaPlatform), np.sin(ThetaPlatform)
        errorLocal = np.array([cosine*errorWorld[0] + sine*errorWorld[1], -sine*errorWorld[0] + cosine*errorWorld[1]])
        velocity = par.driveGain*errorLocal
        speed = np.linalg.norm(velocity)
        if speed > par.maximumDriveSpeed:
            velocity *= par.maximumDriveSpeed/speed
        omega = np.clip(par.turnGain*errorYaw, -par.maximumTurnRate, par.maximumTurnRate)
        reached = np.linalg.norm(errorWorld) < par.positionTolerance and abs(errorYaw) < par.angleTolerance
        return [velocity[0], velocity[1], omega], reached
    else:
        ## TODO 2: Drive to the target, either with the trajectory planning of exercise sheet 3 (ramp profile,
        ##         max. 0.3 m/s, 0.4 rad/s and 0.3 m/s^2, 0.3 rad/s^2: at the first call, driveData empty, store t
        ##         and the start pose in driveData, then command the velocity of the planned profile at t plus a P
        ##         term on the deviation from the planned pose) or with a P control without integral part. The
        ##         velocity is given in the local frame of the platform (rotate the error in K0 by -ThetaPlatform);
        ##         target reached within a tolerance (par.positionTolerance, par.angleTolerance). Mind negative
        ##         directions (driving back). Test alone: runTask = 'drive' (to ApproachPose) or 'return'.
        #start of a ramp profile; a P control does not need driveData
        if not driveData:                                   #first call of this drive
            driveData['startTime'] = t
            driveData['startPose'] = [posPlatform[0], posPlatform[1], ThetaPlatform]
            #optionally computed once here: distance, direction and duration T of the profile
        tau = t - driveData['startTime']                    #time in s since the start of this drive
        #planned pose and velocity of the profile at tau, starting at driveData['startPose']
        return [0., 0., 0.], False


def ArmBasePose(posPlatform, ThetaPlatform):
    """Pose of the UR5 base in K0, T_0,base (TODO 3). It maps
    coordinates in the arm base frame to K0: p_0 = T_0,base @ p_base.

    Input:
        posPlatform   position [x, y, z] in m of the platform in K0, z: height of the arm base (numpy array of
                      shape (3,))
        ThetaPlatform yaw in rad of the platform (float, in [-pi, pi))
    Output:
        T0Base        4x4 homogeneous transformation (numpy array of shape (4, 4)), lengths in m
    """
    if not isTemplate:
        mountPosition = [par.mountPosition[0], par.mountPosition[1], 0.]
        return HTtranslate(posPlatform) @ HTrotateZ(ThetaPlatform) @ HTtranslate(mountPosition) @ HTrotateZ(par.mountYaw)
    else:
        ## TODO 3: The pose of the arm base in K0: the platform at posPlatform turned by ThetaPlatform, the arm base
        ##         at par.mountPosition (x, y) on the platform and turned by par.mountYaw (HTtranslate, HTrotateZ).
        ##         Check in the renderer: frame 'arm base' (showArmBase) must sit on the base of the UR5 and move
        ##         with the platform.
        return HTtranslate(posPlatform)


def GraspPoses(PosObj):
    """TCP poses in K0 of grasping the object (TODO 4).

    Input:
        PosObj        position [x, y, z] in m of the object marker (center of the object top) in K0;
                      numpy array of shape (3,)
    Output:
        poses         dict with exactly the keys 'pregrasp' (in front of the object), 'grasp' (at the object) and
                      'lift' (the object lifted), each a 4x4 homogeneous transformation T_0,TCP (numpy array of shape
                      (4, 4), lengths in m); runTask = 'grasp' and showGraspPoses use these keys
    """
    if not isTemplate:
        graspPose = HTtranslate(np.array(PosObj) - [0., 0., par.graspDistanceBelowMarker]) @ graspRotation
        return {'pregrasp': graspPose @ HTtranslate([0., 0., -par.objectApproachDistance]), 'grasp': graspPose,
                'lift': HTtranslate(par.objectRetreatOffset) @ graspPose}
    else:
        ## TODO 4: The TCP poses (4x4 in K0) of the grasp. The TCP lies between the finger tips, its z axis points
        ##         out of the gripper, the fingers close along its x axis (EE in the figure of the task sheet).
        ##         'grasp': the TCP at the object axis, par.graspDistanceBelowMarker below the marker PosObj, z
        ##         horizontal in +x of K0 (towards the object seen from the approach pose), x horizontal (+y or -y of
        ##         K0, the gripper is symmetric), y then vertical; 'pregrasp': par.objectApproachDistance before it
        ##         along z; 'lift': lifted and pulled back (e.g. par.objectRetreatOffset in K0). Check in the renderer:
        ##         frames 'pregrasp', 'grasp', 'lift' at the object (showGraspPoses); runTask = 'grasp' moves the arm.
        ##         Rotation: the columns of the 3x3 rotation of a pose are the TCP axes x, y, z written in K0 (or a
        ##         product of HTrotateY, HTrotateZ, ...).
        graspPose = HTtranslate(PosObj)         #position below the marker and rotation of the TCP
        pregraspPose = graspPose                #moved back along the z axis of the TCP
        liftPose = graspPose                    #lifted and pulled back
        return {'pregrasp': pregraspPose, 'grasp': graspPose, 'lift': liftPose}


def ArmTarget(posPlatform, ThetaPlatform, worldPose):
    """TCP target relative to the arm base for a TCP pose in K0: TArm = T_base,0 @ T_0,TCP = inv(ArmBasePose) @ worldPose
    (given, no TODO).

    Input:
        posPlatform   position [x, y, z] in m of the platform in K0 (numpy array of shape (3,))
        ThetaPlatform yaw in rad of the platform (float)
        worldPose     TCP pose T_0,TCP in K0 (4x4 numpy array, e.g. of GraspPoses)
    Output:
        TArm          TCP pose T_base,TCP relative to the arm base (4x4 numpy array), the output TArm of
                      functionStateMachine
    """
    return np.linalg.inv(ArmBasePose(posPlatform, ThetaPlatform)) @ worldPose


def functionStateMachine(t, posPlatform, ThetaPlatform, PosObj, myState, myDict):
    """State machine of the sample return mission (TODO 5). It is called in every control step (every 2 ms) while arm
    and gripper are at rest, not once per state; during an arm or gripper motion it is not called. The names PosObj
    and ThetaPlatform are those of the interface of the real robot (bringup_lecture.py).

    Input:
        t             time in s (float), starts at 0
        posPlatform   position [x, y, z] in m of the mobile platform in K0, z: height of the arm base (numpy array of
                      shape (3,))
        ThetaPlatform yaw in rad of the platform about the vertical axis (float, in [-pi, pi))
        PosObj        position [x, y, z] in m of the object marker (center of the object top) in K0 (numpy array of
                      shape (3,)); in the lab updated from Vicon after every drive
        myState       state of the state machine (int, 0 at the first call)
        myDict        dict kept between the calls (empty at the first call)
    Output:
        v             velocity [vx in m/s, vy in m/s, omega in rad/s] of the platform in its local frame (x forward,
                      y left), list or array of 3 floats, limited to 0.3 m/s and 0.4 rad/s; None: the platform
                      stands still; ignored (zero) in a call that returns TArm
        TArm          4x4 TCP target pose T_base,TCP relative to the arm base (numpy array, e.g. of ArmTarget); a new
                      target starts an arm motion; None: no new arm motion. An unreachable target is only reported as a
                      warning, the state machine goes on
        grasp         True: close the gripper, False: open it (bool); None: the gripper stays as it is
        myState       new state (int); -1 ends the task
        myDict        the dict for the next call
    """
    v = [0., 0., 0.]
    TArm = None
    grasp = None
    if not isTemplate:
        if 'startPose' not in myDict:
            myDict['startPose'] = [posPlatform[0], posPlatform[1], ThetaPlatform]
            myDict['approachPose'] = ApproachPose(PosObj)
        if myState == stateApproach:
            v, reached = DriveCommand(t, posPlatform, ThetaPlatform, myDict['approachPose'], myDict.setdefault('driveApproach', {}))
            if reached:
                v, myState = [0., 0., 0.], stateCheckApproach
        elif myState == stateCheckApproach:
            #in the lab the drive is simulated, then replayed on the robot; this state is called with the measured
            #(Vicon) pose afterwards: drive again from there if it missed the approach pose (in the simulation it
            #is reached already)
            error = np.array(myDict['approachPose'][:2]) - posPlatform[:2]
            yawError = (myDict['approachPose'][2] - ThetaPlatform + np.pi) % (2.*np.pi) - np.pi
            missed = np.linalg.norm(error) > par.positionTolerance or abs(yawError) > par.angleTolerance
            corrections = myDict.get('approachCorrections', 0)
            if missed and corrections < maximumApproachCorrections:
                print('approach pose missed by %.1f mm, %.2f deg: correction drive %d'
                      % (1e3*np.linalg.norm(error), np.degrees(yawError), corrections + 1))
                myDict['approachCorrections'] = corrections + 1
                myDict['driveApproach'] = {}                #a new drive from the measured pose
                myState = stateApproach
            else:
                myState = statePregrasp
        elif myState == statePregrasp:
            #grasp poses at the object as measured after the drive (in the lab PosObj is taken
            #over from Vicon after every drive), not at the start
            myDict['graspPoses'] = GraspPoses(PosObj)
            TArm = ArmTarget(posPlatform, ThetaPlatform, myDict['graspPoses']['pregrasp'])
            myState = stateGrasp
        elif myState == stateGrasp:
            TArm = ArmTarget(posPlatform, ThetaPlatform, myDict['graspPoses']['grasp'])
            myState = stateClose
        elif myState == stateClose:
            grasp = True
            myState = stateLift
        elif myState == stateLift:
            TArm = ArmTarget(posPlatform, ThetaPlatform, myDict['graspPoses']['lift'])
            myState = stateReturn
        elif myState == stateReturn:
            v, reached = DriveCommand(t, posPlatform, ThetaPlatform, myDict['startPose'], myDict.setdefault('driveReturn', {}))
            if reached:
                v, myState = [0., 0., 0.], stateFinished
    else:
        ## TODO 5: Implement the state machine of the sample return mission (task sheet) with the helper functions:
        ##         drive to ApproachPose (DriveCommand(t, posPlatform, ThetaPlatform, target, driveData) with one dict
        ##         per drive, e.g. myDict.setdefault('driveApproach', {})), move the arm to the pregrasp and grasp pose (ArmTarget of
        ##         GraspPoses), close the gripper, lift the object and drive back to the start pose; return
        ##         myState = -1 at the end. In one state either the platform, the arm or the gripper moves. Store data
        ##         between the calls in myDict (e.g. the start pose at the first call). Compute the grasp poses when
        ##         the platform stands at the approach pose: in the lab PosObj is updated from Vicon after every drive.
        ##         Run it with runTask = 'stateMachine'.
        #myState is the number of the present step (names: top of the file); the first state shows the pattern
        if 'startPose' not in myDict:                   #first call: store what you need later
            myDict['startPose'] = [posPlatform[0], posPlatform[1], ThetaPlatform]
            myDict['approachPose'] = ApproachPose(PosObj)
        if myState == stateDriveToTable:                    #a drive state stays until the target is reached
            v, reached = DriveCommand(t, posPlatform, ThetaPlatform, myDict['approachPose'],
                                      myDict.setdefault('driveApproach', {}))
            if reached:
                v, myState = [0., 0., 0.], stateDone    #here the number of your next state instead of the end
        #next states: add the name at the top of the file and one 'elif' block here per state, e.g. an arm state
        #(one call: it returns TArm and switches at once; the function is called again when the arm has arrived):
        #elif myState == statePregrasp:
        #    TArm = ArmTarget(posPlatform, ThetaPlatform, <pose T_0,TCP of GraspPoses>)
        #    myState = stateGrasp
        #a gripper state the same with grasp = True instead of TArm
    return v, TArm, grasp, myState, myDict


if __name__ == '__main__':
    import runSimulation
    sys.argv = [sys.argv[0], '--taskFile', __file__] + sys.argv[1:]
    runSimulation.Main()
