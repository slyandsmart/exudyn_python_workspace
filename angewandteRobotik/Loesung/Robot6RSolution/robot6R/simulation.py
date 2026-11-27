# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: simulation.py
#
# Details:  Dynamic simulation of the 6R robot executing a lab PTP matrix Q:
#           Exudyn KinematicTree (minimal coordinates) with CAD inertia, PD joint
#           control following the PTP trajectory (Robot.SetControl/PreStepControl
#           of Exudyn), static gravity compensation,
#           gripper jaws and paper cup (grasp/release with payload update of
#           link 6), STL visualization of robot and lab, evaluation of TCP pose,
#           tracking errors, drive torques and cup tilt.
#           SimulateBahnfahrt(Q, 'rad') is the simulation counterpart of
#           BahnfahrtPython(Q, 'rad') in the lab.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-03
# Notes:    Drive torque = feed forward (gravity compensation) + PD control;
#           compare with schunkNominalTorques / schunkPeakTorques (axes 1-4).
#           joint control, gravity compensation and joint
#           sensors (incl. drive torques) by exudyn.robotics.Robot.
#
# License:  BSD-3 license
#
# Installation:
#           pip install exudyn numpy-stl
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
import numpy as np

if __package__ in (None, ''):
    #file started directly (e.g. F5 in the IDE): make package robot6R importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exudyn as exu
from exudyn.itemInterface import ObjectGround, VObjectGround
from exudyn.rigidBodyUtilities import InverseHT, RigidBodyInertia, RotationMatrixZ, RotationMatrixX
import exudyn.graphics as graphics

from robot6R import parameters as par
from labRobots.robots.robot6R.model import ManipulatorRobot6R, ForwardKinematics, GetLinkLocalHT
from robot6R.labObjects import LoadModelData, GetPayloadInertia
from robot6R import visualization
from labRobots.robots.robot6R import visualization as robotVisualization
from labRobots.robotics.simulation.solverFiles import SolutionFileForSolver
from robot6R.labTrajectory import LabTrajectory

projectDirectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
defaultSolutionFileName = os.path.join(projectDirectory, 'solution', 'robot6RSolution')
settlingTime = 0.5                  #s, simulated after the end of the trajectory
cupPlacementTolerance = 0.01        #m, max. height error for a correct placement of the cup
cupMaxGraspAxisCosine = 0.5         #|y6 . cup axis| must be smaller (closing direction perpendicular to cup axis)
cupMaxPlacementTiltDeg = 10.        #deg, max. tilt for a correct placement

#view of the renderer: looking from the front right onto robot and tables
initialViewCenter = [0.3, 0.2, 0.4]
initialViewZoom = 0.9
initialViewRotation = (RotationMatrixX(-0.5*np.pi + 0.35) @ RotationMatrixZ(-0.25*np.pi)).T  #renderer uses transposed rotation
rendererShadow = 0.     #shadows are computed on the CPU for every triangle (1 instead of 20 frames/s); set 0.25 for screenshots
gravityCompensationMode = 'static'  #compensation of Robot.SetControl: joint torques that hold the robot against gravity
jointSensorVariables = [exu.OutputVariableType.Coordinates, exu.OutputVariableType.Coordinates_t,
                        exu.OutputVariableType.Force]  #Robot.AddSensors: q, q_t and drive torques (PD + compensation)


#! Length in m of the labeled frames and of the frame of toolFrameFunction (AddFrameGraphics), colors of the latter
#! (darker than the frames, to tell them apart).
frameGraphicsLength, toolFrameGraphicsLength = 0.08, 0.12
#! Length and arrow radius in m of the labeled frames of toolFrameFunction (e.g. the DH frames K1..K6): short and thick
#! to tell them apart at the joints; labels of frames at the same origin are joined.
movingFrameGraphicsLength, movingFrameGraphicsRadius = 0.10, 0.003
sameOriginTolerance = 1e-6
#! Radius in m of the links and radius, length in m of the joint cylinders, color of the chain of the labeled frames
#! drawn without the lab robot (showRobot=False, AddFrameGraphics): the own DH model moving alone.
frameChainLinkRadius, frameChainJointRadius, frameChainJointLength = 0.008, 0.016, 0.05
frameChainColor = [0.55, 0.6, 0.75, 1.]
toolFrameColors = [[0.6, 0., 0., 1.], [0., 0.5, 0., 1.], [0., 0., 0.6, 1.]]


def FrameChainGraphics(poses):
    """Return graphics of a kinematic chain of frames (e.g. the DH frames K1..K6 of an own model, Claude Code,
    2026-11-16): links between the origins of K0 (identity) and the frames, a joint cylinder along z at K0 and at each
    frame except the last (in DH notation the axis of the next joint); invalid poses (None, wrong shape) are skipped."""
    chain = [np.eye(4)] + [np.asarray(pose, dtype=float) for pose in poses
                           if pose is not None and np.shape(pose) == (4, 4) and np.all(np.isfinite(pose))]
    graphicsList = []
    for index, pose in enumerate(chain[:-1]):
        link = chain[index + 1][0:3, 3] - pose[0:3, 3]
        if np.linalg.norm(link) > sameOriginTolerance:
            graphicsList.append(graphics.Cylinder(list(pose[0:3, 3]), list(link), frameChainLinkRadius, frameChainColor))
        axis = pose[0:3, 2]
        graphicsList.append(graphics.Cylinder(list(pose[0:3, 3] - 0.5*frameChainJointLength*axis),
                                              list(frameChainJointLength*axis), frameChainJointRadius, frameChainColor))
    return graphicsList


class ControlReference:
    """Joint reference for exudyn.robotics.Robot.SetControl: Evaluate(t) returns [q, q_t, q_tt] of all links from the
    reference function of the simulation (the trajectory, the jaw opening or the live command of the interactive mode)."""

    def __init__(self, referenceFunction):
        """referenceFunction(t) returns [q, q_t, q_tt] of all links of the robot."""
        self.referenceFunction = referenceFunction

    def Evaluate(self, t):
        """Return [q, q_t, q_tt] at time t."""
        return self.referenceFunction(t)


class Robot6RSimulation:
    """Dynamic simulation of the 6R robot following a LabTrajectory (PTP matrix Q of the lab)."""

    solverType = exu.DynamicSolverType.TrapezoidalIndex2

    def __init__(self, labTrajectory, cupMass=par.cupMass, liquidMass=0., includeCup=True, includeEnvironment=True,
                 stepSize=par.simulationStepSize, writePeriod=par.simulationWritePeriod,
                 bandwidthHz=par.controlBandwidthHz, dampingRatio=par.controlDampingRatio,
                 gravityCompensation=True, solutionFileName=defaultSolutionFileName, verbose=True, frames=None,
                 toolFrameFunction=None, showRobot=True, includeLabTables=True):
        """Create Exudyn model.

        Args:
            labTrajectory (LabTrajectory): trajectory to be followed (joint reference and gripper timeline)
            cupMass (float): mass of the empty cup in kg
            liquidMass (float): mass of the liquid in the cup in kg
            includeCup (bool): if True, a cup stands on table 1 and can be grasped
            includeEnvironment (bool): draw lab table, small tables and cup holders
            includeLabTables (bool): with includeEnvironment also the small tables 1, 2 and the cup holders (False:
                only the lab table, e.g. without the cup task)
            stepSize (float): time step of the implicit integrator in s
            writePeriod (float): output period of sensors and solution file in s
            bandwidthHz (float): natural frequency of the joint PD control
            dampingRatio (float): damping ratio of the joint PD control
            gravityCompensation (bool): add static gravity torques as feed forward
            solutionFileName (str): solution file for SolutionViewer (without extension)
            verbose (bool): print grasp/release events and simulation info
            frames (list): [(label, 4x4 pose in K0)] drawn as labeled frames, e.g. the grasp poses (AddFrameGraphics)
            toolFrameFunction (function): joint angles in rad -> 4x4 pose in K0 (or None) drawn as frame at the present
                joint angles, e.g. the TCP of an own kinematic model, or a list [(label, 4x4 pose)] of labeled frames,
                e.g. the DH frames K1..K6 (AddFrameGraphics)
            showRobot (bool): False hides the lab robot (it is still simulated); the labeled frames of
                toolFrameFunction are drawn as kinematic chain, e.g. the own DH model moving alone (AddFrameGraphics)
        """
        self.labTrajectory = labTrajectory
        self.payloadMass = cupMass + liquidMass
        self.includeCup = includeCup
        self.stepSize = stepSize
        self.writePeriod = writePeriod
        self.gravityCompensation = gravityCompensation
        self.solutionFileName = solutionFileName
        self.verbose = verbose
        self.modelData = LoadModelData()

        self.SC = exu.SystemContainer()
        self.mbs = self.SC.AddSystem()
        self.robot = ManipulatorRobot6R(useCadData=True, includeGripper=True, bandwidthHz=bandwidthHz, dampingRatio=dampingRatio)
        #inertia of link 6 (with gripper) in DH frame 6 without payload, used to add/remove the cup
        self.link6InertiaWithoutPayload = RigidBodyInertia()
        self.link6InertiaWithoutPayload.SetWithCOMinertia(self.robot.links[5].mass, self.robot.links[5].inertia, self.robot.links[5].COM)

        robotDict = self.robot.CreateKinematicTree(self.mbs)
        self.objectKinematicTree = robotDict['objectKinematicTree']
        self.nodeKinematicTree = robotDict['nodeGeneric']
        self.mbs.SetNodeParameter(self.nodeKinematicTree, 'initialCoordinates', list(labTrajectory.initialJointAngles))

        #cup state: list of events (time, type, data); free cup pose at start
        self.cupEvents = []
        self.initialCupHT = visualization.GetInitialCupHT()
        self.processedGripperEvents = 0

        if includeEnvironment:
            self.mbs.AddObject(ObjectGround(visualization=VObjectGround(
                graphicsData=visualization.CreateEnvironmentGraphics(self.modelData, includeLabTables=includeLabTables))))
        self.jawGraphics = robotVisualization.CreateJawGraphics(self.modelData)
        self.cupGraphics = visualization.CreateCupGraphics()
        self.mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsDataUserFunction=self.GraphicsUserFunction)))
        self.showRobot = showRobot
        if not showRobot:
            self.mbs.SetObjectParameter(self.objectKinematicTree, 'Vshow', False)
        self.AddFrameGraphics(frames, toolFrameFunction, drawFrameChain=not showRobot)

        self.SetupControlAndSensors()
        self.mbs.SetPreStepUserFunction(self.PreStepUserFunction)
        self.robot.PreStepControl(self.mbs, 0.)    #consistent reference for the initial accelerations
        self.mbs.Assemble()
        self.SetVisualizationSettings()

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #control and gripper/cup logic
    def SetupControlAndSensors(self):
        """Set the joint control of the robot (Robot.SetControl with ComputeReference and the gravity compensation) and
        add the joint sensors (Robot.AddSensors); call after Robot.CreateKinematicTree."""
        self.controlReference = ControlReference(self.ComputeReference)
        self.UpdateControl()
        self.sensors = self.robot.AddSensors(self.mbs, outputVariables=jointSensorVariables)

    def UpdateControl(self):
        """(Re-)set the control of the robot; needed after the inertia of a link changed (gravity compensation)."""
        self.robot.SetControl(self.controlReference, compensation=gravityCompensationMode if self.gravityCompensation else None)

    def ComputeReference(self, t):
        """Return the joint reference [q, q_t, q_tt] of the PD control at time t: the lab trajectory."""
        return self.labTrajectory.Evaluate(t)

    def PreStepUserFunction(self, mbs, t):
        """Set joint reference and feed forward torques (Robot.PreStepControl); process grasp and release of the cup."""
        self.robot.PreStepControl(mbs, t)
        if self.includeCup:
            self.ProcessGripperEvents(t)
        return True

    def GetCurrentTCP(self):
        """Return TCP pose of the current (simulated) state."""
        q = self.mbs.GetNodeOutput(self.nodeKinematicTree, exu.OutputVariableType.Coordinates)
        return ForwardKinematics(q)

    def ProcessGripperEvents(self, t, trajectoryStartTime=0.):
        """Grasp the cup at the end of a closing action, release it at the start of an opening action.

        Args:
            t (float): simulation time in s
            trajectoryStartTime (float): simulation time at which the trajectory started (event times are relative to it)
        """
        #Michael Pieber 2026-11-03: trajectoryStartTime added for trajectories started during a live simulation
        gripperEvents = self.labTrajectory.gripperEvents
        trajectoryTime = t - trajectoryStartTime
        while self.processedGripperEvents < len(gripperEvents):
            event = gripperEvents[self.processedGripperEvents]
            isClosing = event['endPercent'] >= par.gripperGraspPercent
            if isClosing and trajectoryTime >= event['endTime']:
                self.TryGrasp(trajectoryStartTime + event['endTime'])
            elif not isClosing and trajectoryTime >= event['startTime']:
                self.Release(trajectoryStartTime + event['startTime'])
            else:
                break
            self.processedGripperEvents += 1

    def IsCupAttached(self, time):
        """Return (attached, data) of the cup at time from the cup events."""
        attached = False
        data = self.initialCupHT
        for eventTime, eventType, eventData in self.cupEvents:
            if eventTime > time:
                break
            attached = (eventType == 'grasp')
            data = eventData
        return attached, data

    def TryGrasp(self, time):
        """Attach the cup to the gripper if the TCP lies on the cup axis within the cup height."""
        attached, cupHT = self.IsCupAttached(time)
        if attached:
            return
        HTtcp = self.GetCurrentTCP()
        cupAxis = cupHT[0:3, 2]
        relativePosition = HTtcp[0:3, 3] - cupHT[0:3, 3]
        heightOnCup = relativePosition @ cupAxis
        distanceToAxis = np.linalg.norm(relativePosition - heightOnCup*cupAxis)
        closingAlongAxis = abs(HTtcp[0:3, 1] @ cupAxis)
        if (distanceToAxis <= par.gripperCaptureDistance and 0. < heightOnCup < par.cupHeight
                and closingAlongAxis < cupMaxGraspAxisCosine):
            relativeHT = InverseHT(HTtcp) @ cupHT
            self.cupEvents.append((time, 'grasp', relativeHT))
            self.SetPayload(relativeHT)
            self.Print('t = ' + str(round(time, 3)) + ' s: cup grasped (TCP distance to cup axis ' +
                       str(round(distanceToAxis*1000., 1)) + ' mm, grasp height ' + str(round(heightOnCup*1000., 1)) + ' mm)')
        else:
            self.cupEvents.append((time, 'graspFailed', cupHT))
            self.Print('t = ' + str(round(time, 3)) + ' s: gripper closed, but cup NOT grasped (TCP distance to cup axis ' +
                       str(round(distanceToAxis*1000., 1)) + ' mm, height on cup ' + str(round(heightOnCup*1000., 1)) + ' mm)')

    def Release(self, time):
        """Detach the cup at its current pose and check if it stands on a support."""
        attached, relativeHT = self.IsCupAttached(time)
        if not attached:
            return
        cupHT = self.GetCurrentTCP() @ relativeHT
        supportHeight = visualization.GetCupPlacementHeight(cupHT[0:3, 3])
        heightError = cupHT[2, 3] - supportHeight
        tiltDeg = np.degrees(np.arccos(np.clip(cupHT[2, 2], -1., 1.)))
        isPlaced = abs(heightError) <= cupPlacementTolerance and tiltDeg <= cupMaxPlacementTiltDeg
        self.cupEvents.append((time, 'release', cupHT))
        self.SetPayload(None)
        self.Print('t = ' + str(round(time, 3)) + ' s: cup released at ' + str(np.round(cupHT[0:3, 3], 4)) + ' m, ' +
                   ('placed on support' if isPlaced else 'NOT placed correctly') + ' (height above support ' +
                   str(round(heightError*1000., 1)) + ' mm, tilt ' + str(round(tiltDeg, 2)) + ' deg)')

    def SetPayload(self, relativeHT):
        """Add (relativeHT = cup pose in TCP frame) or remove (None) the cup inertia to link 6 in robot and KinematicTree."""
        inertiaDH = RigidBodyInertia() + self.link6InertiaWithoutPayload
        if relativeHT is not None:
            payloadJoint = GetPayloadInertia(self.payloadMass, relativeHT)     #joint frame 6
            inertiaDH = inertiaDH + payloadJoint.Transformed(InverseHT(GetLinkLocalHT(5)))
        link6 = self.robot.links[5]
        link6.mass = inertiaDH.Mass()
        link6.COM = inertiaDH.COM()
        link6.inertia = inertiaDH.InertiaCOM()
        inertiaJoint = inertiaDH.Transformed(GetLinkLocalHT(5))
        oKT = self.objectKinematicTree
        masses = self.mbs.GetObjectParameter(oKT, 'linkMasses')
        masses[5] = inertiaJoint.Mass()
        comList = self.mbs.GetObjectParameter(oKT, 'linkCOMs')
        coms = [np.array(comList[i]) for i in range(par.numberOfJoints)]
        coms[5] = inertiaJoint.COM()
        inertiaList = self.mbs.GetObjectParameter(oKT, 'linkInertiasCOM')
        inertias = [np.array(inertiaList[i]) for i in range(par.numberOfJoints)]
        inertias[5] = inertiaJoint.InertiaCOM()
        self.mbs.SetObjectParameter(oKT, 'linkMasses', masses)
        self.mbs.SetObjectParameter(oKT, 'linkCOMs', exu.Vector3DList(coms))
        self.mbs.SetObjectParameter(oKT, 'linkInertiasCOM', exu.Matrix3DList(inertias))
        self.UpdateControl()     #gravity compensation with the new inertia of link 6

    def GetCupHT(self, time, HTtcp):
        """Return cup pose at time for TCP pose HTtcp (attached cup moves with the TCP)."""
        attached, data = self.IsCupAttached(time)
        if attached:
            return HTtcp @ data
        return data

    def Print(self, text):
        """Print text if verbose."""
        if self.verbose:
            print('Robot6RSimulation:', text)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #visualization
    def AddFrameGraphics(self, frames, toolFrameFunction, drawFrameChain=False):
        """Draw the frames [(label, 4x4 pose in K0)] (without their labels: no text at the
        poses in the renderer) and the frame toolFrameFunction(joint angles in rad) at the
        present joint angles (a pose, or a list [(label, pose)] of labeled frames), also in the SolutionViewer; with
        drawFrameChain the labeled frames as kinematic chain (FrameChainGraphics); a function raising an error or
        returning None draws nothing
        (each subtask of the preparation seen in the renderer, also before the others are
        solved)."""
        frames = [] if frames is None else list(frames)
        if not frames and toolFrameFunction is None:
            return
        staticGraphics = []
        for label, pose in frames:
            pose = np.asarray(pose, dtype=float)
            staticGraphics.append(graphics.Frame(HT=pose, length=frameGraphicsLength))

        def FrameUserFunction(mbs, objectNumber):
            graphicsList = list(staticGraphics)
            if toolFrameFunction is not None:
                q = mbs.systemData.GetODE2Coordinates(exu.ConfigurationType.Visualization)[0:par.numberOfJoints]
                try:
                    result = toolFrameFunction(np.array(q))
                except (ValueError, TypeError, IndexError, ArithmeticError, np.linalg.LinAlgError):   #unsolved: no frame
                    result = None
                #a list [(label, pose)] draws several labeled moving frames (e.g. the DH frames)
                labeled = isinstance(result, list)
                movingFrames = result if labeled else [('', result)]
                length = movingFrameGraphicsLength if labeled else toolFrameGraphicsLength
                labelsAtOrigin = []         #[origin, [labels]]: one text per origin, e.g. 'K2, K3'
                for label, pose in movingFrames:
                    if pose is None or np.shape(pose) != (4, 4) or not np.all(np.isfinite(pose)):
                        continue
                    pose = np.asarray(pose, dtype=float)
                    graphicsList.append(graphics.Frame(HT=pose, length=length, colors=toolFrameColors,
                                                       radius=movingFrameGraphicsRadius))
                    if label == '':
                        continue
                    for origin, labels in labelsAtOrigin:
                        if np.linalg.norm(origin - pose[0:3, 3]) < sameOriginTolerance:
                            labels.append(label)
                            break
                    else:
                        labelsAtOrigin.append([pose[0:3, 3], [label]])
                for origin, labels in labelsAtOrigin:
                    graphicsList.append(graphics.Text(point=list(origin), text=', '.join(labels), color=graphics.color.black,
                                                      offset=[0.5, 0.5]))
                if drawFrameChain and labeled:
                    graphicsList += FrameChainGraphics([pose for label, pose in movingFrames])
            return graphicsList

        self.frameGraphicsFunction = FrameUserFunction   #! Graphics of the frames (mbs, objectNumber) -> list.
        self.mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsDataUserFunction=FrameUserFunction)))

    def GraphicsUserFunction(self, mbs, objectNumber):
        """Draw jaws (opening from gripper timeline) and cup for the current visualization state (also in SolutionViewer)."""
        time = mbs.systemData.GetTime(exu.ConfigurationType.Visualization)
        q = mbs.systemData.GetODE2Coordinates(exu.ConfigurationType.Visualization)[0:par.numberOfJoints]
        HTtcp = ForwardKinematics(q)
        graphicsList = []
        if self.showRobot:
            graphicsList = robotVisualization.MoveJawGraphics(self.jawGraphics, HTtcp, self.labTrajectory.GetGripperPercent(time))
        if self.includeCup:
            cupHT = self.GetCupHT(time, HTtcp)
            graphicsList.append(graphics.Move(self.cupGraphics, cupHT[0:3, 3], cupHT[0:3, 0:3]))
        return graphicsList

    def SetVisualizationSettings(self):
        """Set renderer settings (view, light, hidden frames)."""
        settings = self.SC.visualizationSettings
        settings.general.autoFitScene = False
        settings.general.showSolverInformation = False
        settings.general.graphicsUpdateInterval = 0.02
        settings.view0.scene.drawWorldBasis = True
        settings.view0.scene.worldBasisSize = 0.15
        settings.view0.window.renderWindowSize = [1600, 1000]
        settings.openGL.multiSampling = 4
        settings.openGL.advanced.initialCenterPoint = initialViewCenter
        settings.openGL.advanced.initialZoom = initialViewZoom
        settings.openGL.advanced.initialModelRotation = initialViewRotation
        settings.openGL.light0.shadow = rendererShadow
        settings.openGL.light0.position = [1., -1., 1.5, 0.]   #directional light from the camera side
        settings.openGL.light0.diffuse = 0.55
        settings.openGL.light1.position = [-1., 1., 3., 0.]   #fill light from above
        settings.openGL.light1.diffuse = 0.25
        settings.nodes.show = False
        settings.bodies.kinematicTree.showJointFrames = False
        settings.bodies.kinematicTree.showCOMframes = False
        settings.markers.show = False
        settings.sensors.show = False
        settings.loads.show = False

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #simulation and results
    def ConfigureSimulationSettings(self, simulationSettings):
        """Configure solver settings for this simulation."""
        simulationSettings.timeIntegration.newton.useModifiedNewton = True
        #central worker count, also inherited by physical/VR simulation.
        simulationSettings.parallel.numberOfThreads = par.contactNumberOfThreads

    def Simulate(self, useRenderer=False, realtimeFactor=None, rendererAnimation=None):
        """Run the dynamic simulation over the trajectory duration plus settling time.

        Args:
            useRenderer (bool): show the simulation live in the Exudyn renderer
            realtimeFactor (float): if given, simulation is slowed down to realtimeFactor x real time (with renderer)
            rendererAnimation (function): with the renderer, function(simulation) played in it before the simulation
                (which then starts with Space), e.g. DHRowAnimation.Run

        Returns:
            dict: results, see GetResults()
        """
        endTime = self.labTrajectory.GetDuration() + settlingTime
        simulationSettings = exu.SimulationSettings()
        simulationSettings.timeIntegration.endTime = endTime
        simulationSettings.timeIntegration.numberOfSteps = int(np.ceil(endTime/self.stepSize))
        simulationSettings.timeIntegration.verboseMode = 1 if self.verbose else 0
        self.ConfigureSimulationSettings(simulationSettings)
        simulationSettings.solution.file.write = True
        simulationSettings.solution.file.writePeriod = self.writePeriod
        simulationSettings.solution.sensors.writePeriod = self.writePeriod
        simulationSettings.solution.file.binary = True
        simulationSettings.solution.file.name = self.solutionFileName
        if realtimeFactor is not None:
            simulationSettings.timeIntegration.realtime.active = True
            simulationSettings.timeIntegration.realtime.factor = realtimeFactor
        directory = os.path.dirname(self.solutionFileName)
        if directory != '':
            os.makedirs(directory, exist_ok=True)
        self.simulationSettings = simulationSettings

        if useRenderer:
            self.SC.renderer.Start()
            if rendererAnimation is not None:
                rendererAnimation(self)
            self.SC.renderer.DoIdleTasks()
        with SolutionFileForSolver(simulationSettings):     #folders with umlauts
            self.mbs.SolveDynamic(simulationSettings, solverType=self.solverType)
        if useRenderer:
            self.SC.renderer.DoIdleTasks()
            self.SC.renderer.Stop()
        return self.GetResults()

    def ShowSolution(self, rowIncrement=2, runMode=0):
        """Replay the stored solution in the Exudyn SolutionViewer.

        Args:
            rowIncrement (int): show every rowIncrement-th stored frame; 2 gives about real time at 20 frames/s
            runMode (int): 0 = continuous run, 1 = one cycle, 2 = static (slider)
        """
        self.mbs.SolutionViewer(rowIncrement=rowIncrement, timeout=self.writePeriod, runMode=runMode)

    def GetCupResults(self, time, q):
        """Return cup positions, tilt and attachment state for the stored robot states."""
        cupPosition = np.zeros((len(time), 3))
        cupTiltDeg = np.zeros(len(time))
        cupAttached = np.zeros(len(time), dtype=bool)
        for index, t in enumerate(time):
            cupHT = self.GetCupHT(t, ForwardKinematics(q[index]))
            cupPosition[index] = cupHT[0:3, 3]
            cupTiltDeg[index] = np.degrees(np.arccos(np.clip(cupHT[2, 2], -1., 1.)))
            cupAttached[index] = self.IsCupAttached(t)[0]
        return cupPosition, cupTiltDeg, cupAttached

    def GetResults(self):
        """Evaluate stored sensor data.

        Returns:
            dict: 'time', 'q', 'q_t', 'qReference', 'q_tReference', 'trackingError' (n x 6), 'driveTorques' (n x 6, Nm),
                  'positionTCP' (n x 3), 'rotationTCP' (n x 3 x 3), 'cupPosition' (n x 3), 'cupTiltDeg' (n),
                  'cupAttached' (n, bool), 'cupEvents', 'gripperPercent' (n),
                  'maxTorqueRatioNominal', 'maxTorqueRatioPeak' (4, axes 1-4 max |torque|/Schunk torque), 'skippedRows'
        """
        OVT = exu.OutputVariableType
        coordinatesData = self.mbs.GetSensorStoredData(self.sensors[OVT.Coordinates])
        velocitiesData = self.mbs.GetSensorStoredData(self.sensors[OVT.Coordinates_t])
        torquesData = self.mbs.GetSensorStoredData(self.sensors[OVT.Force])
        time = coordinatesData[:, 0]
        q = coordinatesData[:, 1:1+par.numberOfJoints]
        q_t = velocitiesData[:, 1:1+par.numberOfJoints]
        driveTorques = torquesData[:, 1:1+par.numberOfJoints]
        qReference = np.zeros_like(q)
        q_tReference = np.zeros_like(q)
        for index, t in enumerate(time):
            [qReference[index], q_tReference[index], acceleration] = self.labTrajectory.Evaluate(t)

        positionTCP = np.zeros((len(time), 3))
        rotationTCP = np.zeros((len(time), 3, 3))
        gripperPercent = np.zeros(len(time))
        for index, t in enumerate(time):
            HTtcp = ForwardKinematics(q[index])
            positionTCP[index] = HTtcp[0:3, 3]
            rotationTCP[index] = HTtcp[0:3, 0:3]
            gripperPercent[index] = self.labTrajectory.GetGripperPercent(t)
        cupPosition, cupTiltDeg, cupAttached = self.GetCupResults(time, q)
        maxTorqueRatioNominal = np.max(np.abs(driveTorques[:, 0:4]), axis=0)/np.array(par.schunkNominalTorques)
        maxTorqueRatioPeak = np.max(np.abs(driveTorques[:, 0:4]), axis=0)/np.array(par.schunkPeakTorques)
        return {'time': time, 'q': q, 'q_t': q_t, 'qReference': qReference, 'q_tReference': q_tReference,
                'trackingError': q - qReference, 'driveTorques': driveTorques,
                'positionTCP': positionTCP, 'rotationTCP': rotationTCP,
                'cupPosition': cupPosition, 'cupTiltDeg': cupTiltDeg, 'cupAttached': cupAttached,
                'cupEvents': list(self.cupEvents), 'gripperPercent': gripperPercent,
                'maxTorqueRatioNominal': maxTorqueRatioNominal, 'maxTorqueRatioPeak': maxTorqueRatioPeak,
                'skippedRows': list(self.labTrajectory.skippedRows)}


def SimulateBahnfahrt(matrixQ, angleUnit='rad', showSolution=True, useRenderer=False, mode='fullySynchronous',
                      initialJointAngles=None, verbose=True, contactMode='physical', rendererAnimation=None,
                      **simulationArguments):
    """Simulation counterpart of BahnfahrtPython(Q, str) of the lab: simulate the PTP matrix Q and optionally replay it.

    Args:
        matrixQ (array): n x 8 matrix [q1..q6, speed percent, gripper command]
        angleUnit (str): 'rad' or 'deg'
        showSolution (bool): open the Exudyn SolutionViewer after the simulation
        useRenderer (bool): show the simulation live
        mode (str): PTP interpolation ('fullySynchronous' as in the lab, 'synchronous', 'asynchronous')
        initialJointAngles (array): start configuration in rad; default: zero configuration (start in the lab)
        verbose (bool): print information
        contactMode (str): 'physical' for a free cup with contact, 'kinematic' for the previous approximation
        rendererAnimation (function): function(simulation) played in the renderer before the simulation (Simulate)
        simulationArguments: further arguments of Robot6RSimulation (e.g. liquidMass, includeCup, stepSize)

    Returns:
        tuple: (results dict, Robot6RSimulation object)
    """
    labTrajectory = LabTrajectory(matrixQ, angleUnit=angleUnit, initialJointAngles=initialJointAngles, mode=mode, verbose=verbose)
    if contactMode not in ('physical', 'kinematic'):
        raise ValueError("contactMode must be 'physical' or 'kinematic'")
    if contactMode == 'physical' and simulationArguments.get('includeCup', True):
        from robot6R.physicalSimulation import PhysicalRobot6RSimulation
        simulation = PhysicalRobot6RSimulation(labTrajectory, verbose=verbose, **simulationArguments)
    else:
        simulation = Robot6RSimulation(labTrajectory, verbose=verbose, **simulationArguments)
    results = simulation.Simulate(useRenderer=useRenderer, rendererAnimation=rendererAnimation)
    if showSolution:
        simulation.ShowSolution()
    return results, simulation


if __name__ == '__main__':
    #demo (example of RobotUtils.py): three PTP motions in degree with gripper commands, replay in the SolutionViewer
    demoMatrixQ = np.array([[90., 30., 50., 0., 0., 0., 40., par.gripperCommandOpen],
                            [0., 20., 20., 0., 0., 0., 40., par.gripperCommandNone],
                            [0., 0., 0., 0., 0., 0., 40., par.gripperCommandNone]])
    demoResults, demoSimulation = SimulateBahnfahrt(demoMatrixQ, 'deg', showSolution=True)
