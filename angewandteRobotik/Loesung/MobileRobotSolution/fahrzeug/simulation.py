# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: simulation.py
# Details:  Base simulation of the mobile robot: physical model, task (action list or state machine of a
#           student file), start/stop/pause/reset, renderer and SolutionViewer.
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-05 (split into base, interactive and lab simulation: 2026-11-07)
# Notes:    Pause freezes simulation time; stop cancels drive and arm references. VR input (fahrzeug/vr) and the
#           real lab (fahrzeug/lab) extend VehicleSimulation through the hooks WaitBeforeStep, ControlVelocity,
#           AfterControl, ConfigureRenderer and OnRunEnd; this module needs neither of them.
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Base simulation runner: base of the interactive and lab classes."""
import time
from pathlib import Path
import numpy as np
import exudyn as exu
from exudyn.utilities import LoadSolutionFile
from exudyn.rigidBodyUtilities import RotationMatrixX, RotationMatrixZ, HTtranslate
from exudyn.itemInterface import ObjectGround, VObjectGround
import exudyn.graphics as graphics
try:
    from exuxr import HighTimerResolution   #1 ms Windows timer resolution, as in VR
except ImportError:   #student package without ExuXR: default timer resolution
    from contextlib import nullcontext as HighTimerResolution
from . import parameters as par
from .model import VehicleModel
from .stateMachine import PreparationController
from .taskFile import LoadTask, SelectStudentTask, studentDirectory
from labRobots.robotics.collision.display import CollisionDisplay
from labRobots.robotics.simulation.solverFiles import SolutionFileForSolver
from labRobots.robots.kairos.treeBody import GetBodyOutput
from .preparationTests import PreparationFrames

buttonStart, buttonStop, buttonPause, buttonReset, buttonNext = 'start', 'stop', 'pause', 'reset', 'next'
buttonStudentPrevious, buttonStudentNext, buttonStudentLoad = 'studentPrevious', 'studentNext', 'studentLoad'
#! Buttons that start the loaded task (state machine, or action list of the upper layers).
taskButtons = (buttonStart, buttonNext)
solverTypes = {'implicit': exu.DynamicSolverType.GeneralizedAlpha, 'explicit': exu.DynamicSolverType.ExplicitEuler}
defaultSolverName = 'implicit'
defaultVRSolverName = 'explicit'
#! Length in m of the frames of the preparation (PreparationFrameGraphics).
preparationFrameLength = 0.15
#! Length in m of the frame 'arm base': it must reach out of the base housing of the UR5 (radius 0.076 m) and its
#! flange to be seen.
armBaseFrameLength = 0.35
#! Radius in m of the arrows of the preparation frames (graphics.Frame default: 1 % of the length, 1.5 mm hardly seen).
preparationFrameRadius = 0.005
#! Label of the arm base frame of fahrzeug.preparationTests.PreparationFrames.
armBaseFrameLabel = 'arm base'
#! Length in m of the global frame K0 at the origin and of the platform frame (center of the top plate, x forward,
#! y left), drawn with the preparation frames as in the sketch of the task sheet.
worldFrameLength = 0.5
platformFrameLength = 0.35


class VehicleSimulation:
    """Physical simulation with a task runner; VR and the normal renderer use the same model."""

    def __init__(self, actionsInit=None, callbackInit=None, settingsInit=None, initialPoseInit=None, objectPositionInit=None,
                 studentDirectoryInit=studentDirectory, armInitialAnglesInit=None, objectTrackingInit=False):
        """Create a fresh, assembled system and persistent display settings for scene reset.

        @param armInitialAnglesInit Explicit initial arm angles in rad.
               also supports offline reconstruction from recorded joint states.
        @param objectTrackingInit True: the object is coupled to a live target (LabVehicleSimulation with Vicon).
        """
        if actionsInit is not None and callbackInit is not None:
            raise ValueError('Choose an action list or a preparation callback, not both')
        self.settings = {} if settingsInit is None else settingsInit  #! Display preferences and loaded task survive reset.
        if actionsInit is None and callbackInit is None and self.settings.get('studentTask') is not None:
            actionsInit, callbackInit = self.settings['studentTask']
        self.studentDirectory = Path(studentDirectoryInit)  #! Local student Python files.
        self.systemContainer = exu.SystemContainer()  #! Keeps the Exudyn system alive.
        self.mbs = self.systemContainer.AddSystem()  #! Simulation system.
        self.model = VehicleModel(self.mbs, initialPoseInit, objectPositionInit, objectTrackingInit=objectTrackingInit,
                                  armInitialAnglesInit=armInitialAnglesInit)  #! Shared physical model.
        #simulated S300/VLP-16 beams (settings 'sensorRaysEnabled'; shown: 'sensorRaysShown')
        #! SimulatedSensors of the platform, or None if not enabled.
        #optional subpackages sensors/ and lab/ are imported only where they are used
        self.sensors = None
        if self.settings.get('sensorRaysEnabled', False):
            from .sensors.sensorSimulation import SimulatedSensors
            self.sensors = SimulatedSensors(self.mbs, self.model, self.settings.get('sensorRaysShown', False),
                                            self.settings.get('sensorRaysFull', False))
        #simulated front camera view (settings 'cameraViewEnabled'), own window with the renderer
        #! SimulatedCamera of the front camera, or None if not enabled.
        self.camera = None
        if self.settings.get('cameraViewEnabled', False):
            from .sensors.cameraSimulation import SimulatedCamera
            self.camera = SimulatedCamera(self.mbs, self.model)
        #shapes of the arm collision checker (settings 'collisionShapesShown')
        #! CollisionDisplay of the planner's collision shapes.
        self.collisionDisplay = CollisionDisplay(self.mbs, self.model, self.settings.get('collisionShapesShown', False))
        #the results of the subtasks of the preparation as frames (switches show... of the file)
        #! Task file whose results are drawn (preparationModule of the loaded callback), None: nothing drawn.
        self.preparationModule = getattr(callbackInit, 'preparationModule', None)
        #! Frame graphics at the origin, moved to each pose by PreparationFrameGraphics (building
        #! graphics.Frame for every rendered frame took about 0.1 s in the lab and slowed the simulation down).
        self.preparationFrameTemplate = graphics.Frame(length=preparationFrameLength, radius=preparationFrameRadius)
        self.armBaseFrameTemplate = graphics.Frame(length=armBaseFrameLength, radius=preparationFrameRadius)
        self.worldFrameGraphics = graphics.Frame(HT=HTtranslate([0., 0., preparationFrameRadius]),
                                                 length=worldFrameLength, radius=2.*preparationFrameRadius)
        self.platformFrameTemplate = graphics.Frame(length=platformFrameLength, radius=preparationFrameRadius)
        self.mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsDataUserFunction=self.PreparationFrameGraphics)))
        self.mbs.Assemble()
        self.model.Initialize()
        #! TaskSequence of an action list (fahrzeug/reference, upper layers and lab drives), None for a state machine.
        self.sequence = self.CreateSequence(self.DefaultActions() if actionsInit is None else actionsInit)
        self.callback = PreparationController(callbackInit) if callbackInit is not None else None
        for key, default in (('framesShown', False), ('targetShown', False), ('trailEnabled', False), ('trailId', 1),
                             ('sensorRaysEnabled', False), ('sensorRaysShown', False), ('sensorRaysFull', False),
                             ('collisionShapesShown', False), ('framesOverlay', True), ('jointMonitorShown', False), ('wheelMonitorShown', False),
                             ('commandsShown', False)):
            self.settings.setdefault(key, default)
        self.control = {'time': 0., 'paused': False, 'reset': False, 'selected': buttonStart,
                        'callbackActive': False, 'error': '', 'nextPoll': 0., 'nextRecord': 0.,
                        'nextControl': 0., 'velocity': np.zeros(par.dimension3D)}  #! Runtime control state and held drive command.
        self.link = None  #! Optional ExuXR bridge; no viewer is required for offline runs.
        self.pacing = None  #! Wall-clock and simulation time anchor.
        self.history = []  #! Offline diagnostics [time,x,y,yaw].
        self.fingerHistory = None  #! Finger poses recorded for ShowSolution, None if no solution file is written.
        self.mbs.SetPreStepUserFunction(self.PreStep)
        self.RefreshStudentFile()

    def PreparationFrameGraphics(self, mbs, objectNumber):
        """Draw the frames of the preparation (fahrzeug.preparationTests.PreparationFrames) for the visualized state:
        the platform as it is (arm base), the object marker where the object started (approach and grasp poses);
        without labels (no text at the poses in the renderer)."""
        if self.preparationModule is None:
            return []
        configuration = exu.ConfigurationType.Visualization
        platform = self.model.BodyPose(self.model.bodies['platform'], configuration)
        posPlatform = [platform[0, 3], platform[1, 3], self.model.ArmBasePose(configuration)[2, 3]]
        yaw = float(np.arctan2(platform[1, 0], platform[0, 0]))
        marker = self.model.objectInitialPosition + [0., 0., par.viconObjectMarkerHeight]
        #K0 and the platform frame on the top plate (lifted by the arrow radius to be seen)
        plateHeight = par.baseLinkHeight + par.chassisTopHeight + preparationFrameRadius
        graphicsList = [self.worldFrameGraphics,
                        graphics.Move(self.platformFrameTemplate, [posPlatform[0], posPlatform[1], plateHeight],
                                      RotationMatrixZ(yaw))]
        for label, pose in PreparationFrames(self.preparationModule, posPlatform, yaw, marker):
            template = self.armBaseFrameTemplate if label == armBaseFrameLabel else self.preparationFrameTemplate
            graphicsList.append(graphics.Move(template, pose[0:3, 3], pose[0:3, 0:3]))
        return graphicsList

    def DefaultActions(self):
        """Return the action list run without a task (InteractiveSimulation: sample return mission); None here."""
        return None

    def CreateSequence(self, actions):
        """Return the TaskSequence of an action list or of a factory of the model, None without actions.

        action lists and their drive controller are in fahrzeug/reference (not in the student
        package, which runs state machines only); imported only when an action list is used.
        """
        if actions is None:
            return None
        from .reference.taskSequence import TaskSequence
        return TaskSequence(actions(self.model) if callable(actions) else actions)

    def GripperClosed(self):
        """Return True if the simulated gripper is commanded closed (grasp, also without object)."""
        return bool(self.model.grasp['openingTarget'] < par.gripperOpenWidth)

    def RefreshStudentFile(self, step=0):
        """Discover files added while the viewer runs and preserve selection by filename."""
        path = SelectStudentTask(self.studentDirectory, self.settings.get('studentFileName'), step)
        self.settings['studentFileName'] = path.name if path is not None else None
        return path

    def LoadStudentTask(self):
        """Validate and start the selected submission, retaining it for replay and reset."""
        path = self.RefreshStudentFile()
        if path is None:
            raise FileNotFoundError('Keine .py-Datei in studentFiles')
        actions, callback = LoadTask(path)
        sequence = self.CreateSequence(self.DefaultActions() if actions is None else actions)
        preparedCallback = PreparationController(callback) if callback is not None else None
        self.preparationModule = getattr(callback, 'preparationModule', None)
        self.Stop()
        self.sequence, self.callback = sequence, preparedCallback
        self.settings['studentTask'] = (actions, callback)
        self.settings['loadedStudentFileName'] = path.name
        self.control.update({'callbackActive': preparedCallback is not None, 'selected': buttonStudentLoad,
                             'paused': False, 'error': ''})
        self.pacing = None
        if preparedCallback is None:
            self.sequence.Start()   #an action list (state machine: callbackActive)
        self.settings['trailId'] += 1

    def Stop(self, keepManual=False):
        """Cancel the task and arm motion while wheel motors brake to zero velocity.

        @param keepManual Keep a manual mode of the interactive simulation (no manual modes here).
        """
        if self.sequence is not None:
            self.sequence.Stop()
        self.control['callbackActive'] = False
        self.model.StopArm(self.control['time'])
        self.control['selected'] = buttonStop

    def HandleCommand(self, command):
        """Apply one validated button command of the task controls (the interactive simulation adds VR controls)."""
        if command.get('type') != 'button':
            raise ValueError('Vehicle controls accept button commands only')
        button = command['id']
        if button == buttonReset:
            self.control['reset'] = True
        elif button == buttonPause:
            self.control['paused'] = not self.control['paused']
            self.pacing = None
        elif button == buttonStop:
            self.Stop()
        elif button in (buttonStudentPrevious, buttonStudentNext):
            self.RefreshStudentFile(-1 if button == buttonStudentPrevious else 1)
        elif button == buttonStudentLoad:
            self.LoadStudentTask()
        elif button in taskButtons:
            self.StartTask(button)
        else:
            raise ValueError('Unknown vehicle button: ' + str(button))

    def StartTask(self, button):
        """Start the state machine (button start) or the action list (start, next: only its next action)."""
        if self.callback is not None and button == buttonStart:
            self.callback = PreparationController(self.callback.callback)
            self.control['callbackActive'] = True
        elif self.sequence is not None:
            self.control['callbackActive'] = False
            self.sequence.Start(singleAction=button == buttonNext)
        else:
            raise ValueError('No task loaded (state machine or action list)')
        self.control['selected'] = button
        self.control['error'] = ''
        self.settings['trailId'] += 1

    def ProcessCommands(self):
        """Process viewer commands and report expected input/model errors without stopping the viewer."""
        if self.link is None:
            return
        for command in self.link.GetCommands():
            try:
                self.HandleCommand(command)
            except (ValueError, TypeError, OSError) as error:
                self.Stop()
                self.control['error'] = str(error)

    def IsStopRequested(self):
        """True on reset or when the viewer closes; checked even while paused."""
        return self.control['reset'] or (self.link is not None and self.link.IsStopRequested())

    def IsManualMode(self):
        """True while the platform or the arm is controlled manually (never here: InteractiveSimulation)."""
        return False

    def IsTaskFinished(self):
        """True if the action list is completed or stopped, or the state machine reached state -1."""
        if self.callback is not None and self.callback.finished:
            return True
        sequenceRunning = self.sequence is not None and self.sequence.running
        return not self.control['callbackActive'] and not sequenceRunning and not self.IsManualMode()

    def WaitBeforeStep(self, timeValue):
        """Hook before a time step (LabVehicleSimulation: wait for the real robot); nothing here."""

    def ControlVelocity(self, timeValue):
        """Return the vehicle-local drive command [vx, vy, omega] of the task (state machine or action list)."""
        if self.control['callbackActive']:
            return self.callback.Update(self.model, timeValue)
        if self.sequence is not None:
            return self.sequence.Update(self.model, timeValue)
        return np.zeros(par.dimension3D)

    def AfterControl(self, timeValue):
        """Hook after each feedback control update (LabVehicleSimulation: object tracking, commands to the robot); the
        collision display takes the obstacles from the planner itself."""

    def PreStep(self, mbs, timeValue):
        """Handle commands, freeze on pause, run feedback control and stream the current state."""
        self.control['time'] = timeValue
        if timeValue >= self.control['nextPoll']:
            self.control['nextPoll'] = timeValue + par.commandPollPeriod
            self.ProcessCommands()
        while self.control['paused'] and not self.IsStopRequested():
            if self.link is not None:
                self.link.Publish(timeValue)
            time.sleep(par.commandPollPeriod)
            self.ProcessCommands()
        self.WaitBeforeStep(timeValue)
        if self.IsStopRequested():
            return False
        #offline runs end shortly after the task (or with its error), not at the maximum duration
        if self.control.get('stopWhenFinished', False) and self.IsTaskFinished():
            self.control.setdefault('finishedTime', timeValue)
            if timeValue - self.control['finishedTime'] >= par.finishedSettlingTime:
                return False
        #feedback at 500 Hz; physics and actuator references retain their full rate.
        if timeValue >= self.control['nextControl'] or timeValue < self.model.lastControlTime:
            try:
                velocity = self.ControlVelocity(timeValue)
            except (ValueError, TimeoutError) as error:
                self.Stop()
                self.control['error'] = str(error)
                print('Vehicle task stopped:', error)
                velocity = np.zeros(par.dimension3D)
            self.control['velocity'] = velocity
            self.AfterControl(timeValue)
            self.control['nextControl'] = par.controlUpdatePeriod*(np.floor(timeValue/par.controlUpdatePeriod) + 1.)
        self.model.Update(timeValue, self.control['velocity'])
        if self.fingerHistory is not None:
            self.fingerHistory['time'].append(timeValue)
            self.fingerHistory['poses'].append([self.model.FingerPose(index) for index in range(par.numberOfGripperFingers)])
        if self.link is not None:
            self.link.Publish(timeValue)
        elif timeValue >= self.control['nextRecord']:
            self.control['nextRecord'] = timeValue + 0.1
            self.history.append(np.r_[timeValue, self.model.GetState()['pose']])
        self.PaceRealtime(timeValue)
        return True

    def PaceRealtime(self, timeValue):
        """Pace to the requested factor; reset the anchor after pause or substantial lag."""
        factor = self.control.get('realtimeFactor')
        if factor is None:
            return
        wallTime = time.perf_counter()
        if self.pacing is None:
            self.pacing = (wallTime, timeValue)
        ahead = (timeValue - self.pacing[1])/factor - (wallTime - self.pacing[0])
        if ahead > 0.002:
            time.sleep(ahead)
        elif ahead < -0.1:
            self.pacing = (wallTime, timeValue)

    def SetRendererSettings(self):
        """Hide nodes, connectors and loads; center the view between the start pose and the object."""
        visualization = self.systemContainer.visualizationSettings
        visualization.nodes.show = False
        visualization.connectors.show = False
        visualization.loads.show = False
        #the rigid markers (platform footprint, wheels, sensors) were drawn as small moving triads
        visualization.markers.show = False
        #joint frames of the kinematic tree (arm joints, wheels) only with the button
        visualization.bodies.kinematicTree.showJointFrames = self.settings['framesShown']
        self.mbs.SetObjectParameter(self.model.bodies['viconOrigin'], 'Vshow', self.settings['framesShown'])
        #the whole drive stays in view, also from a distant start pose (demo)
        start, target = self.model.initialPose[:2], self.model.objectInitialPosition[:2]
        visualization.openGL.advanced.initialCenterPoint = [(start[0] + target[0])/2., (start[1] + target[1])/2., 0.5]
        visualization.openGL.advanced.initialZoom = max(par.minimumViewZoom,
                                                        par.viewZoomPerDistance*np.linalg.norm(target - start))
        #OpenGL convention: the transposed matrix of the world-to-view rotation
        visualization.openGL.advanced.initialModelRotation = (RotationMatrixX(par.viewElevation - np.pi/2.) @
                                                             RotationMatrixZ(par.viewAzimuth)).T.tolist()
        visualization.general.autoFitScene = False
        visualization.general.showSolverInformation = False

    def ConfigureRenderer(self, simulationSettings):
        """Hook before the renderer starts (LabVehicleSimulation: video frames); nothing here."""

    def OnRunEnd(self):
        """Hook at the end of Run, also after errors, before the renderer stops (LabVehicleSimulation: stop the
        real robot); nothing here."""

    def Run(self, duration=par.maximumSimulationTime, realtimeFactor=None, useRenderer=False, solutionFile=None,
            stopWhenFinished=False, solverType=exu.DynamicSolverType.GeneralizedAlpha,
            stepSize=None, numberOfThreads=par.numberOfThreads):
        """Solve from the present time for duration seconds, optionally with the native renderer.

        @param solutionFile If given, the solution is written there for ShowSolution (SolutionViewer).
        @param stopWhenFinished End par.finishedSettlingTime after the task is finished; duration is the maximum.
        @param solverType GeneralizedAlpha or ExplicitEuler.
        @param stepSize Optional physics step in s; defaults to 2 ms implicit or 0.2 ms explicit.
        @param numberOfThreads Exudyn worker threads; defaults to par.numberOfThreads (four).
        """
        if not np.isfinite(duration) or duration <= 0.:
            raise ValueError('Simulation duration must be finite and positive')
        if realtimeFactor is not None and (not np.isfinite(realtimeFactor) or realtimeFactor <= 0.):
            raise ValueError('Realtime factor must be finite and positive')
        #compare integrators on the same unconstrained vehicle tree.
        if solverType not in solverTypes.values():
            raise ValueError('Unsupported vehicle solver type')
        if stepSize is None:
            stepSize = par.explicitStepSize if solverType == exu.DynamicSolverType.ExplicitEuler else par.stepSize
        if not np.isfinite(stepSize) or stepSize <= 0.:
            raise ValueError('Physics step size must be finite and positive')
        if not isinstance(numberOfThreads, int) or isinstance(numberOfThreads, bool) or numberOfThreads < 1:
            raise ValueError('Number of threads must be a positive integer')
        self.control['realtimeFactor'] = realtimeFactor
        self.control['stopWhenFinished'] = stopWhenFinished
        self.control.pop('finishedTime', None)
        #finger graphics are ground objects moved in PreStep; recorded for the replay
        self.model.grasp['nativeGraphics'] = useRenderer or solutionFile is not None
        self.pacing = None
        settings = exu.SimulationSettings()
        startTime = self.mbs.systemData.GetTime()
        settings.timeIntegration.startTime = startTime
        settings.timeIntegration.endTime = startTime + duration
        settings.timeIntegration.numberOfSteps = max(1, int(np.ceil(duration/stepSize)))
        settings.timeIntegration.verboseMode = 0
        settings.timeIntegration.newton.useModifiedNewton = True
        settings.timeIntegration.generalizedAlpha.computeInitialAccelerations = True
        settings.timeIntegration.generalizedAlpha.spectralRadius = par.implicitSpectralRadius   #
        settings.timeIntegration.explicit.computeMassMatrixInversePerBody = True
        settings.timeIntegration.explicit.computeEndOfStepAccelerations = False
        settings.parallel.numberOfThreads = numberOfThreads
        settings.solution.file.write = False
        settings.solution.sensors.writePeriod = 0.1
        #optional solution file and finger poses for the SolutionViewer
        self.fingerHistory = None
        if solutionFile is not None:
            Path(solutionFile).parent.mkdir(parents=True, exist_ok=True)
            settings.solution.file.write = True
            settings.solution.file.name = str(solutionFile)
            settings.solution.file.writePeriod = par.solutionWritePeriod
            self.fingerHistory = {'time': [], 'poses': []}  #! Finger carriage poses per step for the replay.
        if useRenderer:
            self.SetRendererSettings()
            self.ConfigureRenderer(settings)
            self.systemContainer.renderer.Start()
            if self.camera is not None:
                self.camera.StartView(self.systemContainer)
        try:
            #match VR's Windows timer resolution for waits and worker scheduling.
            with HighTimerResolution(), SolutionFileForSolver(settings):     #folders with umlauts
                self.mbs.SolveDynamic(settings, solverType=solverType, updateInitialValues=True)
        finally:
            self.OnRunEnd()
            if useRenderer:
                if self.camera is not None:
                    self.camera.StopView(self.systemContainer)   #renderer.Stop() fails with the camera window still open
                self.systemContainer.renderer.Stop()
        return self.model.GetState()

    def ShowSolution(self, solutionFile=par.solutionFileName, runMode=0):
        """Replay a solution written by Run(..., solutionFile) in the Exudyn SolutionViewer.

        The finger carriages are graphics-only ground objects; their poses recorded during Run are set
        before each frame (callPreStepUserFunction), the physical bodies come from the solution file.
        The viewer opens paused; press Run to start playback after its Tk window is fully initialized.
        @param runMode 0 = continuous run, 1 = one cycle, 2 = static (slider).
        """
        if self.fingerHistory is None or not self.fingerHistory['time']:
            raise ValueError('ShowSolution needs a preceding Run(..., solutionFile=...)')
        recordedTimes = np.asarray(self.fingerHistory['time'])
        recordedPoses = self.fingerHistory['poses']

        def ReplayPreStep(mbs, timeValue):
            """Set the finger graphics to the poses recorded at the replayed time."""
            index = max(0, int(np.searchsorted(recordedTimes, timeValue, side='right')) - 1)
            for body, pose in zip(self.model.bodies['fingers'], recordedPoses[index]):
                mbs.SetObjectParameter(body, 'referencePosition', pose[:3, 3])
                mbs.SetObjectParameter(body, 'referenceRotation', pose[:3, :3])
            return True

        self.SetRendererSettings()
        graphicsUpdateInterval = self.systemContainer.visualizationSettings.general.graphicsUpdateInterval
        self.mbs.SetPreStepUserFunction(ReplayPreStep)
        try:
            #automatic playback can close Tk during dialog construction,
            #before Exudyn calls deiconify(). Start from the ready dialog and preserve normal close handling.
            self.mbs.SolutionViewer(solution=LoadSolutionFile(str(solutionFile)), timeout=par.solutionWritePeriod,
                                    runMode=runMode, runOnStart=False, callPreStepUserFunction=True)
        finally:
            self.mbs.SetPreStepUserFunction(self.PreStep)
            #Exudyn's viewer cleanup is skipped if a dialog raises; restore these resources on every exit.
            self.systemContainer.visualizationSettings.general.graphicsUpdateInterval = graphicsUpdateInterval
            self.systemContainer.renderer.Stop()
