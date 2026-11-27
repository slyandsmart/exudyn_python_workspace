# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: robotsSideBySide.py
#
# Details:  Several 6R robots side by side in the Exudyn renderer, each moving along its own joint trajectory at the
#           same time (exudyn.robotics.motion.Trajectory), with the trace of its TCP, a label (name, duration) and
#           optional marker points, the small tables 1, 2 with their cup holders at each robot, and below each robot a scope of the renderer with its joint velocities in percent
#           of the maximum (a SensorUserFunction per robot): the comparison of PTP motion modes and of blending in the
#           preparation (task 1.5).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-16
# Notes:    Kinematics only: the robots are drawn at the joint angles of their trajectories; a solver run in real time
#           without coordinates advances the time and fills the sensors of the scopes. Space in the renderer starts and
#           repeats the motion, Q ends.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Robots side by side moving along their joint trajectories, joint velocities in the scopes."""
import numpy as np
import exudyn as exu
import exudyn.graphics as graphics
from exudyn.itemInterface import ObjectGround, VObjectGround, SensorUserFunction
from exudyn.rigidBodyUtilities import RotationMatrixX, RotationMatrixZ

from labRobots.robots.robot6R.model import ManipulatorRobot6R, ForwardKinematics
from labRobots.robots.robot6R import visualization as robotVisualization
from robot6R.labObjects import LoadModelData
from robot6R.visualization import RobotGraphicsAtJointAngles, CreateLabTablesGraphics

#! Distance in m of the robots along y (side by side), time step in s of the motion (solver, sensors) and of the traces.
robotSpacing, sideBySideTimeStep = 1.2, 0.02
#! Height in m of the labels above the base, colors of the TCP traces and of the marker points, marker radius in m.
labelHeight = 1.0
traceColor, markerColor, markerRadius = [0.85, 0.2, 0.1, 1.], [0.1, 0.6, 0.1, 1.], 0.012
#! View of the renderer: center point in m, zoom per robot, rotation (seen obliquely from the front, y to the right;
#! the renderer uses the transposed rotation).
sideBySideViewCenter, sideBySideZoomPerRobot = [0.3, 0., 0.45], 0.42
sideBySideViewRotation = (RotationMatrixX(-0.5*np.pi + 0.35) @ RotationMatrixZ(-0.5*np.pi - 0.35)).T
#! Scopes of the joint velocities: height of the scope row (fraction of the window, at its bottom), range of the
#! velocity axis in percent of the maximum; at most maximumNumberOfScopes robots (scopes of the renderer).
scopeRowHeight, scopeVelocityRange, maximumNumberOfScopes = 0.32, [-105., 105.], 3


def SetSideBySideView(settings, numberOfRobots):
    """Set the renderer view (robots side by side along y, seen obliquely from the front) and the light as in the
    simulation (Robot6RSimulation.SetVisualizationSettings)."""
    settings.general.autoFitScene = False
    settings.general.showSolverInformation = False
    settings.general.graphicsUpdateInterval = sideBySideTimeStep
    settings.view0.scene.drawWorldBasis = True
    settings.view0.scene.worldBasisSize = 0.15
    settings.view0.window.renderWindowSize = [1600, 1000]
    settings.openGL.multiSampling = 4
    settings.openGL.advanced.initialCenterPoint = sideBySideViewCenter
    settings.openGL.advanced.initialZoom = sideBySideZoomPerRobot*max(numberOfRobots, maximumNumberOfScopes)   #same size for 2 or 3
    settings.openGL.advanced.initialModelRotation = sideBySideViewRotation
    settings.openGL.light0.position = [1., -1., 1.5, 0.]
    settings.openGL.light0.diffuse = 0.55
    settings.openGL.light1.position = [-1., 1., 3., 0.]
    settings.openGL.light1.diffuse = 0.25


def TrajectoryDuration(trajectory):
    """Return the duration in s of an exudyn.robotics.motion.Trajectory (starting at time 0)."""
    return float(trajectory.GetTimes()[-1])


def TCPTrace(trajectory, timeStep=sideBySideTimeStep):
    """Return (times, TCP positions [n x 3] in K0) of a joint trajectory sampled every timeStep s."""
    times = np.append(np.arange(0., TrajectoryDuration(trajectory), timeStep), TrajectoryDuration(trajectory))
    positions = np.array([ForwardKinematics(trajectory.Evaluate(time)[0])[0:3, 3] for time in times])
    return times, positions


class RobotsSideBySide:
    """Robots side by side in one renderer, each following its trajectory, with the scopes of their joint velocities
    (Show)."""

    def __init__(self, motions, maxVelocities, markerPoints=None, showScopesInit=True):
        """Args:
            motions (list): [(label, Trajectory)], one robot each (at most maximumNumberOfScopes), placed along y at
                robotSpacing
            maxVelocities (array): max. joint velocities in rad/s, 100 % of the scopes
            markerPoints (list): points in K0 of each robot (e.g. via points) drawn as small spheres at every robot
            showScopesInit (bool): show the scopes of the joint velocities below the robots (
                False for the configurations of the preparation task 1.3, only the robots)
        """
        self.showScopes = bool(showScopesInit)  #! Show the scopes of the joint velocities.
        if len(motions) > maximumNumberOfScopes:
            raise ValueError('RobotsSideBySide: at most %d robots (one scope each)' % maximumNumberOfScopes)
        self.motions = motions
        self.maxVelocities = np.asarray(maxVelocities, dtype=float)
        self.markerPoints = [] if markerPoints is None else [np.asarray(point, dtype=float) for point in markerPoints]
        self.offsets = [np.array([0., (index - 0.5*(len(motions) - 1))*robotSpacing, 0.]) for index in range(len(motions))]
        self.traces = [TCPTrace(trajectory) for label, trajectory in motions]
        self.totalDuration = max(TrajectoryDuration(trajectory) for label, trajectory in motions)
        robot = ManipulatorRobot6R(useCadData=True, includeGripper=True)
        self.linkGraphics = [link.visualization.graphicsData for link in robot.links]
        self.baseGraphics = robot.base.visualization.graphicsData
        modelData = LoadModelData()
        self.jawGraphics = robotVisualization.CreateJawGraphics(modelData)
        #! Static graphics: the small tables with the cup holders at every robot (the lab table is left out, it would
        #! overlap the neighbors).
        self.tablesGraphics = [item for offset in self.offsets for item in CreateLabTablesGraphics(modelData, offset)]

    def VelocityPercent(self, robotIndex, time):
        """Return the joint velocities in percent of maxVelocities of a robot at time (constant after its end)."""
        trajectory = self.motions[robotIndex][1]
        return 100.*trajectory.Evaluate(min(time, TrajectoryDuration(trajectory)))[1]/self.maxVelocities

    def RobotGraphics(self, time):
        """Return the graphics of all robots at time with their traces up to it, labels and marker points."""
        graphicsList = []
        for (label, trajectory), offset, (times, positions) in zip(self.motions, self.offsets, self.traces):
            duration = TrajectoryDuration(trajectory)
            q = trajectory.Evaluate(min(time, duration))[0]
            graphicsList += RobotGraphicsAtJointAngles(self.linkGraphics, self.jawGraphics, q, offset, self.baseGraphics)
            shown = positions[times <= time + 1e-9]
            if len(shown) > 1:
                graphicsList.append(graphics.Lines(list(shown + offset), color=traceColor))
            status = 'finished' if time >= duration else 't = %.1f s' % time
            graphicsList.append(graphics.Text(point=list(offset + [0., 0., labelHeight]), color=graphics.color.black,
                                              text='%s\n%.2f s, %s' % (label, duration, status)))
            for point in self.markerPoints:
                graphicsList.append(graphics.Sphere(point=list(point + offset), radius=markerRadius, color=markerColor, nTiles=12))
        return graphicsList

    def CreateSystem(self, systemContainer):
        """Create the system: a ground drawing the robots at the visualized time and one sensor per robot with its
        joint velocities in percent (for the scopes); return (mbs, sensor numbers)."""
        mbs = systemContainer.AddSystem()
        mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsData=self.tablesGraphics)))
        mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsDataUserFunction=lambda mbs, objectNumber:
            self.RobotGraphics(mbs.systemData.GetTime(exu.ConfigurationType.Visualization)))))
        sensors = []
        for robotIndex, (label, unusedTrajectory) in enumerate(self.motions):
            sensors.append(mbs.AddSensor(SensorUserFunction(
                sensorNumbers=[], factors=[], storeInternal=True, name=label,      #names must be unique (the labels are)
                sensorUserFunction=lambda mbs, t, sensorNumbers, factors, configuration, robotIndex=robotIndex:
                    list(self.VelocityPercent(robotIndex, t)))))
        mbs.Assemble()
        return mbs, sensors

    def ConfigureScopes(self, settings, sensors):
        """One scope per robot in a row at the bottom of the window, below the robots: its joint velocities in percent
        of the maximum over the time (legend: sensor name with [0] .. [5] for the joints 1..6)."""
        window = settings.view0.window
        window.showScopes = True
        numberOfScopes = len(sensors)
        for scope, sensor in enumerate(sensors):
            scopeSettings = getattr(window, 'scope%d' % scope)
            scopeSettings.location = [[scope/numberOfScopes, 1. - scopeRowHeight], [(scope + 1)/numberOfScopes, 1.]]
            scopeSettings.sensors = [[sensor, -1]]
            scopeSettings.minMaxX = [0., self.totalDuration]
            scopeSettings.minMaxY = scopeVelocityRange
            scopeSettings.xLabel = 'time t in s'
            scopeSettings.yLabel = 'joint velocity in % of max.'
            scopeSettings.showLegend = scope == 0

    def SimulationSettings(self):
        """Return the settings of the run in real time over the longest motion (no coordinates, no solution file)."""
        simulationSettings = exu.SimulationSettings()
        simulationSettings.timeIntegration.endTime = self.totalDuration
        simulationSettings.timeIntegration.numberOfSteps = int(np.ceil(self.totalDuration/sideBySideTimeStep))
        simulationSettings.timeIntegration.simulateInRealtime = True
        simulationSettings.timeIntegration.verboseMode = 0
        simulationSettings.solution.file.write = False
        simulationSettings.solution.sensors.writePeriod = sideBySideTimeStep
        return simulationSettings

    def Show(self):
        """Open the renderer: Space starts the motion of all robots at once (in real time), Space again repeats it,
        Q or closing the window ends."""
        systemContainer = exu.SystemContainer()
        mbs, sensors = self.CreateSystem(systemContainer)
        SetSideBySideView(systemContainer.visualizationSettings, len(self.motions))
        if self.showScopes:
            self.ConfigureScopes(systemContainer.visualizationSettings, sensors)
        simulationSettings = self.SimulationSettings()
        systemContainer.renderer.Start()
        print('robots side by side: Space in the renderer starts (and repeats) the motion, Q ends')
        try:
            while True:
                systemContainer.renderer.DoIdleTasks()          #wait for Space (start, repeat) or Q
                if mbs.GetRenderEngineStopFlag() or not systemContainer.renderer.IsActive():
                    break
                mbs.SolveDynamic(simulationSettings)
                if mbs.GetRenderEngineStopFlag() or not systemContainer.renderer.IsActive():
                    break
        finally:
            systemContainer.renderer.Stop()
