# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: physicalSimulation.py
# Details:  Free-cup contact simulation with physical robot and compliant grasp.
# Authors:  Project contributors; Michael Pieber (created with Claude Code)
# Date:     2026-11-05
# Notes:    Worker count is inherited from the shared simulation configuration.
#           joint and jaw control by Robot.SetControl/PreStepControl of Exudyn (static
#           gravity compensation of all 8 links in C++ at every step), joint sensors by Robot.AddSensors.
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Free-cup simulation with GeneralContact and a compliant holding element."""

import warnings
import numpy as np

import exudyn as exu
from exudyn.itemInterface import (CartesianSpringDamper, MarkerBodyRigid, MarkerKinematicTreeRigid,
                                   SensorBody, VCartesianSpringDamper)

from . import parameters as par
from labRobots.robots.robot6R import visualization as robotVisualization
from .contact import CreateCupContact, JawContactPoints
from labRobots.robots.robot6R.model import ManipulatorRobot6R, ForwardKinematics
from .labObjects import LoadModelData
from .simulation import Robot6RSimulation


class PhysicalRobot6RSimulation(Robot6RSimulation):
    """6R simulation with a free cup; useGripperHold=False disables optional grasp assistance."""

    solverType = exu.DynamicSolverType.ExplicitEuler

    def __init__(self, labTrajectory, cupMass=par.cupMass, liquidMass=0., includeCup=True, includeEnvironment=True,
                 stepSize=par.physicalContactStepSize, writePeriod=par.simulationWritePeriod,
                 bandwidthHz=par.controlBandwidthHz, dampingRatio=par.controlDampingRatio,
                 gravityCompensation=True, solutionFileName=None, verbose=True, tableContactGeometry='simplified',
                 useGripperHold=True, frames=None, toolFrameFunction=None, showRobot=True):
        if not includeCup:
            raise ValueError('PhysicalRobot6RSimulation requires includeCup=True')
        if not np.isfinite(stepSize) or stepSize <= 0.:
            raise ValueError('stepSize must be positive and finite')
        if stepSize > par.physicalContactStepSize:
            warnings.warn('stepSize above %g s may make cup contact unstable' % par.physicalContactStepSize,
                          RuntimeWarning, stacklevel=2)
        from .simulation import defaultSolutionFileName
        self.labTrajectory = labTrajectory
        self.payloadMass = cupMass + liquidMass
        self.includeCup = True
        self.stepSize = stepSize
        self.writePeriod = writePeriod
        self.gravityCompensation = gravityCompensation
        self.useGripperHold = useGripperHold
        self.solutionFileName = defaultSolutionFileName if solutionFileName is None else solutionFileName
        self.verbose = verbose
        self.modelData = LoadModelData()
        self.SC = exu.SystemContainer()
        self.mbs = self.SC.AddSystem()
        self.robot = ManipulatorRobot6R(useCadData=True, includeGripper=True, includeJaws=True,
                                  bandwidthHz=bandwidthHz, dampingRatio=dampingRatio)
        robotDict = self.robot.CreateKinematicTree(self.mbs)
        #coordinate systems of the joints off for all kinematic trees (at the user's request)
        self.mbs.GetSystemContainer().visualizationSettings.bodies.kinematicTree.showJointFrames = False
        self.objectKinematicTree = robotDict['objectKinematicTree']
        self.nodeKinematicTree = robotDict['nodeGeneric']
        jawOffset = robotVisualization.ComputeJawOffset(labTrajectory.initialGripperPercent)
        initial = list(labTrajectory.initialJointAngles) + [jawOffset, jawOffset]
        self.mbs.SetNodeParameter(self.nodeKinematicTree, 'initialCoordinates', initial)

        self.initialCupHT = np.eye(4)
        self.initialCupHT[0:3, 3] = np.array(par.table1Position) + [0., 0., par.table1Height + 0.0005]
        self.cupEvents = []
        self.processedGripperEvents = 0
        self.contact, self.objectCup = CreateCupContact(self.mbs, self.modelData, self.initialCupHT[0:3, 3],
                                                        self.payloadMass, includeEnvironment, tableContactGeometry)
        self.markerCup = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=self.objectCup,
                                                          localPosition=[0., 0., 0.5*par.cupHeight]))
        tcpPosition = [0., 0., par.toolLength]
        self.markerTCP = self.mbs.AddMarker(MarkerKinematicTreeRigid(objectNumber=self.objectKinematicTree,
                                                                     linkNumber=5, localPosition=tcpPosition))
        self.gripperConnector = self.mbs.AddObject(CartesianSpringDamper(
            markerNumbers=[self.markerTCP, self.markerCup], stiffness=[par.gripperHoldStiffness]*3,
            damping=[par.gripperHoldDamping]*3,
            activeConnector=False, visualization=VCartesianSpringDamper(show=False)))

        stiffness = par.contactStiffness
        damping = stiffness*par.contactDampingFactor
        #spheres on the tooth tips of the jaw mesh (JawContactPoints), not on the fitted arc
        for index, name in enumerate(par.jawNames):
            for point in JawContactPoints(self.modelData, name):
                marker = self.mbs.AddMarker(MarkerKinematicTreeRigid(objectNumber=self.objectKinematicTree,
                                                                      linkNumber=6 + index, localPosition=point))
                self.contact.AddSphereWithMarker(markerIndex=marker, radius=par.contactSphereRadius,
                                                 contactStiffness=stiffness, contactDamping=damping,
                                                 frictionMaterialIndex=0)

        self.showRobot = showRobot                           #hide the lab robot
        if not showRobot:
            self.mbs.SetObjectParameter(self.objectKinematicTree, 'Vshow', False)
        self.AddFrameGraphics(frames, toolFrameFunction, drawFrameChain=not showRobot)     #
        self.SetupControlAndSensors()
        self.sensorCupPosition = self.mbs.AddSensor(SensorBody(bodyNumber=self.objectCup, storeInternal=True,
                                                               writeToFile=False, outputVariableType=exu.OutputVariableType.Position))
        self.sensorCupRotation = self.mbs.AddSensor(SensorBody(bodyNumber=self.objectCup, storeInternal=True,
                                                               writeToFile=False, outputVariableType=exu.OutputVariableType.RotationMatrix))
        self.mbs.SetPreStepUserFunction(self.PreStepUserFunction)
        self.robot.PreStepControl(self.mbs, 0.)    #consistent reference for the initial accelerations
        self.mbs.Assemble()
        self.SetVisualizationSettings()

    def ConfigureSimulationSettings(self, simulationSettings):
        super().ConfigureSimulationSettings(simulationSettings)
        simulationSettings.timeIntegration.explicit.computeMassMatrixInversePerBody = True
        simulationSettings.timeIntegration.explicit.computeEndOfStepAccelerations = False
        simulationSettings.linearSolver.solverType = exu.LinearSolverType.EigenSparse

    def ComputeReference(self, t):
        """Return the reference [q, q_t, q_tt] of the 6 joints and the 2 jaws at time t from the lab trajectory."""
        q, q_t, _ = self.labTrajectory.Evaluate(t)
        return self.JointAndJawReference(q, q_t, self.labTrajectory.GetGripperPercent(t))

    #Michael Pieber 2026-11-03: split from the control reference, reused by robot6R.vr.interactiveSimulation
    def JointAndJawReference(self, q, q_t, gripperPercent):
        """Return the reference [q, q_t, q_tt] of all 8 links: arm joints q, q_t (q_tt not used by the static
        compensation) and both jaws at the opening of gripperPercent, at rest."""
        offset = robotVisualization.ComputeJawOffset(gripperPercent)
        numberOfLinks = len(self.robot.links)
        return [np.concatenate((q, [offset, offset])), np.concatenate((q_t, [0., 0.])), np.zeros(numberOfLinks)]

    def GetCurrentTCP(self):
        q = self.mbs.GetNodeOutput(self.nodeKinematicTree, exu.OutputVariableType.Coordinates)
        return ForwardKinematics(q[:par.numberOfJoints])

    def CurrentCupHT(self):
        HT = np.eye(4)
        HT[:3, 3] = self.mbs.GetObjectOutputBody(self.objectCup, exu.OutputVariableType.Position, [0., 0., 0.])
        HT[:3, :3] = np.array(self.mbs.GetObjectOutputBody(self.objectCup, exu.OutputVariableType.RotationMatrix,
                                                          [0., 0., 0.])).reshape(3, 3)
        return HT

    def TryGrasp(self, time):
        if not self.useGripperHold:
            self.Print('t = %.3f s: holding assistance disabled; grasp depends on contact and friction' % time)
            return
        if self.IsCupAttached(time)[0]:
            return
        cupHT = self.CurrentCupHT()
        tcp = self.GetCurrentTCP()
        axis = cupHT[:3, 2]
        delta = tcp[:3, 3] - cupHT[:3, 3]
        height = delta @ axis
        distance = np.linalg.norm(delta - height*axis)
        if distance <= par.gripperCaptureDistance and 0. < height < par.cupHeight and abs(tcp[:3, 1] @ axis) < 0.5:
            self.mbs.SetMarkerParameter(self.markerCup, 'localPosition', [0., 0., height])
            self.mbs.SetObjectParameter(self.gripperConnector, 'activeConnector', True)
            self.cupEvents.append((time, 'grasp', cupHT))
            self.Print('t = %.3f s: compliant gripper activated' % time)
        else:
            self.cupEvents.append((time, 'graspFailed', cupHT))
            self.Print('t = %.3f s: cup outside gripper range' % time)

    def Release(self, time):
        if self.IsCupAttached(time)[0]:
            self.mbs.SetObjectParameter(self.gripperConnector, 'activeConnector', False)
            self.cupEvents.append((time, 'release', self.CurrentCupHT()))
            self.Print('t = %.3f s: compliant gripper released' % time)

    def GetCupHT(self, time, HTtcp):
        raise NotImplementedError('Use GetResults() for the stored free-cup trajectory or CurrentCupHT() for the current state')

    def GetCupResults(self, time, q):
        position = self.mbs.GetSensorStoredData(self.sensorCupPosition)
        rotation = self.mbs.GetSensorStoredData(self.sensorCupRotation)
        if not np.allclose(position[:, 0], time) or not np.allclose(rotation[:, 0], time):
            raise RuntimeError('cup and robot sensor time grids do not match')
        matrices = rotation[:, 1:].reshape(-1, 3, 3)
        tilt = np.degrees(np.arccos(np.clip(matrices[:, 2, 2], -1., 1.)))
        attached = np.array([self.IsCupAttached(t)[0] for t in time], dtype=bool)
        return position[:, 1:4], tilt, attached
