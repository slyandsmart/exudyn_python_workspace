# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: runSimulation.py
# Details:  Run a vehicle task (action list or state machine of a task file) with the native renderer or headless.
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-05 (lab options moved to runLab.py: 2026-11-07)
# Notes:    Base model only (no VR, no lab connection). With renderer (e.g. F5), the SolutionViewer replays
#           solution/vehicleSolution.txt afterwards; --noSolutionViewer disables it.
#           python runSimulation.py --taskFile studentFiles/myStateMachine.py
#           Vicon, real robot, sensors and videos: runLab.py; VR/AR viewer: runVR.py.
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Offline vehicle entry point."""
import argparse
from pathlib import Path
import numpy as np
from fahrzeug import parameters as par
from fahrzeug.simulation import VehicleSimulation, buttonStart, solverTypes, defaultSolverName
from fahrzeug.taskFile import LoadTask
from fahrzeug.preparationTests import TaskOf, taskReturn, returnTarget


#! State machine of the preparation, run without --taskFile.
defaultTaskFile = Path(__file__).resolve().parent / 'preparationStateMachine.py'


def AddSimulationArguments(parser):
    """Add the options of the base simulation (task, duration, renderer, solver, output) to an ArgumentParser."""
    parser.add_argument('--noGraphics', action='store_true', help='run without the native renderer')
    #600 s (was 120 s): the long measurement drive takes about 4 min; the run ends anyway
    #1 s after the task is finished
    parser.add_argument('--duration', type=float, default=600.,
                        help='maximum simulation time in seconds (default 600); the run ends 1 s after the task is finished')
    #default is the state machine of the preparation (F5 runs it)
    parser.add_argument('--taskFile', default=str(defaultTaskFile),
                        help='Python file with the state machine functionStateMachine (default: preparationStateMachine.py) '
                             'or, with runLab.py, an action list')
    parser.add_argument('--output', help='optional trajectory text file: time, world x, world y, yaw (SI units)')
    parser.add_argument('--initialPose', type=float, nargs=3, metavar=('X', 'Y', 'YAW'),
                        help='start pose: world x and y in m, yaw in deg (default 0 0 0, or the Vicon pose with --vicon)')
    parser.add_argument('--noSolutionViewer', action='store_true',
                        help='do not write solution/vehicleSolution.txt and open the SolutionViewer afterwards '
                             '(default with renderer, e.g. started with F5)')
    #select the integrator and worker count for the same physical tree.
    parser.add_argument('--solver', choices=solverTypes, default=defaultSolverName,
                        help='implicit: generalized-alpha, 2 ms; explicit: Euler, 0.2 ms')
    parser.add_argument('--threads', type=int, default=par.numberOfThreads, help='Exudyn worker threads (default: 4)')
    parser.add_argument('--showCollision', action='store_true',
                        help='show the shapes of the arm collision checker (translucent; red: closer than the safety '
                             'distance); also with the button Kollisionskörper an/aus')
    parser.add_argument('--realtime', action='store_true',
                        help='pace the simulation to the wall clock (default: as fast as possible, also with --vicon)')


def ParseArguments():
    """Return duration, renderer and task options for an offline simulation."""
    parser = argparse.ArgumentParser(description='Simulate the mobile robot task without a VR server')
    AddSimulationArguments(parser)
    return parser.parse_args()


def InitialPose(arguments):
    """Return the start pose [x in m, y in m, yaw in rad] of --initialPose, or None (default start pose)."""
    if arguments.initialPose is None:
        return None
    return [arguments.initialPose[0], arguments.initialPose[1], np.radians(arguments.initialPose[2])]


def SolutionFile(arguments):
    """Return the solution file for the SolutionViewer, or None in headless runs and with --noSolutionViewer."""
    #replay in the SolutionViewer by default; never in headless runs
    showSolution = not arguments.noGraphics and not arguments.noSolutionViewer
    return par.solutionFileName if showSolution else None


def RunOptions(arguments):
    """Return the keyword arguments of VehicleSimulation.Run for the parsed options."""
    #as fast as possible; real time only on request (--realtime, e.g. watching a live object)
    return {'duration': arguments.duration, 'realtimeFactor': 1. if arguments.realtime else None,
            'useRenderer': not arguments.noGraphics, 'solutionFile': SolutionFile(arguments), 'stopWhenFinished': True,
            'solverType': solverTypes[arguments.solver], 'numberOfThreads': arguments.threads}


def ReportResult(simulation, arguments, state):
    """Print time, task state and final pose, write --output, open the SolutionViewer; raise the task error."""
    print('Simulated time in s: %.2f' % simulation.mbs.systemData.GetTime())
    if simulation.callback is not None:
        print('Preparation state: %s' % simulation.callback.state)
    elif simulation.sequence is not None:
        print(simulation.sequence.Description())
    print('Final world pose [x in m, y in m, yaw in rad]:', state['pose'])
    #what a test of the preparation is checked by: object grasped and lifted, platform back
    lift = state['objectPosition'][2] - simulation.model.objectInitialPosition[2]
    #the distance to the start pose of this run (testStartPose, --initialPose), before always
    #to [0, 0]; the test 'return' drives to returnTarget
    reference, referenceName = simulation.model.initialPose[0:2], 'start pose'
    module = getattr(simulation, 'preparationModule', None)
    if module is not None and TaskOf(module) == taskReturn:
        reference, referenceName = returnTarget[0:2], 'return target'
    print('Object grasped: %s, lifted by %.3f m; platform %.3f m from the %s [%.3f, %.3f]'
          % ('yes' if state['grasped'] else 'no', lift,
             np.linalg.norm(np.asarray(state['pose'][0:2]) - reference), referenceName, reference[0], reference[1]))
    warnings = getattr(simulation.callback, 'warnings', [])
    if warnings:
        print('Warnings of the state machine: %d (see above), e.g. t = %.2f s: %s' % (len(warnings), *warnings[0]))
    if arguments.output:
        outputPath = Path(arguments.output)
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(outputPath, simulation.history, header='time in s, x in m, y in m, yaw in rad')
        print('Trajectory written to', outputPath)
    solutionFile = SolutionFile(arguments)
    if solutionFile is not None:
        #replay also after a task error, to see where it stopped
        print('Solution written to', solutionFile)
        #defer playback until the SolutionViewer dialog is ready.
        print('SolutionViewer opens paused; press Run to start playback.')
        simulation.ShowSolution(solutionFile)
    if simulation.control['error']:
        raise RuntimeError(simulation.control['error'])


def Main():
    """Run a selected task and report actual final pose and task status."""
    arguments = ParseArguments()
    actions, callback = LoadTask(arguments.taskFile)
    initialPose = InitialPose(arguments)
    testStartPose = getattr(callback, 'testStartPose', None)        #a subtask of the preparation that starts at the table
    if initialPose is None and testStartPose is not None:
        print('start pose of the test [x in m, y in m, yaw in deg]:', np.round(testStartPose, 3))
        initialPose = [testStartPose[0], testStartPose[1], np.radians(testStartPose[2])]
    simulation = VehicleSimulation(actionsInit=actions, callbackInit=callback, initialPoseInit=initialPose,
                                   settingsInit={'collisionShapesShown': arguments.showCollision})
    simulation.HandleCommand({'type': 'button', 'id': buttonStart})
    state = simulation.Run(**RunOptions(arguments))
    ReportResult(simulation, arguments, state)


if __name__ == '__main__':
    Main()
