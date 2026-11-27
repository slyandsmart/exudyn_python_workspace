# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: plotting.py
#
# Details:  Central plotting utilities for the 6R robot simulations: common
#           matplotlib style and standard plots of joint angles, tracking
#           errors, drive torques (with Schunk nominal torques), TCP position
#           and cup tilt.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-03
# Notes:    Axis labels follow the form "time t in s"; sentence case labels.
#
# License:  BSD-3 license
#
# Installation:
#           pip install matplotlib
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import numpy as np
import matplotlib.pyplot as plt

from . import parameters as par

fontSize = 12
lineWidth = 1.5
figureSize = (10, 6)


def SetPlotStyle(fontSizeValue=fontSize):
    """Set the global matplotlib style used for all figures; the default style is restored first, so that a style
    changed by another package does not apply."""
    plt.style.use('default')
    plt.rc('lines', linewidth=lineWidth)
    plt.rc('font', size=fontSizeValue)
    plt.rc('axes', titlesize=fontSizeValue)
    plt.rc('axes', labelsize=fontSizeValue)
    plt.rc('xtick', labelsize=fontSizeValue)
    plt.rc('ytick', labelsize=fontSizeValue)
    plt.rc('legend', fontsize=fontSizeValue)


def SaveFigure(figureHandle, fileName):
    """Apply tight layout and save figure (creates the directory if needed)."""
    plt.tight_layout()
    directory = os.path.dirname(fileName)
    if directory != '':
        os.makedirs(directory, exist_ok=True)
    figureHandle.savefig(fileName, dpi=150)


def PlotSimulationResults(results, fileNamePrefix, title=''):
    """Create standard plots of a simulation: joint angles, tracking errors, drive torques, TCP position and cup tilt.

    Args:
        results (dict): results of Robot6RSimulation.GetResults()
        fileNamePrefix (str): prefix of PNG files, e.g. 'solution/labTask1'
        title (str): optional title added to the figures

    Returns:
        list: figure handles
    """
    SetPlotStyle()
    time = results['time']
    figures = []

    figureHandle, axes = plt.subplots(2, 1, figsize=figureSize, sharex=True)
    for jointIndex in range(par.numberOfJoints):
        axes[0].plot(time, results['q'][:, jointIndex]*par.radToDeg, label='joint ' + str(jointIndex+1))
        axes[1].plot(time, results['trackingError'][:, jointIndex]*par.radToDeg, label='joint ' + str(jointIndex+1))
    axes[0].set_ylabel('joint angle q in deg')
    axes[1].set_ylabel('tracking error in deg')
    axes[1].set_xlabel('time t in s')
    axes[0].legend(ncol=3)
    axes[0].set_title(title + ' joint angles and tracking errors')
    for axis in axes:
        axis.grid(True)
    SaveFigure(figureHandle, fileNamePrefix + 'JointAngles.png')
    figures.append(figureHandle)

    figureHandle = plt.figure(figsize=figureSize)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for jointIndex in range(4):
        plt.plot(time, results['driveTorques'][:, jointIndex], color=colors[jointIndex], label='joint ' + str(jointIndex+1))
        for sign in [-1., 1.]:
            plt.axhline(sign*par.schunkNominalTorques[jointIndex], color=colors[jointIndex], linestyle='--', linewidth=0.8)
            plt.axhline(sign*par.schunkPeakTorques[jointIndex], color=colors[jointIndex], linestyle=':', linewidth=0.8)
    plt.xlabel('time t in s')
    plt.ylabel('drive torque in Nm')
    plt.title(title + ' drive torques of Schunk axes (dashed: nominal, dotted: peak torque)')
    plt.legend()
    plt.grid(True)
    SaveFigure(figureHandle, fileNamePrefix + 'DriveTorques.png')
    figures.append(figureHandle)

    figureHandle, axes = plt.subplots(2, 1, figsize=figureSize, sharex=True)
    for coordinateIndex, coordinateName in enumerate(['x', 'y', 'z']):
        axes[0].plot(time, results['positionTCP'][:, coordinateIndex], label='TCP ' + coordinateName)
    axes[1].plot(time, results['cupTiltDeg'], label='cup tilt')
    axes[1].plot(time, results['gripperPercent']/10., label='gripper command in % / 10')
    axes[0].set_ylabel('TCP position in m')
    axes[1].set_ylabel('cup tilt in deg')
    axes[1].set_xlabel('time t in s')
    axes[0].legend()
    axes[1].legend()
    axes[0].set_title(title + ' TCP position and cup tilt')
    for axis in axes:
        axis.grid(True)
    SaveFigure(figureHandle, fileNamePrefix + 'TCPAndCup.png')
    figures.append(figureHandle)
    return figures


def PrintSimulationSummary(results):
    """Print the most important simulation results."""
    print('simulated time:', round(results['time'][-1], 3), 's')
    print('max. tracking error per joint in deg:', np.round(np.max(np.abs(results['trackingError']), axis=0)*par.radToDeg, 4))
    print('max. |drive torque| / nominal torque (axes 1-4):', np.round(results['maxTorqueRatioNominal'], 3))
    print('max. |drive torque| / peak torque (axes 1-4):', np.round(results['maxTorqueRatioPeak'], 3))
    if np.any(results['cupAttached']):
        print('max. cup tilt while grasped:', round(float(np.max(results['cupTiltDeg'][results['cupAttached']])), 3), 'deg')
    for eventTime, eventType, eventData in results['cupEvents']:
        print('cup event at t =', round(eventTime, 3), 's:', eventType)
    print('final cup position:', np.round(results['cupPosition'][-1], 4), 'm')
    if len(results['skippedRows']) > 0:
        print('rows rejected by the lab check:', results['skippedRows'])
