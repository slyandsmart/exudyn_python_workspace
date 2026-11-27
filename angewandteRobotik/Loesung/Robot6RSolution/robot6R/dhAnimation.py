# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: dhAnimation.py
#
# Details:  The DH frames of an own DH table (DHFrames) and the animation of one row of it in the renderer of the
#           simulation (DHRowAnimation): the frames of the rows before stand, the frame of the animated row moves from
#           K(i-1) in the four steps of the standard DH transformation, rotation about z by theta, translation along z
#           by d, translation along x by a, rotation about x by alpha, each labeled with its value. Used by the
#           preparation (task 1.2), where the students build the DH table row by row.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-16
# Notes:    Kinematics only: the renderer is driven by SendRedrawSignal and DoIdleTasks, no solver.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""DH frames of an own DH table and the animation of one row."""
import numpy as np
from exudyn.robotics import StdDH2HT
from exudyn.rigidBodyUtilities import HTrotateX, HTrotateZ, HTtranslate

#! Number of the elementary steps of a DH row: rotation about z, translation along z, along x, rotation about x.
numberOfDHSteps = 4
#! Duration in s of one step of the animation, pause in s before the first and after the last step, frame period in s.
dhStepTime, dhPauseTime, dhFramePeriod = 1.5, 1., 0.03
radToDeg = 180./np.pi
#! Value of rowNumber of DHRowAnimation that animates all rows of the table one after the other.
animateAllRows = 'all'


def DHFrames(q, dhParameters):
    """Return [('K1', 4x4 pose in K0), ...] of the rows of the DH table dhParameters (rows [theta offset, d, a, alpha],
    any number up to len(q)) at joint angles q in rad."""
    pose = np.eye(4)
    frames = []
    for jointIndex, (thetaOffset, d, a, alpha) in enumerate(dhParameters):
        pose = pose @ StdDH2HT([q[jointIndex] + thetaOffset, d, a, alpha])
        frames.append(('K%d' % (jointIndex + 1), pose))
    return frames


def DHStepPose(theta, d, a, alpha, progress):
    """Return the 4x4 transformation of a DH row after progress (0..numberOfDHSteps) of its steps: RotZ(theta),
    TransZ(d), TransX(a), RotX(alpha), each step scaled by the fraction of it done (StdDH2HT at progress 4)."""
    fractions = [float(np.clip(progress - step, 0., 1.)) for step in range(numberOfDHSteps)]
    return (HTrotateZ(fractions[0]*theta) @ HTtranslate([0., 0., fractions[1]*d]) @ HTtranslate([fractions[2]*a, 0., 0.])
            @ HTrotateX(fractions[3]*alpha))


def DHStepText(rowNumber, theta, d, a, alpha, progress):
    """Return the label of the step at progress of the row rowNumber (1..), e.g. 'row 2: rotation about z1 by -90 deg'."""
    previous = rowNumber - 1
    step = min(int(progress), numberOfDHSteps - 1)
    texts = ['rotation about z%d by theta = %.1f deg' % (previous, theta*radToDeg),
             'translation along z%d by d = %.4f m' % (previous, d),
             'translation along x%d by a = %.4f m' % (rowNumber, a),
             'rotation about x%d by alpha = %.1f deg' % (rowNumber, alpha*radToDeg)]
    return 'row %d: %s' % (rowNumber, texts[step])


class DHRowAnimation:
    """Frames of an own DH table for the renderer (Frames, as toolFrameFunction of the simulation) with the animation of
    one row (Run): while it runs, the frames of the rows before stand and the frame of the animated row moves through
    the four DH steps; otherwise all frames of the table are returned."""

    def __init__(self, dhParametersFunction, rowNumber=None):
        """Args:
            dhParametersFunction: function returning the DH table (rows [theta offset, d, a, alpha]), e.g.
                DHParameters of the preparation; called at every frame (unsolved: may raise)
            rowNumber (int): row (1..) to animate; None: the last row of the table; animateAllRows ('all'): all rows
                one after the other
        """
        self.dhParametersFunction = dhParametersFunction
        self.rowNumber = rowNumber
        self.progress = None    #! Progress (0..numberOfDHSteps) of the running animation, None: not running.
        self.activeRow = 0      #! Number (1..) of the row the running animation moves.

    def AnimatedRowNumbers(self):
        """Return the numbers (1..) of the rows to animate ([] for an empty table); raise ValueError for a row number
        beyond the table."""
        numberOfRows = len(self.dhParametersFunction())
        if self.rowNumber == animateAllRows:
            return list(range(1, numberOfRows + 1))
        rowNumber = numberOfRows if self.rowNumber is None else self.rowNumber
        if rowNumber > numberOfRows or rowNumber < 0:
            raise ValueError('animated DH row %s, the table has %d rows' % (rowNumber, numberOfRows))
        return [rowNumber] if rowNumber > 0 else []

    def Frames(self, q):
        """Return [(label, 4x4 pose in K0)] of the DH frames at joint angles q in rad: during the animation the frames of
        the rows before the animated one and the moving frame labeled with its step, otherwise all frames."""
        dhParameters = self.dhParametersFunction()
        if self.progress is None:
            return DHFrames(q, dhParameters)
        rowNumber = self.activeRow
        frames = DHFrames(q, dhParameters[0:rowNumber - 1])
        previousPose = frames[-1][1] if frames else np.eye(4)
        thetaOffset, d, a, alpha = dhParameters[rowNumber - 1]
        theta = q[rowNumber - 1] + thetaOffset
        pose = previousPose @ DHStepPose(theta, d, a, alpha, self.progress)
        frames.append((DHStepText(rowNumber, theta, d, a, alpha, self.progress), pose))
        return frames

    def Run(self, simulation):
        """Play the animation in the running renderer of simulation (Robot6RSimulation, rendererAnimation of Simulate):
        each step in dhStepTime s, the rows of AnimatedRowNumbers one after the other; 'Q' in the renderer ends it."""
        rowNumbers = self.AnimatedRowNumbers()
        renderer = simulation.SC.renderer
        times = np.arange(0., numberOfDHSteps*dhStepTime + 2.*dhPauseTime + dhFramePeriod, dhFramePeriod)
        if rowNumbers:
            print('animation of DH row %s (Q in the renderer: skip)' % ', '.join(str(row) for row in rowNumbers))
        try:
            for rowNumber in rowNumbers:
                self.activeRow = rowNumber
                for time in times:
                    self.progress = float(np.clip((time - dhPauseTime)/dhStepTime, 0., numberOfDHSteps))
                    simulation.mbs.SendRedrawSignal()
                    renderer.DoIdleTasks(waitSeconds=dhFramePeriod, printPauseMessage=False)
                    if simulation.mbs.GetRenderEngineStopFlag():
                        simulation.mbs.SetRenderEngineStopFlag(False)    #Q skips the animation, not the simulation
                        return
        finally:
            self.progress = None
            simulation.mbs.SendRedrawSignal()
