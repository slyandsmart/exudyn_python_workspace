# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: solverFiles.py
#
# Details:  The solution file of a simulation in a folder with non-ASCII characters (e.g. "Praktikumsdurchführung"
#           or a user name with an umlaut): the Exudyn solver (C++, 1.13.649) writes it into a wrongly encoded
#           folder ("fÃ¼hrung"), the SolutionViewer then does not find it. SolutionFileForSolver lets the solver
#           write with a relative ASCII name from the folder of the file and puts the full name back afterwards
#           (SolutionViewer and LoadSolutionFile read it with Python, which handles any path).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-16
# Notes:    Remove once the Exudyn solver opens its output files with wide characters on Windows.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Solution files of the Exudyn solver in folders with non-ASCII names."""
import os
from contextlib import contextmanager


@contextmanager
def SolutionFileForSolver(simulationSettings):
    """Context for mbs.SolveDynamic: if simulationSettings.solution.file.name has non-ASCII characters, the solver runs
    in the folder of the file (os.chdir) with its bare (ASCII) name; afterwards the working directory and the full
    name are restored. Other names are left as they are.

    Args:
        simulationSettings: exudyn.SimulationSettings with the solution file set

    Raises:
        ValueError: if the bare file name itself is not ASCII
    """
    fileName = simulationSettings.solution.file.name
    if not simulationSettings.solution.file.write or fileName.isascii():
        yield
        return
    directory, baseName = os.path.split(os.path.abspath(fileName))
    if not baseName.isascii():
        raise ValueError('SolutionFileForSolver: the name of the solution file "%s" must be ASCII (Exudyn solver)'
                         % baseName)
    os.makedirs(directory, exist_ok=True)
    workingDirectory = os.getcwd()
    os.chdir(directory)
    simulationSettings.solution.file.name = baseName
    try:
        yield
    finally:
        os.chdir(workingDirectory)
        simulationSettings.solution.file.name = os.path.join(directory, baseName)
