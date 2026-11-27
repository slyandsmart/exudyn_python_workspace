# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: taskFile.py
# Details:  Load an action list/factory or an existing preparation state machine.
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-04 (MoveIt geometry: 2026-11-05)
# Notes:    Python task files are user code and run when explicitly selected.
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Explicit task-file loading."""
import contextlib
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np

#! Submission directory resolved relative to the project, not the shell working directory.
studentDirectory = Path(__file__).resolve().parents[1] / 'studentFiles'
#ROS packages imported at module level by the real-run files (bringup_lecture.py);
#! replaced by placeholders while loading if they are not installed, so functionStateMachine can be used.
rosPackageNames = ('rospy', 'roslib', 'rospkg', 'moveit_commander', 'moveit_msgs', 'geometry_msgs', 'std_msgs',
                   'std_srvs', 'sensor_msgs', 'actionlib', 'move_base_msgs', 'tf', 'tf2_ros', 'tf2_geometry_msgs',
                   'vicon_bridge', 'wsg_50_common', 'robotnik_modules')


class RosPlaceholder:
    """Stand-in for any ROS object: callable, attribute access and indexing return placeholders."""

    def __call__(self, *arguments, **keywordArguments):
        """Return a placeholder, e.g. for rospy.Rate(50)."""
        return RosPlaceholder()

    def __getattr__(self, name):
        """Return a placeholder for any attribute, e.g. geometry_msgs.msg.Twist."""
        if name.startswith('__'):
            raise AttributeError(name)
        return RosPlaceholder()

    def __getitem__(self, key):
        """Return a placeholder for indexing."""
        return RosPlaceholder()


class RosPlaceholderModule(types.ModuleType):
    """Module whose attributes are placeholders; 'from module import *' imports nothing."""

    __all__ = []

    def __getattr__(self, name):
        """Return a placeholder for any module attribute."""
        if name.startswith('__'):
            raise AttributeError(name)
        return RosPlaceholder()


class RosPlaceholderFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import hook creating RosPlaceholderModule objects for ROS packages that are not installed."""

    def __init__(self, packageNamesInit):
        """@param packageNamesInit Top-level package names to replace."""
        self.packageNames = set(packageNamesInit)  #! Top-level names handled by this finder.

    def find_spec(self, fullname, path, target=None):
        """Return a placeholder module spec for the selected packages and their submodules."""
        if fullname.split('.')[0] in self.packageNames:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        """Create the placeholder module."""
        module = RosPlaceholderModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        """Nothing to execute for a placeholder."""


def LegacyHT(rotationMatrix, translation):
    """Homogeneous transformation from rotation matrix and translation, as rigidBodyUtilities.HT of Exudyn 1.9."""
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotationMatrix, dtype=float)
    result[:3, 3] = np.asarray(translation, dtype=float)[:3]
    return result


@contextlib.contextmanager
def RosPlaceholders():
    """Install placeholders for the ROS packages that cannot be imported, and remove them afterwards."""
    missingNames = [name for name in rosPackageNames if importlib.util.find_spec(name) is None]
    finder = RosPlaceholderFinder(missingNames)
    sys.meta_path.insert(0, finder)
    try:
        yield missingNames
    finally:
        sys.meta_path.remove(finder)
        for moduleName in [name for name in sys.modules if name.split('.')[0] in finder.packageNames]:
            del sys.modules[moduleName]


def SelectStudentTask(directory, currentName=None, step=0):
    """Rescan direct Python submissions and return the selected path, or None for an empty folder."""
    directory = Path(directory)
    files = sorted((path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == '.py'
                    and not path.name.startswith('_')), key=lambda path: path.name.casefold()) if directory.is_dir() else []
    names = [path.name for path in files]
    index = names.index(currentName) if currentName in names else 0
    if currentName in names:
        index = (index + step) % len(files)
    return files[index] if files else None


def LoadTask(path):
    """Return (actions or factory, callback) from a Python file; preserve the preparation callback name."""
    if path is None:
        return None, None
    taskPath = Path(path).resolve(strict=True)
    moduleName = 'vehicleUserTask'
    specification = importlib.util.spec_from_file_location(moduleName, taskPath)
    if specification is None or specification.loader is None:
        raise ValueError('Task file must be an importable Python file')
    module = importlib.util.module_from_spec(specification)
    previousPath = list(sys.path)
    try:
        sys.path.insert(0, str(taskPath.parent))
        sys.modules[moduleName] = module
        #read the source on every explicit load, even for edits within the same second.
        source = taskPath.read_text(encoding='utf-8-sig')
        with RosPlaceholders():
            exec(compile(source, str(taskPath), 'exec'), module.__dict__)
    except (SyntaxError, ImportError, NameError, AttributeError, ArithmeticError, LookupError, RuntimeError, AssertionError) as error:
        raise ValueError('Task file could not be loaded (%s): %s' % (taskPath.name, error)) from error
    finally:
        sys.path[:] = previousPath
    #older Exudyn (lab PC, 1.9) had rigidBodyUtilities.HT(A, r), used by some groups
    if 'HT' not in module.__dict__:
        module.HT = LegacyHT
    actions = getattr(module, 'CreateActions', getattr(module, 'actions', None))
    callback = getattr(module, 'functionStateMachine', None)
    if (actions is None) == (callback is None):
        raise ValueError('Task file must define either CreateActions(model), actions, or functionStateMachine')
    if callback is not None and not callable(callback):
        raise TypeError('functionStateMachine must be callable')
    if callback is not None:
        callback = PreparationTask(module, callback, taskPath.name)
    return actions, callback


def PreparationTask(module, callback, fileName):
    """Return the state machine to run of a preparation task file: functionStateMachine, or
    with runTask of the file a test of one subtask (fahrzeug.preparationTests: 'drive', 'grasp', 'return') with its
    start pose (attribute testStartPose, [x, y, yaw in deg], None: the default start); attribute preparationModule:
    the task file, whose results the simulation draws (switches show... of the file)."""
    from .preparationTests import TaskOf, TestCallback, TestStartPose, ConfiguredStartPose, taskStateMachine, studentErrors
    task = TaskOf(module)
    ConfiguredStartPose(module)     #a wrong testStartPose stops here, not the default start
    if task != taskStateMachine:
        print('%s: runTask = %r, only this subtask runs (the whole mission and the lab: runTask = %r)'
              % (fileName, task, taskStateMachine))
        callback = TestCallback(module, task)
        try:
            callback.testStartPose = TestStartPose(module, task)
        except studentErrors as error:
            print('%s: start pose of the test not computed (ApproachPose: %s), default start' % (fileName, error))
            callback.testStartPose = None
        return callback
    try:
        callback.preparationModule = module
        callback.testStartPose = TestStartPose(module, task)     #also the whole mission
    except AttributeError:      #a callable without attributes (e.g. a bound method): nothing is drawn
        pass
    return callback
