# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: model.py
# Details:  Scene of the lab task of the Kairos: the Kairos (kairosModel.KairosModel) with floor, Vicon origin, table,
#           free cup with contact, assisted grasp and the live Vicon object.
# Authors:  Peter Manzl; Michael Pieber (created with Claude Code)
# Date:     2026-11-05 (floating-base KinematicTree; live Vicon object 2026-11-06); 2026-11-12 robot in KairosModel,
#           this file keeps the scene
# Notes:    Derived from ExudynModelV3. No renderer, global system or solver calls.
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Scene of the lab task of the Kairos."""
import numpy as np
import exudyn as exu
from exudyn.itemInterface import MarkerBodyRigid, RigidBodySpringDamper, VRigidBodySpringDamper, ObjectGround
from exudyn.rigidBodyUtilities import HTtranslate, HTrotateZ, RotationMatrix2RotationVector
from exudyn.robotics import RobotBase
from . import parameters as par
from labRobots.robots.kairos.model import KairosModel
from labRobots.robots.kairos.treeBody import GetBodyOutput
from .collisionModel import TableObstacle
from .scene import CreateLabScene

#! Name of the KinematicTree object of the Kairos in the scene.
vehicleName = 'kairos'


class VehicleModel(KairosModel):
    """Scene of the lab task: the Kairos (KairosModel) with floor, table, free cup and assisted grasp."""

    def __init__(self, mbsInit, initialPoseInit=None, objectPositionInit=None, objectTrackingInit=False,
                 armInitialAnglesInit=None):
        """Build the unassembled scene of the lab task in mbsInit; poses and velocities use SI units.

        @param initialPoseInit Platform pose [x, y, yaw] in the world (Vicon) frame; None: the origin.
        @param objectPositionInit Center of the object (cup) in world coordinates; None: par.objectPosition. The table
               stands below it.
        @param objectTrackingInit If True, the free object follows a moving target (live Vicon object,
               UpdateObjectTracking) through a spring-damper while it is not grasped.
        @param armInitialAnglesInit UR5 joint angles in rad at the start (e.g. of the real robot);
               None: par.armInitialAngles.
        """
        initialPose = np.asarray([0., 0., 0.] if initialPoseInit is None else initialPoseInit, dtype=float)
        KairosModel.__init__(self, base=RobotBase(HT=HTtranslate([initialPose[0], initialPose[1], 0.]) @ HTrotateZ(initialPose[2])),
                             q0=armInitialAnglesInit)
        self.initialPose = initialPose  #! Start pose [x, y, yaw] as given (yaw not wrapped).
        self.objectInitialPosition = np.asarray(par.objectPosition if objectPositionInit is None else objectPositionInit, dtype=float)
        #the lab scene (floor, Vicon origin, contact, table, cup) first, the Kairos built into it
        scene = CreateLabScene(mbsInit, self.objectInitialPosition)
        self.contact = scene['generalContact']  #! GeneralContact of floor, table, object and gripper.
        self.CreateMBS(mbsInit, name=vehicleName, markerGround=scene['markerGround'], generalContact=self.contact)
        self.bodies.update({'ground': scene['ground'], 'viconOrigin': scene['viconOrigin'], 'object': scene['cup'],
                            'table': scene['table']})
        #! World x, y in m of the table center (movable in the mirror mode).
        self.tablePosition = np.array(scene['tablePosition'], dtype=float)
        #! Height in m of the table top minus par.tableHeight.
        self.tableHeightOffset = scene['tableTopHeight'] - par.tableHeight
        self.grasp.update(self.CreateGrasp())  #! Assisted connector and marker indices.
        #live object (Vicon) coupling, None without tracking.
        self.tracking = self.CreateObjectTracking() if objectTrackingInit else None
        self.SetEnvironmentObstacles(scene['obstacles'])

    def CreateGrasp(self):
        """Add the inactive compliant grasp between TCP and object (the object of the lab scene)."""
        objectMarker = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=self.bodies['object']))
        connector = self.mbs.AddObject(RigidBodySpringDamper(
            #distinct translational and rotational units avoid artificial rotational stiffness.
            markerNumbers=[self.toolMarker, objectMarker],
            stiffness=np.diag([par.graspStiffness]*par.dimension3D + [par.graspRotationalStiffness]*par.dimension3D),
            damping=np.diag([par.graspDamping]*par.dimension3D + [par.graspRotationalDamping]*par.dimension3D), activeConnector=False,
            visualization=VRigidBodySpringDamper(show=False)))
        return {'connector': connector, 'objectMarker': objectMarker}

    def CreateObjectTracking(self):
        """Add the spring-damper from a movable ground anchor (tracking target) to the object center."""
        anchor = self.mbs.AddObject(ObjectGround(referencePosition=self.objectInitialPosition.tolist()))
        anchorMarker = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=anchor))
        objectCenterMarker = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=self.bodies['object']))
        noStiffness = 0.  #the yaw of the cylinder is not measured
        connector = self.mbs.AddObject(RigidBodySpringDamper(
            markerNumbers=[anchorMarker, objectCenterMarker],
            stiffness=np.diag([par.objectTrackingStiffness]*par.dimension3D +
                              [par.objectTrackingRotationalStiffness]*2 + [noStiffness]),
            damping=np.diag([par.objectTrackingDamping]*par.dimension3D +
                            [par.objectTrackingRotationalDamping]*2 + [noStiffness]),
            visualization=VRigidBodySpringDamper(show=False)))
        return {'anchor': anchor, 'connector': connector, 'position': self.objectInitialPosition.copy(),
                'target': self.objectInitialPosition.copy(), 'lastTime': 0.}

    def UpdateObjectTracking(self, targetPosition, time):
        """Move the tracking anchor towards targetPosition with at most par.objectTrackingMaximumSpeed.

        @param targetPosition Object center target in world coordinates in m; None keeps the last target.
        @param time Simulation time in s; steps back in time (rejected solver steps) do not move the anchor.
        """
        if targetPosition is not None:
            self.tracking['target'] = np.asarray(targetPosition, dtype=float).copy()
        duration = time - self.tracking['lastTime']
        self.tracking['lastTime'] = time
        if duration <= 0.:
            return
        step = self.tracking['target'] - self.tracking['position']
        distance = np.linalg.norm(step)
        maximumStep = par.objectTrackingMaximumSpeed*duration
        if distance > maximumStep:
            step *= maximumStep/distance
        self.tracking['position'] += step
        self.mbs.SetObjectParameter(self.tracking['anchor'], 'referencePosition', self.tracking['position'].tolist())

    def SetObjectTracking(self, enabled):
        """Release (grasp) or resume the live object coupling; it resumes from the actual object position."""
        if enabled:
            self.tracking['position'] = self.BodyPose(self.bodies['object'])[:3, 3].copy()
            self.mbs.SetObjectParameter(self.tracking['anchor'], 'referencePosition', self.tracking['position'].tolist())
        self.mbs.SetObjectParameter(self.tracking['connector'], 'activeConnector', bool(enabled))

    def SetPlatformPose(self, pose):
        """Move the standing vehicle to a world pose [x in m, y in m, yaw in rad] (KairosModel.SetPlatformPose); a
        grasped object moves along."""
        pose = np.asarray(pose, dtype=float)
        oldPose = self.GetState()['pose']
        #planar rigid motion of the vehicle in world coordinates (rotation about z through the origin, then shift)
        motion = HTtranslate([pose[0], pose[1], 0.]) @ HTrotateZ(pose[2] - oldPose[2]) @ HTtranslate([-oldPose[0], -oldPose[1], 0.])
        objectPose = motion @ self.BodyPose(self.bodies['object']) if self.grasp['active'] else None
        KairosModel.SetPlatformPose(self, pose)
        if objectPose is not None:
            self.SetObjectPose(objectPose)

    def SetObjectPose(self, objectPose):
        """Set the object (cup) to a world pose (4x4 homogeneous transformation) at rest, in the present and the start
        of step configuration (from SetPlatformPose, also used by SetObjectPosition)."""
        objectNode = self.mbs.GetObject(self.bodies['object'])['nodeNumber']
        objectIndex = self.mbs.GetNodeODE2Index(objectNode)
        objectReference = np.asarray(self.mbs.GetNodeParameter(objectNode, 'referenceCoordinates'), dtype=float)
        total = np.r_[objectPose[:3, 3], RotationMatrix2RotationVector(objectPose[:3, :3])]
        for configuration in (exu.ConfigurationType.Current, exu.ConfigurationType.StartOfStep):
            coordinates = np.array(self.mbs.systemData.GetODE2Coordinates(configuration=configuration))
            velocities = np.array(self.mbs.systemData.GetODE2Coordinates_t(configuration=configuration))
            coordinates[objectIndex:objectIndex + len(total)] = total - objectReference
            velocities[objectIndex:objectIndex + len(total)] = 0.
            self.mbs.systemData.SetODE2Coordinates(coordinates.tolist(), configuration=configuration)
            self.mbs.systemData.SetODE2Coordinates_t(velocities.tolist(), configuration=configuration)

    def SetObjectPosition(self, position):
        """Move the free object (not grasped) to the world x, y position in m; height and orientation stay (it stands
        on the table). Takes over the Vicon object marker once instead of the live coupling.

        @param position World position [x, y] or [x, y, z] in m; z is ignored.
        @return Horizontal shift in m of the object.
        """
        if self.grasp['active']:
            raise ValueError('A grasped object cannot be moved to a Vicon position')
        objectPose = self.BodyPose(self.bodies['object'])
        shift = np.asarray(position[:2], dtype=float) - objectPose[:2, 3]
        objectPose[:2, 3] += shift
        self.SetObjectPose(objectPose)
        return float(np.linalg.norm(shift))

    def GetState(self):
        """Return actual platform pose, velocity, object position and arm/grasp readiness (KairosModel.GetState)."""
        state = KairosModel.GetState(self)
        state['objectPosition'] = np.asarray(GetBodyOutput(self.mbs, self.bodies['object'], exu.OutputVariableType.Position))
        return state

    def TableOffset(self):
        """Return the shift [x, y, z] in m of the table from its graphics (built under the initial object at
        par.tableHeight)."""
        return np.r_[self.tablePosition - self.objectInitialPosition[:2], self.tableHeightOffset]

    def TableTopHeight(self):
        """Return the height in m of the table top."""
        return par.tableHeight + self.tableHeightOffset

    def SetTablePosition(self, position):
        """Move the table center to the world x, y position in m (graphics and contact; mirror mode)."""
        self.tablePosition = np.array(position[:2], dtype=float)
        self.mbs.SetObjectParameter(self.bodies['table'], 'referencePosition', self.TableOffset().tolist())
        self.SetEnvironmentObstacles([TableObstacle(self.tablePosition, self.TableTopHeight())])

    def SetGrasp(self, enabled, requireObject=True):
        """Attach only a nearby object, preserving its pose, or release the assisted connector.

        @param requireObject If False, closing without an object in reach only closes the fingers (as the real
               gripper does) instead of raising ValueError.
        """
        if enabled and not self.grasp['active']:
            toolPose = self.ToolPose()
            objectPose = self.BodyPose(self.bodies['object'])
            if np.linalg.norm(toolPose[:3, 3] - objectPose[:3, 3]) > par.graspDistance:
                if requireObject:
                    raise ValueError('Object is too far from the TCP to grasp')
                #fingers close on air
                self.SetGripperWidth(par.gripperClosedWidth)
                return
            localPoint = objectPose[:3, :3].T @ (toolPose[:3, 3] - objectPose[:3, 3])
            #grasp frame (tool orientation at the grasped point) as localHT of the object
            #marker; rotationMarker1 of the connector is deprecated
            self.mbs.SetMarkerParameter(self.grasp['objectMarker'], 'localHT',
                                        exu.HT(rotation=objectPose[:3, :3].T @ toolPose[:3, :3], translation=localPoint))
        self.mbs.SetObjectParameter(self.grasp['connector'], 'activeConnector', bool(enabled))
        #a grasped object follows the gripper, not the live Vicon target
        if self.tracking is not None and bool(enabled) != self.grasp['active']:
            self.SetObjectTracking(not enabled)
        self.grasp['active'] = bool(enabled)
        self.SetGripperWidth(par.gripperGraspWidth if enabled else par.gripperOpenWidth)
