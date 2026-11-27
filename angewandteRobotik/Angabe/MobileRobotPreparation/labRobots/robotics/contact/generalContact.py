# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: contact.py
#
# Details:  Contact geometry of the grippers of the lab robots in the GeneralContact of a scene: an outward oriented
#           triangle mesh on a rigid body marker (AddContactMesh) and one box per visual component fitted to its points
#           (AddContactBoxes), used by the gripper contact of the Kairos and of the Leobot; the penalty values of contact
#           spheres from the supported mass, the step size and the allowed penetration (ContactSphereParameters; 
#           ) instead of values tried by hand.
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-12 (from labRobots/robots/kairos/contact.py, with the contact values as arguments)
# Notes:    GeneralContact of Exudyn (theoryContact.md): the stiffnesses of the two partners act in series, the dampings
#           in parallel. A sphere of mass m on stiffness k vibrates with omega = sqrt(k/m); with omega*h of about 0.5
#           (step size h) the implicit solver follows the contact without the bouncing that switches it on and off each
#           step (Newton fails, the step is cut: the chess pieces with 2e4 N/m at 2 ms were 5 times slower than real
#           time); critical damping ends the touch-down without rebound.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Contact meshes and boxes of the grippers, penalty values of contact spheres."""
import numpy as np
import exudyn.graphics as graphics

#! Product of the contact frequency and the step size for the implicit solver (omega*h); damping ratio of the contact.
#! 0.5, not 1: at omega*h = 1 the resting chess pieces sank only 0.04 mm, the contact of their
#! spheres switched on and off within the Newton iterations, Newton diverged every 40 ms and the step was cut (a new
#! Jacobian nearly every step: board with Leobot 1.13 times slower than real time); at 0.5 (0.16 mm) 0.63 times.
implicitFrequencyStep, contactDampingRatio = 0.5, 1.
#! Stiffness of static triangles (tables, floors) relative to the stiffest sphere on them: the sphere sets the series
#! stiffness (within 1 %).
staticStiffnessFactor = 100.
#! Gravity in m/s^2 for the static penetration.
standardGravity = 9.81


#! Width of the regularized friction zone in multiples of the change of the sliding speed by friction within one step.
frictionZoneFactor = 5.


def FrictionZone(frictionCoefficient, stepSize):
    """Return the velocity in m/s below which GeneralContact regularizes the friction (SetFrictionProportionalZone):
    frictionZoneFactor times mu*g*h, the speed friction removes in one step. A narrower zone
    makes friction reverse its direction from step to step at rest (the chess pieces with 0.01 m/s at 2 ms: 1.6 times
    slower; RobotAI_LNF_NEW uses 0.01 m/s at 0.4 ms, also 5 mu g h); a wider one lets a held object creep slowly."""
    return frictionZoneFactor*frictionCoefficient*standardGravity*stepSize


def ContactSphereParameters(supportedMass, stepSize, maximumPenetration, frequencyStep=implicitFrequencyStep,
                            dampingRatio=contactDampingRatio, partnerStiffness=None, partnerDamping=0.):
    """Return (stiffness in N/m, damping in N s/m) of a contact sphere of GeneralContact.

    The effective contact (in series with the partner) is the stiffest the step size allows, omega*h = frequencyStep
    with omega = sqrt(k/m) of the mass the sphere carries, damped with dampingRatio; ValueError if it would sink deeper
    than maximumPenetration under gravity (a smaller step is needed).

    Args:
        supportedMass: mass in kg the sphere carries at rest (e.g. body mass / number of spheres it stands on)
        stepSize: time step in s of the solver
        maximumPenetration: largest static penetration in m allowed
        frequencyStep: omega*h of the contact (implicitFrequencyStep for the implicit solver)
        dampingRatio: damping ratio of the contact (1: critical)
        partnerStiffness: stiffness in N/m of the contact partner (triangles, spheres), None: rigid
        partnerDamping: damping in N s/m of the partner (added to that of the sphere)
    """
    stiffness = supportedMass*(frequencyStep/stepSize)**2              #effective
    penetration = supportedMass*standardGravity/stiffness
    if penetration > maximumPenetration:
        raise ValueError('ContactSphereParameters: %.3g kg would sink %.3g m (more than %.3g m) at the step %.3g s; '
                         'use a smaller step' % (supportedMass, penetration, maximumPenetration, stepSize))
    damping = 2.*dampingRatio*np.sqrt(stiffness*supportedMass)
    sphereStiffness = stiffness
    if partnerStiffness is not None:
        if partnerStiffness <= stiffness:
            raise ValueError('ContactSphereParameters: the partner (%.3g N/m) is softer than the contact needs (%.3g N/m)'
                             % (partnerStiffness, stiffness))
        sphereStiffness = stiffness*partnerStiffness/(partnerStiffness - stiffness)     #series: 1/k = 1/ks + 1/kp
    return sphereStiffness, max(damping - partnerDamping, 0.)


def AddContactMesh(contact, marker, mesh, contactStiffness, contactDamping, static=False, rotation=None):
    """Register an outward oriented triangle mesh on a rigid body marker in the GeneralContact (friction material 0).

    Args:
        contact: GeneralContact of the scene.
        marker: Rigid body marker of the body carrying the mesh.
        mesh: Graphics data (triangle list) in the marker frame.
        contactStiffness, contactDamping: Contact values in N/m and N s/m.
        static: True: the triangles do not move.
        rotation: 3x3 rotation of the mesh relative to the marker, or None.

    Returns:
        Index of the triangles in the GeneralContact.
    """
    points, triangles = graphics.ToPointsAndTrigs(mesh)
    if rotation is not None:
        points = np.asarray(points) @ rotation.T
    return contact.AddTrianglesRigidBodyBased(
        rigidBodyMarkerIndex=marker, contactStiffness=contactStiffness, contactDamping=contactDamping,
        frictionMaterialIndex=0, pointList=points, triangleList=triangles, staticTriangles=static)


def AddContactBoxes(contact, marker, graphicsList, contactStiffness, contactDamping, rotation=None):
    """Fit one inexpensive contact box to each visual component of a graphics list (in the marker frame) and register it.

    Args:
        rotation: 3x3 rotation of the boxes relative to the marker, or None.
    """
    for mesh in graphicsList:
        points = np.asarray(mesh['points']).reshape(-1, 3)
        lower, upper = points.min(axis=0), points.max(axis=0)
        AddContactMesh(contact, marker, graphics.Brick((lower + upper)/2., upper - lower), contactStiffness, contactDamping,
                       rotation=rotation)
