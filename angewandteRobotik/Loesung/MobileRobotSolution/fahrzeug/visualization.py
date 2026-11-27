# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
# Filename: visualization.py
# Details:  UR5 CAD meshes, detailed mecanum rollers and coordinate arrows.
# Authors:  Peter Manzl; Martin Sereinig; Michael Pieber (created with Claude Code)
# Date:     2026-11-04 (MoveIt geometry: 2026-11-05): the parts of the robot are in labRobots/robots/kairos/visualization.py
# Notes:    Mesh attribution is retained in mektcommon2/data/Exudyn/robotics/URDF/kairos_description/ur5_visual_assets/LICENSE.
# License:  BSD-3 license
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Graphics derived from the preparation model."""
import numpy as np
import exudyn.graphics as graphics
from . import parameters as par
from labRobots.robots.kairos.visualization import CoordinateGraphics


#! Length in m of the axes of the Vicon origin drawn on the floor.
viconOriginFrameLength = 0.5


def FloorGraphics():
    """Return a matte slate floor with a subtle 0.5 m grid, exported as actual triangle geometry."""
    #texture-based checkerboards lose their texture in GLB export.
    halfLength = par.floorDrawHalfLength
    lineWidth = 0.004
    items = [graphics.Brick([0., 0., -0.012], [2.*halfLength, 2.*halfLength, 0.02], color=par.floorColor)]
    for coordinate in np.arange(-halfLength, halfLength + 0.25, 0.5):
        items.append(graphics.Brick([coordinate, 0., -0.001], [lineWidth, 2.*halfLength, 0.001], color=par.floorGridColor))
        items.append(graphics.Brick([0., coordinate, -0.001], [2.*halfLength, lineWidth, 0.001], color=par.floorGridColor))
    return items


def ViconOriginGraphics():
    """Return the Vicon origin (world frame) fixed on the floor, shown with the coordinate frames; the triad in the
    lower left corner of the renderer only shows the orientation of the world axes."""
    return CoordinateGraphics(viconOriginFrameLength)


#! Contour points of the rounded widening and largest contour angle in rad smoothed in the object graphics
#! (the bottom and rim edges stay sharp).
objectWideningPoints, objectSmoothingAngle = 9, np.radians(30.)


def ObjectGraphics(color=par.objectColor):
    """Return the graphics of the grasp object in its body frame (origin at the center): cup with a rounded widening
    to the rim and opening, and the Vicon sphere in the opening."""
    bottom, top = -par.objectHeight/2., par.objectHeight/2.
    innerRadius = par.objectRimRadius - par.objectWallThickness
    #rounded widening: radius = body + (rim - body)*(1 - cos(pi/2 s)), tangent to the body at its start (s = 0)
    widening = np.linspace(0., 1., objectWideningPoints)
    flare = [[par.objectHeight - par.objectRimHeight*(1. - s),
              par.objectRadius + (par.objectRimRadius - par.objectRadius)*(1. - np.cos(0.5*np.pi*s))] for s in widening]
    #contour (height from the bottom, radius): bottom, body, widening, rim, opening
    contour = ([[0., 0.], [0., par.objectRadius]] + flare +
               [[par.objectHeight, innerRadius], [par.objectHeight - par.objectOpeningDepth, innerRadius],
                [par.objectHeight - par.objectOpeningDepth, 0.]])
    return [graphics.SolidOfRevolution([0., 0., bottom], [0., 0., 1.], contour, color=color, nTiles=32,
                                       smoothContour=True, smoothingAngle=objectSmoothingAngle),
            graphics.Sphere([0., 0., top], par.objectMarkerSphereRadius, color=graphics.color.lightgrey, nTiles=16)]
