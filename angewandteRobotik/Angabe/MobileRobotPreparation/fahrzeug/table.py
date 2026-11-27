# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: table.py
#
# Details:  Geometry of the lab table as boxes: two side walls, a middle shelf between them and the top plate; the
#           walls are perpendicular to the world x axis (the vehicle faces the long closed wall along +x), the table is
#           open in y. Used by the graphics (model.py), the object contact (contact.py), the simulated distance
#           sensors (sensorSimulation.py) and the arm collision checker (collisionModel.py).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-07
# Notes:    Lab table (photo of 2026-11-07): outer footprint par.tableSize, walls par.tableWallThickness, shelf between
#           par.tableShelfHeight - par.tablePlateThickness and par.tableShelfHeight above the floor, top plate below
#           par.tableHeight. The small corner pieces under the top plate are left out.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Lab table as boxes."""
import numpy as np
import exudyn.graphics as graphics
from . import parameters as par


def TableBoxes(tablePosition):
    """Return the table parts as list of (name, center, size) in world coordinates (m).

    @param tablePosition World x, y of the table center in m.
    """
    x, y = float(tablePosition[0]), float(tablePosition[1])
    sizeX, sizeY = par.tableSize
    wallHeight = par.tableHeight - par.tablePlateThickness
    innerWidth = sizeX - 2.*par.tableWallThickness
    boxes = []
    for side, name in ((-1., 'tableWallFront'), (1., 'tableWallBack')):
        boxes.append((name, np.array([x + side*(sizeX - par.tableWallThickness)/2., y, wallHeight/2.]),
                      np.array([par.tableWallThickness, sizeY, wallHeight])))
    boxes.append(('tableShelf', np.array([x, y, par.tableShelfHeight - par.tablePlateThickness/2.]),
                  np.array([innerWidth, sizeY, par.tablePlateThickness])))
    boxes.append(('tableTop', np.array([x, y, par.tableHeight - par.tablePlateThickness/2.]),
                  np.array([sizeX, sizeY, par.tablePlateThickness])))
    return boxes


def TableGraphics(tablePosition, color=graphics.color.darkgrey):
    """Return one graphics brick per table part (world coordinates).

    @param tablePosition World x, y of the table center in m. @param color RGBA color.
    """
    return [graphics.Brick(center.tolist(), size.tolist(), color=color) for unusedName, center, size in TableBoxes(tablePosition)]
