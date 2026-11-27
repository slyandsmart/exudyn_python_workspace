# -*- coding: utf-8 -*-
"""Contact geometry for the 6R simulation, in SI units."""

import numpy as np

import exudyn as exu
import exudyn.graphics as graphics
from exudyn.itemInterface import MarkerBodyRigid, ObjectGround, VObjectGround

from . import parameters as par
from . import visualization
from labRobots.robots.robot6R import visualization as robotVisualization
from .labObjects import GetCupInertia


def CupContactPoints():
    """Return sphere centers approximating the cup bottom and tapered outside."""
    radius = par.contactSphereRadius
    points = [[0., 0., radius]]
    for height in par.cupSphereRingHeights:
        cupRadius = par.cupBottomRadius + (par.cupTopRadius - par.cupBottomRadius)*height/par.cupHeight
        centerRadius = cupRadius - radius
        for angle in np.linspace(0., 2.*np.pi, par.cupSpheresPerRing, endpoint=False):
            points.append([centerRadius*np.cos(angle), centerRadius*np.sin(angle), height])
    return points


def JawContactPoints(modelData, jawName):
    """Return the sphere centers (jaw link frame, m) of the contact of a jaw: per layer of jawSphereLayers (x) and angle
    of jawSphereAnglesDeg (about x from the closing axis y), the sphere touches the tooth tips of the jaw mesh from the
    cup side (center at the smallest distance of the mesh in this sector from the jaw arc center at the link origin
    plus the sphere radius).

    the spheres were placed on the fitted arc (25.6 mm, the back of the jaw ring); the tooth
    tips are at 21.0-23.2 mm, so the teeth visibly penetrated the cup before the spheres touched it.
    """
    points, triangles, normals = visualization.LoadMesh(modelData['labObjects'][jawName]['file'])
    points = points @ robotVisualization.GetJawLinkRotation(jawName)     #TCP frame -> jaw link frame (rows: R^T p)
    distances = np.linalg.norm(points[:, 1:3], axis=1)
    pointAngles = np.arctan2(points[:, 2], points[:, 1])
    angles = np.radians(par.jawSphereAnglesDeg)
    halfAngle = 0.5*np.min(np.diff(angles))
    halfLayer = 0.5*np.min(np.diff(par.jawSphereLayers))
    centers = []
    for layer in par.jawSphereLayers:
        for angle in angles:
            sector = (np.abs(pointAngles - angle) <= halfAngle) & (np.abs(points[:, 0] - layer) <= halfLayer)
            if not sector.any():
                raise ValueError('JawContactPoints: no mesh points of %s at %.1f deg, layer %.3f m'
                                 % (jawName, np.degrees(angle), layer))
            radius = np.min(distances[sector]) + par.contactSphereRadius
            centers.append([layer, radius*np.cos(angle), radius*np.sin(angle)])
    return centers


def CreateTableContactGeometry(tableName):
    """Approximate a small table by its top plate and four rectangular legs, in table coordinates."""
    size = np.array(par.tableContactSizes[tableName], dtype=float)
    thickness = par.tableContactPlateThickness[tableName]
    legWidth = par.tableContactLegWidth[tableName]
    legHeight = size[2] - thickness
    table = graphics.Brick([0., 0., size[2] - 0.5*thickness], [size[0], size[1], thickness])
    for sideX in [-1., 1.]:
        for sideY in [-1., 1.]:
            center = [sideX*0.5*(size[0] - legWidth), sideY*0.5*(size[1] - legWidth), 0.5*legHeight]
            table = graphics.MergeTriangleLists(table, graphics.Brick(center, [legWidth, legWidth, legHeight]))
    points, triangles = graphics.ToPointsAndTrigs(table)
    return np.asarray(points, dtype=float), np.asarray(triangles, dtype=int)


def CreateCupContact(mbs, modelData, initialPosition, mass, includeEnvironment=True, tableContactGeometry='simplified'):
    """Create a free cup and contact with the lab table, small tables and cup holders.

    The environment flag controls visualization only; contact supports remain active.
    tableContactGeometry selects simplified table bodies or the original table meshes.
    """
    if not np.isfinite(mass) or mass <= 0.:
        raise ValueError('cup mass including liquid must be positive and finite')
    if tableContactGeometry not in ('simplified', 'mesh'):
        raise ValueError("tableContactGeometry must be 'simplified' or 'mesh'")
    contact = mbs.AddGeneralContact()
    contact.SetFrictionPairings(np.array([[par.contactFrictionCoefficient]]))
    contact.SetFrictionProportionalZone(par.contactFrictionProportionalZone)
    contact.SetSearchTreeCellSize(numberOfCells=[24, 24, 32])
    contact.SetSearchTreeBox(pMin=[-1., -1., -1.], pMax=[1.5, 1.5, 1.5])
    groundGraphics = visualization.CreateEnvironmentGraphics(modelData, includeLabTables=True) if includeEnvironment else []
    ground = mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsData=groundGraphics)))
    groundMarker = mbs.AddMarker(MarkerBodyRigid(bodyNumber=ground))
    stiffness = par.contactStiffness
    damping = stiffness*par.contactDampingFactor

    def AddTriangles(points, triangles, offset):
        contact.AddTrianglesRigidBodyBased(rigidBodyMarkerIndex=groundMarker, contactStiffness=stiffness,
                                           contactDamping=damping, frictionMaterialIndex=0,
                                           pointList=points + offset, triangleList=triangles, staticTriangles=True)

    for entry in modelData['meshFiles']:
        if entry['link'] == -2:
            points, triangles, _ = visualization.LoadMesh(entry['file'])
            AddTriangles(points, triangles, np.zeros(3))
    for tableName, (position, height) in zip(['table1', 'table2'], visualization.GetTablePoses()):
        if tableContactGeometry == 'mesh':
            points, triangles, _ = visualization.LoadMesh(modelData['labObjects'][tableName]['file'])
        else:
            points, triangles = CreateTableContactGeometry(tableName)
        AddTriangles(points, triangles, position)
        points, triangles, _ = visualization.LoadMesh(modelData['labObjects']['cupHolder']['file'])
        AddTriangles(points, triangles, position + [0., 0., height])

    cup = mbs.CreateRigidBody(referencePosition=initialPosition, inertia=GetCupInertia(mass), gravity=par.gravity,
                              nodeType=exu.NodeType.RotationRotationVector,
                              graphicsDataList=[visualization.CreateCupGraphics()])
    for point in CupContactPoints():
        marker = mbs.AddMarker(MarkerBodyRigid(bodyNumber=cup, localPosition=point))
        contact.AddSphereWithMarker(markerIndex=marker, radius=par.contactSphereRadius,
                                    contactStiffness=stiffness, contactDamping=damping, frictionMaterialIndex=0)
    return contact, cup
