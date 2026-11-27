# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: collision.py
#
# Details:  Collision shapes of the lab robots as data: capsules and oriented boxes (dicts) created, moved and fitted
#           to mesh points (the link shapes of the Kairos), and point/segment distances to boxes and segments (task
#           checks of the Kairos lab task), ray distances to boxes and capsules (sensor rays hitting
#           moving shapes). The collision check itself is Exudyn's (exudyn.cUtilities.CollisionModel;
#           the shapes become Exudyn shapes by nativeCollision.ExudynShape).
#
# Authors:  Michael Pieber (created with Claude Code)
# Date:     2026-11-05; 2026-11-12 moved from LabCourse/MobileRobot/tools/robotics; 2026-11-12 RobotCollisionChecker
#           removed (the robots check with Exudyn only)
# Notes:    Pure numpy.
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Capsule and box shapes of the lab robots."""
import numpy as np

#! Shape type identifiers.
shapeCapsule, shapeBox = 'capsule', 'box'
#! Number of spatial dimensions.
dimension3D = 3
#! Golden-section line search on segments: ratio and iterations (0.618^24 = 1e-5 of the segment length).
goldenRatio = (np.sqrt(5.) - 1.)/2.
numberOfLineSearchIterations = 24
#! Tolerance for degenerate segments and parallel separating axes.
geometricTolerance = 1e-12


def CreateCapsule(point0, point1, radius, name=''):
    """Create a capsule (segment swept by a sphere).

    Args:
        point0: First segment end point (3D) in the shape's reference frame.
        point1: Second segment end point; equal to point0 for a sphere.
        radius: Capsule radius, >= 0.
        name: Name used in collision reports.

    Returns:
        Shape dictionary.
    """
    if radius < 0.:
        raise ValueError('CreateCapsule: radius must be nonnegative')
    return {'type': shapeCapsule, 'point0': np.asarray(point0, dtype=float), 'point1': np.asarray(point1, dtype=float),
            'radius': float(radius), 'name': name}


def CreateBox(center, halfSize, rotation=None, name=''):
    """Create an oriented box.

    Args:
        center: Box center (3D) in the shape's reference frame.
        halfSize: Half side lengths along the box axes, all >= 0.
        rotation: 3x3 rotation matrix whose columns are the box axes; identity if None.
        name: Name used in collision reports.

    Returns:
        Shape dictionary.
    """
    halfSize = np.asarray(halfSize, dtype=float)
    if np.any(halfSize < 0.):
        raise ValueError('CreateBox: halfSize must be nonnegative')
    return {'type': shapeBox, 'center': np.asarray(center, dtype=float), 'halfSize': halfSize,
            'rotation': np.eye(dimension3D) if rotation is None else np.asarray(rotation, dtype=float), 'name': name}


def TransformShape(shape, homogeneousTransformation, name=None):
    """Return the shape expressed in the parent frame of a 4x4 homogeneous transformation."""
    rotation = np.asarray(homogeneousTransformation)[:dimension3D, :dimension3D]
    translation = np.asarray(homogeneousTransformation)[:dimension3D, dimension3D]
    newName = shape['name'] if name is None else name
    if shape['type'] == shapeCapsule:
        return CreateCapsule(rotation @ shape['point0'] + translation, rotation @ shape['point1'] + translation,
                             shape['radius'], newName)
    return CreateBox(rotation @ shape['center'] + translation, shape['halfSize'], rotation @ shape['rotation'], newName)


def BoxFromPoints(points, rotation=None, margin=0., name=''):
    """Return the smallest box with the given axes containing all points.

    Args:
        points: Array (n,3) in the reference frame.
        rotation: Box axes as columns (3x3); identity (axis aligned) if None.
        margin: Added to every half size.
        name: Name of the shape.
    """
    rotation = np.eye(dimension3D) if rotation is None else np.asarray(rotation, dtype=float)
    localPoints = np.asarray(points, dtype=float).reshape(-1, dimension3D) @ rotation
    lower, upper = localPoints.min(axis=0), localPoints.max(axis=0)
    return CreateBox(rotation @ ((lower + upper)/2.), (upper - lower)/2. + margin, rotation, name)



def YawBoxFromPoints(points, margin=0., name=''):
    """Return the box around points turned about z along the principal axis of the points in the xy plane (e.g. a wall
    panel standing at an angle), for the cell parts of CAD assemblies.

    Args:
        points: Array (n,3) in the reference frame (z up).
        margin: Added to every half size.
        name: Name of the shape.
    """
    points = np.asarray(points, dtype=float).reshape(-1, dimension3D)
    planar = points[:, 0:2] - points[:, 0:2].mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(planar.T @ planar)
    axis = eigenvectors[:, np.argmax(eigenvalues)]
    yaw = np.arctan2(axis[1], axis[0])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.], [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
    return BoxFromPoints(points, rotation, margin, name)

def CapsuleFromPoints(points, margin=0., name=''):
    """Return a capsule containing all points with the smallest volume among the candidate axes:
    the principal axes and the coordinate axes (link frames are usually aligned with joint axes).

    The axis passes through the center of the cross-section's bounding rectangle (exact for cylinders,
    unlike the mean); the radius is the largest distance from the axis, the segment the shortest one.

    Args:
        points: Array (n,3) in the reference frame.
        margin: Added to the radius.
        name: Name of the shape.
    """
    points = np.asarray(points, dtype=float).reshape(-1, dimension3D)
    mean = points.mean(axis=0)
    unusedU, unusedS, principalAxes = np.linalg.svd(points - mean, full_matrices=False)
    bestCapsule, bestVolume = None, np.inf
    for axis in np.vstack([principalAxes, np.eye(dimension3D)]):
        #orthonormal basis of the cross-section plane
        helper = np.eye(dimension3D)[np.argmin(np.abs(axis))]
        first = np.cross(axis, helper)
        first /= np.linalg.norm(first)
        second = np.cross(axis, first)
        crossSection = (points - mean) @ np.column_stack([first, second])
        center = mean + ((crossSection.min(axis=0) + crossSection.max(axis=0))/2.) @ np.vstack([first, second])
        axial = (points - center) @ axis
        radial = np.linalg.norm((points - center) - np.outer(axial, axis), axis=1)
        radius = radial.max() + margin
        #a point (axial t, radial r) is inside if start <= t + reach and end >= t - reach,
        #reach = sqrt(radius^2 - r^2): shortest segment = [min(t + reach), max(t - reach)]
        reach = np.sqrt(np.maximum(radius**2 - radial**2, 0.))
        start, end = np.min(axial + reach), np.max(axial - reach)
        if start > end:
            start = end = (start + end)/2.
        volume = np.pi*radius**2*(end - start) + 4./3.*np.pi*radius**3
        if volume < bestVolume:
            bestCapsule, bestVolume = CreateCapsule(center + start*axis, center + end*axis, radius, name), volume
    return bestCapsule


def SampleTriangleSurface(points, triangles, spacing):
    """Return points on a triangle mesh with at most the given spacing (vertices of long triangles alone
    do not represent e.g. a tube, whose vertices are only at its ends).

    Args:
        points: Mesh points, flat list or array (n,3).
        triangles: Point indices, flat list or array (m,3).
        spacing: Maximum distance of samples along the triangle edges.
    """
    points = np.asarray(points, dtype=float).reshape(-1, dimension3D)
    triangles = np.asarray(triangles, dtype=int).reshape(-1, dimension3D)
    corner0, corner1, corner2 = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    longestEdge = np.max([np.linalg.norm(corner1 - corner0, axis=1), np.linalg.norm(corner2 - corner0, axis=1),
                          np.linalg.norm(corner2 - corner1, axis=1)], axis=0)
    divisions = np.maximum(1, np.ceil(longestEdge/spacing)).astype(int)
    samples = [points]
    for division in np.unique(divisions):
        selected = divisions == division
        first, second = np.meshgrid(np.arange(division + 1), np.arange(division + 1), indexing='ij')
        inside = first + second <= division
        weight1, weight2 = first[inside]/division, second[inside]/division
        edge1, edge2 = corner1[selected] - corner0[selected], corner2[selected] - corner0[selected]
        samples.append((corner0[selected][:, None, :] + weight1[None, :, None]*edge1[:, None, :] +
                        weight2[None, :, None]*edge2[:, None, :]).reshape(-1, dimension3D))
    return np.vstack(samples)


def ShapeSelectionMeasure(shape):
    """Return the size measure used to choose between capsule and box for a slice: box volume, and for a
    capsule the volume of its cylinder part, at least of a cylinder of length 2 radius (the spherical caps
    of neighboring slices overlap; counting them would turn round tubes into boxes with protruding corners)."""
    if shape['type'] == shapeCapsule:
        length = np.linalg.norm(shape['point1'] - shape['point0'])
        return np.pi*shape['radius']**2*max(length, 2.*shape['radius'])
    return 8.*np.prod(shape['halfSize'])


def AxisCapsuleFromPoints(points, axisPoint, axisDirection, housingRadius, margin=0., name=''):
    """Capsule exactly along a given axis line (e.g. a joint axis) around all points within housingRadius
    of the line; a joint housing is a cylinder around its joint axis, so this fit is tight in every joint
    position, unlike a fitted capsule or box.

    Args:
        points: Array (n,3) in the reference frame.
        axisPoint: Point on the axis line (3D).
        axisDirection: Direction of the axis line (3D, any length).
        housingRadius: Points closer than this to the line belong to the housing.
        margin: Added to the radius.
        name: Name of the shape.

    Returns:
        (capsule or None if no point is close, boolean mask of the covered points).
    """
    points = np.asarray(points, dtype=float).reshape(-1, dimension3D)
    direction = np.asarray(axisDirection, dtype=float)/np.linalg.norm(axisDirection)
    axial = (points - axisPoint) @ direction
    radial = np.linalg.norm((points - axisPoint) - np.outer(axial, direction), axis=1)
    covered = radial <= housingRadius
    if not np.any(covered):
        return None, covered
    radius = radial[covered].max() + margin
    #shortest segment whose spherical caps still cover the end faces (see CapsuleFromPoints)
    reach = np.sqrt(np.maximum(radius**2 - radial[covered]**2, 0.))
    start, end = np.min(axial[covered] + reach), np.max(axial[covered] - reach)
    if start > end:
        start = end = (start + end)/2.
    return CreateCapsule(axisPoint + start*direction, axisPoint + end*direction, radius, name), covered


def ShapesFromPoints(points, numberOfShapes, margin=0., name='', axisHousings=()):
    """Approximate a point cloud (e.g. SampleTriangleSurface of a link mesh) by convex shapes.

    Points around the given axes (joint housings) are covered by AxisCapsuleFromPoints. The remaining
    points are split into equally long slices along their first principal axis, and per slice the
    smallest shape (ShapeSelectionMeasure) is taken of: capsule (CapsuleFromPoints), box along the slice's
    principal axes, box along the coordinate axes; e.g. a robot link tube becomes a chain of capsules.

    Args:
        points: Array (n,3) in the reference frame (sample surfaces, vertices alone may leave gaps).
        numberOfShapes: Number of slices of the remaining points (>= 1).
        margin: Added to every radius / half size.
        name: Base name; housings are named nameJ0, nameJ1, ..., slices name0, name1, ...
        axisHousings: List of (axisPoint, axisDirection, housingRadius) in the reference frame.

    Returns:
        List of shapes (empty slices are skipped).
    """
    points = np.asarray(points, dtype=float).reshape(-1, dimension3D)
    if numberOfShapes < 1:
        raise ValueError('ShapesFromPoints: numberOfShapes must be >= 1')
    shapes = []
    remaining = np.ones(len(points), dtype=bool)
    for housingIndex, (axisPoint, axisDirection, housingRadius) in enumerate(axisHousings):
        capsule, covered = AxisCapsuleFromPoints(points, axisPoint, axisDirection, housingRadius, margin,
                                                 name + 'J' + str(housingIndex))
        if capsule is not None:
            shapes.append(capsule)
            remaining &= ~covered
    points = points[remaining]
    if len(points) == 0:
        return shapes
    mean = points.mean(axis=0)
    unusedU, unusedS, principalAxes = np.linalg.svd(points - mean, full_matrices=False)
    axial = (points - mean) @ principalAxes[0]
    boundaries = np.linspace(axial.min(), axial.max(), numberOfShapes + 1)
    sliceIndices = np.clip(np.searchsorted(boundaries, axial, side='right') - 1, 0, numberOfShapes - 1)
    for index in range(numberOfShapes):
        slicePoints = points[sliceIndices == index]
        if len(slicePoints) == 0:
            continue
        unusedU, unusedS, sliceAxes = np.linalg.svd(slicePoints - slicePoints.mean(axis=0), full_matrices=False)
        sliceRotation = sliceAxes.T if np.linalg.det(sliceAxes) > 0. else (sliceAxes*[[1.], [1.], [-1.]]).T
        candidates = [CapsuleFromPoints(slicePoints, margin, name + str(index)),
                      BoxFromPoints(slicePoints, sliceRotation, margin, name + str(index)),
                      BoxFromPoints(slicePoints, None, margin, name + str(index))]
        shapes.append(min(candidates, key=ShapeSelectionMeasure))
    return shapes


def PointSegmentDistance(points, segmentStarts, segmentEnds):
    """Vectorized distance of points (n,3) to segments (n,3)-(n,3)."""
    direction = segmentEnds - segmentStarts
    lengthSquared = np.maximum(np.einsum('ij,ij->i', direction, direction), geometricTolerance)
    parameter = np.clip(np.einsum('ij,ij->i', points - segmentStarts, direction)/lengthSquared, 0., 1.)
    return np.linalg.norm(points - segmentStarts - parameter[:, None]*direction, axis=1)


def PointBoxDistance(points, centers, halfSizes, rotations):
    """Vectorized distance of points (n,3) to oriented boxes; 0 inside."""
    localPoints = np.einsum('nji,nj->ni', rotations, points - centers)
    return np.linalg.norm(np.maximum(np.abs(localPoints) - halfSizes, 0.), axis=1)


def SegmentDistance(segmentStarts, segmentEnds, PointDistanceFunction):
    """Minimum over each segment of a convex point distance function (golden-section search).

    Args:
        segmentStarts: Array (n,3).
        segmentEnds: Array (n,3).
        PointDistanceFunction: Maps points (n,3) to distances (n); convex along each segment (true for the distance to a
            segment or a box).

    Returns:
        Array (n) of minimum distances.
    """
    direction = segmentEnds - segmentStarts

    def Evaluate(parameter):
        """Distance at segment parameters (n) in [0,1]."""
        return PointDistanceFunction(segmentStarts + parameter[:, None]*direction)

    lower = np.zeros(len(segmentStarts))
    upper = np.ones(len(segmentStarts))
    left, right = upper - goldenRatio, lower + goldenRatio
    leftValue, rightValue = Evaluate(left), Evaluate(right)
    for unusedIteration in range(numberOfLineSearchIterations):
        keepLeft = leftValue < rightValue   #minimum in [lower, right]
        newUpper = np.where(keepLeft, right, upper)
        newLower = np.where(keepLeft, lower, left)
        newLeft = np.where(keepLeft, newUpper - goldenRatio*(newUpper - newLower), right)
        newRight = np.where(keepLeft, left, newLower + goldenRatio*(newUpper - newLower))
        newValue = Evaluate(np.where(keepLeft, newLeft, newRight))   #one new evaluation per iteration
        leftValue, rightValue = np.where(keepLeft, newValue, rightValue), np.where(keepLeft, leftValue, newValue)
        lower, upper, left, right = newLower, newUpper, newLeft, newRight
    return np.minimum.reduce([leftValue, rightValue, Evaluate(np.zeros(len(lower))), Evaluate(np.ones(len(lower)))])


#simulated sensor rays hitting moving shapes (other robots; labRobots.robotics.sensors.movingShapes)
def RaySphereDistances(origin, directions, center, radius):
    """Return the distances [n] along rays from origin with unit directions [n x 3] to a sphere (inf: no hit; 0: origin
    inside); hits behind the origin do not count."""
    offset = np.asarray(origin, dtype=float) - np.asarray(center, dtype=float)
    halfLinear = directions @ offset
    constant = offset @ offset - radius**2
    discriminant = halfLinear**2 - constant
    with np.errstate(invalid='ignore'):
        entry = -halfLinear - np.sqrt(discriminant)
    distances = np.where((discriminant >= 0.) & (entry >= 0.), entry, np.inf)
    return np.zeros(len(directions)) if constant < 0. else distances


def RayCapsuleDistances(origin, directions, shape):
    """Return the distances [n] along rays from origin with unit directions [n x 3] to a capsule (CreateCapsule): the
    nearer of the cylinder between the end points and the spheres at them (inf: no hit; 0: origin inside)."""
    origin = np.asarray(origin, dtype=float)
    point0, point1, radius = shape['point0'], shape['point1'], shape['radius']
    distances = np.minimum(RaySphereDistances(origin, directions, point0, radius),
                           RaySphereDistances(origin, directions, point1, radius))
    length = np.linalg.norm(point1 - point0)
    if length < geometricTolerance:
        return distances
    axis = (point1 - point0)/length
    offset = origin - point0
    directionsAcross = directions - np.outer(directions @ axis, axis)
    offsetAcross = offset - (offset @ axis)*axis
    quadratic = np.einsum('ij,ij->i', directionsAcross, directionsAcross)
    halfLinear = directionsAcross @ offsetAcross
    constant = offsetAcross @ offsetAcross - radius**2
    if constant < 0. and 0. <= offset @ axis <= length:
        return np.zeros(len(directions))           #origin inside the cylinder
    discriminant = halfLinear**2 - quadratic*constant
    with np.errstate(invalid='ignore', divide='ignore'):
        entry = (-halfLinear - np.sqrt(discriminant))/quadratic
    along = offset @ axis + entry*(directions @ axis)
    cylinder = (quadratic > geometricTolerance) & (discriminant >= 0.) & (entry >= 0.) & (along >= 0.) & (along <= length)
    return np.minimum(distances, np.where(cylinder, entry, np.inf))


def RayBoxDistances(origin, directions, shape):
    """Return the distances [n] along rays from origin with unit directions [n x 3] to an oriented box (CreateBox; slab
    method in the box frame; inf: no hit; 0: origin inside)."""
    rotation, halfSize = shape['rotation'], shape['halfSize']
    localOrigin = (np.asarray(origin, dtype=float) - shape['center']) @ rotation
    localDirections = directions @ rotation
    with np.errstate(divide='ignore', invalid='ignore'):
        lower = (-halfSize - localOrigin)/localDirections
        upper = (halfSize - localOrigin)/localDirections
    parallel = np.abs(localDirections) < geometricTolerance
    outsideSlab = parallel & (np.abs(localOrigin) > halfSize)
    lower = np.where(parallel, -np.inf, lower)
    upper = np.where(parallel, np.inf, upper)
    entry = np.max(np.minimum(lower, upper), axis=1)
    leave = np.min(np.maximum(lower, upper), axis=1)
    hit = ~np.any(outsideSlab, axis=1) & (entry <= leave) & (leave >= 0.)
    return np.where(hit, np.maximum(entry, 0.), np.inf)


def RayShapesDistances(origin, directions, shapes):
    """Return the distances [n] along rays from origin with unit directions [n x 3] to the nearest of the shapes (boxes
    and capsules in the frame of origin and directions; inf: no hit, also without shapes).

    Args:
        origin: start point of the rays (3D)
        directions: unit directions [n x 3]
        shapes: list of shapes of CreateBox and CreateCapsule
    """
    directions = np.asarray(directions, dtype=float).reshape(-1, dimension3D)
    distances = np.full(len(directions), np.inf)
    for shape in shapes:
        Distances = RayCapsuleDistances if shape['type'] == shapeCapsule else RayBoxDistances
        distances = np.minimum(distances, Distances(origin, directions, shape))
    return distances
