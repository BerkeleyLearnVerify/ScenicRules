# This file includes the some of the classes in Scenic's roads.py, but they are simplified to keep only the necessary properties for the benchmark.
import numbers
from typing import Tuple, Union

import numpy as np
import shapely
from cached_property import cached_property
from scenic.core.object_types import Point
from scenic.core.vectors import Vector
from scenic.domains.driving.roads import _toVector

from rulebook_benchmark.utils import angle_between, normalize_vector

Vectorlike = Union[Vector, Point, Tuple[numbers.Real, numbers.Real]]


class MultiLinePlaceholder:  # placeholder class for getting the linestring from attributes such as centerline, leftEdge etc
    def __init__(self, linestring):
        self.lineString = linestring


class Maneuver:
    def __init__(self, startLane, endLane, connectingLane=None):
        self.startLane = startLane
        self.endLane = endLane
        self.connectingLane = connectingLane


class OrientationVectorPlaceholder:
    def __init__(self, angle):
        self.yaw = angle  # assuming angle is in radians


class ElementOrientation:
    def __init__(self, element):
        self.element = element

    def value(self, point):
        if isinstance(self.element, Lane):
            return self._get_centerline_orientation(point)
        elif isinstance(self.element, NetworkElement):
            lane = self.element.network.findPointIn(
                point, self.element.network.lanes, reject=False
            )
            if lane:
                return lane.orientation.value(point)
            else:
                raise Exception(
                    f"Error: Point {point} not found in any lane of the network, even though it lies on a NetworkElement."
                )
        else:
            raise NotImplementedError(
                "Orientation not implemented for this element type"
            )

    def _get_centerline_orientation(self, point, epsilon=1e-6):
        centerline = self.element.centerline.lineString
        point = shapely.Point(point)
        projection = centerline.project(point, normalized=True)
        if projection <= epsilon:
            previous_point = centerline.interpolate(0, normalized=True)
        else:
            previous_point = centerline.interpolate(
                projection - epsilon, normalized=True
            )

        if projection >= 1 - epsilon:
            next_point = centerline.interpolate(1, normalized=True)
        else:
            next_point = centerline.interpolate(projection + epsilon, normalized=True)
        direction = np.array([next_point.x, next_point.y]) - np.array(
            [previous_point.x, previous_point.y]
        )
        # direction = Vector(next_point.x, next_point.y) - Vector(previous_point.x, previous_point.y)
        zero_radian = np.array([1, 0])  # assuming right is 0 radians
        direction = direction / np.linalg.norm(direction)  # normalize direction
        angle = angle_between(zero_radian, direction)
        return angle


class NetworkElement:
    def __init__(self, polygon, name, network=None, speed_limit=None):
        self.polygon = polygon
        self.name = name  # will also act as unique id
        self.uid = name
        self.id = name  # for compatibility with Scenic
        self.network = network
        self.speedLimit = speed_limit

    @cached_property
    def orientation(self):
        return ElementOrientation(self)

    def __eq__(self, other):
        if not isinstance(other, NetworkElement):
            return NotImplemented
        return self.network is other.network and self.uid == other.uid


class LinearElement(NetworkElement):
    def __init__(
        self,
        polygon,
        name,
        centerline,
        leftEdge,
        rightEdge,
        network=None,
        speed_limit=None,
    ):
        super().__init__(polygon, name, network, speed_limit)
        self.centerline = MultiLinePlaceholder(centerline)
        self.leftEdge = MultiLinePlaceholder(leftEdge)
        self.rightEdge = MultiLinePlaceholder(rightEdge)


class Road(NetworkElement):
    def __init__(self, polygon, name, network=None, speed_limit=None):
        super().__init__(polygon, name, network, speed_limit)
        self.laneGroups = None
        self.lanes = None

    def laneGroupAt(self, point, reject=False):
        return self.network.findPointIn(point, self.laneGroups, reject)


class Lane(LinearElement):
    def __init__(
        self,
        polygon,
        name,
        centerline,
        leftEdge,
        rightEdge,
        network=None,
        speed_limit=None,
    ):
        super().__init__(
            polygon, name, centerline, leftEdge, rightEdge, network, speed_limit
        )
        self.maneuvers = None
        self.predecessor = None
        self.successor = None
        self.group = None
        self.road = None


class LaneGroup(NetworkElement):
    def __init__(self, polygon, name, network=None, speed_limit=None):
        super().__init__(polygon, name, network, speed_limit)
        self.lanes = None
        self.road = None


class Intersection(NetworkElement):
    def __init__(
        self,
        polygon,
        name,
        network=None,
        speed_limit=None,
        incomingLanes=None,
        outgoingLanes=None,
        connectingLanes=None,
    ):
        super().__init__(polygon, name, network, speed_limit)
        self.incomingLanes = incomingLanes
        self.outgoingLanes = outgoingLanes
        self.connectingLanes = connectingLanes
        self.roads = None


class RegionPlaceholder:
    def __init__(self, roads, lanes, intersections):
        self.roads = roads
        self.lanes = lanes
        self.intersections = intersections
        self.__attrs_post_init__()

    def __attrs_post_init__(self):
        polygon = shapely.Polygon()  # Placeholder for the actual polygon
        for road in self.roads:
            polygon = polygon.union(road.polygon)
        for lane in self.lanes:
            polygon = polygon.union(lane.polygon)
        for intersection in self.intersections:
            polygon = polygon.union(intersection.polygon)
        self.polygons = polygon
        self.polygon = polygon


class Network:
    def __init__(
        self, elements, roads, connectingRoads, lanes, laneGroups, intersections
    ):
        self.elements = elements
        self.roads = roads
        self.connectingRoads = connectingRoads
        self.lanes = lanes
        self.laneGroups = laneGroups
        self.intersections = intersections
        self.tolerance = 0
        self.allRoads = roads + connectingRoads
        self.__attrs_post_init__()

    def __attrs_post_init__(self):
        for elem in self.elements.values():
            elem.network = self
        self.drivableRegion = RegionPlaceholder(
            self.roads, self.lanes, self.intersections
        )
        self._uidForIndex = tuple(self.elements)
        self._rtree = shapely.STRtree([elem.polygon for elem in self.elements.values()])

    def findPointIn(self, point, elems, reject):
        point = shapely.geometry.Point(_toVector(point))

        def findElementWithin(distance):
            target = point if distance == 0 else point.buffer(distance)
            indices = self._rtree.query(target, predicate="intersects")
            candidates = {self._uidForIndex[index] for index in indices}
            if candidates:
                for elem in elems:
                    if elem.uid in candidates:
                        return elem
            return None

        # First pass: check for elements containing the point.
        if elem := findElementWithin(0):
            return elem

        # Second pass: check for elements within tolerance of the point.
        if self.tolerance > 0 and (elem := findElementWithin(self.tolerance)):
            return elem

        # No matches found.
        if reject:
            if isinstance(reject, str):
                message = reject
            else:
                message = "requested element does not exist"
            raise Exception(message)
        return None

    def elementAt(self, point: Vectorlike, reject=False) -> Union[NetworkElement, None]:
        """Get the highest-level `NetworkElement` at a given point, if any.

        If the point lies in an `Intersection`, we return that; otherwise if the point
        lies in a `Road`, we return that; otherwise we return :obj:`None`, or reject the
        simulation if **reject** is true (default false).
        """
        point = _toVector(point)
        intersection = self.intersectionAt(point)
        if intersection is not None:
            return intersection
        return self.roadAt(point, reject=reject)

    def roadAt(self, point: Vectorlike, reject=False) -> Union[Road, None]:
        """Get the `Road` passing through a given point."""
        return self.findPointIn(point, self.allRoads, reject)

    def laneAt(self, point: Vectorlike, reject=False) -> Union[Lane, None]:
        """Get the `Lane` passing through a given point."""
        return self.findPointIn(point, self.lanes, reject)

    def laneGroupAt(self, point: Vectorlike, reject=False) -> Union[LaneGroup, None]:
        """Get the `LaneGroup` passing through a given point."""
        point = _toVector(point)
        road = self.roadAt(point, reject=reject)
        return None if road is None else road.laneGroupAt(point, reject=reject)

    def intersectionAt(
        self, point: Vectorlike, reject=False
    ) -> Union[Intersection, None]:
        """Get the `Intersection` at a given point."""
        return self.findPointIn(point, self.intersections, reject)
