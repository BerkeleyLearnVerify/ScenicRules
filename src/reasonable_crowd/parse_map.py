import os

import geopandas as gpd
import shapely

from reasonable_crowd.roads import (
    Intersection,
    Lane,
    LaneGroup,
    Maneuver,
    Network,
    Road,
)

S_main_lane_boundaries = {
    0: (7, 0, -1),
    1: (11, 0, 1),
    2: (14, 1, -1),
    3: (15, 1, 1),
    4: (6, 2, -1),
    5: (8, 2, 1),
    6: (16, 3, 1),
    7: (9, 3, -1),
    8: (17, 4, 1),
    9: (12, 4, -1),
    10: (10, 5, -1),
    11: (13, 5, 1),
}
S_main_boundary_max = 17
S_connecting_boundary_max = 53
S_main_max = 11
S_max = 125
j = 12
for i in range(S_main_boundary_max + 1, S_connecting_boundary_max + 1, 2):
    S_main_lane_boundaries[j] = (i + 1, i, -1)
    j += 1


S_lane_in_groups = 1
S_group_in_roads = 2


U_lane_in_groups = 3
U_group_in_roads = 2


U_main_max = 71
# 83
U_main_boundary_max = 83
U_connecting_boundary_max = 307
U_max = 499
U_main_lane_boundaries = {
    0: (60, 7, -1),
    1: (7, 27, -1),
    2: (27, 30, -1),
    3: (12, 30, 1),
    4: (8, 12, -1),
    5: (79, 8, -1),
    6: (61, 20, -1),
    7: (20, 58, -1),
    8: (58, 33, -1),
    9: (6, 33, 1),
    10: (0, 6, -1),
    11: (76, 0, -1),
    12: (77, 16, -1),
    13: (16, 25, -1),
    14: (25, 36, 1),
    15: (4, 36, -1),
    16: (10, 4, -1),
    17: (62, 10, -1),
    18: (63, 14, -1),
    19: (14, 26, -1),
    20: (26, 31, -1),
    21: (9, 31, 1),
    22: (28, 9, -1),
    23: (81, 28, -1),
    24: (83, 18, -1),
    25: (18, 21, -1),
    26: (21, 32, 1),
    27: (2, 32, -1),
    28: (5, 2, -1),
    29: (64, 5, -1),
    30: (65, 11, -1),
    31: (11, 29, -1),
    32: (29, 59, -1),
    33: (15, 59, 1),
    34: (19, 15, -1),
    35: (80, 19, -1),
    36: (66, 22, -1),
    37: (22, 23, -1),
    38: (23, 34, -1),
    39: (17, 34, 1),
    40: (1, 17, -1),
    41: (82, 1, -1),
    42: (78, 24, -1),
    43: (24, 13, -1),
    44: (13, 35, 1),
    45: (3, 35, -1),
    46: (57, 3, -1),
    47: (67, 57, -1),
    48: (74, 50, -1),
    49: (50, 39, -1),
    50: (39, 52, 1),
    51: (54, 52, -1),
    52: (43, 54, -1),
    53: (68, 43, -1),
    54: (69, 51, -1),
    55: (51, 38, -1),
    56: (38, 41, -1),
    57: (46, 41, 1),
    58: (56, 46, -1),
    59: (72, 56, -1),
    60: (70, 48, -1),
    61: (48, 45, -1),
    62: (45, 55, -1),
    63: (49, 55, 1),
    64: (37, 49, -1),
    65: (75, 37, -1),
    66: (71, 44, -1),
    67: (44, 47, -1),
    68: (47, 53, -1),
    69: (40, 53, 1),
    70: (42, 40, -1),
    71: (73, 42, -1),
}

j = 72
for i in range(U_main_boundary_max + 1, U_connecting_boundary_max + 1, 2):
    U_main_lane_boundaries[j] = (i + 1, i, -1)
    j += 1


def approximate_centerline(linestring_1, linestring_2):
    centerline_points = []
    coords_1 = linestring_1.coords
    coords_2 = linestring_2.coords

    projections = set()

    projections.update(
        {
            linestring_1.project(shapely.Point(coord), normalized=True)
            for coord in coords_1
        }
    )
    projections.update(
        {
            linestring_2.project(shapely.Point(coord), normalized=True)
            for coord in coords_2
        }
    )

    projections = list(projections)
    projections.sort()

    for projection in projections:
        point_1 = linestring_1.interpolate(projection, normalized=True)
        point_2 = linestring_2.interpolate(projection, normalized=True)
        center = shapely.Point((point_1.x + point_2.x) / 2, (point_1.y + point_2.y) / 2)
        centerline_points.append(center)

    return shapely.LineString(centerline_points)


class ReasonableCrowdMapParser:
    def __init__(self, directory, S_U="S"):
        self.directory = directory
        self.S_U = S_U
        self.lane_in_groups = S_lane_in_groups if S_U == "S" else U_lane_in_groups
        self.group_in_roads = S_group_in_roads if S_U == "S" else U_group_in_roads
        self.main_lane_boundaries = (
            S_main_lane_boundaries if S_U == "S" else U_main_lane_boundaries
        )
        self.main_max = S_main_max if S_U == "S" else U_main_max
        (
            self.boundary_df,
            self.road_df,
            self.intersection_df,
            self.lane_group_df,
            self.lane_df,
        ) = self._get_map_files(directory, S_U)
        self._create_lanes_from_boundaries()
        self._create_maneuvers()
        self._get_intersections()
        self._get_lane_groups()
        self._get_roads()
        self._set_intersection_roads()
        self._create_elements()

    def create_network(self):
        return Network(
            self.elements,
            self.roads,
            self.connectingRoads,
            self.lanes,
            self.laneGroups,
            self.intersections,
        )

    def _get_map_files(self, directory, S_U="S"):
        boundary_path = os.path.join(directory, S_U + "_boundaries.gpkg")
        road_path = os.path.join(directory, S_U + "_road_segments.gpkg")
        intersection_path = os.path.join(directory, S_U + "_intersections.gpkg")
        lane_group_path = os.path.join(directory, S_U + "_lane_groups_polygons.gpkg")
        lane_path = os.path.join(directory, S_U + "_lanes_polygons.gpkg")

        boundary_df = gpd.read_file(boundary_path)
        road_df = gpd.read_file(road_path)
        intersection_df = gpd.read_file(intersection_path)
        lane_group_df = gpd.read_file(lane_group_path)
        lane_df = gpd.read_file(lane_path)

        return boundary_df, road_df, intersection_df, lane_group_df, lane_df

    def _create_lanes_from_boundaries(self):
        lanes = []
        for key, value in self.main_lane_boundaries.items():
            lane_id = "lane" + str(key)
            left_edge = self.boundary_df.geometry[value[1]]
            left_edge_coords = list(left_edge.coords)
            right_edge = self.boundary_df.geometry[value[0]]
            right_edge_coords = list(right_edge.coords)

            if (
                value[2] == 1
            ):  # reverse for lane and centerline, do not reverse for polygon
                polygon = shapely.Polygon((right_edge_coords + left_edge_coords))
                left_edge = shapely.LineString(left_edge_coords[::-1])

            elif (
                value[2] == -1
            ):  # reverse the order for polygon, do not reverse for lane and centerline
                polygon = shapely.Polygon((right_edge_coords + left_edge_coords[::-1]))
            else:
                raise ValueError("Invalid boundary direction value, must be 1 or -1")

            centerline = approximate_centerline(right_edge, left_edge)
            lanes.append(
                Lane(
                    polygon=polygon,
                    name=lane_id,
                    centerline=centerline,
                    leftEdge=left_edge,
                    rightEdge=right_edge,
                )
            )
        lanes = tuple(lanes)

        self.lanes = lanes
        self.main_lanes = lanes[: self.main_max + 1]
        self.connecting_lanes = lanes[self.main_max + 1 :]

    def _create_maneuvers(self):

        for lane in self.connecting_lanes:
            next_lanes = []
            previous_lanes = []
            left_edge_endpoint = shapely.Point(lane.leftEdge.lineString.coords[-1])
            right_edge_endpoint = shapely.Point(lane.rightEdge.lineString.coords[-1])
            left_edge_startpoint = shapely.Point(lane.leftEdge.lineString.coords[0])
            right_edge_startpoint = shapely.Point(lane.rightEdge.lineString.coords[0])
            for main_lane in self.main_lanes:
                if main_lane.leftEdge.lineString.intersects(
                    left_edge_endpoint
                ) and main_lane.rightEdge.lineString.intersects(right_edge_endpoint):
                    next_lanes.append(main_lane)
                    lane.successor = main_lane
                    lane.maneuvers = (Maneuver(lane, main_lane),)
                elif main_lane.leftEdge.lineString.intersects(
                    left_edge_startpoint
                ) and main_lane.rightEdge.lineString.intersects(right_edge_startpoint):
                    previous_lanes.append(main_lane)
                    lane.predecessor = main_lane

            # print(lane.name, "next lanes:", [l.name for l in next_lanes], "previous lanes:", [l.name for l in previous_lanes])
            # if len(next_lanes) == 0 or len(previous_lanes) == 0 or len(next_lanes) > 1 or len(previous_lanes) > 1:
            #    print(lane.name)
            #    print("next lanes:", [l.name for l in next_lanes], "previous lanes:", [l.name for l in previous_lanes])

            assert len(next_lanes) == len(previous_lanes) == 1

        for lane in self.main_lanes:
            maneuvers = []
            next_lanes = []
            previous_lanes = []
            left_edge_endpoint = shapely.Point(lane.leftEdge.lineString.coords[-1])
            right_edge_endpoint = shapely.Point(lane.rightEdge.lineString.coords[-1])
            left_edge_startpoint = shapely.Point(lane.leftEdge.lineString.coords[0])
            right_edge_startpoint = shapely.Point(lane.rightEdge.lineString.coords[0])
            for connecting_lane in self.connecting_lanes:
                if connecting_lane.leftEdge.lineString.intersects(
                    left_edge_endpoint
                ) and connecting_lane.rightEdge.lineString.intersects(
                    right_edge_endpoint
                ):
                    next_lanes.append(connecting_lane)
                    maneuvers.append(
                        Maneuver(lane, connecting_lane.successor, connecting_lane)
                    )
                elif connecting_lane.leftEdge.lineString.intersects(
                    left_edge_startpoint
                ) and connecting_lane.rightEdge.lineString.intersects(
                    right_edge_startpoint
                ):
                    previous_lanes.append(connecting_lane)

            if len(next_lanes) == 1:
                lane.successor = next_lanes[0]
            if len(previous_lanes) == 1:
                lane.predecessor = previous_lanes[0]

            lane.maneuvers = tuple(maneuvers)

    def _get_intersections(self):
        intersections = []
        for i, row in self.intersection_df.iterrows():
            processed_lanes = []
            incoming_lanes = []
            outgoing_lanes = []
            connecting_lanes = []
            intersection_polygon = row.geometry
            intersection_id = "intersection" + str(i)
            for lane in self.connecting_lanes:
                if (
                    intersection_polygon.intersects(lane.polygon)
                    and lane.name not in processed_lanes
                ):
                    connecting_lanes.append(lane)
                    incoming_lanes.append(lane.predecessor)
                    outgoing_lanes.append(lane.successor)
                    processed_lanes.append(lane.name)

            incoming_lanes = tuple(incoming_lanes)
            outgoing_lanes = tuple(outgoing_lanes)
            connecting_lanes = tuple(connecting_lanes)

            intersection = Intersection(
                polygon=intersection_polygon,
                name=intersection_id,
                incomingLanes=incoming_lanes,
                outgoingLanes=outgoing_lanes,
                connectingLanes=connecting_lanes,
            )

            intersections.append(intersection)
        intersections = tuple(intersections)
        self.intersections = intersections

    def _get_lane_groups(self):
        lane_groups = []
        for i, row in self.lane_group_df.iterrows():
            lane_group_polygon = row.geometry
            lane_group_id = "lane_group" + str(i)
            lane_group = LaneGroup(polygon=lane_group_polygon, name=lane_group_id)
            lanes = [
                lane
                for lane in self.main_lanes
                if lane_group_polygon.intersects(lane.polygon)
            ]
            # for lane in lanes:
            #    print(lane.name)
            #    print(lane.polygon.is_valid)
            #    print(lane.polygon.is_simple)
            intersection_areas = [
                lane_group.polygon.intersection(lane.polygon).area for lane in lanes
            ]
            zipped = zip(intersection_areas, lanes)
            zipped = sorted(zipped, key=lambda x: x[0], reverse=True)
            lanes = [lane for _, lane in zipped][: self.lane_in_groups]
            for lane in lanes:
                lane.laneGroup = lane_group
            lane_group.lanes = tuple(lanes)
            lane_groups.append(lane_group)

        i += 1
        for lane in self.connecting_lanes:
            lane_group = LaneGroup(polygon=lane.polygon, name="lane_group" + str(i))
            lane_group.lanes = (lane,)
            lane.laneGroup = lane_group
            lane_groups.append(lane_group)

            i += 1

        lane_groups = tuple(lane_groups)
        self.laneGroups = lane_groups

    def _get_roads(self):
        roads = []
        for i, row in self.road_df.iterrows():
            road_polygon = row.geometry
            road_id = "road" + str(i)
            road = Road(polygon=road_polygon, name=road_id)

            lane_groups = [
                lane_group
                for lane_group in self.laneGroups
                if road_polygon.intersects(lane_group.polygon)
            ]
            intersection_areas = [
                lane_group.polygon.intersection(road_polygon).area
                for lane_group in lane_groups
            ]
            zipped = zip(intersection_areas, lane_groups)
            zipped = sorted(zipped, key=lambda x: x[0], reverse=True)
            lane_groups = [lane_group for _, lane_group in zipped][
                : self.group_in_roads
            ]
            # append all tuples
            road.lanes = tuple(
                [lane for lane_group in lane_groups for lane in lane_group.lanes]
            )
            road.laneGroups = tuple(lane_groups)

            for lane_group in road.laneGroups:
                lane_group.road = road
            for lane in road.lanes:
                lane.road = road

            roads.append(road)
        i += 1
        connecting_roads = []
        for lane in self.connecting_lanes:
            road = Road(polygon=lane.polygon, name="road" + str(i))
            road.lanes = (lane,)
            road.laneGroups = (lane.laneGroup,)
            for lane_group in road.laneGroups:
                lane_group.road = road
            for lane in road.lanes:
                lane.road = road
            connecting_roads.append(road)
            lane.road = road

            i += 1

        roads = tuple(roads)
        self.roads = roads
        connecting_roads = tuple(connecting_roads)
        self.connectingRoads = connecting_roads

    def _set_intersection_roads(self):

        for intersection in self.intersections:
            roads = []
            for road in self.connectingRoads:
                if intersection.polygon.intersects(road.polygon):
                    roads.append(road)

            roads = tuple(roads)
            intersection.roads = roads

    def _create_elements(self):
        elements = {}
        for road in self.roads:
            elements[road.name] = road
        for lane in self.lanes:
            elements[lane.name] = lane
        for lane_group in self.laneGroups:
            elements[lane_group.name] = lane_group
        for intersection in self.intersections:
            elements[intersection.name] = intersection
        for road in self.connectingRoads:
            elements[road.name] = road
        self.elements = elements


def parse_map(directory, S_U="S"):
    parser = ReasonableCrowdMapParser(directory, S_U)
    network = parser.create_network()
    return network
