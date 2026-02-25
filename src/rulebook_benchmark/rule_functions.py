import math

import numpy as np
import shapely
from numpy.linalg import norm

from rulebook_benchmark.rulebook import Rule
from rulebook_benchmark.utils import (
    angle_between,
    continuous_ttc,
    early_ttc,
    generalized_collision,
    normalize_vector,
)


def vru_collision(handler, step, car_mass=1500, vru_mass=70, momentum=False):
    vru_states = handler(step).vrus_in_proximity
    return generalized_collision(
        handler,
        handler.collision_timeline,
        vru_states,
        step,
        car_mass,
        vru_mass,
        momentum,
        VRU=True,
    )


def vehicle_collision(handler, step, car_mass=1500, momentum=False):
    vehicle_states = handler(step).vehicles_in_proximity
    return generalized_collision(
        handler,
        handler.collision_timeline,
        vehicle_states,
        step,
        car_mass,
        car_mass,
        momentum,
    )


def vru_ttc(handler, step, threshold=1.0):
    pool = handler(step)
    ego_state = pool.ego_state
    ego_velocity = ego_state.velocity
    ego_position = ego_state.position
    ego_polygon = ego_state.polygon.convex_hull  # comment/uncomment

    violation = 0

    for state in pool.vru_states:
        obj_velocity = state.velocity
        obj_pos = state.position
        if not early_ttc(ego_position, ego_velocity, obj_pos, obj_velocity, threshold):
            continue

        v_rel = (obj_velocity[0] - ego_velocity[0], obj_velocity[1] - ego_velocity[1])
        ttc = continuous_ttc(
            ego_polygon.exterior.coords[:-1],
            state.polygon.convex_hull.exterior.coords[:-1],
            v_rel,
            threshold,
        )
        if ttc is not None:
            violation = max(violation, threshold - ttc)

    return violation


def vehicle_ttc(handler, step, threshold=0.8):
    pool = handler(step)
    ego_state = pool.ego_state
    ego_velocity = ego_state.velocity
    ego_position = ego_state.position
    ego_polygon = ego_state.polygon
    ego_polygon = ego_polygon.convex_hull  # comment/uncomment
    violation = 0

    for state in pool.other_vehicle_states:

        obj_velocity = state.velocity
        obj_polygon = state.polygon
        obj_polygon = obj_polygon.convex_hull  # comment/uncomment
        obj_pos = state.position

        if not early_ttc(ego_position, ego_velocity, obj_pos, obj_velocity, threshold):
            continue
        v_rel = (obj_velocity[0] - ego_velocity[0], obj_velocity[1] - ego_velocity[1])
        ttc = continuous_ttc(
            ego_polygon.exterior.coords[:-1],
            obj_polygon.exterior.coords[:-1],
            v_rel,
            threshold,
        )
        if ttc is not None:
            violation = max(violation, threshold - ttc)

    return violation


def stay_in_drivable_area(handler, step, **kwargs):
    ego = handler.ego
    ego_state = ego.get_state(step)
    drivable_region = handler.network.drivableRegion.polygons

    difference = ego_state.polygon.difference(drivable_region)
    area = difference.area

    distance = shapely.distance(drivable_region, ego_state.polygon)
    violation = area + distance**2

    return violation


def vru_clearance(handler, step, on_road, threshold):
    pool = handler(step)
    vru_states = pool.vrus_in_proximity
    violation = 0
    distance = np.inf
    for vru_state in vru_states:
        if on_road and vru_state.lane is not None:
            distance = pool.distance(vru_state)
        elif not on_road and vru_state.lane is None:
            distance = pool.distance(vru_state)

        violation = max(violation, threshold - distance)

    return violation


def vru_acknowledgement(handler, step, threshold=0, timesteps=20, velocity=4):
    candidates = set()
    violation = 0
    num_vrus = len(handler.vru_uids)
    for i in range(step, min(step + timesteps, len(handler.realization))):
        pool = handler(i)
        vrus = pool.vrus_in_proximity
        for vru_state in vrus:
            candidates.add(vru_state.uid)
        if len(candidates) == num_vrus:
            break

    pool = handler(step)
    ego_acceleration = pool.ego_state.acceleration
    for uid in candidates:
        state = pool.world_state[uid]
        relative_position = state.position - pool.ego_state.position
        relative_position = normalize_vector(relative_position)
        # first check if projected velocity towards ego is above threshold
        projected_velocity = np.dot(state.velocity, relative_position)
        if projected_velocity > velocity:
            ego_acceleration_projected = np.dot(ego_acceleration, relative_position)
            violation = max(0, ego_acceleration_projected - threshold, violation)
        else:
            continue
    return violation


def correct_side(
    handler, step, relax_at_intersections=False
):  # use relax_at_intersections if your lane polygons do not cover all correct sides at intersections
    ego_state = handler(step).ego_state

    correct_area = shapely.Polygon()
    incorrect_area = shapely.Polygon()

    for lane in ego_state.correct_lanes:
        lane_polygon = lane.polygon
        if (
            handler.realization.network.intersectionAt(ego_state.position) is not None
            and relax_at_intersections
        ):
            # lane_polygon = lane.polygon.buffer(0.8)  # allow some buffer at intersections
            return 0  # if at intersection, touching correct lane is enough
        correct_area = correct_area.union(lane_polygon)

    for lane in ego_state.incorrect_lanes:
        incorrect_area = incorrect_area.union(lane.polygon)

    pure_incorrect_area = incorrect_area.difference(correct_area)
    ego_polygon = ego_state.polygon

    ego_violation_area = ego_polygon.intersection(pure_incorrect_area).area

    return ego_violation_area


def correct_side_alt(
    handler, step, relax_at_intersections=False, fine_grained=True
):  # use relax_at_intersections if your lane polygons do not cover all correct sides at intersections
    ego_state = handler(step).ego_state

    isScenic = handler.realization.isScenic
    rot = 0
    ego_lane = ego_state.lane

    if isScenic:
        rot = np.pi / 2

    if (
        ego_lane is None
        or handler.realization.network.intersectionAt(ego_state.position) is not None
        and relax_at_intersections
    ):
        return 0

    ego_lane_heading = ego_lane.orientation.value(ego_state.position) + rot

    if math.cos(ego_lane_heading - ego_state.orientation.yaw) < 0:
        if fine_grained:
            return shapely.intersection(ego_state.polygon, ego_lane.polygon).area
        return 1
    else:
        return 0


def speed_limit(handler, step, threshold=15):  # speed limit
    ego_state = handler(step).ego_state
    ego_velocity = norm(ego_state.velocity)
    if ego_state.lane is None or ego_state.lane.speedLimit is None:
        speed_limit = threshold
    else:
        speed_limit = ego_state.lane.speedLimit

    return max(0, ego_velocity - speed_limit) ** 2


def lane_keeping(handler, step):
    if step == 0:
        return 0
    ego_state = handler(step).ego_state
    ego_prev_state = handler(step - 1).ego_state
    ego_lane = ego_state.lane
    ego_prev_lane = ego_prev_state.lane

    if ego_lane == ego_prev_lane:
        return 0
    elif (ego_prev_lane is None and ego_lane is not None) or (
        ego_prev_lane is not None and ego_lane is None
    ):  # TODO: ask about this
        return 1
    else:
        for maneuver in ego_prev_lane.maneuvers:
            if maneuver.endLane == ego_lane or maneuver.connectingLane == ego_lane:
                return 0
        return 1


def jerk(handler, step):
    if step == 0:
        return 0

    ego_prev_state = handler(step - 1).ego_state
    ego_state = handler(step).ego_state

    jerk_value = norm(ego_state.acceleration - ego_prev_state.acceleration)
    return jerk_value


def longitudinal_acceleration(handler, step):
    if step == 0:
        return 0
    ego_state = handler(step).ego_state
    ego_orientation = ego_state.orientation.yaw
    ego_orientation_vector = normalize_vector(
        np.array([math.cos(ego_orientation), math.sin(ego_orientation)])
    )
    ego_acceleration = ego_state.acceleration
    longitudinal_acceleration = ego_acceleration.dot(ego_orientation_vector)
    return norm(longitudinal_acceleration)


def lateral_acceleration(handler, step):
    if step == 0:
        return 0
    ego_state = handler(step).ego_state
    ego = ego_state.object
    ego_velocity = ego_state.velocity
    turning_radius = ego.length / math.sin(ego.steer * math.pi / 2)
    lateral_acceleration = (
        norm(ego_velocity) ** 2 / turning_radius if turning_radius != 0 else 0
    )
    return abs(lateral_acceleration)


def lane_centering(handler, step, buffer=0.3):  # lane centering
    ego_state = handler(step).ego_state
    ego_pos = ego_state.position
    ego_lane = ego_state.lane
    if ego_lane is None:
        return 0
    centerline = ego_lane.centerline.lineString
    if buffer > 0:
        centerline = centerline.buffer(buffer)
    # double check shapely distance function for sparse centerline
    ego_pos_point = shapely.Point(ego_pos)
    distance = centerline.distance(ego_pos_point)
    return distance


def front_clearance(handler, step, threshold=0.8):
    if step == len(handler.realization) - 1:
        return 0
    pool = handler(step)
    front_ls = pool.trajectory_front_linestring
    ego_width = handler.ego.width

    states = pool.vehicles_in_proximity

    violation = 0
    for state in states:
        if front_ls.distance(state.polygon) < ego_width / 2:
            distance = pool.distance(state)
            violation = max(violation, threshold - distance)

    return violation


def side_clearance(handler, step, left=True, threshold=0.8):
    pool = handler(step)
    if step == 0:
        ls = pool.trajectory_front_linestring
    elif step == len(handler.realization) - 1:
        ls = pool.trajectory_behind_linestring
    else:
        ls = shapely.union(
            pool.trajectory_front_linestring, pool.trajectory_behind_linestring
        )

    violation = 0
    states = pool.vehicles_in_proximity
    ego_state = pool.ego_state
    width = handler.ego.width
    ego_heading_vector = normalize_vector(
        np.array(
            [math.cos(ego_state.orientation.yaw), math.sin(ego_state.orientation.yaw)]
        )
    )
    for state in states:
        if ls.distance(state.polygon) > width / 2:
            ego_to_object = normalize_vector(state.position - ego_state.position)
            angle = angle_between(ego_heading_vector, ego_to_object)
            if (angle >= 0 and left) or (angle < 0 and not left):
                violation = max(violation, threshold - pool.distance(state))

    return violation


def clearance_vector_based(handler, step, threshold=0.8, side_angle=90, side="front"):
    side_angle = math.radians(side_angle)
    pool = handler(step)
    ego_state = pool.ego_state
    states = pool.vehicles_in_proximity

    violation = 0

    ego_heading_vector = normalize_vector(
        np.array(
            [math.cos(ego_state.orientation.yaw), math.sin(ego_state.orientation.yaw)]
        )
    )

    for state in states:
        state_vector = normalize_vector(
            np.array([math.cos(state.orientation.yaw), math.sin(state.orientation.yaw)])
        )
        angle = angle_between(ego_heading_vector, state_vector)
        if (
            (abs(angle) <= side_angle / 2 and side == "front")
            or (-side_angle * 3 / 2 < angle < -side_angle / 2 and side == "right")
            or (side_angle * 3 / 2 > angle > side_angle / 2 and side == "left")
        ):
            violation = max(violation, threshold - pool.distance(state))

    return violation


f1 = Rule(vru_collision, max, "vru_collision", 1)
f2 = Rule(vehicle_collision, max, "vehicle_collision", 2)
f3 = Rule(stay_in_drivable_area, max, "stay_in_drivable_area", 3)
f4 = Rule(vru_ttc, max, "vru_ttc", 4, threshold=1.0)
f5 = Rule(
    vru_acknowledgement,
    max,
    "vru_acknowledgement",
    5,
    threshold=-1,
    timesteps=30,
    velocity=4,
)
f6 = Rule(vehicle_ttc, max, "vehicle_ttc", 6, threshold=0.8)
f7 = Rule(correct_side, sum, "correct_side", 7, relax_at_intersections=True)
f8 = Rule(vru_clearance, max, "vru_offroad_clearance", 8, on_road=False, threshold=1)
f9 = Rule(vru_clearance, max, "vru_onroad_clearance", 9, on_road=True, threshold=1)
f10 = Rule(front_clearance, max, "front_vehicle_clearance", 10, threshold=0.8)
f11 = Rule(side_clearance, max, "left_vehicle_clearance", 11, left=True, threshold=0.8)
f12 = Rule(
    side_clearance, max, "right_vehicle_clearance", 12, left=False, threshold=0.8
)
f13 = Rule(speed_limit, max, "speed_limit", 13, threshold=15)
f14 = Rule(lane_keeping, sum, "lane_keeping", 14)
f15 = Rule(lane_centering, sum, "lane_centering", 15, buffer=0.3)
f17 = Rule(jerk, sum, "jerk", 17)
f18 = Rule(longitudinal_acceleration, max, "longitudinal_acceleration", 18)
f19 = Rule(lateral_acceleration, max, "lateral_acceleration", 19)

f7_alt = Rule(
    correct_side_alt,
    sum,
    "correct_side_alt",
    7,
    relax_at_intersections=True,
    fine_grained=True,
)

f10_sum = Rule(front_clearance, sum, "front_vehicle_clearance_sum", 10, threshold=0.8)
f11_sum = Rule(
    side_clearance, sum, "left_vehicle_clearance_sum", 11, left=True, threshold=0.8
)
f12_sum = Rule(
    side_clearance, sum, "right_vehicle_clearance_sum", 12, left=False, threshold=0.8
)

f10_v = Rule(
    clearance_vector_based,
    max,
    "front_vehicle_clearance_vector",
    10,
    threshold=0.8,
    side_angle=90,
    side="front",
)
f11_v = Rule(
    clearance_vector_based,
    max,
    "left_vehicle_clearance_vector",
    11,
    side="left",
    threshold=0.8,
    side_angle=90,
)
f12_v = Rule(
    clearance_vector_based,
    max,
    "right_vehicle_clearance_vector",
    12,
    side="right",
    threshold=0.8,
    side_angle=90,
)
