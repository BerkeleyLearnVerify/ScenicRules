import math

import numpy as np
import shapely
from shapely.strtree import STRtree

from rulebook_benchmark.utils import normalize_angle


def isObjectInLane(state, lane):  # check if the object's center is in the lane
    lane_polygon = lane.polygon
    object_point = shapely.Point(state.position)
    return lane_polygon.contains(object_point)


def firstPass(
    obj, str_tree, lanes, isScenic=False
):  # process the states where the lane is not ambiguous
    possible_lanes = {}
    ambiguous_lanes = {}
    for i in range(len(obj.trajectory)):
        state = obj.get_state(i)
        point = shapely.Point(state.position)
        polygon = state.polygon
        possible_lanes[i] = get_possible_lanes(point, str_tree, lanes)
        polygon_intersected_lanes = get_possible_lanes(polygon, str_tree, lanes)
        correct, incorrect = correct_incorrect_lanes(
            state, polygon_intersected_lanes, isScenic=isScenic
        )
        state.correct_lanes = correct
        state.incorrect_lanes = incorrect

        if len(possible_lanes[i]) == 1:
            obj.trajectory[i].lane = possible_lanes[i][0]
        elif len(possible_lanes[i]) == 0:  # out of road
            obj.trajectory[i].lane = None
        else:  # needs second pass
            ambiguous_lanes[i] = possible_lanes[i]

    # print(len(ambiguous_lanes), "states with ambiguous lanes")
    return ambiguous_lanes


def get_closest_orientation_lane(state, lanes, isScenic=False):
    rot = 0
    if isScenic:
        rot = np.pi / 2

    orientation = state.orientation.yaw
    pos = state.position
    similarities = []
    for lane in lanes:
        lane_orientation = lane.orientation.value(pos) + rot
        similarities.append(math.cos(lane_orientation - orientation))

    max_idx = similarities.index(max(similarities))
    return lanes[max_idx]


def get_most_recent_lane(obj, step):
    for i in range(step - 1, -1, -1):
        lane = obj.get_state(i).lane
        if lane is not None:
            return lane
    return None


def get_next_lane(obj, step):
    for i in range(step + 1, len(obj.trajectory)):
        lane = obj.get_state(i).lane
        if lane is not None:
            return lane
    return None


def correct_incorrect_lanes(state, lanes, isScenic=False):
    pos = state.position
    ego_orientation = state.orientation.yaw
    rot = 0
    if isScenic:
        rot = np.pi / 2

    corrects = []
    incorrects = []
    for lane in lanes:
        lane_orientation = lane.orientation.value(pos) + rot
        if math.cos(lane_orientation - ego_orientation) > 0:
            corrects.append(lane)
        else:
            incorrects.append(lane)

    return corrects, incorrects


def secondPass(obj, ambiguous_lanes, network, isScenic=False):
    if isScenic:
        rot = np.pi / 2
    else:
        rot = 0
    for step, lanes in ambiguous_lanes.items():
        pos = obj.get_state(step).position

        intersection = network.intersectionAt(pos)

        found = False
        if intersection is not None:
            if step > 0:
                prev_lane = obj.get_state(step - 1).lane
                for lane in lanes:
                    if prev_lane == lane:
                        obj.trajectory[step].lane = lane
                        found = True
                        # print("found same lane as previous", step)
                        break
                if found:
                    continue

            j = step + 1
            while (
                j < len(obj.trajectory)
                and obj.trajectory[j].lane not in intersection.outgoingLanes
            ):
                j += 1

            candidate_lanes = []
            future_lane = None
            if j < len(obj.trajectory):
                future_lane = obj.get_state(j).lane

            if (
                future_lane is not None
            ):  # non-ambiguous lane or lane group found in the future
                future_lane = obj.get_state(j).lane
                for lane in lanes:
                    end_lane = lane.successor
                    if end_lane == future_lane:
                        found = True
                        obj.trajectory[step].lane = lane
                        candidate_lanes.append(lane)
                    elif end_lane.group == future_lane.group:
                        candidate_lanes.append(lane)

                if len(candidate_lanes) == 1:
                    obj.trajectory[step].lane = candidate_lanes.pop()
                    found = True
                    continue
                elif found == True:
                    continue

            k = step - 1
            while k >= 0 and obj.trajectory[k].lane not in intersection.incomingLanes:
                k -= 1

            candidate_lanes_2 = []
            prev_lane = None
            if k >= 0:
                prev_lane = obj.get_state(k).lane

            if (
                prev_lane is not None and future_lane is not None
            ):  # non-ambiguous lane or lane group found in the future and past
                for lane in candidate_lanes:
                    end_lane = lane.successor
                    start_lane = lane.predecessor

                    if end_lane == future_lane and start_lane == prev_lane:
                        candidate_lanes_2.append(lane)
                    elif (
                        end_lane.group == future_lane.group
                        and start_lane.group == prev_lane.group
                    ):
                        candidate_lanes_2.append(lane)
                if len(candidate_lanes_2) == 1:
                    obj.trajectory[step].lane = candidate_lanes_2.pop()
                    found = True
                    continue
                else:
                    pass
                    # for lane in candidate_lanes_2:
                    #    print(lane.id)

            angles = []

            last_resort = lanes

            obj_orientation = obj.get_state(step).orientation.yaw
            obj_orientation = obj_orientation % (2 * np.pi)
            for lane in last_resort:  # workaround for when no candidate lanes are found
                lane_orientation = (
                    lane.orientation.value(obj.get_state(step).position) + rot
                )
                lane_orientation = lane_orientation % (2 * np.pi)
                angles.append(abs(normalize_angle(lane_orientation - obj_orientation)))
            min_idx = angles.index(min(angles))
            obj.trajectory[step].lane = lanes[min_idx]
            # print(obj.object_id, "found lane with closest orientation", step)

        else:
            # print("this should not happen: secondPass no intersection found", step)
            prev_lane = obj.get_state(step - 1).lane if step > 0 else None
            if prev_lane is not None:
                for lane in lanes:
                    if prev_lane == lane or lane in [
                        maneuver.endLane for maneuver in prev_lane.maneuvers
                    ]:
                        obj.trajectory[step].lane = lane
                        found = True
                        # print("found same lane as previous", step)
                        break
            else:
                angles = []
                for lane in lanes:
                    lane_orientation = (
                        lane.orientation.value(obj.get_state(step).position)
                        % (2 * np.pi)
                        + rot
                    )
                    obj_orientation = obj.get_state(step).orientation.yaw % (2 * np.pi)
                    # ensure both angles are in the range [0, 2*pi)
                    lane_orientation = lane_orientation % (2 * np.pi)
                    obj_orientation = obj_orientation % (2 * np.pi)

                    angles.append(
                        abs(normalize_angle(lane_orientation - obj_orientation))
                    )
                min_idx = angles.index(min(angles))
                obj.trajectory[step].lane = lanes[min_idx]

            # print("found lane with closest orientation", step)


def process_trajectory(
    realization, isScenic=False
):  # given a realization, extract the sequence of lanes followed by each vehicle
    network = realization.network
    objects = realization.objects

    lanes = network.lanes
    lane_polygons = [lane.polygon for lane in lanes]
    strtree = STRtree(lane_polygons)
    polygon_to_lane = dict(zip(lane_polygons, lanes))

    for obj in objects:
        ambiguous_lanes = firstPass(obj, strtree, lanes, isScenic=isScenic)
        if len(ambiguous_lanes) > 0:
            secondPass(obj, ambiguous_lanes, network, isScenic=isScenic)

        # print(f"Object {obj.object_type} {obj.mesh} has trajectory: {[state.lane for state in obj.trajectory]}")


def get_possible_lanes(shapely_obj, tree, lanes):
    indices = tree.query(shapely_obj, predicate="intersects")
    return [lanes[ind] for ind in indices]


def process_trajectory_old(realization):
    network = realization.network
    objects = realization.objects
    for obj in objects:
        for i in range(len(obj.trajectory)):
            state = obj.get_state(i)
            state.lane = network.laneAt(state.position)
            # if state.lane is None:
            #    print(f"Object {obj.object_type} {obj.mesh} is out of road at step {i}")
