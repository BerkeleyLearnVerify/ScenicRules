# TODO: change naming of class methods
import math

import numpy as np
import shapely
from cached_property import cached_property

from rulebook_benchmark.utils import in_proximity, intersects, polygon_distance

DELTA = 0.1


class Realization:
    def __init__(self, ego_index=0, delta=DELTA, proximity_threshold=3):
        self.network = None
        self.objects = None
        self.ego_index = ego_index
        self.delta = delta
        self.proximity_threshold = proximity_threshold
        self.isScenic = False

    def __len__(self):
        return len(self.objects[self.ego_index].trajectory)

    @property
    def ego(self):
        if not isinstance(self.ego_index, int):
            raise Exception(f"Error: Ego index {self.ego_index} is not an integer")
        if self.ego_index < 0 or self.ego_index >= len(self.objects):
            raise Exception(
                f"Error: Ego index {self.ego_index} out of bounds for objects list of length {len(self.objects)}"
            )
        return self.objects[self.ego_index]

    def get_ego(self):
        try:
            return self.objects[self.ego_index]
        except IndexError:
            raise Exception(
                f"Error: Ego index {self.ego_index} not found in objects list"
            )

    def get_world_state(self, step):
        states = []
        for i in range(len(self.objects)):
            states.append(self.objects[i].get_state(step))
        return WorldState(states, step, self.ego_index)

    def set_ego_index(self, ego_index):
        if 0 <= ego_index < len(self.objects):
            self.ego_index = ego_index
        else:
            raise Exception(f"Error: Proposed ego index {ego_index} out of list bounds")

    def get_ego_index(self):
        return self.ego_index

    @property
    def trajectory(self):
        trajectory = []
        for i in range(len(self)):
            states = self.get_world_state(i)
            trajectory.append(states)
        return trajectory

    @property
    def other_objects(self):
        return self.objects[: self.ego_index] + self.objects[self.ego_index + 1 :]

    @cached_property
    def vehicles(self):
        vehicles = []
        for obj in self.objects:
            if obj.object_type == "Car" or obj.object_type == "Truck":
                vehicles.append(obj)
        return vehicles

    @property
    def other_vehicles(self):
        return self.vehicles[: self.ego_index] + self.vehicles[self.ego_index + 1 :]

    @cached_property
    def vrus(self):
        VRUs = []
        for obj in self.objects:
            if obj.object_type == "Pedestrian" or obj.object_type == "Bicycle":
                VRUs.append(obj)
        return VRUs


class RealizationObject:
    def __init__(self, object_id, dimensions, object_type, base_footprint=None):
        self.uid = object_id
        self.dimensions = dimensions
        self.length = dimensions[0]
        self.width = dimensions[1]
        self.base_footprint = base_footprint

        self.object_type = object_type
        self.trajectory = []

    @cached_property
    def radius(self):
        return math.sqrt(self.width**2 + self.length**2) / 2

    def get_state(self, step):
        try:
            return self.trajectory[step]
        except IndexError:
            raise Exception(f"Error: Step {step} not found in object trajectory")


class State:
    def __init__(
        self,
        obj,
        position,
        velocity,
        orientation,
        step,
        steer=None,
        throttle=None,
        brake=None,
    ):
        self.object = obj
        self.position = position
        self.velocity = velocity
        self.orientation = orientation
        self.step = step
        self.steer = steer
        self.throttle = throttle
        self.brake = brake
        self.lane = None  # to be set in process_trajectory
        self.correct_lanes = []  # to be set in process_trajectory
        self.incorrect_lanes = []  # to be set in process_trajectory

    @cached_property
    def orientation_trimesh(self):
        return self.orientation._trimeshEulerAngles()

    @cached_property
    def polygon(self):
        if self.object.base_footprint is not None:
            # use base footprint if provided
            base_footprint = self.object.base_footprint

            cx, cy = self.position
            yaw = self.orientation.yaw  # radians

            # Rotation matrix
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

            # Rotate + translate
            world_corners = (base_footprint @ R.T) + np.array([cx, cy])

            return shapely.Polygon(world_corners)

        else:
            obj_length = self.object.length
            obj_width = self.object.width
            cx, cy = self.position
            yaw = self.orientation.yaw  # radians

            # Half-dimensions
            hl = obj_length / 2
            hw = obj_width / 2

            # Rectangle corners in local frame (centered at origin, no rotation)
            local_corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])

            # Rotation matrix
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

            # Rotate + translate
            world_corners = (local_corners @ R.T) + np.array([cx, cy])

            return shapely.Polygon(world_corners)

    @cached_property
    def coords_np(self):
        return np.array(self.polygon.exterior.coords[:-1])

    @cached_property
    def acceleration(self):
        if self.step == 0:
            return np.zeros(2)
        else:
            prev_state = self.object.get_state(self.step - 1)
            delta_v = self.velocity - prev_state.velocity
            return delta_v / DELTA

    @property
    def uid(self):
        return self.object.uid


class WorldState:
    def __init__(self, states, step, ego_index):
        self.ego_index = ego_index
        self.states = states
        self.step = step

    def __getitem__(self, index):
        if index < 0 or index >= len(self.states):
            raise IndexError(f"Index {index} out of bounds for states list")
        return self.states[index]

    def get_ego_state(self):
        return self[self.ego_index]

    @property
    def other_states(self):
        return self.states[: self.ego_index] + self.states[self.ego_index + 1 :]

    @property
    def other_vehicle_states(self):
        return [
            state
            for state in self.other_states
            if state.object.object_type in ["Car", "Truck"]
        ]

    @property
    def ego_state(self):
        return self.get_ego_state()

    @property
    def vru_states(self):
        return [
            state
            for state in self.states
            if state.object.object_type in ["Pedestrian", "Bicycle"]
        ]


class VariableHandler:
    def __init__(self, realization):
        self.realization = realization
        self.proximity_threshold = self.realization.proximity_threshold
        self.max_steps = len(realization)
        self._pools = {}
        self.ego = realization.ego
        self.objects = realization.objects
        self.network = realization.network
        self._collision_timeline = {}
        self.vehicle_uids = set(obj.uid for obj in self.realization.other_vehicles)
        self.vru_uids = set(obj.uid for obj in self.realization.vrus)

    def __call__(self, step, **kwargs):
        if step not in self._pools:
            self._pools[step] = VariablePool(
                step, self, self.proximity_threshold, **kwargs
            )

        # self._pools.pop(step - 3, None)  # free memory by removing pools for steps that are no longer needed
        return self._pools[step]

    @cached_property
    def trajectory_linestring(self):
        return shapely.LineString([state.position for state in self.ego.trajectory])

    @cached_property
    def isScenic(self):
        return self.realization.isScenic

    @cached_property
    def other_objects(self):
        return self.realization.other_objects

    @cached_property
    def trajectory_buffer(self):
        width = self.ego.width
        polygon = shapely.buffer(
            self.trajectory_linestring, width / 2, cap_style="square"
        )
        polygon = polygon.union(self.ego.trajectory[-1].polygon)
        polygon = polygon.union(self.ego.trajectory[0].polygon)
        return polygon

    @cached_property
    def collision_timeline(self):
        self._collision_timeline = {}

        previous_colliding = set()
        for i in range(len(self.realization)):
            pool = self(i)
            colliding = set(pool.vehicles_colliding.keys())
            colliding.update(pool.vrus_colliding.keys())

            # new collisions
            for uid in colliding - previous_colliding:
                self._collision_timeline.setdefault(uid, []).append([i, i])

            # ongoing collisions
            for uid in colliding & previous_colliding:
                self._collision_timeline[uid][-1][1] = i  # extend end time

            previous_colliding = colliding

        # convert inner [start, end] lists to tuples
        for uid, intervals in self._collision_timeline.items():
            self._collision_timeline[uid] = [tuple(interval) for interval in intervals]

        return self._collision_timeline


class VariablePool:
    def __init__(self, step, handler, proximity_threshold, steps_ahead=None):
        self.handler = handler
        self.realization = self.handler.realization
        self.other_objects = self.handler.other_objects
        self.objects = self.handler.objects
        self.step = step
        self.world_state = self.realization.get_world_state(step)
        self.ego = self.realization.ego
        self.ego_state = self.world_state.ego_state
        self.other_vehicle_states = self.world_state.other_vehicle_states
        self.vru_states = self.world_state.vru_states
        self.proximity_threshold = proximity_threshold
        self._distances = {}
        self._lazy_distances = {}
        self.steps_ahead = len(self.realization) if steps_ahead is None else steps_ahead

    @cached_property
    def ego_state(self):
        return self.ego.get_state(self.step)

    def colliding(self, states):
        colliding = {}
        for state in states:
            if intersects(self.ego_state, state):
                colliding[state.uid] = state
        return colliding

    def distance(self, other_state):
        uid = other_state.uid
        if uid not in self._distances:
            self._distances[uid] = polygon_distance(self.ego_state, other_state)
        return self._distances[uid]

    def center_distance(self, other_state):
        uid = other_state.uid
        if uid not in self._lazy_distances:
            self._lazy_distances[uid] = np.linalg.norm(
                other_state.position - self.ego_state.position
            )
        return self._lazy_distances[uid]

    @cached_property
    def vehicles_colliding(self):
        states_in_proximity = self.vehicles_in_proximity
        return self.colliding(states_in_proximity)

    @cached_property
    def vrus_colliding(self):
        states_in_proximity = self.vrus_in_proximity
        return self.colliding(states_in_proximity)

    @cached_property
    def vehicles_in_proximity(self):
        return in_proximity(
            self.ego_state, self.other_vehicle_states, self.proximity_threshold
        )

    @cached_property
    def vrus_in_proximity(self):
        return in_proximity(self.ego_state, self.vru_states, self.proximity_threshold)

    @cached_property
    def trajectory_linestring(self):
        return self.handler.trajectory_linestring

    @cached_property
    def trajectory_buffer(self):
        return self.handler.trajectory_buffer

    @cached_property
    def trajectory_front_buffer(self):
        width = self.ego.width
        polygon = shapely.buffer(
            self.trajectory_front_linestring, width / 2, cap_style="square"
        )
        return polygon

    @cached_property
    def trajectory_behind_buffer(self):
        width = self.ego.width
        polygon = shapely.buffer(
            self.trajectory_behind_linestring, width / 2, cap_style="square"
        )
        return polygon

    @cached_property
    def trajectory_front_linestring(self):

        return shapely.LineString(
            [
                state.position
                for state in self.ego.trajectory[
                    self.step : self.step + self.steps_ahead
                ]
            ]
        )

    @cached_property
    def trajectory_behind_linestring(self):
        return shapely.LineString(
            [
                state.position
                for state in self.ego.trajectory[
                    max(self.step - self.steps_ahead, 0) : self.step + 1
                ]
            ]
        )
