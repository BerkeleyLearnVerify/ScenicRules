# TODO: change naming of class methods
import math

import numpy as np
import shapely
from cached_property import cached_property
from shapely import affinity

from rulebook_benchmark.utils import intersects, polygon_distance

DELTA = 0.1


class Realization:
    """
    A Realization represents a single scenario execution, containing the full trajectories of all objects and the road network.
        - ego_index: index of the ego object in the objects list
        - delta: time step between states in seconds
        - proximity_threshold: distance threshold for determining which objects are "close" to the ego and should be considered for collision checking and other interactions
        - isScenic: whether this realization is from a Scenic scenario (True). This is currently used to determine whether we should rotate the coordinate system to be 0 degrees facing east (Scenic north-facing) or keep it as is (Reasonable Crowd).   
    """
    def __init__(self, ego_index: int = 0, delta: float = DELTA, proximity_threshold: float = 3, isScenic: bool = True):
        self.network = None
        self.objects = None
        self.ego_index = ego_index
        self.delta = delta
        self.proximity_threshold = proximity_threshold
        self.isScenic = isScenic

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
        """Get WorldState for a given step by aggregating the states of all objects at that step."""
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
        """Get full trajectory of all objects as a list of WorldStates, one per step."""
        trajectory = []
        for i in range(len(self)):
            states = self.get_world_state(i)
            trajectory.append(states)
        return trajectory

    @property
    def other_objects(self):
        """Get list of all objects except the ego."""
        return self.objects[: self.ego_index] + self.objects[self.ego_index + 1 :]

    @cached_property
    def vehicles(self):
        """Get list of all vehicle objects (Cars and Trucks). Make sure your custom Scenic classes end with "Car" or "Truck" for this to work properly."""
        vehicles = []
        for obj in self.objects:
            # match with regex to see if it ends with "Car" or "Truck" to be more robust to different naming conventions
            if obj.object_type.endswith("Car") or obj.object_type.endswith("Truck"):
                vehicles.append(obj)
        return vehicles

    @property
    def other_vehicles(self):
        """Get list of all vehicle objects except the ego."""
        ego_vidx = self.vehicles.index(self.ego)
        return self.vehicles[: ego_vidx] + self.vehicles[ego_vidx + 1 :]

    @cached_property
    def vrus(self):
        """Get list of all vulnerable road user objects (Pedestrians and Bicycles). Make sure your custom Scenic classes end with "Pedestrian" or "Bicycle" for this to work properly."""
        VRUs = []
        for obj in self.objects:
            if obj.object_type.endswith("Pedestrian") or obj.object_type.endswith(
                "Bicycle"
            ):
                VRUs.append(obj)
        return VRUs


class RealizationObject:
    """
    A RealizationObject represents a single object in the scenario, containing its dimensions, type, and trajectory.
    
    - object_id: unique identifier for the object
    - dimensions: (length, width) tuple representing the object's size in meters
    - object_type: string representing the type of object (e.g. "Car", "Pedestrian", etc.)
    - base_footprint: optional numpy array of local coordinates of the object's footprint corners, centered at the object's position and oriented with the object's heading. If not provided, the footprint will be assumed to be a rectangle defined by the dimensions, centered at the position and oriented with the heading.
    """
    def __init__(self, object_id: int, dimensions: tuple, object_type: str, base_footprint: np.ndarray = None):
        self.uid = object_id
        self.dimensions = dimensions
        self.length = dimensions[0]
        self.width = dimensions[1]
        self.base_footprint = base_footprint

        self.object_type = object_type
        self.trajectory = []
        
        if self.base_footprint is not None:
            self.base_polygon = shapely.Polygon(self.base_footprint)
        else:
            # create base footprint as rectangle centered at origin, oriented with x-axis, with given length and width
            hl = self.length / 2
            hw = self.width / 2
            self.base_footprint = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
            self.base_polygon = shapely.Polygon(self.base_footprint)
            

    @cached_property
    def radius(self):
        return math.sqrt(self.width**2 + self.length**2) / 2

    def get_state(self, step):
        try:
            return self.trajectory[step]
        except IndexError:
            raise Exception(f"Error: Step {step} not found in object trajectory")

class State:
    """A State represents the state of a single object at a single time step, containing its position, velocity, and orientation.
    
        - obj: the RealizationObject this state corresponds to
        - position: numpy array of shape (2,) representing the (x, y) position of the object in meters
        - velocity: numpy array of shape (2,) representing the (vx, vy) velocity of the object in meters per second
        - orientation: can be a Scenic Orientation, OrientationVectorPlaceholder or any object that has a _trimeshEulerAngles method and a yaw attribute (e.g. a custom class we define that wraps the raw heading angle and provides these properties). The orientation should represent the heading of the object in radians, where 0 radians means facing east (positive x direction) and positive rotation is counterclockwise.
        - step: integer representing the time step index
        - steer: optional float representing the steering angle of the object (for vehicles)
        - throttle: optional float representing the throttle of the object (for vehicles)
        - brake: optional float representing the brake of the object (for vehicles)
    """
    def __init__(
        self,
        obj: RealizationObject,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation, # can be a Scenic Orientation, OrientationVectorPlaceholder or any object that has a _trimeshEulerAngles method and a yaw attribute (e.g. a custom class we define that wraps the raw heading angle and provides these properties)
        step: int,
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
        """Compute the world-coordinate polygon representing the object's footprint at this state."""
        cx, cy = self.position
        yaw = self.orientation.yaw
        poly = affinity.rotate(self.object.base_polygon, yaw, origin=(0, 0), use_radians=True)
        poly = affinity.translate(poly, xoff=cx, yoff=cy)
        return poly

        # Old implementation
        if self.object.base_footprint is not None:
            # use base footprint if provided
            base_footprint = self.object.base_footprint

    # radians

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
    """A WorldState represents the state of the entire world at a single time step, containing the states of all objects."""
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
    """A VariableHandler is responsible for computing and caching all the variables needed to evaluate the rules at each time step. It takes in a Realization and provides methods to get the relevant variables for any given step. Your rules will call these methods to get the variables they need for their violation functions. The VariableHandler will compute and cache these variables on demand, so that if multiple rules need the same variable at the same step, it will only be computed once."""
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
        """Get the VariablePool for a given step, computing it if it hasn't been computed before. The VariablePool will compute and cache all relevant variables for that step when it is initialized."""
        if step not in self._pools:
            self._pools[step] = VariablePool(
                step, self, self.proximity_threshold, **kwargs
            )

        # self._pools.pop(step - 3, None)  # free memory by removing pools for steps that are no longer needed
        return self._pools[step]

    @cached_property
    def trajectory_linestring(self):
        """Return a shapely LineString representing the trajectory of the ego object over the entire realization."""
        return shapely.LineString([state.position for state in self.ego.trajectory])

    @cached_property
    def isScenic(self):
        return self.realization.isScenic

    @cached_property
    def other_objects(self):
        return self.realization.other_objects

    @cached_property
    def trajectory_buffer(self):
        """Return a shapely Polygon representing the buffer around the ego's trajectory, which can be used for efficient collision checking and proximity queries."""
        width = self.ego.width
        polygon = shapely.buffer(
            self.trajectory_linestring, width / 2, cap_style="square"
        )
        polygon = polygon.union(self.ego.trajectory[-1].polygon)
        polygon = polygon.union(self.ego.trajectory[0].polygon)
        return polygon

    @cached_property
    def collision_timeline(self):
        """Compute a dictionary mapping each object UID to a list of (start_step, end_step) tuples representing the time intervals during which that object is colliding with the ego. This can be used for efficiently checking if an object is colliding at a given step without having to recompute the collision at every step."""
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
    """VariablePool computes and caches all the variables needed to evaluate the rules at a specific time step. It is initialized with a reference to the VariableHandler, which it can use to access the Realization and other cached properties. When a VariablePool is initialized for a given step, it will compute all the relevant variables for that step and cache them as properties, so that they can be accessed efficiently by the rule violation functions. See rule_functions.py for examples of how to access these variables in your rule violation functions."""
    def __init__(self, step, handler, proximity_threshold, steps_ahead=None):
        self.handler = handler
        self.realization = self.handler.realization
        self.other_objects = self.handler.other_objects
        self.objects = self.handler.objects
        self.step = step
        self.world_state = self.realization.get_world_state(step)
        self.ego = self.realization.ego
        self.ego_state = self.world_state.ego_state
        self.vehicle_states = self.world_state.other_vehicle_states
        self.vru_states = self.world_state.vru_states
        self.proximity_threshold = proximity_threshold
        self._distances = {}
        self._lazy_distances = {}
        self.steps_ahead = len(self.realization) if steps_ahead is None else steps_ahead

    @cached_property
    def ego_state(self):
        return self.ego.get_state(self.step)

    def colliding(self, states):
        """Given a list of states, return a dictionary mapping the UIDs of the states that are colliding with the ego at this step to their corresponding State objects."""
        colliding = {}
        for state in states:
            if intersects(self.ego_state, state):
                colliding[state.uid] = state
        return colliding

    def in_proximity(self, ego_state, object_states, threshold):
        """Given a list of states, return a list of the states that are within the given distance threshold of the ego at this step."""
        if len(object_states) == 0:
            return []
        ego = ego_state.object
        radius = ego.radius + threshold
        ego_pos = ego_state.position
        adv_positions = np.array([v.position for v in object_states])
        adv_radii = np.array([v.object.radius for v in object_states])
        distances = np.linalg.norm(adv_positions - ego_pos, axis=1)
        mask = distances <= (radius + adv_radii)
        return [v for v, m in zip(object_states, mask) if m]

    def distance(self, other_state):
        """Compute the distance between the ego and another state, using the cached polygon distance if it has already been computed for this state, or computing and caching it if not."""
        uid = other_state.uid
        if uid not in self._distances:
            self._distances[uid] = polygon_distance(self.ego_state, other_state)
        return self._distances[uid]

    def center_distance(self, other_state):
        """Compute the distance between the centers of the ego and the other state, which is a cheaper computation than polygon distance and can be used as a preliminary check before computing polygon distance if needed."""
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
        return self.in_proximity(
            self.ego_state, self.vehicle_states, self.proximity_threshold
        )

    @cached_property
    def vrus_in_proximity(self):
        return self.in_proximity(
            self.ego_state, self.vru_states, self.proximity_threshold
        )

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
