# import json
import numpy as np
import orjson
import shapely
from cached_property import cached_property
from scenic.core.vectors import Vector

from reasonable_crowd.roads import OrientationVectorPlaceholder
from rulebook_benchmark.realization import Realization, RealizationObject, State

# example trajectory data


class ReasonableCrowdObject(RealizationObject):
    def __init__(self, object_id, dimensions, object_type, base_footprint=None):
        super().__init__(
            object_id, dimensions, object_type, base_footprint=base_footprint
        )
        if object_type == "ego" or object_type == "vehicle":
            self.object_type = "Car"
        elif object_type == "pedestrian":
            self.object_type = "Pedestrian"
        else:
            raise ValueError(f"Unknown object type: {object_type}")


class ReasonableCrowdTrajectoryParser:
    def __init__(self, path, step_size=100000):
        with open(path, "rb") as f:  # must open in binary mode
            self.trajectory_data = orjson.loads(f.read())
        self.step_size = step_size

        self._preprocess_trajectory_data()

    def _preprocess_trajectory_data(self):
        # aggregate each object into a list of states
        self.object_states = {}
        max_timestamps = {}
        # print(len(self.trajectory_data))

        for state in self.trajectory_data:
            obj_id = state["id"]
            timestamp = state["timestamp"]
            modulus = timestamp % self.step_size
            if modulus != 0 and modulus != 1 and modulus != -1:
                continue
            if obj_id not in self.object_states:
                self.object_states[obj_id] = {}
                max_timestamps[obj_id] = 0
            self.object_states[obj_id][state["timestamp"]] = state
            max_timestamps[obj_id] = max(max_timestamps[obj_id], state["timestamp"])

        max_end_timestamp = min(max_timestamps.values())

        objects = []
        for obj, states in self.object_states.items():
            state_list = list(states.values())
            footprint = state_list[0]["footprint"]
            heading = state_list[0]["heading_radians"]
            position = np.array([state_list[0]["x_meters"], state_list[0]["y_meters"]])
            footprint_arr = np.array(footprint)
            # Translate so position is at origin
            translated = footprint - position

            # Rotate by negative heading to remove orientation
            R = np.array(
                [
                    [np.cos(-heading), -np.sin(-heading)],
                    [np.sin(-heading), np.cos(-heading)],
                ]
            )
            base_footprint = translated @ R.T

            polygon = shapely.Polygon(footprint)
            base_polygon = shapely.affinity.rotate(
                polygon, -heading, origin=polygon.centroid, use_radians=True
            )
            width = base_polygon.bounds[3] - base_polygon.bounds[1]
            length = base_polygon.bounds[2] - base_polygon.bounds[0]
            dimensions = (length, width)
            object_type = state_list[0]["type"]
            obj_id = state_list[0]["id"]
            assert obj_id != 0
            if obj_id == -1:
                obj_id = 0

            obj = ReasonableCrowdObject(
                obj_id, dimensions, object_type, base_footprint=base_footprint
            )

            timestamps = np.array([s["timestamp"] for s in state_list])

            # pull all numeric attributes into arrays
            xs = np.array([s["x_meters"] for s in state_list])
            ys = np.array([s["y_meters"] for s in state_list])
            xvs = np.array([s["x_velocity_meters_per_second"] for s in state_list])
            yvs = np.array([s["y_velocity_meters_per_second"] for s in state_list])
            headings = np.array([s["heading_radians"] for s in state_list])

            # query times
            query_times = np.arange(0, max_end_timestamp + 1, self.step_size)

            # interpolate everything at once
            xs_interp = np.interp(query_times, timestamps, xs)
            ys_interp = np.interp(query_times, timestamps, ys)
            xvs_interp = np.interp(query_times, timestamps, xvs)
            yvs_interp = np.interp(query_times, timestamps, yvs)
            headings = np.unwrap([s["heading_radians"] for s in state_list])
            headings_interp = np.interp(query_times, timestamps, headings)
            headings_interp = (headings_interp + np.pi) % (2 * np.pi) - np.pi

            # build trajectory list (still Python dicts, but now one loop only)
            for step, (t, x, y, xv, yv, h) in enumerate(
                zip(
                    query_times,
                    xs_interp,
                    ys_interp,
                    xvs_interp,
                    yvs_interp,
                    headings_interp,
                )
            ):
                state = State(
                    obj,
                    np.array([x, y]),
                    np.array([xv, yv]),
                    OrientationVectorPlaceholder(h),
                    step,
                )

                obj.trajectory.append(state)

            objects.append(obj)

        self.objects = objects

    def create_realization(self):
        realization = Realization()
        realization.objects = self.objects
        return realization


def parse_trajectory(path, step_size):
    parser = ReasonableCrowdTrajectoryParser(path, step_size)
    return parser.create_realization()
