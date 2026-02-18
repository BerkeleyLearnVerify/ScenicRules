import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import math
import random

import numpy as np
import scenic
from agent import Agent, AgentAction
from metadrive.engine.engine_utils import get_global_config
from metadrive.examples.ppo_expert import numpy_expert
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.utils.math import clip, norm
from scenic.core.simulators import Action
from scenic.simulators.metadrive import MetaDriveSimulator
from shapely.geometry import LineString, Point, Polygon

MAX_STEPS = 200
METADRIVE_PPO_PATH = "assets/expert_weights.npz" # could be downloaded from https://github.com/metadriverse/metadrive/blob/main/metadrive/examples/ppo_expert/expert_weights.npz
VERBOSITY = 1


class MetaDrivePolicyAgent(Agent):
    def __init__(self):
        self._expert_weights = None
        self._expert_observation = None
        self._trajectory_curr_idx = 0  # To keep track of trajectory progress
        self._trajectory_next_idx = 1  # To keep track of trajectory progress

    def run_step(self, actor, scenic_ego, trajectory):
        if VERBOSITY >= 2:
            print(
                f"previous throttle_brake: {actor.throttle_brake}, steer: {actor.steering}"
            )
        if actor is None:
            print("Actor is None!")
            return 0.0, 0.0, 0.0
        actions, obs = self.expert(
            actor, scenic_ego, trajectory, deterministic=True, need_obs=True
        )
        steer = actions[0]
        if actions[1] > 0:
            throttle = actions[1]
            brake = 0.0
        else:
            throttle = 0.0
            brake = -actions[1]
        return throttle, brake, steer

    def expert(
        self, vehicle, scenic_ego, trajectory, deterministic=False, need_obs=False
    ):
        """
        Replicate the expert function from metadrive.examples.ppo_expert
        """
        expert_obs_cfg = dict(
            lidar=dict(
                num_lasers=240,
                distance=50,
                num_others=4,
                gaussian_noise=0.0,
                dropout_prob=0.0,
            ),
            side_detector=dict(
                num_lasers=0, distance=50, gaussian_noise=0.0, dropout_prob=0.0
            ),
            lane_line_detector=dict(
                num_lasers=0, distance=20, gaussian_noise=0.0, dropout_prob=0.0
            ),
            random_agent_model=False,
        )
        origin_obs_cfg = vehicle.config.copy()

        if self._expert_weights is None:
            self._expert_weights = np.load(METADRIVE_PPO_PATH)
            config = get_global_config().copy()
            config["vehicle_config"].update(expert_obs_cfg)
            self._expert_observation = LidarStateObservation(config)
            # assert self._expert_observation.observation_space.shape[0] == 275, "Observation not match"

        vehicle.config.update(expert_obs_cfg)
        state = self._get_vehicle_state(vehicle, scenic_ego, trajectory)
        lidar = self._expert_observation.lidar_observe(vehicle)
        obs = np.concatenate((state, np.asarray(lidar)))
        vehicle.config.update(origin_obs_cfg)
        obs = self._obs_correction(obs)
        weights = self._expert_weights
        obs = obs.reshape(1, -1)
        x = (
            np.matmul(obs, weights["default_policy/fc_1/kernel"])
            + weights["default_policy/fc_1/bias"]
        )
        x = np.tanh(x)
        x = (
            np.matmul(x, weights["default_policy/fc_2/kernel"])
            + weights["default_policy/fc_2/bias"]
        )
        x = np.tanh(x)
        x = (
            np.matmul(x, weights["default_policy/fc_out/kernel"])
            + weights["default_policy/fc_out/bias"]
        )
        x = x.reshape(-1)
        mean, log_std = np.split(x, 2)
        if deterministic:
            return (mean, obs) if need_obs else mean
        std = np.exp(log_std)
        action = np.random.normal(mean, std)
        ret = action
        # ret = np.clip(ret, -1.0, 1.0) all clip should be implemented in env!
        return (ret, obs) if need_obs else ret

    def _obs_correction(self, obs):
        obs[15] = 1 - obs[15]
        obs[10] = 1 - obs[10]
        return obs

    def _get_vehicle_state(self, vehicle, scenic_ego, trajectory):
        """
        Assume no navigation module, so info requiring navigation module is filled with zero.
        Ego states: [
                    [Distance to left yellow Continuous line,
                    Distance to right Side Walk], if NOT use lane_line detector else:
                    [Side_detector cloud_points]

                    Difference of heading between ego vehicle and current lane,
                    Current speed,
                    Current steering,
                    Throttle/brake of last frame,
                    Steering of last frame,
                    Yaw Rate,

                     [Lateral Position on current lane.], if use lane_line detector, else:
                     [lane_line_detector cloud points]
                    ], dim >= 9
        Navi info: [
                    Projection of distance between ego vehicle and checkpoint on ego vehicle's heading direction,
                    Projection of distance between ego vehicle and checkpoint on ego vehicle's side direction,
                    Radius of the lane whose end node is the checkpoint (0 if this lane is straight),
                    Clockwise (1) or anticlockwise (0) (0 if lane is straight),
                    Angle of the lane (0 if lane is straight)
                   ] * 2, dim = 10
        Since agent observes current lane info and next lane info, and two checkpoints exist, the dimension of navi info
        is 10.
        :param vehicle: BaseVehicle
        :return: Vehicle State + Navigation information
        """
        info = []

        # Distance to left yellow Continuous line, Distance to right Side Walk
        lateral_to_left, lateral_to_right = self._get_distance_to_side(scenic_ego)
        info += [clip(lateral_to_left, 0.0, 1.0), clip(lateral_to_right, 0.0, 1.0)]
        if VERBOSITY >= 2:
            print(f"lateral_to_left: {info[0]}, lateral_to_right: {info[1]}")

        info += [
            # The angular difference between vehicle's heading and the lane heading at this location.
            self._get_angular_diff(vehicle, scenic_ego),
            # The velocity of target vehicle
            clip((vehicle.speed_km_h + 1) / (vehicle.max_speed_km_h + 1), 0.0, 1.0),
            # Current steering
            clip((vehicle.steering / vehicle.MAX_STEERING + 1) / 2, 0.0, 1.0),
            # The normalized actions at last steps
            clip((vehicle.last_current_action[1][0] + 1) / 2, 0.0, 1.0),
            clip((vehicle.last_current_action[1][1] + 1) / 2, 0.0, 1.0),
        ]
        if VERBOSITY >= 2:
            print(
                f"speed: {vehicle.speed_km_h}, steering: {vehicle.steering}, last_steering: {vehicle.last_current_action[1][0]}, last_throttle_brake: {vehicle.last_current_action[1][1]}"
            )

        # Current angular acceleration (yaw rate)
        heading_dir_last = vehicle.last_heading_dir
        heading_dir_now = vehicle.heading
        cos_beta = heading_dir_now.dot(heading_dir_last) / (
            norm(*heading_dir_now) * norm(*heading_dir_last)
        )
        beta_diff = np.arccos(clip(cos_beta, 0.0, 1.0))
        yaw_rate = beta_diff / 0.1
        info.append(clip(yaw_rate, 0.0, 1.0))
        if VERBOSITY >= 2:
            print(f"yaw_rate: {yaw_rate}")

        # Lateral Position on current lane
        info += [self._get_lateral_offset(scenic_ego)]
        if VERBOSITY >= 2:
            print(f"lateral_offset: {info[-1]}")

        # Navigation information
        checkpoint_info = self._get_checkpoints(vehicle, scenic_ego, trajectory)
        info += [
            checkpoint_info[0]
        ]  # [1]     # distance to current checkpoint projected on heading direction
        info += [
            checkpoint_info[1]
        ]  # [0.5]   # distance to current checkpoint projected on side direction
        info += [checkpoint_info[2]]  # [0]     # radius of current lane (0 if straight)
        info += [
            checkpoint_info[3]
        ]  # [0.5]   # bending direction of current lane (1 for clockwise, 0 for counterclockwise)
        info += [
            checkpoint_info[4]
        ]  # [0.5]   # angle of current lane (0.5 if straight)
        info += [
            checkpoint_info[5]
        ]  # [1]     # distance to next checkpoint projected on heading direction
        info += [
            checkpoint_info[6]
        ]  # [0.5]   # distance to next checkpoint projected on side direction
        info += [checkpoint_info[7]]  # [0]     # radius of next lane (0 if straight)
        info += [
            checkpoint_info[8]
        ]  # [0.5]   # bending direction of next lane (1 for clockwise, 0 for counterclockwise)
        info += [checkpoint_info[9]]  # [0.5]   # angle of next lane (0.5 if straight)

        ret = ret = np.asarray(info)
        return ret.astype(np.float32)

    def _get_distance_to_side(self, scenic_ego, total_width=10):
        """Metadrive src
        lateral_to_left, lateral_to_right, = vehicle.dist_to_left_side, vehicle.dist_to_right_side
        total_width = 100
        lateral_to_left /= total_width
        lateral_to_right /= total_width
        """
        """Scenic"""
        ego_position = Point(scenic_ego.position[0], scenic_ego.position[1])
        right_sidewalk = scenic_ego.laneGroup.curb.lineString
        lateral_to_right = ego_position.distance(right_sidewalk)
        lateral_to_left = total_width / 2
        if scenic_ego.laneGroup._opposite is not None:
            lateral_to_left = ego_position.distance(
                scenic_ego.laneGroup._opposite.polygon
            )
        if VERBOSITY >= 2:
            print(
                f"lateral_to_left: {lateral_to_left}, lateral_to_right: {lateral_to_right}"
            )
        lateral_to_right /= total_width
        lateral_to_left /= total_width
        return lateral_to_left, lateral_to_right

    def _get_angular_diff(self, vehicle, scenic_ego, k=3.0):
        ego_heading = vehicle.heading
        lane_heading_scenic = scenic_ego.laneGroup._defaultHeadingAt(
            scenic_ego.position
        )
        lane_heading = (
            math.cos(lane_heading_scenic.yaw + math.pi / 2),
            math.sin(lane_heading_scenic.yaw + math.pi / 2),
        )  # Scenic coordinate is 90 degrees ahead of metadrive coordinate
        dot = ego_heading[0] * lane_heading[0] + ego_heading[1] * lane_heading[1]
        cross = ego_heading[0] * lane_heading[1] - ego_heading[1] * lane_heading[0]
        angle = math.atan2(cross, dot)
        # lane heading is to the right of ego heading --> 1, lane heading is to the left of ego heading --> 0
        # use tanh to normalize angle to [0, 1], with more weight on small angle; k is the factor to adjust the curve, larger k means more weight on small angle
        if -math.pi <= angle <= -math.pi / 2:
            ret = 1.0
        elif -math.pi / 2 <= angle <= math.pi / 2:
            denom = math.tanh(k * (math.pi / 2))
            if denom == 0:
                s = math.tanh(k * angle)
            else:
                s = math.tanh(k * angle) / denom
            ret = (1.0 - s) / 2.0
        else:
            ret = 0.0
        # ret = (-angle + math.pi) / (2 * math.pi)  # linear mapping
        if VERBOSITY >= 2:
            print(
                f"ego_heading: {ego_heading}, lane_heading: {lane_heading, lane_heading_scenic.yaw}, angle: {math.degrees(angle)}, ret: {ret}"
            )
        return ret

    def _get_lateral_offset(self, scenic_ego, lane_width=4):
        ego_position = Point(scenic_ego.position[0], scenic_ego.position[1])
        centerline = scenic_ego.lane.centerline.lineString
        lateral_offset = self._signed_distance(ego_position, centerline)
        # lateral_offset = ego_position.distance(centerline) #unsigned distance
        ret = clip((lateral_offset * 2 / lane_width + 1.0) / 2.0, 0.0, 1.0)
        if VERBOSITY >= 2:
            print(f"lateral_offset: {lateral_offset}, ret: {ret}")
        return ret

    def _signed_distance(self, point: Point, line: LineString):
        """Compute signed distance from point to line (negative if point is to the right of line)."""
        # Project point onto line
        proj_dist = line.project(point)
        proj_point = line.interpolate(proj_dist)

        # Vector along the line (tangent)
        # Use a small step to approximate tangent
        delta = 1e-6
        d1 = max(proj_dist - delta, 0)
        d2 = min(proj_dist + delta, line.length)
        p1 = line.interpolate(d1)
        p2 = line.interpolate(d2)
        tangent = (p2.x - p1.x, p2.y - p1.y)

        # Vector from projected point to the external point
        vec = (point.x - proj_point.x, point.y - proj_point.y)

        # Cross product z-component to determine side
        cross = tangent[0] * vec[1] - tangent[1] * vec[0]

        # Signed distance
        dist = point.distance(proj_point)
        return dist if cross > 0 else -dist

    def _get_checkpoints(self, vehicle, scenic_ego, trajectory):
        info = []
        ego_position = Point(scenic_ego.position[0], scenic_ego.position[1])
        navi_dist = 50.0

        # Update trajectory index based on ego position
        if len(trajectory) == 1:
            self._trajectory_curr_idx = 0
            self._trajectory_next_idx = 0
        elif self._trajectory_curr_idx == len(trajectory) - 1:
            self._trajectory_curr_idx = len(trajectory) - 1
            self._trajectory_next_idx = len(trajectory) - 1
        else:
            next_lane = trajectory[self._trajectory_next_idx]
            if next_lane.polygon.contains(ego_position):
                self._trajectory_curr_idx += 1
                self._trajectory_next_idx += 1
                if self._trajectory_next_idx >= len(trajectory):
                    self._trajectory_next_idx = len(trajectory) - 1

        # Get current lane info
        curr_lane = trajectory[self._trajectory_curr_idx]
        curr_checkpoint = Point(
            curr_lane.centerline.lineString.coords[-1][0],
            curr_lane.centerline.lineString.coords[-1][1],
        )
        curr_heading, curr_rhs = self._point_to_vehicle_frame(
            curr_checkpoint, ego_position, vehicle.heading, navi_dist=navi_dist
        )
        info += [
            clip((curr_heading / navi_dist + 1) / 2, 0.0, 1.0)
        ]  # distance to current checkpoint projected on heading direction
        info += [
            clip((curr_rhs / navi_dist + 1) / 2, 0.0, 1.0)
        ]  # distance to current checkpoint projected on side direction
        curr_curvature = self._lane_curvature_score(curr_lane.centerline.lineString)
        info += [clip(curr_curvature, 0.0, 1.0)]  # curvature of current lane
        curr_dir, curr_angle = self._lane_bending_info(curr_lane.centerline.lineString)
        info += [
            clip((curr_dir + 1) / 2, 0.0, 1.0)
        ]  # bending direction of current lane (1 for clockwise, 0 for counterclockwise)
        info += [
            clip((np.rad2deg(curr_angle) / 135 + 1) / 2, 0.0, 1.0)
        ]  # angle of current lane

        # Get next lane info
        next_lane = trajectory[self._trajectory_next_idx]
        next_checkpoint = Point(
            next_lane.centerline.lineString.coords[-1][0],
            next_lane.centerline.lineString.coords[-1][1],
        )
        next_heading, next_rhs = self._point_to_vehicle_frame(
            next_checkpoint, ego_position, vehicle.heading, navi_dist=navi_dist
        )
        info += [
            clip((next_heading / navi_dist + 1) / 2, 0.0, 1.0)
        ]  # distance to next checkpoint projected on heading direction
        info += [
            clip((next_rhs / navi_dist + 1) / 2, 0.0, 1.0)
        ]  # distance to next checkpoint projected on side direction
        next_curvature = self._lane_curvature_score(next_lane.centerline.lineString)
        info += [clip(next_curvature, 0.0, 1.0)]  # curvature of next lane
        next_dir, next_angle = self._lane_bending_info(next_lane.centerline.lineString)
        info += [
            clip((next_dir + 1) / 2, 0.0, 1.0)
        ]  # bending direction of next lane (1 for clockwise, 0 for counterclockwise)
        info += [
            clip((np.rad2deg(next_angle) / 135 + 1) / 2, 0.0, 1.0)
        ]  # angle of next lane

        # Update the trajectory index for next step
        if self._trajectory_curr_idx == len(trajectory) - 1:
            self._trajectory_curr_idx = len(trajectory) - 1
            self._trajectory_next_idx = len(trajectory) - 1
        elif (curr_heading / navi_dist + 1) / 2 < 0.4 and (
            next_heading / navi_dist + 1
        ) / 2 > 0.5:
            self._trajectory_curr_idx += 1
            self._trajectory_next_idx += 1
            if self._trajectory_next_idx >= len(trajectory):
                self._trajectory_next_idx = len(trajectory) - 1

        if VERBOSITY >= 2:
            print(
                f"curr_checkpoint: {curr_checkpoint}, next_checkpoint: {next_checkpoint}"
            )
            print(f"vehicle pos: {ego_position}, heading: {vehicle.heading}")
            print(
                f"curr_heading: {curr_heading}, curr_rhs: {curr_rhs}, next_heading: {next_heading}, next_rhs: {next_rhs}"
            )
            print(f"curr_curvature: {curr_curvature}, next_curvature: {next_curvature}")
        if VERBOSITY >= 2:
            print(
                f"curr_idx: {self._trajectory_curr_idx}, next_idx: {self._trajectory_next_idx}"
            )
            print("checkpoint info:", info)
        if VERBOSITY >= 2:
            print(f"curr linestr: {curr_lane.centerline.lineString}")
            print(f"next linestr: {next_lane.centerline.lineString}")
        return info

    def _point_to_vehicle_frame(
        self, pt: Point, veh_pos: Point, heading_vec, navi_dist=50
    ):
        # normalize heading
        hx, hy = heading_vec
        norm = np.sqrt(hx**2 + hy**2)
        h_hat = np.array([hx, hy]) / norm

        # right-hand side unit vector (90° clockwise)
        r_hat = np.array([hy, -hx]) / norm

        # vector from vehicle to point
        d = np.array([pt.x - veh_pos.x, pt.y - veh_pos.y])
        d_norm = np.sqrt(d[0] ** 2 + d[1] ** 2)
        if d_norm > navi_dist:
            d = d / d_norm * navi_dist

        # projections
        heading_dist = np.dot(d, h_hat)
        rhs_dist = np.dot(d, r_hat)

        return heading_dist, rhs_dist

    def _curvature_from_three_points(self, p1, p2, p3):
        """Compute curvature (1/radius) given 3 points."""
        A = np.linalg.norm(p2 - p1)
        B = np.linalg.norm(p3 - p2)
        C = np.linalg.norm(p1 - p3)
        s = (A + B + C) / 2
        area = max(s * (s - A) * (s - B) * (s - C), 0)
        if area == 0:
            return 0
        area = np.sqrt(area)
        radius = (A * B * C) / (4 * area)
        return 0 if radius == 0 else 1 / radius

    def _lane_curvature_score(
        self, centerline: LineString, num_samples=50, k_max=0.014
    ):
        # k_max is the curvature of a circle with radius 70m, which is a gentle turn
        line = centerline

        # Sample points
        coords = [
            line.interpolate(i / num_samples, normalized=True).coords[0]
            for i in range(num_samples + 1)
        ]
        pts = np.array(coords)
        if len(pts) < 3:
            return 0.0

        curvatures = []
        for i in range(1, len(pts) - 1):
            curvatures.append(
                self._curvature_from_three_points(pts[i - 1], pts[i], pts[i + 1])
            )

        avg_curv = np.mean(curvatures)
        # Normalize
        score = min(1.0, avg_curv / k_max)
        return score

    def _lane_bending_info(self, centerline: LineString):
        if len(centerline.coords) < 3:
            return 0.0, 0.0
        t0 = self._tangent_at(centerline, 0.0)  # start tangent
        t1 = self._tangent_at(centerline, 1.0)  # end tangent

        delta = self._signed_angle(t0, t1)
        delta_abs = abs(delta)

        if abs(delta) < 1e-6:
            direction = 0
        else:
            direction = +1 if delta < 0 else -1  # +1 CW, -1 CCW

        return direction, delta_abs

    def _unit(self, vx, vy):
        n = math.hypot(vx, vy)
        return (0.0, 0.0) if n == 0 else (vx / n, vy / n)

    def _tangent_at(self, ls: LineString, s: float, eps_frac=1e-3):
        """
        Tangent (unit vector) at normalized position s in [0,1] along a LineString.
        """
        s = min(max(s, 0.0), 1.0)
        L = ls.length
        if L == 0:
            return (0.0, 0.0)

        # pick two nearby points around s to estimate tangent
        ds = max(eps_frac * L, 1e-9)
        a = max(0.0, min(L, s * L - ds / 2))
        b = max(0.0, min(L, s * L + ds / 2))
        pa = ls.interpolate(a)
        pb = ls.interpolate(b)
        vx, vy = (pb.x - pa.x), (pb.y - pa.y)
        return self._unit(vx, vy)

    def _signed_angle(self, v_from, v_to):
        """
        Signed angle from v_from to v_to in (-pi, pi], +CCW, -CW.
        """
        (x1, y1) = self._unit(*v_from)
        (x2, y2) = self._unit(*v_to)
        dot = x1 * x2 + y1 * y2
        det = x1 * y2 - y1 * x2  # z-component of 2D cross
        return math.atan2(det, max(-1.0, min(1.0, dot)))

    def reset_idx(self):
        self._trajectory_curr_idx = 0
        self._trajectory_next_idx = 1


class MetaDrivePolicyAction(AgentAction):
    def __init__(self, reset_idx=False):
        super().__init__()
        self.reset_idx = reset_idx

    def applyTo(self, agent, simulation):
        agent.update_actor()
        if self.reset_idx:
            agent.controller.reset_idx()
        throttle, brake, steer = agent.controller.run_step(
            agent.actor, agent, agent.trajectory
        )
        if VERBOSITY >= 2:
            print(f"throttle: {throttle}, brake: {brake}, steer: {steer}")
        agent.setThrottle(throttle)
        agent.setBraking(brake)
        agent.setSteering(steer)

