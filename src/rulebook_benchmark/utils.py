import math

import numpy as np
import shapely


def normalize_angle(angle):
    while angle > math.pi:
        angle -= math.tau
    while angle < -math.pi:
        angle += math.tau
    assert -math.pi <= angle <= math.pi
    return angle


def angle_between(v1, v2):
    x, y = v1
    x2, y2 = v2
    return normalize_angle(math.atan2(y2, x2) - math.atan2(y, x))


def project_vector(v1, v2):  # project v1 onto v2
    v2_norm = np.linalg.norm(v2)
    if v2_norm == 0:
        return np.zeros_like(v1)
    return (np.dot(v1, v2) / v2_norm**2) * v2


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def polygon_distance(x_state, y_state):
    x_polygon = x_state.polygon
    y_polygon = y_state.polygon
    return shapely.distance(x_polygon, y_polygon)


def intersects(x_state, y_state):
    x_polygon = x_state.polygon
    y_polygon = y_state.polygon
    return shapely.intersects(x_polygon, y_polygon)


def in_proximity(ego_state, object_states, threshold):
    if len(object_states) == 0:
        return []
    ego = ego_state.object
    radius = ego.radius + threshold
    ego_pos = ego_state.position
    adv_positions = np.array([v.position for v in object_states])
    adv_radii = np.array([v.object.radius for v in object_states])
    distances = np.linalg.norm(adv_positions - ego_pos, axis=1)
    mask = distances < (radius + adv_radii)
    return [v for v, m in zip(object_states, mask) if m]


def project_polygon(vertices, axis):
    """Project polygon onto axis and return [min, max]."""
    dots = [np.dot(v, axis) for v in vertices]
    return min(dots), max(dots)


def continuous_ttc(ego_vertices, adv_vertices, v_rel, threshold):
    """
    Returns earliest collision time in [0, threshold] or None if no collision.
    """
    # Build candidate separating axes = edge normals of both polygons
    axes = []
    for poly in [ego_vertices, adv_vertices]:
        for i in range(len(poly)):
            p1, p2 = np.array(poly[i]), np.array(poly[(i + 1) % len(poly)])
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])  # perpendicular
            if np.linalg.norm(normal) > 1e-9:
                normal = normal / np.linalg.norm(normal)
                axes.append(normal)

    t_enter, t_exit = 0.0, threshold

    for axis in axes:
        # Project ego (stationary)
        e_min, e_max = project_polygon(ego_vertices, axis)
        # Project adv at t=0
        a_min0, a_max0 = project_polygon(adv_vertices, axis)
        v_proj = np.dot(v_rel, axis)

        if abs(v_proj) < 1e-12:
            # No relative motion along this axis: must overlap at t=0
            if a_max0 < e_min or e_max < a_min0:
                return None
            else:
                continue

        # Times when projections *start* and *stop* overlapping
        t0 = (e_min - a_max0) / v_proj
        t1 = (e_max - a_min0) / v_proj
        t_axis_enter, t_axis_exit = min(t0, t1), max(t0, t1)

        # Shrink global window
        t_enter = max(t_enter, t_axis_enter)
        t_exit = min(t_exit, t_axis_exit)
        if t_enter > t_exit:
            return None  # separating axis → no collision

    if t_enter < 0:
        t_enter = 0
    return t_enter if t_enter <= threshold else None


def kinetic_energy_loss(
    ego_velocity_before,
    ego_velocity_after,
    adv_velocity_before,
    adv_velocity_after,
    ego_mass,
    adv_mass,
    VRU=False,
):
    ego_loss = (
        0.5
        * ego_mass
        * (
            np.linalg.norm(ego_velocity_before) ** 2
            - np.linalg.norm(ego_velocity_after) ** 2
        )
    )
    adv_loss = (
        0.5
        * adv_mass
        * (
            np.linalg.norm(adv_velocity_before) ** 2
            - np.linalg.norm(adv_velocity_after) ** 2
        )
    )

    if VRU:
        # If VRU is involved, we check how much kinetic energy the ego lost, and how much the VRU gained
        adv_loss = -adv_loss

    return ego_loss + adv_loss


def momentum_loss(
    ego_velocity_before,
    ego_velocity_after,
    adv_velocity_before,
    adv_velocity_after,
    ego_mass,
    adv_mass,
):
    ego_momentum_loss = np.linalg.norm(
        ego_mass * (ego_velocity_after - ego_velocity_before)
    )
    adv_momentum_loss = np.linalg.norm(
        adv_mass * (adv_velocity_after - adv_velocity_before)
    )

    return ego_momentum_loss + adv_momentum_loss


def generalized_collision(
    handler,
    collision_timeline,
    states,
    step,
    ego_mass,
    adv_mass,
    momentum,
    epsilon=1e-6,
    VRU=False,
):
    violation = 0
    for state in states:
        uid = state.uid
        if uid not in collision_timeline:
            continue
        collisions = collision_timeline[uid]
        for collision in collisions:
            collision_start, collision_end = collision
            if collision_start == 0:
                before_collision = collision_start
            else:
                before_collision = collision_start - 1
            if before_collision > step:
                break
            elif before_collision < step:
                continue
            else:
                prev_state = handler(before_collision).ego_state
                after_state = handler(collision_end).ego_state

                adv_prev_state = handler(before_collision).world_state[uid]
                adv_after_state = handler(collision_end).world_state[uid]

                if momentum:
                    curr_violation = max(
                        0,
                        momentum_loss(
                            ego_velocity_before=prev_state.velocity,
                            ego_velocity_after=after_state.velocity,
                            adv_velocity_before=adv_prev_state.velocity,
                            adv_velocity_after=adv_after_state.velocity,
                            ego_mass=ego_mass,
                            adv_mass=adv_mass,
                        ),
                    )

                else:
                    curr_violation = max(
                        0,
                        kinetic_energy_loss(
                            ego_velocity_before=prev_state.velocity,
                            ego_velocity_after=after_state.velocity,
                            adv_velocity_before=adv_prev_state.velocity,
                            adv_velocity_after=adv_after_state.velocity,
                            ego_mass=ego_mass,
                            adv_mass=adv_mass,
                            VRU=VRU,
                        ),
                    )

                curr_violation = max(
                    curr_violation, epsilon
                )  # ensure non-zero violation for any collision
                violation += curr_violation

    return violation


def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def lines_intersect(p1, p2, q1, q2):
    r = p2 - p1
    s = q2 - q1
    denom = cross2d(r, s)
    if abs(denom) < 1e-9:  # parallel
        return False, (None, None)
    t = cross2d(q1 - p1, s) / denom
    u = cross2d(q1 - p1, r) / denom
    return True, (t, u)


def early_ttc(ego_pos, ego_vel, adv_pos, adv_vel, threshold, times=3):
    horizon = threshold * times

    ego_end = ego_pos + ego_vel * horizon
    adv_end = adv_pos + adv_vel * horizon

    intersect, (t, u) = lines_intersect(ego_pos, ego_end, adv_pos, adv_end)
    if not intersect:
        return False  # no intersection ever

    ego_vel_normalized = normalize_vector(ego_vel)
    ego_to_adv = normalize_vector(adv_pos - ego_pos)
    projection = np.dot(ego_vel_normalized, ego_to_adv)
    # if intersection happens after horizon, skip expensive TTC
    if t > horizon or u > horizon or t < 0 or u < 0 or projection <= 0:
        return False

    return True  # possible interaction, run continuous_ttc
