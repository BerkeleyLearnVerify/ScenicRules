import matplotlib.pyplot as plt
import matplotx
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle, Polygon
from matplotlib.ticker import MaxNLocator


def animate_realization(
    realization, dpi=100, interval=100, margin=50, buffer=0.5, show_step=True
):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    colors = {
        "Car": "blue",
        "Truck": "yellow",
        "Pedestrian": "purple",
        "Bicycle": "green",
    }
    patches = []
    dummy = np.zeros((3, 2))

    # Lane base layer
    for lane in realization.network.lanes:
        poly = Polygon(
            lane.polygon.exterior.coords[:-1],
            closed=True,
            facecolor="lightgray",
            edgecolor="black",
            alpha=0.5,
        )
        ax.add_patch(poly)

    # Ego lane highlight
    ego_lane_patch = Polygon(dummy, closed=True, facecolor="yellow", alpha=0.3)
    ax.add_patch(ego_lane_patch)

    # Object patches
    for obj in realization.objects:
        color = "red" if obj is realization.ego else colors.get(obj.object_type, "gray")
        poly = Polygon(dummy, closed=True, facecolor=color, alpha=0.6)
        ax.add_patch(poly)
        patches.append(poly)

    ego_buffer = Circle(
        (0, 0),
        radius=buffer,
        facecolor="none",
        edgecolor="cyan",
        linestyle="--",
        linewidth=1,
    )
    ax.add_patch(ego_buffer)

    # Arrows only for ego
    ego_arrow = [ax.arrow(0, 0, 0, 0, head_width=2, head_length=4, fc="red", ec="red")]
    lane_arrow = [
        ax.arrow(0, 0, 0, 0, head_width=2, head_length=4, fc="yellow", ec="yellow")
    ]
    text = ax.text(0, 0, "", fontsize=8, color="red")

    # Optional step display (in axes fraction coords so it stays in corner)
    if show_step:
        step_text = ax.text(
            0.02, 0.95, "", transform=ax.transAxes, fontsize=8, color="black"
        )
    else:
        step_text = None

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    def init():
        for patch in patches:
            patch.set_xy(dummy)
        ego_lane_patch.set_xy(dummy)
        if show_step:
            step_text.set_text("")
            return patches + [
                ego_lane_patch,
                ego_arrow[0],
                lane_arrow[0],
                ego_buffer,
                text,
                step_text,
            ]
        return patches + [ego_lane_patch, ego_arrow[0], lane_arrow[0], ego_buffer, text]

    def update(frame):
        ws = realization.get_world_state(min(frame, len(realization) - 1))
        ego = ws.ego_state
        ego_pos = ego.position
        ego_yaw = ego.orientation.yaw

        # Ego center buffer
        ego_buffer.center = ego_pos

        # Update ego lane highlight
        lane = getattr(ego, "lane", None)
        if lane is not None:
            ego_lane_patch.set_xy(lane.polygon.exterior.coords[:-1])
            lane_yaw = lane.orientation.value(ego_pos)
            lane_arrow[0].remove()
            ldx, ldy = 8 * np.cos(lane_yaw), 8 * np.sin(lane_yaw)
            lane_arrow[0] = ax.arrow(
                ego_pos[0],
                ego_pos[1],
                ldx,
                ldy,
                head_width=2,
                head_length=4,
                fc="yellow",
                ec="yellow",
            )
        else:
            ego_lane_patch.set_xy(dummy)
            lane_arrow[0].set_visible(False)

        # Update all object polygons
        for patch, state in zip(patches, ws.states):
            patch.set_xy(state.polygon.exterior.coords[:-1])

        # Ego heading arrow
        ego_arrow[0].remove()
        dx, dy = 8 * np.cos(ego_yaw), 8 * np.sin(ego_yaw)
        ego_arrow[0] = ax.arrow(
            ego_pos[0],
            ego_pos[1],
            dx,
            dy,
            head_width=2,
            head_length=4,
            fc="red",
            ec="red",
        )

        # Ego yaw text
        text.set_text(f"{ego_yaw:.2f} rad")
        text.set_position((ego_pos[0] + dx + 2, ego_pos[1] + dy + 2))

        # Viewport follows ego
        cx, cy = ego_pos
        ax.set_xlim(cx - margin, cx + margin)
        ax.set_ylim(cy - margin, cy + margin)

        ret = patches + [ego_lane_patch, ego_arrow[0], lane_arrow[0], ego_buffer, text]
        if show_step:
            step_text.set_text(f"Step: {frame}")
            ret.append(step_text)

        return ret

    max_frames = len(realization)
    anim = animation.FuncAnimation(
        fig, update, frames=max_frames, init_func=init, interval=interval, blit=True
    )
    return anim


def compare_realizations_gif(
    realization_model_pref,
    realization_human_pref,
    reason,
    agreement,
    dpi=100,
    interval=100,
    margin=50,
    buffer=0.5,
    show_step=True,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
    colors = {
        "Car": "blue",
        "Truck": "yellow",
        "Pedestrian": "purple",
        "Bicycle": "green",
    }
    patches1, patches2 = [], []
    dummy = np.zeros((3, 2))

    for lane in realization_human_pref.network.lanes:
        axes[0].add_patch(
            Polygon(
                lane.polygon.exterior.coords[:-1],
                closed=True,
                facecolor="lightgray",
                edgecolor="black",
                alpha=0.5,
            )
        )
    for lane in realization_model_pref.network.lanes:
        axes[1].add_patch(
            Polygon(
                lane.polygon.exterior.coords[:-1],
                closed=True,
                facecolor="lightgray",
                edgecolor="black",
                alpha=0.5,
            )
        )

    ego_lane_patch1 = Polygon(dummy, closed=True, facecolor="yellow", alpha=0.3)
    ego_lane_patch2 = Polygon(dummy, closed=True, facecolor="yellow", alpha=0.3)
    axes[0].add_patch(ego_lane_patch1)
    axes[1].add_patch(ego_lane_patch2)

    ego_buffer_human = Circle(
        (0, 0),
        radius=buffer,
        facecolor="none",
        edgecolor="cyan",
        linestyle="--",
        linewidth=1,
    )
    ego_buffer_model = Circle(
        (0, 0),
        radius=buffer,
        facecolor="none",
        edgecolor="cyan",
        linestyle="--",
        linewidth=1,
    )
    axes[0].add_patch(ego_buffer_human)
    axes[1].add_patch(ego_buffer_model)

    for obj in realization_human_pref.objects:
        color = (
            "red"
            if obj is realization_human_pref.ego
            else colors.get(obj.object_type, "gray")
        )
        poly = Polygon(dummy, closed=True, facecolor=color, alpha=0.6)
        axes[0].add_patch(poly)
        patches1.append(poly)
    for obj in realization_model_pref.objects:
        color = (
            "red"
            if obj is realization_model_pref.ego
            else colors.get(obj.object_type, "gray")
        )
        poly = Polygon(dummy, closed=True, facecolor=color, alpha=0.6)
        axes[1].add_patch(poly)
        patches2.append(poly)

    ego_arrow1 = [
        axes[0].arrow(0, 0, 0, 0, head_width=2, head_length=4, fc="red", ec="red")
    ]
    ego_arrow2 = [
        axes[1].arrow(0, 0, 0, 0, head_width=2, head_length=4, fc="red", ec="red")
    ]
    lane_arrow1 = [
        axes[0].arrow(0, 0, 0, 0, head_width=2, head_length=4, fc="yellow", ec="yellow")
    ]
    lane_arrow2 = [
        axes[1].arrow(0, 0, 0, 0, head_width=2, head_length=4, fc="yellow", ec="yellow")
    ]

    text1 = axes[0].text(0, 0, "", fontsize=8, color="red")
    text2 = axes[1].text(0, 0, "", fontsize=8, color="red")

    # Optional step display (in axes fraction coords so it stays in corner)
    if show_step:
        step_text1 = axes[0].text(
            0.02, 0.95, "", transform=axes[0].transAxes, fontsize=8, color="black"
        )
        step_text2 = axes[1].text(
            0.02, 0.95, "", transform=axes[1].transAxes, fontsize=8, color="black"
        )
    else:
        step_text1 = step_text2 = None

    axes[0].set_title("Human Preference - Agreement: " + str(agreement))
    axes[1].set_title("Model Preference - Reason: " + reason)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    def init():
        for patch in patches1 + patches2:
            patch.set_xy(dummy)
        ego_lane_patch1.set_xy(dummy)
        ego_lane_patch2.set_xy(dummy)
        ret = (
            patches1
            + patches2
            + [
                ego_lane_patch1,
                ego_lane_patch2,
                ego_arrow1[0],
                ego_arrow2[0],
                lane_arrow1[0],
                lane_arrow2[0],
                text1,
                text2,
            ]
        )
        if show_step:
            step_text1.set_text("")
            step_text2.set_text("")
            ret += [step_text1, step_text2]
        return ret

    def update(frame):
        # Human side
        ws1 = realization_human_pref.get_world_state(
            min(frame, len(realization_human_pref) - 1)
        )
        ego1 = ws1.ego_state
        ego_pos1 = ego1.position
        ego_yaw1 = ego1.orientation.yaw
        ego_buffer_human.center = ego_pos1
        lane1 = getattr(ego1, "lane", None)
        if lane1 is not None:
            ego_lane_patch1.set_xy(lane1.polygon.exterior.coords[:-1])
            lane_yaw1 = lane1.orientation.value(ego_pos1)
            lane_arrow1[0].remove()
            ldx1, ldy1 = 8 * np.cos(lane_yaw1), 8 * np.sin(lane_yaw1)
            lane_arrow1[0] = axes[0].arrow(
                ego_pos1[0],
                ego_pos1[1],
                ldx1,
                ldy1,
                head_width=2,
                head_length=4,
                fc="yellow",
                ec="yellow",
            )
        else:
            ego_lane_patch1.set_xy(dummy)
            lane_arrow1[0].set_visible(False)

        for patch, state in zip(patches1, ws1.states):
            patch.set_xy(state.polygon.exterior.coords[:-1])

        ego_arrow1[0].remove()
        dx1, dy1 = 8 * np.cos(ego_yaw1), 8 * np.sin(ego_yaw1)
        ego_arrow1[0] = axes[0].arrow(
            ego_pos1[0],
            ego_pos1[1],
            dx1,
            dy1,
            head_width=2,
            head_length=4,
            fc="red",
            ec="red",
        )
        text1.set_text(f"{ego_yaw1:.2f} rad")
        text1.set_position((ego_pos1[0] + dx1 + 2, ego_pos1[1] + dy1 + 2))
        axes[0].set_xlim(ego_pos1[0] - margin, ego_pos1[0] + margin)
        axes[0].set_ylim(ego_pos1[1] - margin, ego_pos1[1] + margin)

        # Model side
        ws2 = realization_model_pref.get_world_state(
            min(frame, len(realization_model_pref) - 1)
        )
        ego2 = ws2.ego_state
        ego_pos2 = ego2.position
        ego_yaw2 = ego2.orientation.yaw
        ego_buffer_model.center = ego_pos2
        lane2 = getattr(ego2, "lane", None)
        if lane2 is not None:
            ego_lane_patch2.set_xy(lane2.polygon.exterior.coords[:-1])
            lane_yaw2 = lane2.orientation.value(ego_pos2)
            lane_arrow2[0].remove()
            ldx2, ldy2 = 8 * np.cos(lane_yaw2), 8 * np.sin(lane_yaw2)
            lane_arrow2[0] = axes[1].arrow(
                ego_pos2[0],
                ego_pos2[1],
                ldx2,
                ldy2,
                head_width=2,
                head_length=4,
                fc="yellow",
                ec="yellow",
            )
        else:
            ego_lane_patch2.set_xy(dummy)
            lane_arrow2[0].set_visible(False)

        for patch, state in zip(patches2, ws2.states):
            patch.set_xy(state.polygon.exterior.coords[:-1])

        ego_arrow2[0].remove()
        dx2, dy2 = 8 * np.cos(ego_yaw2), 8 * np.sin(ego_yaw2)
        ego_arrow2[0] = axes[1].arrow(
            ego_pos2[0],
            ego_pos2[1],
            dx2,
            dy2,
            head_width=2,
            head_length=4,
            fc="red",
            ec="red",
        )
        text2.set_text(f"{ego_yaw2:.2f} rad")
        text2.set_position((ego_pos2[0] + dx2 + 2, ego_pos2[1] + dy2 + 2))
        axes[1].set_xlim(ego_pos2[0] - margin, ego_pos2[0] + margin)
        axes[1].set_ylim(ego_pos2[1] - margin, ego_pos2[1] + margin)

        # Step display
        ret = (
            patches1
            + patches2
            + [
                ego_lane_patch1,
                ego_lane_patch2,
                ego_arrow1[0],
                ego_arrow2[0],
                lane_arrow1[0],
                lane_arrow2[0],
                text1,
                text2,
            ]
        )
        if show_step:
            step_text1.set_text(f"Step: {frame}")
            step_text2.set_text(f"Step: {frame}")
            ret += [step_text1, step_text2]

        return ret

    max_frames = max(len(realization_human_pref), len(realization_model_pref))
    anim = animation.FuncAnimation(
        fig, update, frames=max_frames, init_func=init, interval=interval, blit=True
    )
    return anim


def animate_trajectory_with_violations(
    realization,
    evaluation_results,
    dpi=100,
    interval=100,
    margin=50,
    buffer=0.5,
    show_step=True,
):
    """
    Create a side-by-side animation showing trajectory and rule violations updating in real time.

    Parameters
    ----------
    realization : Realization
        The trajectory realization object containing objects and world states.
    evaluation_results : dict
        Dictionary mapping rule names to Result objects or numeric values.
        Can contain either Result objects (with violation_history) or simple numeric values.
    dpi : int, optional
        DPI for the figure (default: 100).
    interval : int, optional
        Animation interval in milliseconds (default: 100).
    margin : float, optional
        Camera follow margin around ego (default: 50).
    buffer : float, optional
        Ego buffer radius visualization (default: 0.5).
    show_step : bool, optional
        Whether to display step counter (default: True).

    Returns
    -------
    animation.FuncAnimation
        Animation object that can be saved or displayed.
    """
    fig, (ax_traj, ax_viol) = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)

    colors = {
        "Car": "blue",
        "Truck": "yellow",
        "Pedestrian": "purple",
        "Bicycle": "green",
    }
    patches = []
    dummy = np.zeros((3, 2))

    # === LEFT SIDE: TRAJECTORY VISUALIZATION ===
    # Lane base layer
    for lane in realization.network.lanes:
        poly = Polygon(
            lane.polygon.exterior.coords[:-1],
            closed=True,
            facecolor="lightgray",
            edgecolor="black",
            alpha=0.5,
        )
        ax_traj.add_patch(poly)

    # Ego lane highlight
    ego_lane_patch = Polygon(dummy, closed=True, facecolor="yellow", alpha=0.3)
    ax_traj.add_patch(ego_lane_patch)

    # Object patches
    for obj in realization.objects:
        color = "red" if obj is realization.ego else colors.get(obj.object_type, "gray")
        poly = Polygon(dummy, closed=True, facecolor=color, alpha=0.6)
        ax_traj.add_patch(poly)
        patches.append(poly)

    ego_buffer = Circle(
        (0, 0),
        radius=buffer,
        facecolor="none",
        edgecolor="cyan",
        linestyle="--",
        linewidth=1,
    )
    ax_traj.add_patch(ego_buffer)

    # Arrows only for ego
    ego_arrow = [
        ax_traj.arrow(0, 0, 0, 0, head_width=2, head_length=4, fc="red", ec="red")
    ]
    lane_arrow = [
        ax_traj.arrow(0, 0, 0, 0, head_width=2, head_length=4, fc="yellow", ec="yellow")
    ]
    text = ax_traj.text(0, 0, "", fontsize=8, color="red")

    # Optional step display
    if show_step:
        step_text = ax_traj.text(
            0.02, 0.95, "", transform=ax_traj.transAxes, fontsize=8, color="black"
        )
    else:
        step_text = None

    ax_traj.set_aspect("equal")
    ax_traj.set_xticks([])
    ax_traj.set_yticks([])
    ax_traj.axis("off")
    ax_traj.set_title("Trajectory")

    # === RIGHT SIDE: RULE VIOLATIONS VISUALIZATION ===
    # Extract violation histories from evaluation results
    rule_names = []
    violation_histories = []

    for rule_name, result in evaluation_results.items():
        rule_names.append(rule_name)
        # Handle both Result objects and numeric values
        if hasattr(result, "violation_history"):
            violation_histories.append(result.violation_history)
        else:
            # If it's a numeric value, create a constant history
            violation_histories.append([result] * len(realization))

    # Initialize violation lines for matplotx
    max_steps = len(realization)
    violation_lines = []

    # Use matplotx for clean line drawing
    with matplotx.line_labels():
        for i, (rule_name, viol_hist) in enumerate(
            zip(rule_names, violation_histories)
        ):
            # Pad violation history to max_steps if needed
            if len(viol_hist) < max_steps:
                viol_hist = list(viol_hist) + [viol_hist[-1] if viol_hist else 0] * (
                    max_steps - len(viol_hist)
                )

            (line,) = ax_viol.plot([], [], label=rule_name, linewidth=2)
            violation_lines.append((line, viol_hist))

        ax_viol.legend(loc="upper left", fontsize=8)

    ax_viol.set_xlabel("Time Step", fontsize=10)
    ax_viol.set_ylabel("Violation", fontsize=10)
    ax_viol.set_title("Rule Violations Over Time")
    ax_viol.grid(True, alpha=0.3)
    ax_viol.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Set initial limits
    ax_viol.set_xlim(0, max_steps - 1)
    min_viol = (
        min(min(hist) for hist in violation_histories) if violation_histories else 0
    )
    max_viol = (
        max(max(hist) for hist in violation_histories) if violation_histories else 1
    )
    margin_viol = (max_viol - min_viol) * 0.1 if max_viol > min_viol else 0.5
    ax_viol.set_ylim(min_viol - margin_viol, max_viol + margin_viol)

    def init():
        # Trajectory side
        for patch in patches:
            patch.set_xy(dummy)
        ego_lane_patch.set_xy(dummy)

        # Violations side
        for line, _ in violation_lines:
            line.set_data([], [])

        ret = patches + [ego_lane_patch, ego_arrow[0], lane_arrow[0], ego_buffer, text]
        if show_step:
            step_text.set_text("")
            ret.append(step_text)
        ret.extend([line for line, _ in violation_lines])
        return ret

    def update(frame):
        current_step = min(frame, len(realization) - 1)

        # === UPDATE TRAJECTORY ===
        ws = realization.get_world_state(current_step)
        ego = ws.ego_state
        ego_pos = ego.position
        ego_yaw = ego.orientation.yaw

        # Ego center buffer
        ego_buffer.center = ego_pos

        # Update ego lane highlight
        lane = getattr(ego, "lane", None)
        if lane is not None:
            ego_lane_patch.set_xy(lane.polygon.exterior.coords[:-1])
            lane_yaw = lane.orientation.value(ego_pos)
            lane_arrow[0].remove()
            ldx, ldy = 8 * np.cos(lane_yaw), 8 * np.sin(lane_yaw)
            lane_arrow[0] = ax_traj.arrow(
                ego_pos[0],
                ego_pos[1],
                ldx,
                ldy,
                head_width=2,
                head_length=4,
                fc="yellow",
                ec="yellow",
            )
        else:
            ego_lane_patch.set_xy(dummy)
            lane_arrow[0].set_visible(False)

        # Update all object polygons
        for patch, state in zip(patches, ws.states):
            patch.set_xy(state.polygon.exterior.coords[:-1])

        # Ego heading arrow
        ego_arrow[0].remove()
        dx, dy = 8 * np.cos(ego_yaw), 8 * np.sin(ego_yaw)
        ego_arrow[0] = ax_traj.arrow(
            ego_pos[0],
            ego_pos[1],
            dx,
            dy,
            head_width=2,
            head_length=4,
            fc="red",
            ec="red",
        )

        # Ego yaw text
        text.set_text(f"{ego_yaw:.2f} rad")
        text.set_position((ego_pos[0] + dx + 2, ego_pos[1] + dy + 2))

        # Viewport follows ego
        cx, cy = ego_pos
        ax_traj.set_xlim(cx - margin, cx + margin)
        ax_traj.set_ylim(cy - margin, cy + margin)

        # === UPDATE VIOLATIONS ===
        for line, viol_hist in violation_lines:
            # Update line data up to current step
            x_data = np.arange(min(frame + 1, len(viol_hist)))
            y_data = np.array(viol_hist[: frame + 1])
            line.set_data(x_data, y_data)

        # Update x-axis to follow current frame
        ax_viol.set_xlim(0, max(current_step + 1, 5))

        ret = patches + [ego_lane_patch, ego_arrow[0], lane_arrow[0], ego_buffer, text]
        if show_step:
            step_text.set_text(f"Step: {frame}")
            ret.append(step_text)
        ret.extend([line for line, _ in violation_lines])

        return ret

    max_frames = len(realization)
    anim = animation.FuncAnimation(
        fig, update, frames=max_frames, init_func=init, interval=interval, blit=True
    )
    return anim
