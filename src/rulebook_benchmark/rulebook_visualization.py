import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_topological_graph(
    G,
    figsize=(8, 6),
    node_size=500,
    layer_gap=1.5,
    horiz_gap=1.5,
    scenario_name=None,
    weights=None,
    rule_id_to_name=None,
):
    """
    Plot a DAG so that topologically earlier nodes are higher.
    Nodes at the same depth (no. of hops from a root) share the same row.
    Optional:
      - weights: {node_id: weight} -> darker node color for higher weights.
      - rule_id_to_name: {node_id: display_name} -> shown next to node.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a directed acyclic graph (DAG).")

    # Compute depth
    depth = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        depth[node] = max((depth[p] + 1 for p in preds), default=0)

    # Group by level
    levels = {}
    for n, d in depth.items():
        levels.setdefault(d, []).append(n)

    # Layout coordinates
    pos = {}
    for lvl, nodes in levels.items():
        n = len(nodes)
        xs = [i * horiz_gap - (n - 1) * horiz_gap / 2 for i in range(n)]
        y = -lvl * layer_gap
        for x, node in zip(xs, nodes):
            pos[node] = (x, y)

    # Normalize weights → alpha
    def normalize_colors(G, weights):
        if not weights:
            return {n: (0, 0, 0, 0.3) for n in G.nodes()}
        vals = list(weights.values()) + [0]  # include zero baseline
        mn, mx = min(vals), max(vals)
        colors = {}
        for n in G.nodes():
            w = weights.get(n, 0)
            if mx == mn:
                a = 0.3 if w == 0 else 1.0  # make weighted nodes dark, others light
            else:
                a = 0.3 + 0.7 * ((w - mn) / (mx - mn))
            colors[n] = (0, 0, 0, a)
        return colors

    colors = normalize_colors(G, weights)

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=10, width=1)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[colors[n] for n in G.nodes()],
        node_size=node_size,
        edgecolors="black",
    )
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=8)

    # Add rule names if provided
    if rule_id_to_name:
        for n, (x, y) in pos.items():
            if n in rule_id_to_name:
                plt.text(
                    x + 0.25,
                    y,
                    rule_id_to_name[n],
                    fontsize=7,
                    color="black",
                    va="center",
                )

    if scenario_name:
        plt.title(f"Rulebook Priority Graph for Scenario: {scenario_name}", pad=10)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_group_topological_graph(
    G,
    colors,
    figsize=(8, 6),
    node_size=500,
    layer_gap=1.5,
    horiz_gap=1.5,
    scenario_name=None,
    rule_id_to_name=None,
):
    """
    Plot a DAG so that topologically earlier nodes are higher.
    Nodes at the same depth (no. of hops from a root) share the same row.
    Optional:
      - weights: {node_id: weight} -> darker node color for higher weights.
      - rule_id_to_name: {node_id: display_name} -> shown next to node.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a directed acyclic graph (DAG).")

    # Compute depth
    depth = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        depth[node] = max((depth[p] + 1 for p in preds), default=0)

    # Group by level
    levels = {}
    for n, d in depth.items():
        levels.setdefault(d, []).append(n)

    # Layout coordinates
    pos = {}
    for lvl, nodes in levels.items():
        n = len(nodes)
        xs = [i * horiz_gap - (n - 1) * horiz_gap / 2 for i in range(n)]
        y = -lvl * layer_gap
        for x, node in zip(xs, nodes):
            pos[node] = (x, y)

    # Color node groups - colors = {rule_id:color}
    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=10, width=1)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[colors.get(n, (0, 0, 0, 0.3)) for n in G.nodes()],
        node_size=node_size,
        edgecolors="black",
    )
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=8)

    # Add rule names if provided
    if rule_id_to_name:
        for n, (x, y) in pos.items():
            if n in rule_id_to_name:
                plt.text(
                    x + 0.25,
                    y,
                    rule_id_to_name[n],
                    fontsize=7,
                    color="black",
                    va="center",
                )

    if scenario_name:
        plt.title(f"Rulebook Priority Graph for Scenario: {scenario_name}", pad=10)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_two_rulebooks_side_by_side(
    G1,
    G2,
    weights1=None,
    weights2=None,
    figsize=(12, 6),
    node_size=500,
    layer_gap=1.5,
    horiz_gap=1.5,
    x_offset=8,
    scenario_name=None,
    scenario_name2=None,
    rule_id_to_name=None,
):

    def topological_positions(G, x_shift=0):
        depth = {}
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            depth[node] = max((depth[p] + 1 for p in preds), default=0)

        levels = {}
        for n, d in depth.items():
            levels.setdefault(d, []).append(n)

        pos = {}
        for lvl, nodes in levels.items():
            n = len(nodes)
            xs = [x_shift + i * horiz_gap - (n - 1) * horiz_gap / 2 for i in range(n)]
            y = -lvl * layer_gap
            for x, node in zip(xs, nodes):
                pos[node] = (x, y)
        return pos

    def normalize_colors(G, weights):
        if not weights:
            return {n: (0, 0, 0, 0.3) for n in G.nodes()}
        vals = list(weights.values()) + [0]  # include zero baseline
        mn, mx = min(vals), max(vals)
        colors = {}
        for n in G.nodes():
            w = weights.get(n, 0)
            if mx == mn:
                a = 0.3 if w == 0 else 1.0  # make weighted nodes dark, others light
            else:
                a = 0.3 + 0.7 * ((w - mn) / (mx - mn))
            colors[n] = (0, 0, 0, a)
        return colors

    pos1 = topological_positions(G1, x_shift=-x_offset / 2)
    pos2 = topological_positions(G2, x_shift=+x_offset / 2)

    colors1 = normalize_colors(G1, weights1)
    colors2 = normalize_colors(G2, weights2)

    plt.figure(figsize=figsize)

    def draw_graph(G, pos, colors, label_offset=0.25):
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=10, width=1)
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=[colors[n] for n in G.nodes()],
            node_size=node_size,
            edgecolors="black",
        )
        nx.draw_networkx_labels(G, pos, font_color="white", font_size=8)

        # Add rule names if available
        if rule_id_to_name:
            for n, (x, y) in pos.items():
                if n in rule_id_to_name:
                    plt.text(
                        x + label_offset,
                        y,
                        rule_id_to_name[n],
                        fontsize=7,
                        color="black",
                        va="center",
                    )

    draw_graph(G1, pos1, colors1)
    draw_graph(G2, pos2, colors2)

    plt.axis("off")
    if scenario_name and scenario_name2:
        plt.suptitle(
            f"Left: {scenario_name}    |    Right: {scenario_name2}",
            y=1.02,
            fontsize=14,
        )
    elif scenario_name:
        plt.suptitle(f"{scenario_name}", y=1.02, fontsize=14)
    elif scenario_name2:
        plt.suptitle(f"{scenario_name2}", y=1.02, fontsize=14)

    plt.tight_layout()
    plt.show()
