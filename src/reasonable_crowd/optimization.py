import itertools
import math
import random
from collections import defaultdict

import networkx as nx
import numpy as np
from tqdm import tqdm

from reasonable_crowd.evaluation import (
    evaluate_rule_with_cache,
    evaluate_rulebook_with_cache,
)


def cache_rule_evaluations(
    rbook,
    rule_id_to_params,
    rule_id_to_values,
    X,
    y,
    rule_parameter_result_dict,
    trajectories_dict,
    verbose=False,
):
    default_config = rbook.get_config()
    rule_id_to_rule = rbook.rule_id_to_rule
    pbar = tqdm(
        total=len(rule_id_to_rule), desc="Caching rule evaluations", leave=False
    )

    for rule_id, r in rule_id_to_rule.items():

        if rule_id in rule_parameter_result_dict:  # rule already cached
            continue

        if verbose:
            print(f"\n[optimize_rules_grid] Evaluating {rule_id} (id={rule_id})")

        param_names = rule_id_to_params.get(rule_id, [])
        value_lists = rule_id_to_values.get(rule_id, {})

        # Case 1: rule has tunable params → grid search
        if param_names and value_lists and rule_id:
            for values in itertools.product(*(value_lists[p] for p in param_names)):
                trial_params = dict(zip(param_names, values))
                rule = rule_id_to_rule[rule_id]
                rule.parameters.update(trial_params)

                correct = evaluate_rule_with_cache(
                    rule, X, y, rule_parameter_result_dict, rule_id, trajectories_dict
                )

        # Case 2: no tunable params → just evaluate
        else:
            rule = rule_id_to_rule[rule_id]
            correct = evaluate_rule_with_cache(
                rule, X, y, rule_parameter_result_dict, rule_id, trajectories_dict
            )

        pbar.update(1)

    rbook.apply_config(default_config)  # reset to default after caching
    pbar.close()


def optimize_rulebook_greedy_by_priority(
    rulebook,
    training_data,
    training_labels,
    training_votes,
    rule_id_to_params,
    rule_id_to_values,
    trajectories_dict,
    epochs=1,
    rule_parameter_result_dict=None,
    skip=None,
    verbose=0,
):
    """
    Greedy optimization in rule priority order (training-only).
    For each rule, try all its parameter combinations while keeping
    previously chosen parameters fixed. Accept a change only if it
    improves the training score.

    skip: set of rule IDs to skip during optimization
    """
    default_config = rulebook.get_config()

    graph = rulebook.priority_graph
    try:
        priority_order = list(nx.topological_sort(graph))
    except Exception:
        priority_order = list(graph.nodes)

    if rule_parameter_result_dict is None:
        rule_parameter_result_dict = {}

    best_config = {}
    best_train_score = evaluate_rulebook_with_cache(
        rulebook,
        training_data,
        training_labels,
        training_votes,
        rule_parameter_result_dict,
        trajectories_dict,
    )[0]

    if verbose:
        print(f"[greedy] Initial Train={best_train_score:.6f}")

    for epoch in range(epochs):
        improved = False
        for node_id in priority_order:
            for rule_id in graph.nodes[node_id]["rules"]:
                # print(rule_id)
                if rule_id not in rule_id_to_params or (
                    skip is not None and rule_id in skip
                ):
                    continue

                param_names = rule_id_to_params[rule_id]
                value_lists = rule_id_to_values.get(rule_id, {})
                for p in param_names:
                    if p not in value_lists:
                        raise ValueError(
                            f"No candidate values for {p} in rule {rule_id}"
                        )

                combos = itertools.product(*[value_lists[p] for p in param_names])
                combos = tqdm(
                    list(combos),
                    desc=f"Rule {rule_id}",
                    leave=False,
                    disable=not bool(verbose),
                )

                current_rule = rulebook.rule_id_to_rule[rule_id]
                old_params = current_rule.parameters.copy()
                best_local_params = old_params.copy()
                local_best_train = best_train_score

                for combo in combos:
                    trial_params = dict(zip(param_names, combo))
                    current_rule.parameters.update(trial_params)

                    train_score = evaluate_rulebook_with_cache(
                        rulebook,
                        training_data,
                        training_labels,
                        training_votes,
                        rule_parameter_result_dict,
                        trajectories_dict,
                    )[0]

                    if verbose >= 2:
                        print(f"  {trial_params} -> Train={train_score:.6f}")

                    if train_score > local_best_train:
                        # print(train_score, local_best_train)
                        best_local_params = trial_params.copy()
                        local_best_train = train_score
                        # print(best_local_params)
                        improved = True

                current_rule.parameters.update(best_local_params)
                best_config[rule_id] = best_local_params
                # print(best_config[rule_id])
                best_train_score = local_best_train

                if verbose:
                    print(f"[greedy] {rule_id} -> Train={best_train_score:.6f}")
        if not improved:
            break

    # print(best_config)
    rulebook.apply_config(best_config)
    return best_config, best_train_score


def swap_nodes(g, u, v):
    # Create a mapping that swaps u and v labels
    mapping = {u: v, v: u}
    # Relabel the graph with copy=False to modify in place
    return nx.relabel_nodes(g, mapping, copy=True)


def is_acyclic(graph):
    return nx.is_directed_acyclic_graph(graph)


def no_redundant_edges(graph):
    for n in graph.nodes:
        successors = set(graph.successors(n))
        for s in successors:
            if len(list(nx.all_simple_paths(graph, n, s))) > 1:
                return False
    return True


def is_weakly_connected(graph):
    return nx.is_weakly_connected(graph)


def random_action(rulebook, max_attempts=10):
    """
    Perform a random modification on the rulebook's priority_graph.
    The action space includes adding, removing, or swapping edges.
    Returns a new rulebook with an acyclic priority_graph.
    If no valid action is found after max_attempts, returns the original rulebook.
    """
    for _ in range(max_attempts):
        new_rulebook = rulebook.copy()
        g = new_rulebook.priority_graph
        nodes = list(g.nodes)
        edges = list(g.edges)
        # choices = ["add", "remove", "swap"]
        choices = ["swap"]
        action_type = random.choice(choices)

        if action_type == "add":
            u, v = random.sample(nodes, 2)
            if not g.has_edge(u, v):
                g.add_edge(u, v)
            else:
                continue  # resample if edge already exists

        elif action_type == "remove":
            if edges:
                u, v = random.choice(edges)
                g.remove_edge(u, v)
            else:
                continue  # resample if no edges to remove

        elif action_type == "swap":
            if len(nodes) >= 2:
                u, v = random.sample(nodes, 2)
                g = swap_nodes(g, u, v)
        else:
            continue

        # Validate the modified graph
        if is_acyclic(g) and no_redundant_edges(g) and is_weakly_connected(g):
            new_rulebook.priority_graph = g
            return new_rulebook

    # If no valid action found after max_attempts, return original rulebook
    return rulebook


def simulated_annealing(
    rulebook,
    train_data,
    train_labels,
    train_votes,
    rule_parameter_result_dict,
    trajectories_dict,
    max_iter=1000,
    start_temp=10.0,
    alpha=0.995,
    seed=None,
):
    if seed is not None:
        random.seed(seed)

    current_rulebook = rulebook.copy()
    current_score = evaluate_rulebook_with_cache(
        current_rulebook,
        train_data,
        train_labels,
        train_votes,
        rule_parameter_result_dict,
        trajectories_dict,
    )[0]

    best = current_rulebook.copy()
    best_score = current_score
    print("Initial Score:", best_score)

    T = start_temp

    pbar = tqdm(total=max_iter, desc="Simulated Annealing", leave=False)

    for i in range(max_iter):

        candidate = random_action(current_rulebook)

        candidate_score = evaluate_rulebook_with_cache(
            candidate,
            train_data,
            train_labels,
            train_votes,
            rule_parameter_result_dict,
            trajectories_dict,
        )[0]

        delta = candidate_score - current_score

        if delta > 0 or random.random() < math.exp(min(0, max(-delta / T, -700))):
            current_rulebook = candidate
            current_score = candidate_score

            if current_score > best_score:
                best = current_rulebook.copy()
                best_score = current_score
                pbar.set_description(f"New best: {best_score:.4f}")

        T *= alpha
        pbar.update(1)

    pbar.close()
    return best, best_score


def simulated_annealing_with_validation(
    rulebook,
    train_data,
    train_labels,
    train_votes,
    val_data,
    val_labels,
    val_votes,
    rule_parameter_result_dict,
    trajectories_dict,
    max_iter=1000,
    start_temp=10.0,
    alpha=0.995,
    seed=None,
):
    if seed is not None:
        random.seed(seed)

    current_rulebook = rulebook
    current_score = evaluate_rulebook_with_cache(
        current_rulebook,
        train_data,
        train_labels,
        train_votes,
        rule_parameter_result_dict,
        trajectories_dict,
    )[0]
    current_val_score = evaluate_rulebook_with_cache(
        current_rulebook,
        val_data,
        val_labels,
        val_votes,
        rule_parameter_result_dict,
        trajectories_dict,
    )[0]

    best = current_rulebook
    best_score = current_score
    best_val_score = current_val_score

    T = start_temp

    pbar = tqdm(total=max_iter, desc="Simulated Annealing", leave=False)

    for i in range(max_iter):
        candidate = random_action(current_rulebook)
        candidate_score = evaluate_rulebook_with_cache(
            candidate,
            train_data,
            train_labels,
            train_votes,
            rule_parameter_result_dict,
            trajectories_dict,
        )[0]
        candidate_val_score = evaluate_rulebook_with_cache(
            candidate,
            val_data,
            val_labels,
            val_votes,
            rule_parameter_result_dict,
            trajectories_dict,
        )[0]

        delta = candidate_score - current_score
        delta_val = candidate_val_score - current_val_score
        if (delta > 0 and delta_val > 0) or (
            random.random() < math.exp(min(0, max(-delta / T, -700)))
            and random.random() < math.exp(min(0, max(-delta_val / T, -700)))
        ):
            current_rulebook = candidate
            current_score = candidate_score
            current_val_score = candidate_val_score

            if current_score > best_score and current_val_score > best_val_score:
                best = current_rulebook.copy()
                best_score = current_score
                best_val_score = current_val_score
                pbar.set_description(
                    f"New best: Train {best_score:.4f}, Val {best_val_score:.4f}"
                )
        pbar.update(1)
        T *= alpha

    pbar.close()
    return best, best_score, best_val_score


def simulated_annealing_small_set(
    rulebook,
    train_data,
    train_labels,
    train_votes,
    rule_parameter_result_dict,
    trajectories_dict,
    max_iter=1000,
    start_temp=10.0,
    alpha=0.995,
    seed=None,
):
    if seed is not None:
        random.seed(seed)

    current_rulebook = rulebook.copy()
    current_score = evaluate_rulebook_with_cache(
        current_rulebook,
        train_data,
        train_labels,
        train_votes,
        rule_parameter_result_dict,
        trajectories_dict,
    )[0]

    best = current_rulebook.copy()
    best_score = current_score
    if best_score == len(train_data):
        return best, best_score

    T = start_temp

    pbar = tqdm(total=max_iter, desc="Simulated Annealing", leave=False)

    for i in range(max_iter):

        candidate = random_action(current_rulebook)

        candidate_score = evaluate_rulebook_with_cache(
            candidate,
            train_data,
            train_labels,
            train_votes,
            rule_parameter_result_dict,
            trajectories_dict,
        )[0]

        delta = candidate_score - current_score

        if delta > 0 or random.random() < math.exp(min(0, max(-delta / T, -700))):
            current_rulebook = candidate
            current_score = candidate_score

            if current_score > best_score:
                best = current_rulebook.copy()
                best_score = current_score
                pbar.set_description(f"New best: {best_score:.4f}")
            if best_score == len(train_data):  # perfect score
                return best, best_score

        T *= alpha
        pbar.update(1)

    pbar.close()
    return best, best_score


def number_of_unique_rulebooks(
    rulebook,
    train_data,
    train_labels,
    train_votes,
    rule_parameter_result_dict,
    trajectories_dict,
    seed=None,
):

    score = 0
    total = len(train_data)
    unique_rulebooks = []
    pbar = tqdm(total=total, desc="Finding Unique Rulebooks", leave=False)
    unsatisfiable_samples = []

    for i in range(total):
        sample = train_data[i : i + 1]
        label = train_labels[i : i + 1]
        votes = train_votes[i : i + 1]

        # assert sc > 0
        rb, sc = simulated_annealing_small_set(
            rulebook,
            sample,
            label,
            votes,
            rule_parameter_result_dict,
            trajectories_dict,
            max_iter=1000,
            start_temp=15.0,
            alpha=0.995,
            seed=seed,
        )

        found = False
        if sc > 0:
            for existing_rb in unique_rulebooks:
                if nx.utils.graphs_equal(existing_rb.priority_graph, rb.priority_graph):
                    found = True
                    break

            if not found:
                unique_rulebooks.append(rb)
        else:
            unsatisfiable_samples.append((sample[0], votes[0]))

        score += sc
        pbar.update(1)
    pbar.close()

    return len(unique_rulebooks), score, score / total, unsatisfiable_samples


def get_scenario_to_samples(X, y, y_votes, trajectories_dict):
    scenario_to_samples = {}

    for name, trajectory in trajectories_dict.items():
        parts = name.split("-")
        scenario_name = parts[0]
        if scenario_name not in scenario_to_samples:
            scenario_to_samples[scenario_name] = {}
            scenario_to_samples[scenario_name]["X"] = []
            scenario_to_samples[scenario_name]["y"] = []
            scenario_to_samples[scenario_name]["votes"] = []

    for i in range(len(X)):
        pair = X[i]
        label = y[i]
        votes = y_votes[i]
        parts = pair[0].split("-")
        scenario_name = parts[0]
        scenario_to_samples[scenario_name]["X"].append(pair)
        scenario_to_samples[scenario_name]["y"].append(label)
        scenario_to_samples[scenario_name]["votes"].append(votes)

    return scenario_to_samples


def find_scenario_rulebooks(
    base_rulebook,
    all_pairs,
    all_labels,
    all_votes,
    rule_parameter_result_dict,
    trajectories_dict,
    seed=42,
):
    scenario_to_samples = get_scenario_to_samples(
        all_pairs, all_labels, all_votes, trajectories_dict
    )
    rulebooks = {}
    total = 0
    correct = 0
    num_unique_rulebooks = 0

    pbar = tqdm(
        total=len(scenario_to_samples),
        desc="Finding Rulebooks for Scenarios",
        leave=False,
    )
    # print("Number of scenarios:", len(scenario_to_samples))
    for name, traj_dict in scenario_to_samples.items():
        rulebook, score = simulated_annealing_small_set(
            base_rulebook,
            traj_dict["X"],
            traj_dict["y"],
            traj_dict["votes"],
            rule_parameter_result_dict,
            trajectories_dict,
            max_iter=100,
            start_temp=10.0,
            alpha=0.995,
            seed=seed,
        )
        # print(f"Scenario: {name}, Score: {score}/{len(traj_dict['X'])}")
        traj_dict["rulebook"] = rulebook
        traj_dict["score"] = score / len(traj_dict["X"])
        total += len(traj_dict["X"])
        correct += score

        found = False
        for existing_rb in rulebooks.values():
            if nx.utils.graphs_equal(
                existing_rb.priority_graph, rulebook.priority_graph
            ):
                found = True
                rulebooks[name] = existing_rb
                break
        if not found:
            rulebooks[name] = rulebook
            num_unique_rulebooks += 1

        pbar.update(1)
    pbar.close()

    return (
        rulebooks,
        num_unique_rulebooks,
        correct,
        correct / total,
        scenario_to_samples,
    )


def shuffle_graph_nodes(g, seed=None):
    """
    Returns a new graph identical in structure to g,
    but with node labels randomly permuted.
    """
    if seed is not None:
        random.seed(seed)

    nodes = list(g.nodes())
    shuffled = nodes[:]
    random.shuffle(shuffled)

    mapping = dict(zip(nodes, shuffled))
    return nx.relabel_nodes(g, mapping, copy=True)


def shuffle_rulebook(rulebook, seed=None):
    if seed is not None:
        random.seed(seed)
    new_rulebook = rulebook.copy()
    g = new_rulebook.priority_graph
    g = shuffle_graph_nodes(g, seed=seed)
    new_rulebook.priority_graph = g
    return new_rulebook


def random_dag_from_nodes(nodes, seed=None):
    """
    Create a random connected DAG (no cycles) using the given node IDs.
    Ensures if a path exists from A→B, no redundant edge A→B is added.
    """
    if seed is not None:
        random.seed(seed)

    nodes = list(nodes)
    n = len(nodes)
    g = nx.DiGraph()
    g.add_nodes_from(nodes)

    # random topological order
    order = nodes[:]
    random.shuffle(order)

    # ensure weak connectivity (tree-like backbone)
    for i in range(1, n):
        parent = random.choice(order[:i])
        g.add_edge(parent, order[i])

    # add extra edges while keeping DAG property
    possible_edges = [
        (u, v)
        for i, u in enumerate(order)
        for v in order[i + 1 :]
        if not g.has_edge(u, v)
    ]

    random.shuffle(possible_edges)
    for u, v in possible_edges:
        if not nx.has_path(g, v, u):  # avoid cycle
            g.add_edge(u, v)

    return g


def combine_groups_in_order(g, groups, keep_relations):
    """
    Combine node groups from a DAG into a new ordered graph.
    - Keeps in-group edges unchanged.
    - Connects nodes with no successors in group[i] to nodes with no predecessors in group[i+1].

    Args:
        g: networkx.DiGraph (must be a DAG)
        groups: list[list[node_id]] defining the desired group order.

    Returns:
        new_g: networkx.DiGraph
    """
    new_g = nx.DiGraph()

    # also add data from original graph
    new_g.add_nodes_from(g.nodes())

    # 1. Add intra-group edges
    reach = nx.transitive_closure_dag(g)
    if keep_relations:
        for group in groups:
            for u, v in itertools.permutations(group, 2):
                if reach.has_edge(u, v):
                    new_g.add_edge(u, v)

        # Remove redundant edges
        new_g = nx.transitive_reduction(new_g)

        for n, d in g.nodes(data=True):
            new_g.nodes[n].update(d)

    # 2. Connect consecutive groups
    for i in range(len(groups) - 1):
        current_group = groups[i]
        next_group = groups[i + 1]

        # terminal nodes = nodes with no successors within current group
        terminals = [
            n
            for n in current_group
            if not any(s in current_group for s in new_g.successors(n))
        ]

        # entry nodes = nodes with no predecessors within next group
        entries = [
            n
            for n in next_group
            if not any(p in next_group for p in new_g.predecessors(n))
        ]

        # connect all terminal → entry pairs
        for u in terminals:
            for v in entries:
                new_g.add_edge(u, v)

    # 3. Validate
    assert nx.is_directed_acyclic_graph(new_g), "Resulting graph is not acyclic"

    return new_g


def group_rulebook(rulebook, groups, keep_relations):
    rb = rulebook.copy()
    rb.priority_graph = combine_groups_in_order(
        rulebook.priority_graph, groups, keep_relations
    )
    return rb


def group_nodes_by_level(g):
    """
    Group nodes of a DAG by their topological levels.
    Level 0: nodes with no predecessors.
    Level 1: nodes whose predecessors are all in level 0.
    And so on.

    Args:
        g: networkx.DiGraph (must be a DAG)

    Returns:
        levels: list[list[node_id]] where levels[i] contains nodes at level i.
    """
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Input graph must be a DAG")

    in_degree = {n: d for n, d in g.in_degree()}
    levels = []
    current_level = [n for n, d in in_degree.items() if d == 0]

    while current_level:
        levels.append(current_level)
        next_level = []
        for n in current_level:
            for succ in g.successors(n):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    next_level.append(succ)
        current_level = next_level

    return levels


def greedy_group_optimization(
    rulebook,
    X,
    y,
    y_votes,
    cache_dict,
    trajectories_dict,
    groups=None,
    max_iters=100,
    keep_relations=True,
    restricted=False,
    fixed_level_depth=None,
):
    """
    Greedily swap rulebook levels in the graph. Always choose the maximum improving move.
    Stops when no move can improve the score.
    """

    rb = rulebook.copy()
    g = rb.priority_graph

    levels = groups
    fixed_levels = []
    if fixed_level_depth is not None:
        fixed_levels = groups[: fixed_level_depth + 1]
        levels = groups[fixed_level_depth + 1 :]  # only optimize lower levels

    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        best_score = evaluate_rulebook_with_cache(
            rb, X, y, y_votes, cache_dict, trajectories_dict
        )[0]
        best_rb = rb.copy()
        best_levels = levels[:]

        if restricted:
            for i in range(len(levels) - 1, 0, -1):
                # swap levels i and i+1
                new_levels = levels[:]  # make a copy
                new_levels[i], new_levels[i - 1] = new_levels[i - 1], new_levels[i]

                new_g = combine_groups_in_order(
                    g, fixed_levels + new_levels, keep_relations
                )
                new_rb = rb.copy()
                new_rb.priority_graph = new_g

                new_score = evaluate_rulebook_with_cache(
                    new_rb, X, y, y_votes, cache_dict, trajectories_dict
                )[0]
                if new_score > best_score:
                    best_score = new_score
                    best_rb = new_rb
                    best_levels = new_levels
                    improved = True

        else:
            for i in range(len(levels) - 1, -1, -1):
                for j in range(i - 1, -1, -1):
                    # Swap levels i and j
                    new_levels = levels[:]
                    new_levels[i], new_levels[j] = new_levels[j], new_levels[i]
                    new_g = combine_groups_in_order(
                        g, fixed_levels + new_levels, keep_relations
                    )
                    new_rb = rb.copy()
                    new_rb.priority_graph = new_g

                    new_score = evaluate_rulebook_with_cache(
                        new_rb, X, y, y_votes, cache_dict, trajectories_dict
                    )[0]
                    if new_score > best_score:
                        best_score = new_score
                        best_rb = new_rb
                        best_levels = new_levels
                        improved = True
        rb = best_rb
        g = best_rb.priority_graph
        levels = best_levels

        iter_count += 1

    return rb, best_score


def greedy_group_optimization_with_validation(
    rulebook,
    train_x,
    train_y,
    train_votes,
    val_x,
    val_y,
    val_votes,
    cache_dict,
    trajectories_dict,
    groups,
    max_iters=100,
    keep_relations=True,
    restricted=False,
    fixed_level_depth=None,
):
    """
    Greedily swap rulebook levels in the graph. Always choose the maximum improving move.
    Stops when no move can improve the score.
    """

    rb = rulebook.copy()
    g = rb.priority_graph
    levels = groups
    fixed_levels = []
    if fixed_level_depth is not None:
        fixed_levels = groups[: fixed_level_depth + 1]
        levels = groups[fixed_level_depth + 1 :]  # only optimize lower levels
    improved = True
    iter_count = 0

    while improved and iter_count < max_iters:
        improved = False
        train_best_score = evaluate_rulebook_with_cache(
            rb, train_x, train_y, train_votes, cache_dict, trajectories_dict
        )[0]
        val_best_score = evaluate_rulebook_with_cache(
            rb, val_x, val_y, val_votes, cache_dict, trajectories_dict
        )[0]
        best_rb = rb.copy()
        best_levels = levels[:]

        if restricted:
            for i in range(len(levels) - 1, 0, -1):
                # swap levels i and i+1
                new_levels = levels[:]  # make a copy
                new_levels[i], new_levels[i - 1] = new_levels[i - 1], new_levels[i]
                new_g = combine_groups_in_order(
                    g, fixed_levels + new_levels, keep_relations
                )
                new_rb = rb.copy()
                new_rb.priority_graph = new_g

                train_new_score = evaluate_rulebook_with_cache(
                    new_rb, train_x, train_y, train_votes, cache_dict, trajectories_dict
                )[0]
                val_new_score = evaluate_rulebook_with_cache(
                    new_rb, val_x, val_y, val_votes, cache_dict, trajectories_dict
                )[0]
                if (
                    train_new_score > train_best_score
                    and val_new_score > val_best_score
                ):
                    train_best_score = train_new_score
                    val_best_score = val_new_score
                    best_rb = new_rb
                    best_levels = new_levels
                    improved = True

        else:
            for i in range(len(levels) - 1, -1, -1):
                for j in range(i - 1, -1, -1):
                    # Swap levels i and j
                    new_levels = levels[:]
                    new_levels[i], new_levels[j] = new_levels[j], new_levels[i]
                    new_g = combine_groups_in_order(
                        g, fixed_levels + new_levels, keep_relations
                    )
                    new_rb = rb.copy()
                    new_rb.priority_graph = new_g

                    train_new_score = evaluate_rulebook_with_cache(
                        new_rb,
                        train_x,
                        train_y,
                        train_votes,
                        cache_dict,
                        trajectories_dict,
                    )[0]
                    val_new_score = evaluate_rulebook_with_cache(
                        new_rb, val_x, val_y, val_votes, cache_dict, trajectories_dict
                    )[0]

                    if (
                        train_new_score > train_best_score
                        and val_new_score > val_best_score
                    ):
                        train_best_score = train_new_score
                        val_best_score = val_new_score
                        best_rb = new_rb
                        best_levels = new_levels
                        improved = True
        rb = best_rb
        g = best_rb.priority_graph
        levels = best_levels

        iter_count += 1

    return rb, train_best_score, val_best_score


def brute_force_group_optimization(
    rulebook,
    X,
    y,
    y_votes,
    cache_dict,
    trajectories_dict,
    groups,
    keep_relations=True,
    fixed_level_depth=None,
):
    """
    Try all permutations of rulebook levels in the graph. Choose the best one.
    """
    fixed_levels = []
    if fixed_level_depth is not None:
        fixed_levels = groups[: fixed_level_depth + 1]
        groups = groups[fixed_level_depth + 1 :]  # only optimize lower levels

    rb = rulebook.copy()
    g = rb.priority_graph

    best_score = evaluate_rulebook_with_cache(
        rb, X, y, y_votes, cache_dict, trajectories_dict
    )[0]
    best_rb = rb.copy()

    for perm in itertools.permutations(groups):
        new_g = combine_groups_in_order(g, fixed_levels + list(perm), keep_relations)
        new_rb = rb.copy()
        new_rb.priority_graph = new_g

        new_score = evaluate_rulebook_with_cache(
            new_rb, X, y, y_votes, cache_dict, trajectories_dict
        )[0]

        if new_score > best_score:
            best_score = new_score
            best_rb = new_rb

    return best_rb, best_score


def brute_force_group_optimization_with_validation(
    rulebook,
    X_train,
    y_train,
    votes_train,
    X_val,
    y_val,
    votes_val,
    cache_dict,
    trajectories_dict,
    groups,
    keep_relations=True,
    fixed_level_depth=None,
):
    """
    Try all permutations of rulebook levels in the graph. Choose the best one.
    """
    fixed_levels = []
    if fixed_level_depth is not None:
        fixed_levels = groups[: fixed_level_depth + 1]
        groups = groups[fixed_level_depth + 1 :]  # only optimize lower levels

    rb = rulebook.copy()
    g = rb.priority_graph

    train_best_score = evaluate_rulebook_with_cache(
        rb, X_train, y_train, votes_train, cache_dict, trajectories_dict
    )[0]
    val_best_score = evaluate_rulebook_with_cache(
        rb, X_val, y_val, votes_val, cache_dict, trajectories_dict
    )[0]
    best_rb = rb.copy()

    for perm in itertools.permutations(groups):
        new_g = combine_groups_in_order(g, fixed_levels + list(perm), keep_relations)
        new_rb = rb.copy()
        new_rb.priority_graph = new_g

        new_train_score = evaluate_rulebook_with_cache(
            new_rb, X_train, y_train, votes_train, cache_dict, trajectories_dict
        )[0]
        new_val_score = evaluate_rulebook_with_cache(
            new_rb, X_val, y_val, votes_val, cache_dict, trajectories_dict
        )[0]

        if new_train_score > train_best_score and new_val_score > val_best_score:
            train_best_score = new_train_score
            val_best_score = new_val_score
            best_rb = new_rb

    return best_rb, train_best_score, val_best_score


def find_scenario_groups(
    rulebook,
    X,
    y,
    y_votes,
    rule_parameter_result_dict,
    trajectories_dict,
    optimization_alg,
    groups=None,
    keep_relations=True,
    **kwargs,
):
    scenario_to_samples = get_scenario_to_samples(X, y, y_votes, trajectories_dict)
    total = 0
    correct = 0
    rulebooks = {}
    num_unique_rulebooks = 0
    scenario_to_groups = {}
    if groups is None:
        groups = group_nodes_by_level(rulebook.priority_graph)
    grouped_rulebook = group_rulebook(rulebook, groups, keep_relations)

    pbar = tqdm(
        total=len(scenario_to_samples), desc="Finding Groups for Scenarios", leave=False
    )
    for name, traj_dict in scenario_to_samples.items():
        rulebook, score = optimization_alg(
            grouped_rulebook,
            traj_dict["X"],
            traj_dict["y"],
            traj_dict["votes"],
            rule_parameter_result_dict,
            trajectories_dict,
            groups=groups,
            keep_relations=keep_relations,
            **kwargs,
        )
        traj_dict["rulebook"] = rulebook
        traj_dict["score"] = score / len(traj_dict["X"])
        total += len(traj_dict["X"])
        correct += score
        scenario_to_groups[name] = groups

        found = False
        for existing_rb in rulebooks.values():
            if nx.utils.graphs_equal(
                existing_rb.priority_graph, rulebook.priority_graph
            ):
                found = True
                rulebooks[name] = existing_rb
                break
        if not found:
            rulebooks[name] = rulebook
            num_unique_rulebooks += 1

        pbar.update(1)
    pbar.close()

    return (
        rulebooks,
        num_unique_rulebooks,
        correct,
        correct / total,
        scenario_to_samples,
    )
