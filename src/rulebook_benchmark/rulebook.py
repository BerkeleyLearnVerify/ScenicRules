import ast
from collections import defaultdict
from copy import deepcopy
from enum import Enum

import matplotlib.pyplot as plt
import networkx as nx

from rulebook_benchmark.realization import VariableHandler


class FunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []

    def visit_FunctionDef(self, node):
        self.functions.append(node)


class Relation(Enum):
    LARGER = 1
    SMALLER = 2
    EQUAL = 3
    NONCOMPARABLE = 4


class Rulebook:
    def __init__(self, rule_id_to_rule: dict, rulebook_file=None):
        """
        Args:
            rule_id_to_rule (dict): A dictionary mapping rule ids to Rule objects.
            rulebook_file (string): Path to the rulebook file.
        """
        self.rulebook_file = rulebook_file
        self.verbosity = 1
        self.priority_graph = nx.DiGraph()
        self.rule_ids = set()
        self.rule_id_to_rule = {}
        self.rule_name_to_rule_id = {rule.name: rule_id for rule_id, rule in rule_id_to_rule.items()}
        for rule_id, rule in rule_id_to_rule.items():
            self.rule_id_to_rule[rule_id] = rule.copy()
        self.rule_id_to_node_id = (
            {}
        )  # mapping from rule id to node id in the priority graph
        if rulebook_file:
            self._parse_rulebook_from_file(rulebook_file)

    def copy(self):
        copy_rules = {}
        for rule_id, rule in self.rule_id_to_rule.items():
            copy_rules[rule_id] = rule.copy()

        # create new rulebook, then copy the priority graph structure
        new_rulebook = Rulebook(copy_rules, self.rulebook_file)
        # Sever all edges, then copy edges from original graph
        new_rulebook.priority_graph.remove_edges_from(
            list(new_rulebook.priority_graph.edges())
        )
        new_rulebook.priority_graph.add_edges_from(self.priority_graph.edges())

        return new_rulebook

    def _parse_rulebook_from_file(self, rulebook_file):
        """
        Parses a rulebook from a rulebook file.
        """
        with open(rulebook_file, "r") as f:
            lines = f.readlines()
            header_section = False
            rule_section = False
            same_level_section = False
            edge_section = False
            for line in lines:
                line = line.strip()
                if line == "#header":
                    header_section = True
                    continue
                elif line == "#rules":
                    header_section = False
                    rule_section = True
                    continue
                elif line == "#same-level":
                    rule_section = False
                    same_level_section = True
                    continue
                elif line == "#priorities":
                    same_level_section = False
                    edge_section = True
                    continue

                # Header
                if header_section:
                    continue

                # Node
                if rule_section:
                    rule_id = int(line.strip())
                    assert (
                        rule_id in self.rule_id_to_rule
                    ), f"Rule ID {rule_id} not found in the provided rule_id_to_rule dictionary."
                    rule = self.rule_id_to_rule[rule_id]
                    self.rule_ids.add(rule_id)
                    self.rule_id_to_node_id[rule_id] = rule_id
                    self.priority_graph.add_node(
                        rule_id, rules={rule_id}
                    )  # create set for rules at the node
                    if self.verbosity >= 2:
                        print(
                            f"Add rule {rule_id} with name: {rule.name}, rule function: {rule.calculate_violation}"
                        )

                # Same level rules
                if same_level_section:
                    same_level_info = line.split(" ")
                    rep = int(same_level_info[0])
                    for i in range(1, len(same_level_info)):
                        self.rule_id_to_node_id[int(same_level_info[i])] = rep
                        self.priority_graph.nodes[rep]["rules"].add(
                            int(same_level_info[i])
                        )
                        self.priority_graph.remove_node(int(same_level_info[i]))
                        if self.verbosity >= 2:
                            print(
                                f"Assign rule {int(same_level_info[i])} to the same level as the representative rule {rep}"
                            )

                # Edge
                if edge_section:
                    edge_info = line.split(" ")
                    src = self.rule_id_to_node_id[int(edge_info[0])]
                    dst = self.rule_id_to_node_id[int(edge_info[1])]
                    if src == dst:
                        continue
                    if self.priority_graph.has_edge(src, dst):
                        continue
                    self.priority_graph.add_edge(src, dst)
                    if self.verbosity >= 2:
                        print(f"Add edge from rule {src} to rule {dst}")

            self.check_rulebook()

            if self.verbosity >= 2:
                for node_id in self.priority_graph.nodes():
                    for rule_id in self.priority_graph.nodes[node_id]["rules"]:
                        rule = self.rule_id_to_rule[rule_id]
                        print(
                            f"Node {id} contains rule {rule_id} with name: {rule.name}, rule function: {rule.calculate_violation}"
                        )
                print(f"Nodes: {self.priority_graph.nodes(data=True)}")
                print(f"Edges: {self.priority_graph.edges()}")

    def add_rule(self, rule_object):
        """
        Adds an isolated rule to the rulebook.
        """
        rule_id = rule_object.id
        if rule_id in self.rule_ids:
            raise ValueError(f"Rule ID {rule_id} already exists in the rulebook.")
        self.rule_ids.add(rule_id)
        self.rule_id_to_node_id[rule_id] = rule_id
        self.rule_id_to_rule[rule_id] = rule_object
        self.rule_name_to_rule_id[rule_object.name] = rule_id
        self.priority_graph.add_node(rule_id, rules={rule_id})

    def add_rule_relation(self, rule_id_1, rule_id_2, relation=Relation.LARGER):
        """
        Adds a rule relation to the rulebook.
        """
        if rule_id_1 not in self.rule_ids or rule_id_2 not in self.rule_ids:
            raise ValueError(
                f"Rule IDs {rule_id_1} and {rule_id_2} must exist in the rulebook."
            )
        resp_1 = self.rule_id_to_node_id[rule_id_1]
        resp_2 = self.rule_id_to_node_id[rule_id_2]
        if resp_1 == resp_2:
            raise ValueError(
                f"Node IDs {rule_id_1} and {rule_id_2} are already assigned to the same level."
            )
        if relation == Relation.LARGER:
            self.priority_graph.add_edge(resp_1, resp_2)
        elif relation == Relation.SMALLER:
            self.priority_graph.add_edge(resp_2, resp_1)
        elif relation == Relation.EQUAL:
            for rule_id in self.priority_graph.nodes[resp_2]["rules"]:
                self.priority_graph.nodes[resp_1]["rules"].add(rule_id)
                self.rule_id_to_node_id[rule_id] = resp_1
            preds = list(self.priority_graph.predecessors(resp_2))
            succs = list(self.priority_graph.successors(resp_2))
            self.priority_graph.remove_node(resp_2)
            for pred in preds:
                self.priority_graph.add_edge(pred, resp_1)
            for succ in succs:
                self.priority_graph.add_edge(resp_1, succ)
        elif relation == Relation.NONCOMPARABLE:
            pass
        self.check_rulebook()

    def remove_rule(self, rule_id):
        """
        Removes a rule from the rulebook. The predecessors and successors will be connected.
        """
        if rule_id not in self.rule_ids:
            raise ValueError(f"Rule ID {rule_id} not found in the priority graph.")
        resp = self.rule_id_to_node_id[rule_id]
        if len(self.priority_graph.nodes[resp]["rules"]) == 1:
            preds = list(self.priority_graph.predecessors(resp))
            succs = list(self.priority_graph.successors(resp))
            for pred in preds:
                for succ in succs:
                    if not self.priority_graph.has_edge(pred, succ):
                        self.priority_graph.add_edge(pred, succ)
            self.priority_graph.remove_node(resp)
            self.check_rulebook()
        else:
            self.priority_graph.nodes[resp]["rules"].remove(rule_id)
            if resp == rule_id:
                new_resp = list(self.priority_graph.nodes[resp]["rules"])[0]
                self.priority_graph.add_node(
                    new_resp, **self.priority_graph.nodes[resp]
                )
                for pred in self.priority_graph.predecessors(resp):
                    self.priority_graph.add_edge(pred, new_resp)
                for succ in self.priority_graph.successors(resp):
                    self.priority_graph.add_edge(new_resp, succ)
                self.priority_graph.remove_node(resp)
                for id in self.priority_graph.nodes[new_resp]["rules"]:
                    self.rule_id_to_node_id[id] = new_resp
        self.rule_ids.remove(rule_id)
        self.rule_id_to_node_id.pop(rule_id)
        self.rule_name_to_rule_id.pop(self.rule_id_to_rule[rule_id].name)
        self.rule_id_to_rule.pop(rule_id)

    def remove_rule_relation(self, rule_id_1, rule_id_2):
        """
        Removes a rule edge from the rulebook.
        """
        if rule_id_1 not in self.rule_ids or rule_id_2 not in self.rule_ids:
            raise ValueError(
                f"Rule IDs {rule_id_1} and {rule_id_2} must exist in the rulebook."
            )
        resp_1 = self.rule_id_to_node_id[rule_id_1]
        resp_2 = self.rule_id_to_node_id[rule_id_2]
        if self.priority_graph.has_edge(resp_1, resp_2):
            self.priority_graph.remove_edge(resp_1, resp_2)
        else:
            raise ValueError(f"No edge exists between {rule_id_1} and {rule_id_2}.")

    def get_rule_names(self):
        """
        Returns the names of all rules in the rulebook.
        """
        names = []
        for node in self.priority_graph.nodes():
            for rule_id in self.priority_graph.nodes[node]["rules"]:
                names.append(self.rule_id_to_rule[rule_id].name)
        return names

    def get_rule_relation(self, rule_id_1, rule_id_2, to_print=False):
        """
        Returns the priority relation between rule 1 and rule 2.
        """
        if rule_id_1 not in self.rule_ids or rule_id_2 not in self.rule_ids:
            if to_print:
                print("Rule IDs not found in the rulebook.")
            return
        node_id_1 = self.rule_id_to_node_id[rule_id_1]
        node_id_2 = self.rule_id_to_node_id[rule_id_2]
        if node_id_1 == node_id_2:
            if to_print:
                print(f"Rule {rule_id_1} and Rule {rule_id_2} are equal.")
            return Relation.EQUAL
        if node_id_1 in nx.descendants(self.priority_graph, node_id_2):
            if to_print:
                print(f"Rule {rule_id_1} is smaller than Rule {rule_id_2}.")
            return Relation.SMALLER
        if node_id_2 in nx.descendants(self.priority_graph, node_id_1):
            if to_print:
                print(f"Rule {rule_id_1} is larger than Rule {rule_id_2}.")
            return Relation.LARGER
        if to_print:
            print(f"Rule {rule_id_1} and Rule {rule_id_2} are non-comparable.")
        return Relation.NONCOMPARABLE

    def visualize_rulebook(self, output_file_name="merged_rule_graph.png"):
        ranks = {}
        for node in nx.topological_sort(self.priority_graph):
            preds = list(self.priority_graph.predecessors(node))
            if not preds:
                ranks[node] = 0
            else:
                ranks[node] = max(ranks[p] + 1 for p in preds)

        pos = {}
        levels = defaultdict(list)
        for node, rank in ranks.items():
            levels[rank].append(node)

        for y, (rank, nodes_at_rank) in enumerate(sorted(levels.items(), reverse=True)):
            for x, node in enumerate(nodes_at_rank):
                pos[node] = (x, y)

        labels = {}
        for node in self.priority_graph.nodes():
            rule_ids = self.priority_graph.nodes[node]["rules"]
            labels[node] = ", ".join([str(rule_id) for rule_id in rule_ids])
        plt.figure(figsize=(12, 8))
        nx.draw(
            self.priority_graph,
            pos,
            labels=labels,
            with_labels=True,
            node_color="lightblue",
            node_size=1000,
            font_size=10,
            arrows=True,
        )
        plt.title("Rulebook Graph with Same-Level Nodes Merged")
        plt.savefig(output_file_name)
        plt.show()

    def get_adjecency_list(self):
        """
        Returns the adjacency list of the priority graph.
        """
        return nx.to_dict_of_lists(self.priority_graph)

    def print_adjacency_matrix(self):
        """
        Returns the adjacency matrix of the rulebook.
        >: The row node is of higher priority than the column node.
        <: The row node is of lower priority than the column node.
        =: The row node is of the equal level to the column node.
        x: The row node is not comparable to the column node.
        -: The row node is the same as the column node.
        """
        relation_symbols = {
            Relation.LARGER: ">",
            Relation.SMALLER: "<",
            Relation.EQUAL: "=",
            Relation.NONCOMPARABLE: "x",
        }
        print("      " + "  ".join(f"{id2:>2}" for id2 in self.rule_ids))
        print("    " + "----" * (len(self.rule_ids)))
        for id1 in self.rule_ids:
            row_display = [f"{id1:>2} |"]
            for id2 in self.rule_ids:
                if id1 == id2:
                    row_display.append(" -")
                else:
                    relation = self.get_rule_relation(id1, id2)
                    symbol = relation_symbols[relation]
                    row_display.append(f" {symbol}")
            print("  ".join(row_display))

    def check_rulebook(self):
        """
        Checks the rulebook for consistency.
        """
        if len(list(nx.simple_cycles(self.priority_graph))) > 0:
            print(
                "Cycles in the rulebook:", list(nx.simple_cycles(self.priority_graph))
            )
            raise ValueError("The rulebook contains cycles. Please double check!")

    @property
    def root_nodes(self):
        return [n for n, d in self.priority_graph.in_degree() if d == 0]

    def evaluate(self, realization):
        rule_engine = RuleEngine(self.rule_id_to_rule)
        handler = VariableHandler(realization)
        result = rule_engine.evaluate(handler)
        return result

    def evaluate_with_cache(self, rule_parameter_result_dict, scenario):
        rule_engine = RuleEngine(self.rule_id_to_rule)
        result = rule_engine.evaluate_with_cache(rule_parameter_result_dict, scenario)
        return result

    def compare_trajectories(self, realization1, realization2):
        handler1 = VariableHandler(realization1)
        handler2 = VariableHandler(realization2)

        r1_advocates = set()
        r2_advocates = set()

        for root in self.root_nodes:
            self._compare_trajectories(
                root, handler1, handler2, r1_advocates, r2_advocates
            )

        def win_condition(set1, set2):
            for rule in set1:
                if all(nx.has_path(self.priority_graph, rule, other) for other in set2):
                    return True, rule
            return False, None

        win_1, winning_node_1 = win_condition(r1_advocates, r2_advocates)

        if (len(r1_advocates) > 0 and len(r2_advocates) == 0) or win_1:
            return Relation.LARGER, winning_node_1

        win_2, winning_node_2 = win_condition(r2_advocates, r1_advocates)

        if (len(r2_advocates) > 0 and len(r1_advocates) == 0) or win_2:
            return Relation.SMALLER, winning_node_2
        elif len(r1_advocates) == 0 and len(r2_advocates) == 0:
            return Relation.EQUAL, None
        else:
            return Relation.NONCOMPARABLE, None

    def _compare_trajectories(
        self, node_id, handler1, handler2, r1_advocates, r2_advocates
    ):
        rules = self.priority_graph.nodes[node_id]["rules"]
        result1 = 0
        result2 = 0
        for rule_id in rules:
            rule = self.rule_id_to_rule[rule_id]
            result1 += rule.evaluate(handler1)
            result2 += rule.evaluate(handler2)

        result1 /= len(rules)
        result2 /= len(rules)

        if result1 < result2:
            r1_advocates.add(node_id)
        elif result1 > result2:
            r2_advocates.add(node_id)
        else:
            for child_node_id in self.priority_graph.successors(node_id):
                self._compare_trajectories(
                    child_node_id, handler1, handler2, r1_advocates, r2_advocates
                )

    def compare_results(self, results1, results2):
        r1_advocates = set()
        r2_advocates = set()

        for root in self.root_nodes:
            self._compare_results(root, results1, results2, r1_advocates, r2_advocates)

        def win_condition(set1, set2):
            for rule in set1:
                if all(nx.has_path(self.priority_graph, rule, other) for other in set2):
                    return True, rule
            return False, None

        win_1, winning_node_1 = win_condition(r1_advocates, r2_advocates)

        if (len(r1_advocates) > 0 and len(r2_advocates) == 0) or win_1:
            return Relation.LARGER, winning_node_1

        win_2, winning_node_2 = win_condition(r2_advocates, r1_advocates)
        if (len(r2_advocates) > 0 and len(r1_advocates) == 0) or win_2:
            return Relation.SMALLER, winning_node_2
        elif len(r1_advocates) == 0 and len(r2_advocates) == 0:
            return Relation.EQUAL, None
        else:
            return Relation.NONCOMPARABLE, None

    def _compare_results(self, node_id, results1, results2, r1_advocates, r2_advocates):
        result1 = 0
        result2 = 0
        rules = self.priority_graph.nodes[node_id]["rules"]
        for rule_id in rules:
            result1 += results1[rule_id]
            result2 += results2[rule_id]

        result1 /= len(rules)
        result2 /= len(rules)

        if result1 < result2:
            r1_advocates.add(node_id)
        elif result1 > result2:
            r2_advocates.add(node_id)
        else:
            for child_node_id in self.priority_graph.successors(node_id):
                self._compare_results(
                    child_node_id, results1, results2, r1_advocates, r2_advocates
                )

    def compute_error_weight(self):
        level = {}
        for node in nx.topological_sort(self.priority_graph):
            if self.priority_graph.in_degree(node) == 0:
                level[node] = 0
            else:
                level[node] = (
                    max([level[p] for p in self.priority_graph.predecessors(node)]) + 1
                )

        ranking_map = {}
        ranking_count = {}
        for rank in sorted(level.values()):
            if rank not in ranking_count:
                ranking_count[rank] = 1
            else:
                ranking_count[rank] += 1
        count = 0
        for key, value in reversed(ranking_count.items()):
            ranking_map[key] = count
            count += value

        self.error_weight = {}  # node_id -> weight
        self.sum_error_weight = 0
        for node in level:
            self.error_weight[node] = ranking_map[level[node]]
            self.sum_error_weight += 2 ** self.error_weight[node]
        if self.verbosity >= 2:
            for key, value in sorted(self.error_weight.items()):
                print(
                    f"Node {key} {self.priority_graph.nodes[key]['rules'][key].name}: level = {value}, weight = {2**value}"
                )
            print(f"Sum of error weights: {self.sum_error_weight}")

    def compute_error_value(self, results):
        """Given a result dictionary from rule evaluation, compute the error value of the sample.
        Args:
            results (dict): A dictionary where keys are rule names and values are degrees of violations, where the degree of violation is positive iff the rule is violated.
        Returns:
            error_value (int): The computed error value.
            normalized_error_value (float): The normalized error value in [0, 1].
            violated_rules (list): A list of names of the violated rules.
        """
        if self.verbosity >= 2:
            print(f"Results:")
            for rule_name, result in results.items():
                print(f"  {rule_name}: {result.total_violation}")
        error_value = 0
        violated_rules = []
        for rule_name, result in results.items():
            if rule_name not in self.get_rule_names():
                continue
            rule_id = self.rule_name_to_rule_id[rule_name]
            node_id = self.rule_id_to_node_id[rule_id]
            if result.total_violation > 0:
                error_value += 2 ** self.error_weight[node_id]
                violated_rules.append(rule_name)
        normalized_error_value = (
            error_value / self.sum_error_weight if self.sum_error_weight > 0 else 0
        )
        return error_value, normalized_error_value, violated_rules

    def apply_config(self, config):
        for rule_id, params in config.items():
            self.rule_id_to_rule[rule_id].parameters.update(params)

    def get_config(self):
        config = {}
        for rule_id, rule in self.rule_id_to_rule.items():
            config[rule_id] = deepcopy(rule.parameters)
        return config


class Result:
    def __init__(self, minimum_violation=0, aggregation_method=max):
        self.total_violation = minimum_violation
        self.violation_history = []
        self.aggregation_method = aggregation_method

    def add(self, violation):
        self.total_violation = self.aggregation_method(
            (self.total_violation, violation)
        )
        self.violation_history.append(self.total_violation)


class Rule:
    def __init__(
        self, calculate_violation, aggregation_method, name, rule_id, **kwargs
    ):
        self.calculate_violation = calculate_violation
        self.aggregation_method = aggregation_method
        self.parameters = kwargs
        self.name = name
        self.id = rule_id

    def __call__(self, handler, step, **runtime_params):
        # merge init parameters and runtime ones
        params = {**self.parameters, **runtime_params}
        return self.calculate_violation(handler, step, **params)

    def copy(self):
        # copy parameters
        new_params = deepcopy(self.parameters)
        return Rule(
            self.calculate_violation,
            self.aggregation_method,
            self.name,
            self.id,
            **new_params,
        )

    def evaluate(self, handler, **runtime_params):
        result = Result(aggregation_method=self.aggregation_method)
        for step in range(handler.max_steps):
            result.add(self(handler, step, **runtime_params))
        return result.total_violation

    def evaluate_with_cache(
        self, handler, rule_parameter_result_dict, scenario, rule_id, **runtime_params
    ):
        params = self.parameters
        param_tuple = tuple(sorted(params.items())) if params else ()

        if rule_id in rule_parameter_result_dict:
            pass
        else:
            rule_parameter_result_dict[rule_id] = {}

        if param_tuple in rule_parameter_result_dict[rule_id]:
            pass
        else:
            rule_parameter_result_dict[rule_id][param_tuple] = {}

        if scenario in rule_parameter_result_dict[rule_id][param_tuple]:
            return rule_parameter_result_dict[rule_id][param_tuple][scenario]

        result = Result(aggregation_method=self.aggregation_method)
        for step in range(handler.max_steps):
            result.add(self(handler, step, **runtime_params))

        rule_parameter_result_dict[rule_id][param_tuple][
            scenario
        ] = result.total_violation
        return result.total_violation


class RuleEngine:
    def __init__(self, rule_id_to_rule):
        # rules is a dict: {"rule_name": Rule(...), ...}
        self.rules = rule_id_to_rule

    def evaluate(self, handler, start_index=None, end_index=None, **runtime_params):
        realization = handler.realization
        max_steps = len(realization) - 1

        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = max_steps

        # initialize results per rule
        results = {
            name: Result(aggregation_method=rule.aggregation_method)
            for name, rule in self.rules.items()
        }

        # pad initial history
        for res in results.values():
            res.violation_history += [0] * start_index

        # step loop
        for step in range(start_index, end_index + 1):
            for name, rule in self.rules.items():
                violation_score = rule(handler, step, **runtime_params)
                results[name].add(violation_score)

        # pad final history
        for res in results.values():
            res.violation_history += [res.total_violation] * (max_steps - end_index)

        return results

    def evaluate_with_cache(self, rule_parameter_result_dict, scenario):
        rule_id_to_params = {}

        for name, rule in self.rules.items():
            params = rule.parameters
            rule_id_to_params[name] = tuple(sorted(params.items())) if params else ()

        # initialize results per rule
        results = {}

        cached = set()
        for name in self.rules.keys():
            if name in rule_parameter_result_dict:
                pass
            else:
                rule_parameter_result_dict[name] = {}

            if rule_id_to_params[name] in rule_parameter_result_dict[name]:
                pass
            else:
                rule_parameter_result_dict[name][rule_id_to_params[name]] = {}

            if scenario in rule_parameter_result_dict[name][rule_id_to_params[name]]:
                violation_score = rule_parameter_result_dict[name][
                    rule_id_to_params[name]
                ][scenario]
                results[name] = violation_score
                cached.add(name)
            else:
                for d in rule_parameter_result_dict[name]:
                    print(d)
                pass
        return results
