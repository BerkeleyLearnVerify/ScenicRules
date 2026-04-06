Rulebooks
=======================================

A [Rulebook](https://arxiv.org/abs/1902.09355) is a specification structure that encodes a set of objectives and their priority relations. Formally, a Rulebook can be defined as a tuple $B = (R, \preceq_R)$, where $R$ is a set of objectives and $\preceq_R$ is a preorder over $R$ that encodes the priority relations between the objectives.
A Rulebook can be represented as a directed graph, where each node corresponds to an objective and each directed edge represents a priority relation between two objectives.

In this page, we will show how to define a Rulebook specification in our benchmark and the details of the `Rulebook` class implementation.

Defining a Rulebook Specification
---------------------------------------
A Rulebook specification is defined in a `.graph` file, which is a text file that describes the rules and their priority relations. The format of the `.graph` file is as follows:

```
#header
<header_info>
#rules
<rule_id_1> 
<rule_id_2>
...
#same-level
<rule_id_a> <rule_id_b> <rule_id_c> ...
...
#priorities
<rule_id_x> <rule_id_y>
<rule_id_y> <rule_id_z>
...
```

- The `#header` section contains any additional information about the Rulebook, such as its name or description.
- The `#rules` section lists all the rules in the Rulebook, each identified by a unique `rule_id`.
- The `#same-level` section specifies groups of rules that are considered to be at the same priority level. Rules that are listed in the same line are considered to be at the same priority level.
- The `#priorities` section defines the priority relations between the rules. Each line in this section specifies a priority relation, where the first `rule_id` has higher priority than the second `rule_id`.


The Rulebook Class Implementation ([`rulebook.py`](https://github.com/BerkeleyLearnVerify/ScenicRules/blob/main/src/rulebook_benchmark/rulebook.py))
---------------------------------------

### Initialization and Parsing
Given a `.graph` file (as described above) and a mapping from rule IDs to their corresponding rule functions, the `Rulebook` class initializes by parsing the file to construct the internal representation of the rules and their priority relations:

```python
from rulebook_benchmark.rule_functions import (f1, f2, f3, ..., f16)
rule_id_to_rule = {1: f1, 2: f2, 3: f3, ..., 16: f16} # A dictionary mapping rule IDs to their corresponding functions
rulebook = Rulebook('path/to/rulebook.graph', rule_id_to_rule)
```

### Managing the Rulebook
The `Rulebook` class provides methods to manage the rules and their priority relations, including:

```python
rulebook.add_rule(rule) # Add an isolated rule object to the rulebook
rulebook.add_rule_relation(rule_id_1, rule_id_2, relation) # Add a priority relation between two rules
rulebook.remove_rule(rule_id) # Remove a rule from the rulebook. The predecessors and successors will be connected.
rulebook.remove_rule_relation(rule_id_1, rule_id_2) # Remove a priority relation between two rules
```

### Visualization
The `Rulebook` class also includes methods to visualize the priority relations between rules:

```python
rulebook.visualize_rulebook('path/to/visualization.png') # Visualize the rulebook as a graph and save it as an image
rulebook.print_adjacency_matrix() # Print the adjacency matrix of the rulebook
```

### Evaluation
To evaluate trajectories against a rulebook specification, we provide the following methods:

#### `evaluate_trajectory()`
This method takes a `Realization` object that contains the trajectory data to be evaluated and returns a dictionary that maps each rule to its corresponding violation score.

```python
results = rulebook.evaluate_trajectory(realization)
```
* Input: a `Realization` object that contains the trajectory data to be evaluated.
* Output: a dictionary that maps each rule to its corresponding violation score.

#### `compute_error_weight()` and `compute_error_value()`
Error weight is a measure of the relative importance of each rule, which is derived from the priority relations in the rulebook. Given the results from evaluating a trajectory, we can compute the error value for the trajectory by aggregating the error weights of the violated rules. The larger the error value, the more severe the trajectory violates the rulebook specification. `rulebook.compute_error_weight()` only needs to be called once.

```python
rulebook.compute_error_weight()
error_value, normalized_error_value, violated_rules = rulebook.compute_error_value(results)
```
* Input: a dictionary that maps each rule to its corresponding violation score.
* Output: a tuple containing the overall error value, the normalized error value (between 0 and 1), and a list of violated rules.

#### `compare_trajectories()`
Given two `Realization` objects that contain the trajectory data to be compared, this method returns which trajectory is better according to the rulebook specification. 

```python
relation = rulebook.compare_trajectories(realization_1, realization_2)
```
* Input: two `Realization` objects that contain the trajectory data to be compared.
* Output: the types of relation between the two trajectories, defined using a Enum `Relation` with values `LARGER`, `SMALLER`, `EQUAL`, and `NONCOMPARABLE`.

#### `compare_results()`
Given two dictionaries that map each rule to its corresponding violation score, this method returns which set of results is better according to the rulebook specification.

```python
relation = rulebook.compare_results(results_1, results_2)
```
* Input: two dictionaries that map each rule to its corresponding violation score.
* Output: the types of relation between the two sets of results, defined using a Enum `Relation` with values `LARGER`, `SMALLER`, `EQUAL`, and `NONCOMPARABLE`.
