Rules
=======================================
<!-- Outline:
- Rule class
- Rule functions
- utils
- RuleEngine class
- Result class
-->

This module defines the classes and methods used to organize rules, manage their priority, and calculate how well an agent follows them during a simulation.

The `Rule` class acts as the primary method to evaluate ralizations. It stores the logic for a single rule, which includes a name, a unique ID, a `calculate_violation` function, and an `aggregation_method`. When called, the `Rule` executes its violation function using the provided `VariableHandler` and time `step`. Any parameters passed during rule initialization or at runtime are merged and passed to this function. The `Result` class stores the output, tracking the `total_violation` score and a history of violations across the simulation. The `aggregation_method` typically `max` or `sum` determines how the step-by-step scores are combined into the final result.

## How to Define a Rule

To create a new rule, you must define a Python function that calculates a violation score in a single realization step. This function needs to accept two required arguments: `VariableHandler` and `step`. Inside this function, you use the `handler` to retrieve the `VariablePool` for that specific time step—by calling `handler(step)` and then extract the specific data you need, such as vehicle velocity or object distances. If the criteria for a rule violation are met, the function should return a positive numerical score representing the magnitude of the violation; otherwise, it should return zero. For instance, if you were defining a minimum distance rule, your function would call the handler to get the current state of both the ego vehicle and the nearest adversary, calculate the distance between them, and return the difference between that distance and your safety threshold if the threshold is violated.

Once your function is defined, you pass it into the `Rule` class constructor along with a name, a numeric ID (that would match the ID you wrote for the corresponding rule in the .graph file), and an aggregation method. The aggregation method tells the system how to handle the sequence of scores returned over the entire simulation; if you want to know the single worst violation that occurred, you would pass `max`, but if you want to know the total cumulative violation, you would pass `sum`.

## Rulebook and Priority

The `Rulebook` class maintains a collection of rules and their relative importance using a directed graph. Each node in the graph represents a set of rules that share the same priority level. Edges between nodes define the priority: an edge from one node to another indicates the first is more important.

You can manage the rulebook by adding rules, creating relations between them using `add_rule_relation`, or removing them. The `add_rule_relation` method supports setting relations such as `LARGER`, `SMALLER`, or `EQUAL` to adjust the priority graph. The `Rulebook` also includes methods to visualize this hierarchy as a graph or print a textual adjacency matrix to show the priority relations between rules.

## Evaluation and Scoring

The `RuleEngine` iterates through the realization to evaluate all rules. It applies the violation function of each rule at every time step and collects the scores in `Result` objects. 

The `Result` class tracks how a rule's violation score evolves over the simulation. As the engine processes each step, the `Result.add` method stores the current violation score and updates the `total_violation` value based on the aggregation logic. Beyond the final score, the object maintains a `violation_history`, which is a list containing the violation score recorded at every time step.
