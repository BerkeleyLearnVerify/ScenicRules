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

Below is an example speed limit rule where the violation increases linearly.

```
import numpy as np

def speed_limit_violation(handler, step, limit=20):
    # Retrieve ego state from the pool at this step
    ego_state = handler(step).ego_state
    
    # Calculate magnitude of the velocity vector
    current_speed = np.linalg.norm(ego_state.velocity)
    
    # Return the overshoot, or 0 if within the limit
    return max(0, current_speed - limit)

# Registering the rule with ID 4, this means the rule will correspond to the number 4 in your .graph file
speed_limit_rule = Rule(speed_limit_violation, max, "speed_limit", 4)
```

Below is a clearance rule where the violation is the difference between the minimum required distance and the actual distance to the nearest vehicle in proximity. 

```
def proximity_violation(handler, step, safety_threshold=2.0):
    pool = handler(step)
    # Get states of vehicles within the proximity_threshold
    nearby_vehicles = pool.vehicles_in_proximity
    max_violation = 0

    for veh_state in nearby_vehicles:
        # pool.distance(state) returns the polygonal distance to ego
        dist = pool.distance(veh_state)
        
        if dist < safety_threshold:
            # Calculate how much the safety buffer was breached
            max_violation = max(max_violation, safety_threshold - dist)
            
    return max_violation

# Registering the rule with ID 5
proximity_violation_rule = Rule(proximity_violation, max, "proximity_violation", 5)
```

## Evaluation and Scoring

The `RuleEngine` iterates through the realization to evaluate all rules. It applies the violation function of each rule at every time step and collects the scores in `Result` objects. 

The `Result` class tracks how a rule's violation score evolves over the simulation. As the engine processes each step, the `Result.add` method stores the current violation score and updates the `total_violation` value based on the aggregation logic. Beyond the final score, the object maintains a `violation_history`, which is a list containing the violation score recorded at every time step.
