Preprocessing of Simulation Results
=======================================
<!-- Outline:
- Overview of the preprocessing steps
- process_trajectory.py
- realization.py
- bench.scenic
-->

This document describes the pipeline for collecting trajectory information from Scenic simulations, storing them in relevant data structures, and further processing the trajectories to track each object's occupied lanes throughout the realization.

## Collection (`bench.scenic`)

The `bench.scenic` script acts as a monitor within the Scenic environment. Its function is to extract simulation state data at each time step and map it to internal data structures defined in `realization.py`. Upon starting the simulation, the monitor retrieves the `realization` object from the global parameters and binds Scenic's road network to the realization using the provided map file while initializing the list of objects.

The monitor then collects and stores objects by iterating through the simulation’s initial object list. It instantiates a `RealizationObject` for each entity, recording the unique ID, physical dimensions, and class name, storing these instances in the `realization.objects` list. Finally, the monitor enters a loop that executes once per simulation step to perform state collection. For every object currently in the simulation, a `State` object is generated to capture the position, velocity, orientation, and current step index, which is then appended to the object’s specific `trajectory` list within the `RealizationObject`.

## Trajectory Abstraction (`realization.py`)

The `realization.py` module defines the core data structures used to store and evaluate the recorded simulation data. It provides the abstractions necessary to navigate the state information of the objects throughout the simulation duration.

The `RealizationObject` acts as the primary container for an entity. It holds static information such as physical dimensions and unique IDs while maintaining a `trajectory` list that serves as a sequential log of `State` objects. The `State` class stores kinematic data including position, velocity, and orientation for a specific time step. It also computes geometric representations, such as the object's polygonal footprint.

The `Realization` class aggregates the map, all `RealizationObject` instances, and the ego vehicle index, essentially acting as the central storage for the entire simulation result. To facilitate efficient access to these structures, the `VariableHandler` and `VariablePool` act as interfaces. These classes cache geometric data such as inter-object distances and collision timelines to enable fast evaluation for rulebooks.

## Processing (`process_trajectory.py`)

The `process_trajectory.py` module determines the sequence of lanes an object occupies throughout a simulation. It identifies occupied lanes at each time step, resolving ambiguities that arise at intersections or any other instances where an object might appear to occupy multiple lanes simultaneously.

The module first performs spatial querying by using an `STRtree` to identify candidate lanes that intersect with an object's position or polygonal footprint at every time step. During the initial classification pass, known as `firstPass`, the module evaluates each state based on these polygon intersections. States that map to a single lane are assigned immediately, whereas states mapping to multiple lanes or no lanes are marked as ambiguous.

Finally, the module employs ambiguity resolution in a second pass, `secondPass`, to process these uncertain states by analyzing past and future trajectory data. It first checks for consistency with previously assigned lanes. If a state remains ambiguous, the module traverses the road network to identify valid successors or predecessors within intersections. As a final fallback, it selects the lane with the orientation most closely aligned with the object's actual heading.