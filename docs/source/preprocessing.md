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

To be able to execute the trajectory collection, you have to add `require monitor bench.bench()` to the bottom of your Scenic program, after making the necessary import. Here is an example Scenic program to demonstrate: 

```
param map = localPath('../maps/Town05.xodr')
model scenic.domains.driving.model

ego = new Car with behavior FollowLaneBehavior

from rulebook_benchmark import bench
require monitor bench.bench()
```
**Warning:** Any defined custom Scenic class name should end with the name its the inherited class' traffic actor (Car, Truck, Pedestrian, Bicycle), e.g. `Example(Car)` should end with "Car": `ExampleCar(Car)`, or `Example(Pedestrian)` should be `ExamplePedestrian(Pedestrian)`.

## Trajectory Abstraction (`realization.py`)

The `realization.py` module defines the core data structures used to store and evaluate the recorded simulation data. It provides the abstractions necessary to navigate the state information of the objects throughout the simulation duration.

The `RealizationObject` acts as the primary container for an entity. It holds static information such as physical dimensions and unique IDs while maintaining a `trajectory` list that serves as a sequential log of `State` objects. The `State` class stores kinematic data including position, velocity, and orientation for a specific time step. It also computes geometric representations, such as the object's polygonal footprint.

The `Realization` class aggregates the map, all `RealizationObject` instances, and the ego vehicle index, essentially acting as the central storage for the entire simulation result. To facilitate efficient access to these structures, the `VariableHandler` and `VariablePool` act as interfaces. These classes cache geometric data such as inter-object distances and collision timelines to enable fast evaluation for rulebooks.

### `VariableHandler`

The `VariableHandler` serves as the primary interface for accessing state and trajectory data, and managing the lifecycle of `VariablePool` objects. It handles world state caching and collision tracking for easy rule evaluation, while allowing quick access to object IDs and traffic network. 

### Key Functionalities

- It instantiates and caches `VariablePool` for a specific simulation step when called. This ensures that expensive computations (like spatial queries) are not repeated between multiple rules that check similar information.
    
- It leverages `shapely` to create a `trajectory_buffer`, which represents the entire physical space occupied by the ego vehicle throughout the simulation. This is useful for identifying objects of interest that ever entered the ego's potential path.
    
- The `collision_timeline` property iterates through the simulation to identify contiguous frames where objects intersect. It returns a dictionary mapping object UIDs to a list of `(start_step, end_step)` tuples.
    

### Usage Examples

To retrieve data for a specific step, you simply call the handler instance with the desired step index. This returns a `VariablePool` containing the state of all actors at that moment:

```
# Assuming 'handler' is an instance of VariableHandler
step_20_data = handler(20)

# Accessing vehicles colliding with ego at step 20
colliding_vehs = step_20_data.vehicles_colliding
```

The `collision_timeline` is particularly useful for post-simulation analysis to determine exactly when and for how long a conflict occurred:

```
# Get the timeline of all collisions
timeline = handler.collision_timeline

for uid, intervals in timeline.items():
    for start, end in intervals:
        print(f"Object {uid} collided from step {start} to {end}")
```

### `VariablePool`

While `VariableHandler` is useful for retrieving more global properties, `VariablePool` is especially handy for getting state-specific information regarding ego. It provides a snapshot of the simulation at a specific time step, handling the heavy lifting for spatial queries between the ego vehicle and its surroundings. Instead of calculating distances for every object in the scene, it uses a proximity_threshold to filter for relevant actors.

### Spatial Filtering and Collisions

The pool distinguishes between standard vehicles and **VRUs** (Vulnerable Road Users). It first performs a fast radial check to identify objects "in proximity" before performing more expensive polygonal intersection tests to detect actual collisions.

```
# Check if any pedestrians are currently colliding with the ego
current_step = handler(50)
pedestrian_collisions = current_step.vrus_colliding

if pedestrian_collisions:
    print(f"Collision detected with UIDs: {list(pedestrian_collisions.keys())}")
```

### Trajectory Buffering

A key feature is the ability to generate buffers for the ego's path both ahead of and behind its current position. This is useful for determining if an object is currently blocking the ego's intended route or if an object is behind ego.

```
# Get a polygon representing the ego's path in the future
front_buffer = current_step.trajectory_front_buffer

# Check if a specific vehicle's current position is within that future path
other_veh = current_step.vehicle_states[0]
if front_buffer.contains(other_veh.polygon):
    print("Vehicle is directly in the ego's future path.")
```

The pool also provides both `distance` (precise polygonal distance) and `center_distance` (fast Euclidean distance between centers), allowing you to choose the level of precision needed for your specific rule or heuristic.

## Processing (`process_trajectory.py`)

The `process_trajectory.py` module determines the sequence of lanes an object occupies throughout a simulation. It identifies occupied lanes at each time step, resolving ambiguities that arise at intersections or any other instances where an object might appear to occupy multiple lanes simultaneously.

The module first performs spatial querying by using an `STRtree` to identify candidate lanes that intersect with an object's position or polygonal footprint at every time step. During the initial classification pass, known as `firstPass`, the module evaluates each state based on these polygon intersections. States that map to a single lane are assigned immediately, whereas states mapping to multiple lanes or no lanes are marked as ambiguous.

Finally, the module employs ambiguity resolution in a second pass, `secondPass`, to process these uncertain states by analyzing past and future trajectory data. It first checks for consistency with previously assigned lanes. If a state remains ambiguous, the module traverses the road network to identify valid successors or predecessors within intersections. As a final fallback, it selects the lane with the orientation most closely aligned with the object's actual heading.