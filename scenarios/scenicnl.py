import sys
from google import genai


# Scenic description
def get_scenic_description():
    prompt = ""
    prompt += "You are an expert in the Scenic programming language. Your task is to convert natural language descriptions of scenarios into valid Scenic code.\n"
    # prompt += "The syntax of the Scenic language can be found here: https://docs.scenic-lang.org/en/latest/index.html\n"
    prompt += "A Scenic program consists of several sections, including:\n"
    prompt += "- 'MAP AND MODEL': for specifying the map and driving models used in the scenario\n"
    prompt += "- 'CONSTANTS': for defining parameters and constants\n"
    prompt += "- 'AGENT BEHAVIORS': for defining bahaviors of agents in the scenario\n"
    prompt += "- 'SPATIAL RELATIONS': for specifying the initial positions and trajectories (if needed) of agents\n"
    prompt += "- 'SCENARIO SPECIFICATION': for defining the object instances with their behaviors\n"
    # prompt += "- 'REQUIREMENTS': for specifying the conditions that must be met in the scenario as well as the termination condition\n"
    prompt += "Each section is clearly marked with a header (e.g., '#################################').\n"
    prompt += "When writing the Scenic code, ensure that the code is syntactically correct and all necessary sections are included and properly formatted.\n"

    # Detailed guide
    prompt += "Below is a detailed syntax guide for every section. PLEASE STRICTLY FOLLOW THE SYNTAX. DO NOT HALLUCINATE.\n"
    prompt += _get_model_guide()
    prompt += _get_constants_guide()
    prompt += _get_agent_behaviors_guide()
    prompt += _get_spatial_relations_guide()
    prompt += _get_scenario_specification_guide()

    return prompt


def _get_model_guide():
    prompt = ""
    prompt += "### MAP AND MODEL ###\n"
    prompt += "In this section, just paste the following three lines:\n"
    prompt += "param map = localPath('../../maps/Town04.xodr')\n"
    prompt += "model scenic.simulators.metadrive.model\n"
    prompt += "param POLICY = 'built_in'\n\n"
    return prompt


def _get_constants_guide():
    prompt = ""
    prompt += "### CONSTANTS ###\n"
    prompt += "In this section, define all the constants and parameters used in the scenario.\n"
    prompt += "Constants can be defined using simple assignments (e.g., MODEL = 'vehicle.lincoln.mkz_2017', EGO_INIT_DIST = [20, 25]).\n"
    prompt += "Parameters that need to be sampled during scenario generation can be defined using the 'param' keyword along with 'VerifaiRange' for continuous ranges (e.g., param EGO_SPEED = VerifaiRange(7, 10)). Later when using these parameters in the scenario, refer to them as 'globalParameters.PARAM_NAME'.\n\n"
    return prompt


def _get_agent_behaviors_guide():
    prompt = ""
    prompt += "### AGENT BEHAVIORS ###\n"
    prompt += "In this section, define the behaviors of all the agents involved in the scenario.\n"
    prompt += "Each behavior should be defined using the 'behavior' keyword followed by the behavior name and parameters (if any).\n"
    prompt += "You can adopt the built-in behaviors and actions provided by Scenic.\n"
    prompt += "When using a built-in behavior, use the 'do' keyword followed by the behavior name and its parameters. These are the built-in behaviors you can use:\n"
    prompt += "- FollowTrajectoryBehavior(target_speed, trajectory)\n"
    prompt += "- FollowLaneBehavior(target_speed, laneToFollow=None)\n"
    prompt += "- LaneChangeBehavior(laneSectionToSwitch, target_speed)\n"
    prompt += "When using a built-in action, use the 'take' keyword followed by the action name and its parameters. These are the built-in actions you can use:\n"
    prompt += "- SetBrakeAction(brake)\n"
    prompt += "- SetSteerAction(steer)\n"
    prompt += "- SetThrottleAction(throttle)\n"
    prompt += "- SetSpeedAction(speed)\n"
    prompt += "- SetVelocityAction(xVel, yVel)\n"
    prompt += "You can also use try-interrupt statements to define complex behaviors. The 'try' block contains the main behavior, while the 'interrupt when' clauses specify conditions that can interrupt the main behavior and trigger alternative actions or behaviors.\n"
    prompt += (
        "Here is an example of defining a behavior with try-interrupt statements:\n"
    )
    prompt += "behavior EgoBehavior(trajectory):\n"
    prompt += "    try:\n"
    prompt += "        do FollowTrajectoryBehavior(target_speed=globalParameters.EGO_SPEED, trajectory=trajectory)\n"
    prompt += "    interrupt when withinDistanceToAnyObjs(self, globalParameters.SAFETY_DIST):\n"
    prompt += "        take SetBrakeAction(globalParameters.EGO_BRAKE)\n"
    prompt += "    interrupt when withinDistanceToAnyObjs(self, CRASH_DIST):\n"
    prompt += "        terminate\n"
    prompt += "Here are some predefined functions or operators you may use in the interrupt conditions:\n"
    prompt += "- withinDistanceToAnyObjs(vehicle, distance): checks if the agent is within a certain distance to any other objects.\n"
    prompt += "- withinDistanceToObjsInLane(vehicle, distance): checks whether there exists any obj (1) in front of the vehicle, (2) on the same lane, and (3) within the threshold distance.\n"
    prompt += "- distance [from vector] to vector: computes the distance to the given position from ego (or the position provided with the optional from vector)\n"
    prompt += "- angle [from vector] to vector: computes the heading (azimuth) to the given position from ego (or the position provided with the optional from vector). For example, if angle to taxi is zero, then taxi is due North of ego.\n"
    prompt += "- apparent heading of vector [from vector]: computes the apparent heading of the vector, with respect to the line of sight from ego (or the position provided with the optional from vector)\n"
    prompt += "- vector in region: checks if the given position is inside the specified region\n\n"
    return prompt


def _get_spatial_relations_guide():
    prompt = ""
    prompt += "### SPATIAL RELATIONS ###\n"
    prompt += "In this section, define the initial positions and trajectories (if needed) of the agents.\n"
    prompt += "First, the hierarchy of road network structure is road -> laneGroup -> lane -> laneSection. A 'road' consists of multiple 'laneGroups', each 'laneGroup' contains multiple 'lanes', and each 'lane' is divided into multiple 'laneSections'.\n"
    prompt += "At each level, you can access its ancestors and descendants (descendants may be a list).\n"
    prompt += "In addition, for lanes, you can access its maneuvers, which represent possible driving actions (e.g., straight, left turn, right turn) that can be taken from the end of the lane.\n"
    prompt += "For laneSections (note not for lanes), you can access its fasterLane and slowerLane (if any) to facilitate lane change maneuvers.\n"
    prompt += "When defining initial positions, you can use constructs like 'new OrientedPoint in lane.centerline' to specify a random point along the centerline of a lane.\n"
    prompt += "You can also use relative positioning to define positions based on other agents or points, such as 'new OrientedPoint following roadDirection from egoSpawnPt for ADV_DIST_FROM_EGO_INIT'.\n"
    prompt += "Here are the available specifiers you can use:\n"
    prompt += "- offset by vector: Positions the object at the given coordinates in the local coordinate system of ego (which must already be defined).\n"
    prompt += "- offset along direction by vector: Positions the object at the given coordinates, in a local coordinate system centered at ego and oriented along the given direction.\n"
    prompt += "- following vectorField [from vector] for scalar: Position by following the given vector field for the given distance starting from ego or the given vector.\n"
    prompt += "- (ahead of | behind) vector [by scalar]: Positions the object to the front/back by the given scalar distance\n"
    prompt += "- (left of | right of) vector [by scalar]: Positions the object to the left/right by the given scalar distance\n"
    prompt += "When defining trajectories, you can create a list of lanes that the agent will follow during the scenario.\n\n"
    return prompt


def _get_scenario_specification_guide():
    prompt = ""
    prompt += "### SCENARIO SPECIFICATION ###\n"
    prompt += "In this section, define the object instances involved in the scenario along with their behaviors.\n"
    prompt += "Each object instance should be created using the 'new' keyword followed by the object type (e.g., Car) and its initial position.\n"
    prompt += "You can specify additional properties for each object, such as the blueprint model and the behavior to be used.\n"
    prompt += (
        "Here is an example of defining an ego vehicle and an adversary vehicle:\n"
    )
    prompt += "ego = new Car at egoSpawnPt,\n"
    prompt += "    with blueprint MODEL,\n"
    prompt += "    with behavior EgoBehavior(egoTrajectory)\n"
    prompt += "adversary = new Car at advSpawnPt,\n"
    prompt += "    with blueprint MODEL,\n"
    prompt += "    with behavior FollowLaneBehavior(target_speed=globalParameters.ADV_SPEED)\n"
    prompt += "You can also include requirements and termination conditions for the scenario using 'require' and 'terminate when' statements.\n"
    prompt += "Note that within requirements and termination conditions parameters cannot be accessed, only constants can be used.\n\n"
    return prompt


# Examples
def get_examples():
    prompt = ""
    prompt += "Here are three examples of natural language descriptions with their corresponding Scenic codes:\n"
    prompt += get_example_1()
    prompt += get_example_2()
    prompt += get_example_3()
    # prompt += get_example_4()
    return prompt


# Example 1 (examples/carla/NHTSA_Scenarios/intersection/intersection_01.scenic)
def get_example_1(id=1):
    prompt = ""
    prompt += f"Natural Language Description {id}: 'Ego vehicle goes straight at 4-way intersection and must suddenly stop to avoid collision when adversary vehicle from opposite lane makes a left turn.'\n"
    prompt += f"Scenic Code {id}:\n"
    prompt += "```scenic\n"
    prompt += "#################################\n"
    prompt += "# MAP AND MODEL                 #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "param map = localPath('../../maps/Town04.xodr')\n"
    prompt += "model scenic.simulators.metadrive.model\n"
    prompt += "param POLICY = 'built_in'\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# CONSTANTS                     #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "MODEL = 'vehicle.lincoln.mkz_2017'\n"
    prompt += "\n"
    prompt += "EGO_INIT_DIST = [20, 25]\n"
    prompt += "param EGO_SPEED = VerifaiRange(7, 10)\n"
    prompt += "param EGO_BRAKE = VerifaiRange(0.5, 1.0)\n"
    prompt += "\n"
    prompt += "ADV_INIT_DIST = [15, 20]\n"
    prompt += "param ADV_SPEED = VerifaiRange(7, 10)\n"
    prompt += "\n"
    prompt += "param SAFETY_DIST = VerifaiRange(10, 20)\n"
    prompt += "CRASH_DIST = 5\n"
    prompt += "TERM_DIST = 70\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# AGENT BEHAVIORS               #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "behavior EgoBehavior(trajectory):\n"
    prompt += "    try:\n"
    prompt += "        do FollowTrajectoryBehavior(target_speed=globalParameters.EGO_SPEED, trajectory=trajectory)\n"
    prompt += "    interrupt when withinDistanceToAnyObjs(self, globalParameters.SAFETY_DIST):\n"
    prompt += "        take SetBrakeAction(globalParameters.EGO_BRAKE)\n"
    prompt += "    interrupt when withinDistanceToAnyObjs(self, CRASH_DIST):\n"
    prompt += "        terminate\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# SPATIAL RELATIONS             #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += (
        "intersection = Uniform(*filter(lambda i: i.is4Way, network.intersections))\n"
    )
    prompt += "\n"
    prompt += "egoInitLane = Uniform(*intersection.incomingLanes)\n"
    prompt += "egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, egoInitLane.maneuvers))\n"
    prompt += "egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]\n"
    prompt += "egoSpawnPt = new OrientedPoint in egoInitLane.centerline\n"
    prompt += "\n"
    prompt += "advInitLane = Uniform(*filter(lambda m:\n"
    prompt += "        m.type is ManeuverType.STRAIGHT,\n"
    prompt += "        egoManeuver.reverseManeuvers)\n"
    prompt += "    ).startLane\n"
    prompt += "advManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN, advInitLane.maneuvers))\n"
    prompt += "advTrajectory = [advInitLane, advManeuver.connectingLane, advManeuver.endLane]\n"
    prompt += "advSpawnPt = new OrientedPoint in advInitLane.centerline\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# SCENARIO SPECIFICATION        #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "ego = new Car at egoSpawnPt,\n"
    prompt += "    with blueprint MODEL,\n"
    prompt += "    with behavior EgoBehavior(egoTrajectory)\n"
    prompt += "\n"
    prompt += "adversary = new Car at advSpawnPt,\n"
    prompt += "    with blueprint MODEL,\n"
    prompt += "    with behavior FollowTrajectoryBehavior(target_speed=globalParameters.ADV_SPEED, trajectory=advTrajectory)\n"
    prompt += "\n"
    prompt += (
        "require EGO_INIT_DIST[0] <= (distance to intersection) <= EGO_INIT_DIST[1]\n"
    )
    prompt += "require ADV_INIT_DIST[0] <= (distance from adversary to intersection) <= ADV_INIT_DIST[1]\n"
    prompt += "terminate when (distance to egoSpawnPt) > TERM_DIST\n"
    prompt += "``` \n\n"

    return prompt


# Example 2 (examples/carla/NHTSA_Scenarios/bypassing/bypassing_01.scenic)
def get_example_2(id=2):
    prompt = ""
    prompt += f"Natural Language Description {id}: 'Ego vehicle performs a lane change to bypass a slow adversary vehicle before returning to its original lane.'\n"
    prompt += f"Scenic Code {id}:\n"
    prompt += "```scenic\n"
    prompt += "#################################\n"
    prompt += "# MAP AND MODEL                 #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "param map = localPath('../../maps/Town04.xodr')\n"
    prompt += "model scenic.simulators.metadrive.model\n"
    prompt += "param POLICY = 'built_in'\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# CONSTANTS                     #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "MODEL = 'vehicle.lincoln.mkz_2017'\n"
    prompt += "\n"
    prompt += "param EGO_SPEED = VerifaiRange(7, 10)\n"
    prompt += "\n"
    prompt += "param ADV_DIST = VerifaiRange(10, 25)\n"
    prompt += "param ADV_SPEED = VerifaiRange(2, 4)\n"
    prompt += "\n"
    prompt += "BYPASS_DIST = [15, 10]\n"
    prompt += "INIT_DIST = 50\n"
    prompt += "TERM_TIME = 5\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# AGENT BEHAVIORS               #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "behavior EgoBehavior():\n"
    prompt += "    try:\n"
    prompt += "        do FollowLaneBehavior(target_speed=globalParameters.EGO_SPEED)\n"
    prompt += "    interrupt when withinDistanceToAnyObjs(self, BYPASS_DIST[0]):\n"
    prompt += "        fasterLaneSec = self.laneSection.fasterLane\n"
    prompt += "        do LaneChangeBehavior(\n"
    prompt += "                laneSectionToSwitch=fasterLaneSec,\n"
    prompt += "                target_speed=globalParameters.EGO_SPEED)\n"
    prompt += "        do FollowLaneBehavior(\n"
    prompt += "                target_speed=globalParameters.EGO_SPEED,\n"
    prompt += "                laneToFollow=fasterLaneSec.lane) \ \n"
    prompt += "            until (distance to adversary) > BYPASS_DIST[1]\n"
    prompt += "        slowerLaneSec = self.laneSection.slowerLane\n"
    prompt += "        do LaneChangeBehavior(\n"
    prompt += "                laneSectionToSwitch=slowerLaneSec,\n"
    prompt += "                target_speed=globalParameters.EGO_SPEED)\n"
    prompt += "        do FollowLaneBehavior(target_speed=globalParameters.EGO_SPEED) for TERM_TIME seconds\n"
    prompt += "        terminate \n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# SPATIAL RELATIONS             #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "initLane = Uniform(*network.lanes)\n"
    prompt += "egoSpawnPt = new OrientedPoint in initLane.centerline\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# SCENARIO SPECIFICATION        #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "ego = new Car at egoSpawnPt,\n"
    prompt += "    with blueprint MODEL,\n"
    prompt += "    with behavior EgoBehavior()\n"
    prompt += "\n"
    prompt += (
        "adversary = new Car following roadDirection for globalParameters.ADV_DIST,\n"
    )
    prompt += "    with blueprint MODEL,\n"
    prompt += "    with behavior FollowLaneBehavior(target_speed=globalParameters.ADV_SPEED)\n"
    prompt += "\n"
    prompt += "require (distance to intersection) > INIT_DIST\n"
    prompt += "require (distance from adversary to intersection) > INIT_DIST\n"
    prompt += "require always (adversary.laneSection._fasterLane is not None)\n"
    prompt += "``` \n\n"

    return prompt


# Example 3 (crash_waymo-august-12-2019)
def get_example_3(id=3):
    prompt = ""
    prompt += f"Natural Language Description {id}: 'A Waymo Autonomous Vehicle (“Waymo AV”) was in autonomous mode on northbound S. Rengstorff Avenue at Crisanto Avenue in Mountain View when it was rear-ended. After starting to proceed following a red-to-green traffic light change, the Waymo AV yielded to a bicyclist who merged from the bike lane into the Waymo AV's travel lane, and a passenger vehicle then made contact with the rear bumper of the Waymo AV. The passenger vehicle was traveling at approximately 8 MPH, and the Waymo AV was traveling at approximately 3 MPH.  The Waymo AV sustained minor damage to its rear bumper, and the passenger vehicle sustained minor damage to its front bumper. There were no injuries reported at the scene.'\n"
    prompt += f"Scenic Code {id}:\n"
    prompt += "```scenic\n"
    prompt += "#################################\n"
    prompt += "# MAP AND MODEL                 #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "param map = localPath('../../maps/Town04.xodr')\n"
    prompt += "model scenic.simulators.metadrive.model\n"
    prompt += "param POLICY = 'built_in'\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# CONSTANTS                     #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += 'MODEL = "vehicle.toyota.prius"\n'
    prompt += 'BICYCLE_MODEL = "vehicle.bh.crossbike"\n'
    prompt += "\n"
    prompt += "param WAYMO_SPEED = VerifaiRange(2.5, 3.5)\n"
    prompt += "param WAYMO_BRAKE = VerifaiRange(0.5, 1.0)\n"
    prompt += "param PASSENGER_SPEED = VerifaiRange(7, 9)\n"
    prompt += "param PASSENGER_DIST = VerifaiRange(-6, -8)\n"
    prompt += "param SAFETY_DIST = VerifaiRange(3, 5)\n"
    prompt += "BICYCLE_SPEED = 2\n"
    prompt += "BICYCLE_DIST = 10\n"
    prompt += "INIT_DIST = 20\n"
    prompt += "TERM_DIST = 40\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# AGENT BEHAVIORS               #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "behavior WaymoBehavior(trajectory):\n"
    prompt += "    try:\n"
    prompt += "        do FollowTrajectoryBehavior(target_speed=globalParameters.WAYMO_SPEED, trajectory=trajectory)\n"
    prompt += "    interrupt when withinDistanceToAnyObjs(self, globalParameters.SAFETY_DIST):\n"
    prompt += "        take SetBrakeAction(globalParameters.WAYMO_BRAKE)\n"
    prompt += "\n"
    prompt += "behavior PassengerBehavior(trajectory):\n"
    prompt += "    do FollowTrajectoryBehavior(target_speed=globalParameters.PASSENGER_SPEED, trajectory=trajectory)\n"
    prompt += "\n"
    prompt += "behavior BicycleBehavior():\n"
    prompt += "    fasterLaneSec = self.laneSection.fasterLane\n"
    prompt += "    do LaneChangeBehavior(laneSectionToSwitch=fasterLaneSec, target_speed=BICYCLE_SPEED)\n"
    prompt += "    do FollowLaneBehavior(target_speed=BICYCLE_SPEED, laneToFollow=fasterLaneSec.lane)\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# SPATIAL RELATIONS             #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += (
        "intersection = Uniform(*filter(lambda i: i.is4Way, network.intersections))\n"
    )
    prompt += "\n"
    prompt += "egoInitLane = Uniform(*intersection.incomingLanes)\n"
    prompt += "egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, egoInitLane.maneuvers))\n"
    prompt += "egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]\n"
    prompt += "egoSpawnPt = new OrientedPoint in egoInitLane.centerline\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# SCENARIO SPECIFICATION        #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "ego = new Car at egoSpawnPt,\n"
    prompt += "    with blueprint MODEL,\n"
    prompt += "    with behavior WaymoBehavior(egoTrajectory)\n"
    prompt += "\n"
    prompt += "adversary = new Car following roadDirection for globalParameters.PASSENGER_DIST,\n"
    prompt += "    with blueprint MODEL,\n"
    prompt += "    with behavior FollowLaneBehavior(target_speed=globalParameters.PASSENGER_SPEED)\n"
    prompt += "\n"
    prompt += "bicycle = new Bicycle offset by (4, BICYCLE_DIST),\n"
    prompt += "    with blueprint BICYCLE_MODEL,\n"
    prompt += "    with behavior BicycleBehavior()\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# REQUIREMENTS                  #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "require (distance to intersection) < INIT_DIST\n"
    prompt += "require (distance from adversary to intersection) < INIT_DIST\n"
    prompt += "#require ego.laneSection._slowerLane is not None\n"
    prompt += "require bicycle.laneSection._fasterLane is not None\n"
    prompt += "require next ego.lane is not bicycle.lane\n"
    prompt += (
        "terminate when (distance to adversary) < (ego.length + adversary.length) / 2\n"
    )
    prompt += "terminate when (distance to egoSpawnPt) > TERM_DIST\n"
    prompt += "``` \n\n"

    return prompt


# Example 4 (crash_apple_082321)
def get_example_4(id=4):
    prompt = ""
    prompt += f"Natural Language Description {id}: 'An Apple vehicle, operating in manual mode and stopped at a stop sign in the Main Street Cupertino retail center parking lot, was struck on the left rear bumper by a diagonally-parked Subaru reversing out of a parking space at low speed. Both vehicles sustained minor damage. At the time of the incident, no injuries were reported by either party and law enforcement was not called to the scene.'\n"
    prompt += f"Scenic Code {id}:\n"
    prompt += "```scenic\n"
    prompt += "#################################\n"
    prompt += "# MAP AND MODEL                 #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "param map = localPath('../../maps/Town04.xodr')\n"
    prompt += "model scenic.simulators.metadrive.model\n"
    prompt += "param POLICY = 'built_in'\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# CONSTANTS                     #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += 'APPLE_MODEL = "vehicle.toyota.prius"\n'
    prompt += 'SUBARU_MODEL = "vehicle.toyota.prius"\n'
    prompt += "\n"
    prompt += "param SUBARU_REVERSE_THROTTLE = VerifaiRange(0.3, 0.6)\n"
    prompt += "param ADVERSARY_X_OFFSET = VerifaiRange(-5, -3)\n"
    prompt += "param ADVERSARY_Y_OFFSET = VerifaiRange(-6, -4)\n"
    prompt += "param ADVERSARY_HEADING = VerifaiRange(310, 340) # Facing roughly EW\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# AGENT BEHAVIORS               #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "behavior AppleStoppedBehavior():\n"
    prompt += "    while True:\n"
    prompt += "        take SetBrakeAction(1)\n"
    prompt += "\n"
    prompt += "behavior SubaruReverseBehavior():\n"
    prompt += "    while True:\n"
    prompt += (
        "        take SetThrottleAction(globalParameters.SUBARU_REVERSE_THROTTLE)\n"
    )
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# SPATIAL RELATIONS             #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "egoSpawnPt = new OrientedPoint\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# SCENARIO SPECIFICATION        #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "ego = new Car at egoSpawnPt, \n"
    prompt += "    facing 0 deg,\n"
    prompt += "    with blueprint APPLE_MODEL,\n"
    prompt += "    with behavior AppleStoppedBehavior()\n"
    prompt += "\n"
    prompt += "adversary = new Car at ego offset by (ego.position.x + globalParameters.ADVERSARY_X_OFFSET,\n"
    prompt += "                                      ego.position.y + globalParameters.ADVERSARY_Y_OFFSET),\n"
    prompt += "    facing globalParameters.ADVERSARY_HEADING deg,\n"
    prompt += "    with blueprint SUBARU_MODEL,\n"
    prompt += "    with behavior SubaruReverseBehavior()\n"
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# REQUIREMENTS                  #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += (
        "terminate when (distance to adversary) < (ego.length + adversary.length) / 2\n"
    )
    prompt += "\n"
    prompt += "#################################\n"
    prompt += "# RECORDING                     #\n"
    prompt += "#################################\n"
    prompt += "\n"
    prompt += "record ego._boundingPolygon as egoPoly\n"
    prompt += "record adversary._boundingPolygon as advPoly\n\n"

    return prompt


# Instruction
def get_instruction():
    prompt = ""
    prompt += "Please just return the Scenic program following the format of the provided example. DO NOT include any additional text or explanations. DO NOT ADD COMMENTS in the generated Scenic program!\n"
    return prompt


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        model = lines[1].strip()
    
    print(f"Processing report: {sys.argv[2]}")
    fr = open(sys.argv[2], "r")
    description = fr.read()
    description = " ".join(description.split())
    fr.close()

    prompt = ""
    prompt += get_scenic_description()
    prompt += get_examples()
    prompt += "Now, convert the following natural language description into a Scenic program:\n"
    prompt += description
    prompt += "\n\n"
    prompt += get_instruction()
    # print(prompt)

    print("Waiting for response...")
    client = genai.Client(api_key=api_key)
    chat = client.chats.create(model=model)
    response = chat.send_message(prompt)
    # print(response.text)

    fw = open(sys.argv[3], "w")
    fw.write('"""\n')
    fw.write(f"TITLE: {sys.argv[2]}\n")
    fw.write(f"DESCRIPTION: {description}\n")
    fw.write(f"SOURCE: California DMV Crash Reports\n")
    fw.write(f"GENERATED BY: {model}\n")
    fw.write('"""\n\n')
    scenic_program = "\n".join(response.text.splitlines()[1:-1])
    fw.write(scenic_program)
    fw.close()
