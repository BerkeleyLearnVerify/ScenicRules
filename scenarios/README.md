Scenarios
===

We collect two types of scenarios in this benchmark: **basic maneuver scenarios** and **near-accident scenarios**. The former represent basic behaviors of vehicles (e.g., lane change, turns, etc.) and serve as building blocks for more complicated scenarios. The latter reconstruct scenarios from DMV crash reports and serve as critical traffic situations.

Creating a Scenic Scenario
---
To test different driving policies for the scenario, we introduce a parameter `POLICY` in each Scenic file. Currently, we support three policies:
1. `'built_in'`: Use the behaviors defined in the Scenic files to control the ego vehicle. The behaviors are basically rule-based planners with PID controllers.
2. `'metadrive_ppo'`: Use the MetaDrive PPO agent to control the ego vehicle (see `src/agents/` for more details). We assume the trajectory of the ego vehicle is given. Below is an example of how to set the ego's behavior to `MetaDrivePPOPolicyBehavior`.
```scenic
from metadrive_expert import MetaDrivePPOPolicyCar, MetaDrivePPOPolicyBehavior, MetaDrivePPOUpdateState
ego = new MetaDrivePPOPolicyCar at egoSpawnPt,
    with blueprint MODEL,
    with behavior MetaDrivePPOPolicyBehavior(egoTrajectory)
require monitor MetaDrivePPOUpdateState()
```
3. `'ppo_with_built_in'`: Sometimes if the ego's trajectory is not available, we may need to use rule-based planners to guide the ego vehicle, while using the MetaDrive PPO agent to perform the control. Here is an example, where the MetaDrive PPO agent only focuses on the lane following parts:
```scenic
behavior EgoPPOBehavior():
    try:
        do MetaDrivePPOFollowLaneBehavior()
    interrupt when withinDistanceToAnyObjs(self, BYPASS_DIST[0]):
        fasterLaneSec = self.laneSection.fasterLane
        do LaneChangeBehavior(
                laneSectionToSwitch=fasterLaneSec,
                target_speed=globalParameters.EGO_SPEED)
        do MetaDrivePPOFollowLaneBehavior() \
            until (distance to adversary) > BYPASS_DIST[1] and (apparent heading of adversary) > 1.57
        slowerLaneSec = self.laneSection.slowerLane
        do LaneChangeBehavior(
                laneSectionToSwitch=slowerLaneSec,
                target_speed=globalParameters.EGO_SPEED)
        do MetaDrivePPOFollowLaneBehavior() for TERM_TIME seconds
        terminate 
```

Basic Maneuver Scenarios
---

Near-Accident Scenarios
---
### LLM-Assisted Scenic Code Generation

We provide an LLM-assisted flow to generate Scenic code from DMV crash reports written in natural language ([source of the reports](https://github.com/KE7/ScenarioNL-CA-AV-Crash/tree/86f72268c5320be8a92ec6b3d76ef6963f668cf0/crash_reports/hard)). In the prompt, we guide the LLM by providing the typical structure of a Scenic program and several crash report-Scenic code example pairs. See `scenicnl.py` for more details. To run the flow, users need to paste their Google Gemini API keys in `scenicnl.py`.
