"""
TITLE: crash_apple_082321
DESCRIPTION: An Apple vehicle, operating in manual mode and stopped at a stop sign in the Main Street Cupertino retail center parking lot, was struck on the left rear bumper by a diagonally-parked Subaru reversing out of a parking space at low speed. Both vehicles sustained minor damage. At the time of the incident, no injuries were reported by either party and law enforcement was not called to the scene.
SOURCE: DMV REPORT
POLICY: Scenic Built-in Agent
"""

#################################
# MAP AND MODEL                 #
#################################

param map = localPath('../../maps/Town05.xodr')
model scenic.domains.driving.model
param POLICY = 'built_in'

#################################
# CONSTANTS                     #
#################################

APPLE_MODEL = "vehicle.toyota.prius"
SUBARU_MODEL = "vehicle.toyota.prius"

param SUBARU_REVERSE_THROTTLE = VerifaiRange(0.3, 0.6)
param ADVERSARY_X_OFFSET = VerifaiRange(-5, -3)
param ADVERSARY_Y_OFFSET = VerifaiRange(-6, -4)
param ADVERSARY_HEADING = VerifaiRange(310, 340) # Facing roughly EW

#################################
# AGENT BEHAVIORS               #
#################################

behavior AppleStoppedBehavior():
    while True:
        take SetBrakeAction(1)

behavior SubaruReverseBehavior():
    while True:
        take SetThrottleAction(globalParameters.SUBARU_REVERSE_THROTTLE)

#################################
# SPATIAL RELATIONS             #
#################################

egoSpawnPt = new OrientedPoint

#################################
# SCENARIO SPECIFICATION        #
#################################

if globalParameters.POLICY == 'metadrive_ppo' or globalParameters.POLICY == 'ppo_with_built_in':
    from metadrive_expert import MetaDrivePPOPolicyCar, MetaDrivePPOPolicyBehavior, MetaDrivePPOUpdateState
    egoTrajectory = [network.laneAt(egoSpawnPt)]
    ego = new MetaDrivePPOPolicyCar at egoSpawnPt,
        with blueprint APPLE_MODEL,
        with behavior MetaDrivePPOPolicyBehavior(egoTrajectory)
    require monitor MetaDrivePPOUpdateState()
else:
    ego = new Car at egoSpawnPt, 
        facing 0 deg,
        with blueprint APPLE_MODEL,
        with behavior AppleStoppedBehavior()

adversary = new Car at ego offset by (ego.position.x + globalParameters.ADVERSARY_X_OFFSET,
                                      ego.position.y + globalParameters.ADVERSARY_Y_OFFSET),
    facing globalParameters.ADVERSARY_HEADING deg,
    with blueprint SUBARU_MODEL,
    with behavior SubaruReverseBehavior()

#################################
# REQUIREMENTS                  #
#################################

terminate when (distance to adversary) < (ego.length + adversary.length) / 2

from rulebook_benchmark import bench
require monitor bench.bench()

#################################
# RECORDING                     #
#################################

record True as egoReachedGoal # No specific goal for ego
record ego._boundingPolygon as egoPoly
record adversary._boundingPolygon as advPoly
record ego.lane.polygon as egoLanePoly
record adversary.lane.polygon as advLanePoly
