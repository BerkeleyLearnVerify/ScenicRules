"""
TITLE: crash_apple_10232023
DESCRIPTION: On October 23rd, a test vehicle operating in manual driving mode was stopped at a red light in the righthand lane of Lawrence Expressway at the intersection with Kifer Road in Santa Clara. A Lexus RX300 attempted to pass through traffic in the adjacent bike lane, and made contact with the test vehicle's side mirror. No injuries were reported and law enforcement was not called to the scene. The test vehicle sustained minor damage.
SOURCE: DMV REPORT
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

EGO_MODEL = "vehicle.toyota.prius"
ADVERSARY_MODEL = "vehicle.toyota.prius"

param ADVERSARY_THROTTLE = VerifaiRange(0.2, 0.5)
param ADVERSARY_OFFSET_LONG = VerifaiRange(-8, -4)
param ADVERSARY_OFFSET_LAT = VerifaiRange(1.5, 2.5)

INIT_DIST_TO_INTERSECTION = 20
TERM_DIST = 20

#################################
# AGENT BEHAVIORS               #
#################################

behavior StoppedBehavior():
    while True:
        take SetBrakeAction(1)

behavior AdversaryBikeLanePassBehavior(throttle):
    while True:
        take SetThrottleAction(throttle)

#################################
# SPATIAL RELATIONS             #
#################################

intersection = Uniform(*filter(lambda i: i.is4Way, network.intersections))

egoInitLane = Uniform(*intersection.incomingLanes)
egoSpawnPt = new OrientedPoint in egoInitLane.centerline

#################################
# SCENARIO SPECIFICATION        #
#################################

if globalParameters.POLICY == 'metadrive_ppo' or globalParameters.POLICY == 'ppo_with_built_in':
    from metadrive_expert import MetaDrivePPOPolicyCar, MetaDrivePPOPolicyBehavior, MetaDrivePPOUpdateState
    egoTrajectory = [egoInitLane]
    ego = new MetaDrivePPOPolicyCar at egoSpawnPt,
        with blueprint EGO_MODEL,
        with behavior MetaDrivePPOPolicyBehavior(egoTrajectory)
    require monitor MetaDrivePPOUpdateState()
else:
    ego = new Car at egoSpawnPt,
        with blueprint EGO_MODEL,
        with behavior StoppedBehavior()

adversary = new Car at ego offset by (ego.width, globalParameters.ADVERSARY_OFFSET_LONG),
    with blueprint ADVERSARY_MODEL,
    with behavior AdversaryBikeLanePassBehavior(globalParameters.ADVERSARY_THROTTLE)

#################################
# REQUIREMENTS                  #
#################################

require (distance to intersection) < INIT_DIST_TO_INTERSECTION
require ego.laneSection._slowerLane is not None
require next adversary.lane is not ego.lane

terminate when (distance to adversary) < (ego.width + adversary.width) * 0.52
terminate when (distance from adversary to egoSpawnPt) > TERM_DIST

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
