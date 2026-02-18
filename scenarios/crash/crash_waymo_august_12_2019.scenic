"""
TITLE: crash_waymo-august-12-2019
DESCRIPTION: A Waymo Autonomous Vehicle (“Waymo AV”) was in autonomous mode on northbound S. Rengstorff Avenue at Crisanto Avenue in
Mountain View when it was rear-ended. After starting to proceed following a red-to-green traffic light change, the Waymo AV yielded to a
bicyclist who merged from the bike lane into the Waymo AV's travel lane, and a passenger vehicle then made contact with the rear bumper
of the Waymo AV. The passenger vehicle was traveling at approximately 8 MPH, and the Waymo AV was traveling at approximately 3 MPH. 
The Waymo AV sustained minor damage to its rear bumper, and the passenger vehicle sustained minor damage to its front bumper.
There were no injuries reported at the scene.
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

MODEL = "vehicle.toyota.prius"
BICYCLE_MODEL = "vehicle.bh.crossbike"

param WAYMO_SPEED = VerifaiRange(2.5, 3.5)
param WAYMO_BRAKE = VerifaiRange(0.5, 1.0)
param SAFETY_DIST = VerifaiRange(3, 5)
param PASSENGER_SPEED = VerifaiRange(7, 9)
param PASSENGER_DIST = VerifaiRange(-6, -8)
BICYCLE_SPEED = 2
BICYCLE_DIST = 10
INIT_DIST = 20
TERM_DIST = 40

#################################
# AGENT BEHAVIORS               #
#################################

behavior WaymoBehavior(trajectory):
    try:
        do FollowTrajectoryBehavior(target_speed=globalParameters.WAYMO_SPEED, trajectory=trajectory)
    interrupt when withinDistanceToAnyObjs(self, globalParameters.SAFETY_DIST):
        take SetBrakeAction(globalParameters.WAYMO_BRAKE)

behavior PassengerBehavior(trajectory):
    do FollowTrajectoryBehavior(target_speed=globalParameters.PASSENGER_SPEED, trajectory=trajectory)

behavior BicycleBehavior():
    fasterLaneSec = self.laneSection.fasterLane
    do LaneChangeBehavior(laneSectionToSwitch=fasterLaneSec, target_speed=BICYCLE_SPEED)
    do FollowLaneBehavior(target_speed=BICYCLE_SPEED, laneToFollow=fasterLaneSec.lane)

#################################
# SPATIAL RELATIONS             #
#################################

intersection = Uniform(*filter(lambda i: i.is4Way, network.intersections))

egoInitLane = Uniform(*intersection.incomingLanes)
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, egoInitLane.maneuvers))
egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
egoSpawnPt = new OrientedPoint in egoInitLane.centerline

#################################
# SCENARIO SPECIFICATION        #
#################################

if globalParameters.POLICY == 'metadrive_ppo' or globalParameters.POLICY == 'ppo_with_built_in':
    from metadrive_expert import MetaDrivePPOPolicyCar, MetaDrivePPOPolicyBehavior, MetaDrivePPOUpdateState
    ego = new MetaDrivePPOPolicyCar at egoSpawnPt,
        with blueprint MODEL,
        with behavior MetaDrivePPOPolicyBehavior(egoTrajectory)
    require monitor MetaDrivePPOUpdateState()
else:
    ego = new Car at egoSpawnPt,
        with blueprint MODEL,
        with behavior WaymoBehavior(egoTrajectory)

adversary = new Car following roadDirection for globalParameters.PASSENGER_DIST,
    with blueprint MODEL,
    with behavior FollowLaneBehavior(target_speed=globalParameters.PASSENGER_SPEED)

bicycle = new Car offset by (4, BICYCLE_DIST),
    with blueprint BICYCLE_MODEL,
    with behavior BicycleBehavior()

#################################
# REQUIREMENTS                  #
#################################

require (distance to intersection) < INIT_DIST
require (distance from adversary to intersection) < INIT_DIST
#require ego.laneSection._slowerLane is not None
require bicycle.laneSection._fasterLane is not None
require next ego.lane is not bicycle.lane
#terminate when (distance to adversary) < (ego.length + adversary.length) / 2
terminate when (distance to egoSpawnPt) > TERM_DIST

from rulebook_benchmark import bench
require monitor bench.bench()

#################################
# RECORDING                     #
#################################

record ego in egoManeuver.endLane as egoReachedGoal

record ego._boundingPolygon as egoPoly
record adversary._boundingPolygon as advPoly
record bicycle._boundingPolygon as bicyclePoly

record ego.lane.polygon as egoLanePoly
record adversary.lane.polygon as advLanePoly
record bicycle.lane.polygon as bicycleLanePoly
record egoTrajectory[1].polygon as egoConnectingLanePoly
record egoTrajectory[2].polygon as egoEndLanePoly
