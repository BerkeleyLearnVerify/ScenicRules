model scenic.simulators.metadrive.model
from metadrive_expert_agent import MetaDrivePolicyAgent, MetaDrivePolicyAction

METADRIVE_ACTOR = [None]
TRAJECTORY = [None]

class MetaDrivePPOPolicyCar(Car):
    actor: None
    controller: None
    switched: False

    def update_actor(self):
        self.actor = METADRIVE_ACTOR[-1]
        self.trajectory = TRAJECTORY[-1]

    def startDynamicSimulation(self):
        self.controller = MetaDrivePolicyAgent()

behavior MetaDrivePPOPolicyBehavior(egoTrajectory):
    action = MetaDrivePolicyAction()
    TRAJECTORY.append(egoTrajectory)
    take action
    while True:
        take action

behavior MetaDrivePPOFollowLaneBehavior():
    action = MetaDrivePolicyAction(reset_idx=True)
    while True:
        current_lane = self.lane
        next_lane = current_lane.successor
        egoTrajectory = [current_lane, next_lane]
        TRAJECTORY.append(egoTrajectory)
        action = MetaDrivePolicyAction(reset_idx=True)
        take action

monitor MetaDrivePPOUpdateState():
    while True:
        METADRIVE_ACTOR.append(ego.metaDriveActor)
        wait