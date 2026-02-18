"""
The template classes for interfacing driving agents with Scenic.
"""

from abc import abstractmethod

from scenic.core.simulators import Action


class Agent:
    def __init__(self):
        pass

    @abstractmethod
    def run_step(self, *args, **kwargs):
        """
        Return the next control action for the agent.
        """
        raise NotImplementedError("run_step must be implemented by the subclass.")


class AgentAction(Action):
    def applyTo(self, agent, *args, **kwargs):
        """
        Call the run_step function in the Agent class and apply the action to the Scenic agent.
        """
        raise NotImplementedError("applyTo must be implemented by the subclass.")
