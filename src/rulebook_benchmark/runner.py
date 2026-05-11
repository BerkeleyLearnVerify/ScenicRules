import random

from scenic import scenarioFromFile
from scenic.simulators.metadrive import MetaDriveSimulator

from rulebook_benchmark.process_trajectory import process_trajectory
from rulebook_benchmark.realization import Realization


def run_scenic_metadrive(
    scenic_path,
    max_steps,
    proximity_threshold=3.0,
    ego_index=0,
    seed=None,
    render=False,
):
    if seed is not None:
        random.seed(seed)

    realization = Realization(
        ego_index=ego_index, proximity_threshold=proximity_threshold
    )
    scenario = scenarioFromFile(
        scenic_path,
        params={"realization": realization},
        model="scenic.simulators.metadrive.model",
        mode2D=True,
    )
    scene, _ = scenario.generate()
    simulator = MetaDriveSimulator(sumo_map=scenario.params["sumo_map"], render=render)
    simulation = simulator.simulate(scene, maxSteps=max_steps)
    process_trajectory(realization)
    return realization
