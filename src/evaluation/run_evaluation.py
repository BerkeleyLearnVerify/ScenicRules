import logging
import os
import random
import sys
import time
from pathlib import Path

import hydra
import scenic
import timeout_decorator
from verifai.features import Categorical, Feature, FeatureSpace, Struct
from verifai.samplers import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from rulebook_benchmark.process_trajectory import process_trajectory
from rulebook_benchmark.realization import Realization, VariableHandler
from rulebook_benchmark.rule_functions import (
    f1, 
    f2, 
    f3, 
    f4, 
    f5, 
    f6, 
    f7, 
    f8, 
    f9, 
    f10,
    f11,
    f12,
    f13,
    f14,
    f15
)
from rulebook_benchmark.rulebook import Rulebook, Result, RuleEngine, Rule
from rulebook_benchmark.samplers import CrossEntropySampler, MultiArmedBanditSampler
from scenarios.utils import visualize_simulation

ruleset = {
    "vru_collision": f1,
    "vehicle_collision": f2,
    "drivable_area": f3,
    "vru_ttc": f4,
    "vru_acknowledgement": f5,
    "vehicle_ttc": f6,
    "correct_side": f7,
    "vru_offroad": f8,
    "vru_onroad": f9,
    "front_clearance_buffer": f10,
    "left_clearance_buffer": f11,
    "right_clearance_buffer": f12,
    "speed_limit": f13,
    "lane_keeping": f14,
    "lane_centering": f15
}
def reaching_goal(simulation, step):
    assert (
        "egoReachedGoal" in simulation.records
    ), "egoReachedGoal not recorded in the simulation."
    reached_goal = simulation.records["egoReachedGoal"][step][1]
    return 0 if reached_goal else 1
f16 = Rule(reaching_goal, min, "reaching_goal", 16)

ROOT = Path(__file__).parent
log = logging.getLogger(__name__)


@hydra.main(config_path=str(ROOT / "cfgs"), version_base=None)
def main(cfg):
    random.seed(cfg["experiment"]["rng_seed"])
    seeds = [random.randint(0, 1e6) for _ in range(cfg["experiment"]["num_seeds"])]
    for seed in seeds:
        run_evaluation(cfg, seed)


def run_evaluation(cfg, seed):
    ### Initialization ###
    log.info(
        f"Running evaluation for {cfg['scenic']['file_path'].split('/')[-1]} with seed: {seed}, policy: {cfg['agent']['type']}, and falsifier: {cfg['falsification']['sampler_type']}"
    )
    random.seed(seed)
    # Simulator
    if cfg["scenic"]["simulator"] == "metadrive":
        from scenic.simulators.metadrive import MetaDriveSimulator

        simulator = MetaDriveSimulator(sumo_map=cfg["scenic"]["map_file_path"])
        model = "scenic.simulators.metadrive.model"
    elif cfg["scenic"]["simulator"] == "newtonian":
        from scenic.simulators.newtonian import NewtonianSimulator

        simulator = NewtonianSimulator()
        model = "scenic.simulators.newtonian.driving_model"
    else:
        raise NotImplementedError(
            f"Simulator {cfg['scenic']['simulator']} not supported."
        )
    # Input parameter space and sampler
    scenario = scenic.scenarioFromFile(
        cfg["scenic"]["file_path"], model=model, mode2D=True
    )
    param_domain = {}
    for param, value in scenario.params.items():
        if isinstance(value, scenic.core.external_params.VerifaiRange):
            param_domain[param] = value.domain
    print(f"Parameter domain: {param_domain}")
    sampler = sampler_factory(cfg, param_domain)
    # Rulebook
    rule_id_to_rule = {1: f1, 2: f2, 3: f3, 4: f4, 5: f5, 6: f6, 7: f7, 8: f8, 9: f9, 10: f10, 11: f11, 12: f12, 13: f13, 14: f14, 15: f15, 16: f16}
    rulebook = Rulebook(rule_id_to_rule, cfg["rulebook"]["file_path"])
    rulebook.compute_error_weight()
    # Results
    avg_error_value = 0
    avg_normalized_error_value = 0
    max_error_value = 0
    max_normalized_error_value = 0
    ce_ratio = 0
    rule_violation_count = {rule: 0 for rule in rulebook.get_rule_names()}
    unique_violations = set()
    # Sampling loop
    successful_samples = 0
    retries = 0

    while successful_samples < cfg["experiment"]["num_samples"]:
        if retries >= cfg["scenic"]["max_retries"]:
            log.error(
                f"Exceeded maximum retries ({cfg['scenic']['max_retries']}). Stopping evaluation."
            )
            break

        ### Generate a sample ###
        log.info(f"Sample {successful_samples+1}/{cfg['experiment']['num_samples']}")
        realization = Realization()
        sample = sampler.getSample()
        params = {}
        param_info = "Sampled parameters: "
        if not isinstance(sample, dict):  # random, halton
            for name in param_domain.keys():
                value = getattr(sample[0].params, name)[0]
                params[name] = value
                param_info += f"{name}: {value} "
        else:  # cross-entropy, mab
            for name in param_domain.keys():
                value = sample[name]
                params[name] = value
                param_info += f"{name}: {value} "
        log.info(param_info)
        params["realization"] = realization
        params["POLICY"] = cfg["agent"]["type"]
        scenario = scenic.scenarioFromFile(
            cfg["scenic"]["file_path"], model=model, params=params, mode2D=True
        )

        ### Run the simulation and evaluate ###
        try:
            decorator = timeout_decorator.timeout(
                cfg["scenic"]["timeout"],
                timeout_exception=TimeoutError,
                use_signals=False,
            )
            error_value, normalized_error_value, violated_rules = decorator(
                simulate_and_eval
            )(simulator, scenario, realization, rulebook, cfg, idx=successful_samples)
        except TimeoutError:
            log.warning(
                f"Simulation timeout after {cfg['scenic']['timeout']}s. Retrying..."
            )
            retries += 1
            continue
        except Exception as e:
            log.error(f"Simulation failed with exception: {e}")
            retries += 1
            continue
        log.info(
            f"Error value: {error_value}, Normalized error value: {normalized_error_value}, Violated rules: {violated_rules}"
        )
        avg_error_value += error_value
        avg_normalized_error_value += normalized_error_value
        max_error_value = max(max_error_value, error_value)
        max_normalized_error_value = max(
            max_normalized_error_value, normalized_error_value
        )
        if len(violated_rules) > 0:
            ce_ratio += 1
            for rule in violated_rules:
                rule_violation_count[rule] += 1
        unique_violations.add(tuple(sorted(violated_rules)))
        successful_samples += 1
        retries = 0

        ### Update the sampler ###
        if cfg["falsification"]["active"]:
            sampler.update(sample, normalized_error_value)

    log.info(
        f"Results for {cfg['scenic']['file_path'].split('/')[-1]} with seed: {seed}, policy: {cfg['agent']['type']}, and falsifier: {cfg['falsification']['sampler_type']}"
    )
    log.info(
        f"Average error value: {avg_error_value/cfg['experiment']['num_samples']:.3f}, Average normalized error value: {avg_normalized_error_value/cfg['experiment']['num_samples']:.3f}, Counterexample ratio: {ce_ratio/cfg['experiment']['num_samples']:.3f}, Max error value: {max_error_value:.3f}, Max normalized error value: {max_normalized_error_value:.3f}"
    )
    log.info(f"Rule violation count: {rule_violation_count}")
    log.info(
        "Sample counts: "
        + str({k: [int(x - 1) for x in v] for k, v in sampler.counts.items()})
    )
    log.info(f"Number of unique violations: {len(unique_violations)}")
    unique_violations_lists = [list(s) for s in unique_violations]
    log.info("Unique violations: " + str(unique_violations_lists))

    simulator.destroy()


def simulate_and_eval(simulator, scenario, realization, rulebook, cfg, idx=0):
    """
    Safe simulation and evaluation with timeouts and exceptions.
    """
    ### Run the simulation ###
    scene, _ = scenario.generate()
    simulation = simulator.simulate(
        scene,
        maxSteps=cfg["scenic"]["max_steps_per_simulation"],
        maxIterations=cfg["scenic"]["max_rejection_sampling_iterations"],
    )

    if not simulation:
        log.error("Simulation returned None. Retrying...")
        raise Exception("Simulation returned None.")
    process_trajectory(realization, isScenic=True)

    ### Evaluate the result ###
    results = get_rule_violations(realization)
    if cfg["rulebook"]["add_reaching_goal_rule"]:
        result = Result(minimum_violation=1, aggregation_method=min)
        for t in range(len(simulation.records["egoReachedGoal"])):
            result.add(reaching_goal(simulation, t))
        results["reaching_goal"] = result
    error_value, normalized_error_value, violated_rules = rulebook.compute_error_value(
        results
    )

    if cfg["visualization"]["record_simulation"]:
        visualize_simulation(
            simulation,
            ids=cfg["visualization"]["ids"],
            violated_rules=violated_rules,
            save_path=cfg["visualization"]["record_dir"]
            + cfg["agent"]["type"]
            + f"_{idx+1}.mp4",
        )

    return error_value, normalized_error_value, violated_rules


def get_rule_violations(realization):
    handler = VariableHandler(realization)
    rule_engine = RuleEngine(ruleset)

    # Evaluate rules
    results = rule_engine.evaluate(handler)

    return results


def sampler_factory(cfg, param_domain):
    if cfg["falsification"]["sampler_type"] == "random":
        sampler = FeatureSampler.randomSamplerFor(
            FeatureSpace({"params": Feature(Struct(param_domain))})
        )
    elif cfg["falsification"]["sampler_type"] == "halton":
        sampler = FeatureSampler.haltonSamplerFor(
            FeatureSpace({"params": Feature(Struct(param_domain))})
        )
    elif cfg["falsification"]["sampler_type"] == "ce":
        alpha = cfg["falsification"]["params"]["alpha"]
        thres = cfg["falsification"]["params"]["thres"]
        buckets = cfg["falsification"]["params"]["buckets"]
        sampler = CrossEntropySampler(
            param_domain, alpha=alpha, thres=thres, buckets=buckets
        )
    elif cfg["falsification"]["sampler_type"] == "mab":
        alpha = cfg["falsification"]["params"]["alpha"]
        thres = cfg["falsification"]["params"]["thres"]
        buckets = cfg["falsification"]["params"]["buckets"]
        exploration_ratio = cfg["falsification"]["params"]["exploration_ratio"]
        sampler = MultiArmedBanditSampler(
            param_domain,
            alpha=alpha,
            thres=thres,
            buckets=buckets,
            exploration_ratio=exploration_ratio,
        )
    else:
        raise NotImplementedError(
            f"Sampler {cfg['falsification']['sampler_type']} not supported."
        )
    return sampler


if __name__ == "__main__":
    main()
