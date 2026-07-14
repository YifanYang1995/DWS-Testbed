"""Evaluate a bundled GPHH priority tree on a DWS workload stream."""

from functools import partial
import random
import time

from deap import base, creator, gp, tools
import numpy as np
import pandas as pd

from config.Params import configs
from env.workflow_scheduling_v3.simulator_wf import WFEnv
from utils.evaluation_utils import (
    finalize_evaluation,
    init_wandb,
    prepare_evaluation,
    resolve_repo_path,
    set_seed,
)


def protected_div(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(left, right)
        if isinstance(result, np.ndarray):
            result[np.isinf(result) | np.isnan(result)] = 1
        elif np.isinf(result) or np.isnan(result):
            result = 1
    return result


PSET = gp.PrimitiveSet("main1", 6)
PSET.addPrimitive(np.maximum, 2)
PSET.addPrimitive(np.minimum, 2)
PSET.addPrimitive(np.add, 2)
PSET.addPrimitive(np.subtract, 2)
PSET.addPrimitive(np.multiply, 2)
PSET.addPrimitive(protected_div, 2, name="div")
PSET.addEphemeralConstant("rand101", partial(random.randint, -1, 1))
PSET.renameArguments(ARG0="TS", ARG1="RW", ARG2="ET", ARG3="FT", ARG4="CU", ARG5="UL")

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

TOOLBOX = base.Toolbox()
TOOLBOX.register("compile", gp.compile, pset=PSET)


def select_action(tree, observation):
    rule = TOOLBOX.compile(expr=tree)
    if configs.normalize:
        observation = observation / np.asarray(configs.normalize_features[1:])
    priorities = np.asarray(rule(*observation.T))
    if priorities.ndim == 0:
        priorities = np.ones(len(observation), dtype=np.float64)
    minimum_indices = np.flatnonzero(priorities == np.min(priorities))
    return int(np.random.choice(minimum_indices))


def main():
    started_at = time.time()
    prepare_evaluation(configs)
    checkpoint_path = resolve_repo_path(configs.gphh_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GPHH checkpoint not found: {checkpoint_path}")
    tree = pd.read_pickle(checkpoint_path)

    run = init_wandb("GPHH", configs)
    set_seed(configs.algo_seed, configs.device)
    env = WFEnv(configs.env_name, configs, False)
    observation = env.resetGP()
    while True:
        action = select_action(tree, observation)
        observation, _, done = env.stepGP(action)
        if done:
            finalize_evaluation(
                "GPHH", configs, env.all_flowTime, env.all_workload, started_at, run
            )
            return


if __name__ == "__main__":
    main()
