"""Evaluate a bundled ERL-DWS policy on a DWS workload stream."""

import time

import torch

from config.Params import configs
from env.workflow_scheduling_v3.simulator_wf import WFEnv
from evaluation_utils import (
    finalize_evaluation,
    init_wandb,
    prepare_evaluation,
    resolve_repo_path,
    set_seed,
)
from policy import es_rl as _checkpoint_definitions  # noqa: F401


def main():
    started_at = time.time()
    prepare_evaluation(configs)
    checkpoint_path = resolve_repo_path(configs.erldws_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ERL-DWS checkpoint not found: {checkpoint_path}")

    # The bundled file stores a complete trusted policy object, not only a state_dict.
    policy = torch.load(
        checkpoint_path,
        map_location=torch.device(configs.device),
        weights_only=False,
    )
    if hasattr(policy.model, "eval"):
        policy.model.eval()

    run = init_wandb("ERL-DWS", configs)
    set_seed(configs.algo_seed, configs.device, include_torch=True)
    env = WFEnv(configs.env_name, configs, False)
    observation = env.resetES()
    inference_time = 0.0

    while True:
        with torch.no_grad():
            inference_started = time.time()
            action, _ = policy.model(observation)
            inference_time += time.time() - inference_started
        observation, _, done = env.stepES(action)
        if done:
            finalize_evaluation(
                "ERL-DWS",
                configs,
                env.all_flowTime,
                env.all_workload,
                started_at,
                run,
                inference_time,
            )
            return


if __name__ == "__main__":
    main()
