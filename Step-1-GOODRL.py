"""Evaluate a bundled offline GOODRL actor on a DWS workload stream."""

import time

import torch

from config.Params import configs
from env.workflow_scheduling_v3.simulator_wf import WFEnv
from utils.evaluation_utils import (
    ROOT,
    finalize_evaluation,
    init_wandb,
    prepare_evaluation,
    set_seed,
)
from policy.actor3 import BatchGraph, PPO


def build_actor():
    algorithm = PPO(
        input_dim_wf=configs.input_dim_wf,
        input_dim_vm=configs.input_dim_vm,
        hidden_dim=configs.hidden_dim,
        c_hidden_dim=configs.c_hidden_dim,
        gnn_layers=configs.gnn_layers,
        atten_layers=configs.atten_layers,
        mlp_layers=configs.mlp_layers,
        heads=configs.heads,
        dropout=configs.dropout,
        activate_fn=configs.activate_fn,
    )
    checkpoint_path = (
        ROOT / "pretrained" / "goodrl" / "actors" / f"a_{configs.online_start_ac}.pth"
    )
    if not checkpoint_path.is_file():
        available = ", ".join(path.stem.removeprefix("a_") for path in checkpoint_path.parent.glob("a_*.pth"))
        raise FileNotFoundError(
            f"GOODRL checkpoint not found: {checkpoint_path}. Available identifiers: {available}"
        )
    state_dict = torch.load(
        checkpoint_path,
        map_location=torch.device(configs.device),
        weights_only=True,
    )
    algorithm.actor.load_state_dict(state_dict)
    algorithm.actor.eval()
    return algorithm.actor


def main():
    started_at = time.time()
    prepare_evaluation(configs)
    actor = build_actor()
    run = init_wandb("GOODRL", configs)

    set_seed(configs.algo_seed, configs.device, include_torch=True)
    env = WFEnv(configs.env_name, configs, False)
    observation = env.reset()
    batch = BatchGraph(configs.normalize)
    inference_time = 0.0

    while True:
        with torch.no_grad():
            batch.wrapper(*observation)
            inference_started = time.time()
            action, _, _ = actor(
                state_wf=batch.wf_features,
                state_vm=batch.vm_features,
                edge_index_wf=batch.wf_edges,
                edge_index_vm=batch.vm_edges,
                mask_wf=batch.wf_masks,
                mask_vm=batch.vm_masks,
                batch_wf=batch.wf_batchs,
                batch_vm=batch.vm_batchs,
                candidate_task_index=batch.candidate_taskID,
                deterministic=True,
            )
            inference_time += time.time() - inference_started
        observation, _, done = env.step(action.item())
        if done:
            finalize_evaluation(
                "GOODRL",
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
