"""Shared, deterministic evaluation helpers for the DWS baselines."""

from pathlib import Path
import random
import tempfile
import time

import numpy as np

from env.workflow_scheduling_v3.lib.poissonSampling import (
    sample_poisson_piecewise,
    sample_poisson_shape,
)
from env.workflow_scheduling_v3.simulator_wf import WFEnv


ROOT = Path(__file__).resolve().parents[1]
INSTANCES_DIR = ROOT / "data" / "instances"
GENERATION_TEMPLATE = INSTANCES_DIR / "validation_instance_2026.npy"
ARRIVAL_PATTERNS = {
    "cyclic": [0.0015, 0.0035, 0.0015, 0.0035, 0.0015, 0.0035, 0.0015],
    "multipeak": [0.0015, 0.0035, 0.0015, 0.0015, 0.0035, 0.0015],
    "flash": [0.0015, 0.0015, 0.0015, 0.0035, 0.0015, 0.0015],
}


def resolve_repo_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def set_seed(seed, device="cpu", include_torch=False):
    random.seed(seed)
    np.random.seed(seed)
    if not include_torch:
        return
    try:
        import torch
    except (ImportError, OSError):
        if not str(device).startswith("cpu"):
            raise RuntimeError(f"PyTorch is required for device {device}")
        return
    torch.manual_seed(seed)
    torch_device = torch.device(device)
    if torch.cuda.is_available() and torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _dataset_path(data_name):
    return INSTANCES_DIR / f"validation_instance_{data_name}.npy"


def _generate_dataset(seed, destination):
    if not GENERATION_TEMPLATE.is_file():
        raise FileNotFoundError(
            f"Dataset-generation template not found: {GENERATION_TEMPLATE}"
        )

    template_shape = np.load(GENERATION_TEMPLATE, mmap_mode="r").shape
    generator = np.random.RandomState(seed)
    dataset = generator.randint(0, 4, template_shape, dtype=np.int64)

    # Write atomically so concurrent jobs cannot leave a partially written array.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.stem}-",
            suffix=".npy",
            dir=INSTANCES_DIR,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.save(temporary_file, dataset)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"Generated dataset with seed {seed}: {destination}", flush=True)
    return destination


def _seed_from_dataset_suffix(data_name):
    try:
        seed = int(data_name)
    except ValueError as exc:
        raise ValueError(
            f"Dataset {data_name!r} does not exist and cannot be generated: "
            "its suffix is not an integer seed"
        ) from exc
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError(
            f"Dataset seed {seed} must be between 0 and 2**32 - 1"
        )
    return seed


def _resolve_dataset_path(configs):
    if configs.data_name is None:
        raise ValueError("data_name is required and cannot be None")

    dataset_path = _dataset_path(configs.data_name)
    if dataset_path.is_file():
        print(f"Loading dataset: {dataset_path}", flush=True)
        return dataset_path
    dataset_seed = _seed_from_dataset_suffix(configs.data_name)
    return _generate_dataset(dataset_seed, dataset_path)


def _load_dataset(configs):
    dataset_path = _resolve_dataset_path(configs)

    dataset = np.load(dataset_path).reshape((1, 1, -1))
    if configs.data_name == "2024":
        dataset = np.flip(dataset, axis=2)
    if configs.wf_num > dataset.shape[2]:
        raise ValueError(
            f"wf_num={configs.wf_num} exceeds the {dataset.shape[2]} workflows in {dataset_path.name}"
        )
    if configs.data_name == "12" and configs.wf_size != "all":
        raise ValueError("Dataset '12' contains 12 workflow templates and requires --wf_size all")
    return dataset[:, :, : configs.wf_num]


def _sample_arrivals(configs):
    shape = configs.valid_dataset.shape
    rate = configs.arr_rate
    if configs.rate_dist in {None, "constant"}:
        return sample_poisson_shape(rate, shape)
    if configs.rate_dist == "change10":
        if np.isclose(rate, 0.0015):
            pattern = [0.0015, 0.0025, 0.0035]
        elif np.isclose(rate, 0.0025):
            pattern = [0.0035, 0.0025, 0.0015]
        else:
            raise ValueError("change10 supports --arr_rate 5.4 or 9 only")
        return sample_poisson_piecewise(pattern, shape)
    if configs.rate_dist == "change5":
        if np.isclose(rate, 0.0015):
            pattern = [0.0015, 0.0020, 0.0025]
        elif np.isclose(rate, 0.0025):
            pattern = [0.0025, 0.0020, 0.0015]
        else:
            raise ValueError("change5 supports --arr_rate 5.4 or 9 only")
        return sample_poisson_piecewise(pattern, shape)
    if configs.rate_dist in ARRIVAL_PATTERNS:
        return sample_poisson_piecewise(ARRIVAL_PATTERNS[configs.rate_dist], shape)
    return sample_poisson_shape(rate, shape)


def rate_distribution_name(configs):
    return configs.rate_dist or "constant"


def prepare_evaluation(configs):
    if configs.wf_num <= 0:
        raise ValueError("wf_num must be positive")
    if not 1 <= configs.vm_types <= 6:
        raise ValueError("vm_types must be between 1 and 6")
    if configs.each_vm_type_num <= 0:
        raise ValueError("each_vm_type_num must be positive")
    set_seed(configs.algo_seed, configs.device)
    configs.valid_dataset = _load_dataset(configs)
    configs.GENindex = 0
    configs.indEVALindex = 0
    configs.arr_times = _sample_arrivals(configs)
    return configs


def output_path(method, configs):
    output_dir = resolve_repo_path(configs.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rate_per_hour = configs.arr_rate * 3600
    filename = (
        f"online_{method}_{rate_distribution_name(configs)}_{configs.data_name}_"
        f"{configs.vm_types}_{configs.each_vm_type_num}_{rate_per_hour:.1f}_"
        f"seed{configs.algo_seed}.npy"
    )
    return output_dir / filename


def heft_reference_path(configs):
    rate_per_hour = configs.arr_rate * 3600
    filename = (
        f"online_HEFT_{rate_distribution_name(configs)}_{configs.data_name}_"
        f"{configs.vm_types}_{configs.each_vm_type_num}_{rate_per_hour:.1f}.npy"
    )
    return ROOT / "data" / "heft_reference" / filename


def compare_with_heft(flowtimes, heft_flowtimes, workloads):
    objective = np.mean(np.asarray(flowtimes) / np.asarray(workloads))
    baseline = np.mean(np.asarray(heft_flowtimes) / np.asarray(workloads))
    return 100 * (baseline - objective) / (baseline + 1e-8)


def init_wandb(method, configs):
    if not configs.use_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb or run with --use_wandb false") from exc

    return wandb.init(
        project=configs.wandb_project,
        group=(
            f"{configs.data_name}_{rate_distribution_name(configs)}_"
            f"{configs.vm_types}_{configs.each_vm_type_num}_{configs.arr_rate * 3600:.1f}"
        ),
        name=f"{method}_seed{configs.algo_seed}",
        config={
            "method": method,
            "seed": configs.algo_seed,
            "dataset": configs.data_name,
            "wf_num": configs.wf_num,
            "wf_size": configs.wf_size,
            "vm_types": configs.vm_types,
            "each_vm_type_num": configs.each_vm_type_num,
            "arrival_rate_per_hour": configs.arr_rate * 3600,
            "rate_dist": rate_distribution_name(configs),
            "device": configs.device,
        },
    )


def finalize_evaluation(method, configs, flowtimes, workloads, started_at, run=None, inference_time=None):
    flowtimes = np.asarray(flowtimes)
    workloads = np.asarray(workloads)
    destination = output_path(method, configs)
    np.save(destination, flowtimes)

    elapsed_hours = (time.time() - started_at) / 3600
    metrics = {
        "test/mean_flowtime": float(np.mean(flowtimes)),
        "test/std_flowtime": float(np.std(flowtimes)),
        "test/min_flowtime": float(np.min(flowtimes)),
        "test/max_flowtime": float(np.max(flowtimes)),
        "test/median_flowtime": float(np.median(flowtimes)),
        "test/num_workflows": int(len(flowtimes)),
        "test/total_workload": float(np.sum(workloads)),
        "test/mean_workload": float(np.mean(workloads)),
        "time/evaluation_hours": elapsed_hours,
    }
    if inference_time is not None:
        metrics["time/inference_seconds"] = float(inference_time)

    reference_path = heft_reference_path(configs)
    if reference_path.is_file():
        heft_flowtimes = np.load(reference_path)
        if heft_flowtimes.shape == flowtimes.shape:
            improvement = compare_with_heft(flowtimes, heft_flowtimes, workloads)
            metrics["compare/heft_improvement_percentage"] = float(improvement)
            print(f"HEFT-normalized improvement: {improvement:.4f}%", flush=True)

    print(
        f"{method}: mean flowtime={metrics['test/mean_flowtime']:.5f}, "
        f"workflows={len(flowtimes)}, elapsed={elapsed_hours:.4f} h",
        flush=True,
    )
    print(f"Saved: {destination}", flush=True)

    if run is not None:
        run.log(metrics)
        run.finish()
    return destination, metrics


def run_heuristic(method, configs):
    method = method.upper()
    if method not in {"HEFT", "EST", "PEFT"}:
        raise ValueError(f"Unsupported heuristic: {method}")

    started_at = time.time()
    prepare_evaluation(configs)
    run = init_wandb(method, configs)
    env = WFEnv(configs.env_name, configs, False)
    env.resetGP()
    select_action = getattr(env, method)

    while True:
        action = select_action()
        _, _, done = env.stepGP(action)
        if done:
            return finalize_evaluation(
                method,
                configs,
                env.all_flowTime,
                env.all_workload,
                started_at,
                run,
            )
