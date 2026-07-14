"""Command-line arguments shared by the DWS evaluation drivers."""

import argparse
import re


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def positive_rate_per_second(value):
    rate_per_hour = float(value)
    if rate_per_hour <= 0:
        raise argparse.ArgumentTypeError("Arrival rate must be positive")
    return rate_per_hour / 3600


def comma_separated_tuple(value):
    if isinstance(value, tuple):
        return value
    try:
        return tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated numbers, got: {value}"
        ) from exc


def numpy_seed(value):
    seed = int(value)
    if not 0 <= seed <= 2**32 - 1:
        raise argparse.ArgumentTypeError("Seed must be between 0 and 2**32 - 1")
    return seed


def dataset_suffix(value):
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise argparse.ArgumentTypeError(
            "Dataset suffix may contain only letters, numbers, dots, underscores, and hyphens"
        )
    return value


parser = argparse.ArgumentParser(
    description="Evaluate scheduling baselines in the Dynamic Workflow Scheduling testbed."
)

# Runtime and reproducibility
parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0")
parser.add_argument(
    "--algo_seed",
    type=numpy_seed,
    default=42,
    help="Algorithm and evaluation seed; independent of dataset generation",
)
parser.add_argument("--env_seed", type=int, default=0, help="Simulator seed")
parser.add_argument("--output_dir", default="outputs", help="Directory for generated .npy files")

# Environment
parser.add_argument("--env_name", default="WorkflowScheduling-v3")
parser.add_argument("--wf_num", type=int, default=100, help="Number of workflows to evaluate")
parser.add_argument(
    "--wf_size",
    choices=("S", "M", "L", "XL", "all"),
    default="all",
    help="Workflow template group; use 'all' for the bundled 12-template datasets",
)
parser.add_argument("--valid_num", type=int, default=1)
parser.add_argument("--generateWay", default="rotation")
parser.add_argument("--traf_type", default="CONSTANT")
parser.add_argument(
    "--arr_rate",
    type=positive_rate_per_second,
    default=5.4 / 3600,
    help="Nominal workflow arrival rate per hour (converted internally to per second)",
)
parser.add_argument("--each_vm_type_num", type=int, default=7)
parser.add_argument("--vm_types", type=int, default=5)
parser.add_argument(
    "--rate_dist",
    choices=("constant", "change5", "change10", "cyclic", "multipeak", "flash"),
    default=None,
    help="Optional arrival-rate pattern; omit for a constant --arr_rate",
)
parser.add_argument(
    "--data_name",
    type=dataset_suffix,
    required=True,
    help=(
        "Dataset filename suffix. Load it when present; otherwise use a numeric "
        "suffix as the dataset seed and save the generated file."
    ),
)

# Bundled checkpoints
parser.add_argument(
    "--online_start_ac",
    default="5_5_5.4",
    help="GOODRL actor identifier under pretrained/goodrl/actors",
)
parser.add_argument(
    "--gphh_checkpoint",
    default="pretrained/gphh/5_5_0.0015.pkl",
    help="GPHH tree checkpoint path",
)
parser.add_argument(
    "--erldws_checkpoint",
    default="pretrained/erldws/es_5_5_0.0015.pth",
    help="ERL-DWS policy checkpoint path",
)

# GP and ES-RL definitions retained for checkpoint compatibility
parser.add_argument("--eval_num", type=int, default=3)
parser.add_argument("--pop_size", type=int, default=1024)
parser.add_argument("--gen_num", type=int, default=100)
parser.add_argument("--elite_num", type=int, default=10)
parser.add_argument("--cross_rate", type=float, default=0.8)
parser.add_argument("--mutate_rate", type=float, default=0.15)
parser.add_argument("--sigma_init", type=float, default=0.05)
parser.add_argument("--sigma_decay", type=float, default=1.0)
parser.add_argument("--es_pop_size", type=int, default=4)
parser.add_argument("--es_gen_num", type=int, default=3)

# PPO and network definitions retained for GOODRL checkpoint compatibility
parser.add_argument("--slip_return", type=int, choices=(0, 1), default=1)
parser.add_argument("--grad_control", type=int, choices=(0, 1), default=0)
parser.add_argument("--entropy_control", type=int, choices=(0, 1), default=1)
parser.add_argument("--entropy_min", type=float, default=0.6)
parser.add_argument("--entropy_max", type=float, default=1.2)
parser.add_argument("--warmup_critic", type=int, default=50)
parser.add_argument("--warmup_steps", type=int, default=300)
parser.add_argument("--window_steps", type=int, default=2048)
parser.add_argument("--lr_a", type=float, default=1e-3)
parser.add_argument("--lr_c", type=float, default=1e-3)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--epochs_a", type=int, default=1)
parser.add_argument("--epochs_c", type=int, default=4)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max_updates", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=1.0)
parser.add_argument("--batch_num", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--eps_clip", type=float, default=0.2)
parser.add_argument("--vloss_coef", type=float, default=0.5)
parser.add_argument("--ploss_coef", type=float, default=1.0)
parser.add_argument("--entloss_coef", type=float, default=0.0)
parser.add_argument("--log_interval", type=int, default=20)
parser.add_argument("--require_estimated_features", type=int, choices=(0, 1, 2), default=1)
parser.add_argument("--require_mean", type=int, choices=(0, 1), default=0)
parser.add_argument("--require_undirected", type=int, choices=(0, 1), default=1)
parser.add_argument("--remove_completed", type=int, choices=(0, 1), default=0)
parser.add_argument("--require_clip_value", type=float, default=4.0)
parser.add_argument("--c_gnn_layers", type=int, default=2)
parser.add_argument("--c_layers", type=int, default=4)
parser.add_argument("--c_hidden_dim", type=int, default=128)
parser.add_argument("--atten_layers", type=int, default=1)
parser.add_argument("--gnn_layers", type=int, default=2)
parser.add_argument("--mlp_layers", type=int, default=4)
parser.add_argument("--input_dim_wf", type=int, default=3)
parser.add_argument("--input_dim_vm", type=int, default=4)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--embedding_type", default="gat")
parser.add_argument("--activate_fn", default="relu")
parser.add_argument("--heads", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--neighbor_pooling_type", default="sum")
parser.add_argument("--graph_pooling_type", default="average")
parser.add_argument("--normalize", type=str2bool, default=True)
parser.add_argument(
    "--normalize_features",
    type=comma_separated_tuple,
    default=(3, 39186, 105873, 19593, 50000, 48, 1),
)
parser.add_argument(
    "--normalize_wf",
    type=comma_separated_tuple,
    default=(3, 39186, 105873),
)
parser.add_argument(
    "--normalize_vm",
    type=comma_separated_tuple,
    default=(19593, 50000, 48, 1),
)
parser.add_argument("--normalize_advantage", type=str2bool, default=False)
parser.add_argument("--initial_num", type=int, default=2)
parser.add_argument("--grad_max", type=float, default=2.0)
parser.add_argument("--normalize_rewards", type=int, default=1000)

# Optional experiment logging
parser.add_argument("--use_wandb", type=str2bool, default=False)
parser.add_argument("--wandb_project", default="DWS-Testbed")
parser.add_argument("--wandb_date", default=None)


configs = parser.parse_args()
