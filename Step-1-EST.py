"""Evaluate the EST heuristic on a bundled DWS workload stream."""

from config.Params import configs
from evaluation_utils import run_heuristic


if __name__ == "__main__":
    run_heuristic("EST", configs)
