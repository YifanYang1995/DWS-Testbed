# DWS-Testbed

**DWS-Testbed** is a reproducible, event-driven testbed for the Dynamic Workflow Scheduling (DWS) problem. It provides fixed workflow streams, scientific-workflow DAGs, heterogeneous VM pools, six evaluation baselines, bundled pretrained artifacts, and single-seed reference results.

The simulator and GOODRL evaluator are derived from the public [GOODRL repository](https://github.com/YifanYang1995/GOODRL), which accompanies the ICLR 2025 paper [Graph Assisted Offline-Online Deep Reinforcement Learning for Dynamic Workflow Scheduling](https://openreview.net/forum?id=4PlbIfmX9o).

## 🎯 Scope

This repository is intended for controlled **evaluation**, baseline comparison, and environment testing. It is not a training pipeline. A run processes one fixed stream of workflows and saves the flowtime of every completed workflow as a NumPy array.

Included evaluators:

| Method | Driver | Bundled artifact | Evaluation policy |
|---|---|---|---|
| EST | `Step-1-EST.py` | None | Select the VM with the earliest available time |
| PEFT | `Step-1-PEFT.py` | None | Earliest finish time plus a path-to-exit estimate |
| HEFT | `Step-1-HEFT.py` | None | Select the VM with the earliest task finish time |
| GPHH | `Step-1-GP.py` | `pretrained/gphh/` | Evaluate a saved genetic-programming priority tree |
| ERL-DWS | `Step-1-ESRL.py` | `pretrained/erldws/` | Evaluate a saved evolution-strategy policy |
| GOODRL | `Step-1-GOODRL.py` | `pretrained/goodrl/` | Evaluate a frozen offline GOODRL actor |

Lower mean flowtime is better.

## ⚙️ DWS environment

Each scientific workflow is a directed acyclic graph (DAG). Tasks become ready when their predecessors finish. At every environment step, the scheduler assigns the current ready task to one VM in a heterogeneous pool.

- **Action:** index of a VM queue.
- **Transition:** enqueue the task, advance the event-driven simulator, and expose the next ready task.
- **Reward:** `-flowtime / 1000` when a workflow completes; otherwise `0`.
- **Episode end:** all requested workflows have completed.
- **Primary output:** per-workflow flowtimes in `env.all_flowTime`.
- **Workload normalizer:** per-workflow workload in `env.all_workload`.

The simulator exposes three state interfaces:

- `reset()` / `step()` for graph-based policies such as GOODRL;
- `resetGP()` / `stepGP()` for heuristic and GPHH feature vectors;
- `resetES()` / `stepES()` for ERL-DWS VM feature sequences.

## 📁 Repository layout

```text
.
├── Step-1-{EST,PEFT,HEFT,GP,ESRL,GOODRL}.py  # Evaluation entry points
├── config/Params.py                          # Command-line arguments
├── env/
│   ├── workflow_scheduling_v3/               # Event-driven DWS simulator
│   │   ├── dax/                              # Scientific-workflow XML files
│   │   └── lib/                              # DAG, VM, queue, and simulator components
├── policy/                                   # GOODRL and ERL-DWS policy definitions
├── utils/
│   ├── evaluation_utils.py                   # Shared data, seed, logging, and output logic
│   └── ...                                   # Policy/checkpoint compatibility utilities
├── data/
│   ├── instances/                            # Fixed workflow-type streams
│   └── heft_reference/                       # Matching HEFT arrays for normalized comparison
├── pretrained/
│   ├── erldws/                               # ERL-DWS policy objects
│   ├── gphh/                                 # GPHH trees
│   └── goodrl/{actors,critics}/              # GOODRL state dictionaries
├── reference_results/results_seed42.xlsx     # Two-dataset reference workbook
├── scripts/submit_slurm.sh                   # Portable SLURM example
└── outputs/                                  # Generated arrays; ignored by Git
```

## 📦 Requirements

The reference implementation targets:

- Python 3.11 or newer;
- PyTorch 2.4.1 or newer;
- PyTorch Geometric 2.5.3 or newer;
- CPU execution by default.

Create an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the PyTorch build appropriate for the local CPU/CUDA platform before
the remaining requirements when a non-default CUDA build is required. Weights
& Biases support is installed by `requirements.txt`, but logging remains
disabled unless `--use_wandb true` is passed.

## 🚀 Quick start

Run commands from the repository root because the checkpoint and data defaults are repository-relative.

A short HEFT smoke test:

```bash
python Step-1-HEFT.py \
  --wf_num 100 \
  --wf_size all \
  --vm_types 6 \
  --each_vm_type_num 4 \
  --arr_rate 5.4 \
  --data_name 2024 \
  --algo_seed 42
```

`--rate_dist` is intentionally omitted here. This matches the original GOODRL
environment: arrivals follow the constant rate supplied through `--arr_rate`.

To generate a new dataset, request a numeric suffix whose file does not yet
exist:

```bash
python Step-1-HEFT.py --data_name 42 --algo_seed 7 --wf_num 100 --wf_size all
```

This creates `data/instances/validation_instance_42.npy` with dataset seed `42`;
the independent algorithm/evaluation seed is `7`. See
[Automatic dataset generation](#automatic-dataset-generation) for the exact
rule.

Evaluate the other methods by changing the driver:

```bash
python Step-1-EST.py     [arguments]
python Step-1-PEFT.py    [arguments]
python Step-1-GP.py      [arguments]
python Step-1-ESRL.py    [arguments]
python Step-1-GOODRL.py  [arguments]
```

Generated arrays are written to `outputs/` using this convention:

```text
online_{METHOD}_{RATE_DIST}_{DATASET}_{VM_TYPES}_{VMS_PER_TYPE}_{RATE}_seed{SEED}.npy
```

For example:

```text
outputs/online_HEFT_constant_2024_6_4_5.4_seed42.npy
```

## 🔬 Reproduce the reference setting

The bundled workbook uses `seed=42`, `wf_num=20000`, and `wf_size=all`. A representative GOODRL command is:

```bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

python Step-1-GOODRL.py \
  --wf_num 20000 \
  --wf_size all \
  --vm_types 6 \
  --each_vm_type_num 4 \
  --arr_rate 5.4 \
  --rate_dist change5 \
  --data_name 2024 \
  --online_start_ac 5_5_5.4 \
  --algo_seed 42 \
  --use_wandb false
```

The thread limits are important on shared multi-core systems; unrestricted BLAS/OpenMP thread creation can make these event-driven evaluations substantially slower.

### Pretrained selection

The default saved artifacts reproduce the historical cross-configuration evaluation setup:

```bash
# GPHH default
python Step-1-GP.py --gphh_checkpoint pretrained/gphh/5_5_0.0015.pkl [arguments]

# ERL-DWS default
python Step-1-ESRL.py --erldws_checkpoint pretrained/erldws/es_5_5_0.0015.pth [arguments]

# GOODRL default
python Step-1-GOODRL.py --online_start_ac 5_5_5.4 [arguments]
```

Other bundled identifiers are documented in [`pretrained/README.md`](pretrained/README.md). A GPHH or ERL-DWS artifact may have been trained on a different VM-pool shape from the evaluation pool; record both the checkpoint identifier and evaluation pool when reporting results.

The ERL-DWS `.pth` files contain complete Python policy objects and are loaded with `weights_only=False`. Only load these bundled, trusted files or artifacts from a source you trust.

## 🗂️ Datasets and workflow templates

The bundled validation files have original shape `(1, 10000, 30)` and are
flattened into one workflow stream before taking the first `wf_num` entries.
Their filename suffix is the random seed used to create the array.

| `--data_name` | Type IDs | Loading rule | Reference workbook |
|---|---:|---|---|
| `2024` | 0–3 | Historical reverse-order loading | Yes |
| `2026` | 0–3 | Stored order | Yes |
| `12` | 0–11 | Stored order; requires `--wf_size all` | No |

The `2024` and `2026` files were generated with their suffix as the NumPy seed
and `randint(0, 4, shape)`. The `12` file follows the same suffix-as-seed
convention but uses `randint(0, 12, shape)` to cover all 12 templates.

The `all` template group contains 12 DAGs across CyberShake, Montage, Inspiral, and Sipht, with small, medium, and large variants. See [`data/README.md`](data/README.md) for the exact mapping.

### Automatic dataset generation

Dataset selection follows these rules:

1. If `--data_name <suffix>` is supplied and
   `validation_instance_<suffix>.npy` exists, that file is loaded.
2. If `--data_name <suffix>` is supplied but the file does not exist, `<suffix>`
   is converted with `int(data_name)` and must be a valid NumPy seed. The dataset
   is generated with that seed and
   saved as `validation_instance_<suffix>.npy`. `--algo_seed` is **not** used to
   generate the dataset in this branch.

Generation takes the array shape from `validation_instance_2026.npy` and draws
integer template IDs uniformly from `{0, 1, 2, 3}` with NumPy's legacy seeded
random-number sequence.

For example, `--data_name 123 --algo_seed 42` creates
`validation_instance_123.npy` with dataset seed `123` when the file is missing;
seed `42` controls evaluation randomness only. If that file already exists, it
is reused and no random generation occurs. Generated instance files are local
artifacts and are ignored by Git. Pass an existing suffix such as
`--data_name 2026` to skip generation.

## 📈 Arrival-rate patterns

`--arr_rate` is entered as workflows per hour and converted internally to
workflows per second. By default, `--rate_dist` is unset and this rate remains
constant, matching the original GOODRL environment. Pass `--rate_dist`
explicitly only when evaluating a changing arrival-rate pattern. Piecewise
patterns are defined internally per second; the table below shows the more
readable per-hour values.

| `--rate_dist` | `--arr_rate` | Segment rates per hour |
|---|---:|---|
| omitted or `constant` | any positive value | Constant at the supplied rate (default) |
| `change5` | 5.4 | 5.4 → 7.2 → 9.0 |
| `change5` | 9.0 | 9.0 → 7.2 → 5.4 |
| `change10` | 5.4 | 5.4 → 9.0 → 12.6 |
| `change10` | 9.0 | 12.6 → 9.0 → 5.4 |
| `cyclic` | label only | 5.4 ↔ 12.6, repeated |
| `multipeak` | label only | 5.4 → 12.6 → 5.4 → 5.4 → 12.6 → 5.4 |
| `flash` | label only | 5.4 → 5.4 → 5.4 → 12.6 → 5.4 → 5.4 |

For piecewise patterns marked “label only,” the supplied nominal rate remains part of the experiment label and output filename but does not alter the fixed segment list.

## 📊 Reference results

[`reference_results/results_seed42.xlsx`](reference_results/results_seed42.xlsx) contains mean flowtime results for:

- datasets `2024` and `2026`;
- VM pools `6×4`, `5×7`, and `4×10`;
- upward and downward `change5`/`change10` arrival patterns;
- EST, PEFT, HEFT, GPHH, ERL-DWS, and GOODRL;
- one evaluation seed, `42`.

The arrow labels in the workbook map to command-line arguments as follows:

| Workbook label | Command-line setting |
|---|---|
| `5.4↗` | `--rate_dist change5 --arr_rate 5.4` |
| `5.4↗↗` | `--rate_dist change10 --arr_rate 5.4` |
| `9↘` | `--rate_dist change5 --arr_rate 9` |
| `12.6↘` | `--rate_dist change10 --arr_rate 9` |

A compact check for the `6×4`, `5.4↗` setting is:

| Dataset | EST | PEFT | HEFT | GPHH | ERL-DWS | GOODRL |
|---|---:|---:|---:|---:|---:|---:|
| 2024 | 1037.70316 | 412.10405 | 371.57366 | 296.11411 | 2867.10634 | 295.42759 |
| 2026 | 1034.02708 | 407.89202 | 370.10951 | 293.23080 | 2936.28915 | 291.08634 |

These single-seed values are provided as implementation checks, not as uncertainty estimates or final statistical claims. For a study, evaluate multiple independent seeds and report an aggregate with dispersion or confidence intervals.

## 🖥️ SLURM example

The submission template contains no user-specific cluster paths. Select the Python executable through `PYTHON_BIN` if needed:

```bash
sbatch --export=ALL,PYTHON_BIN=/path/to/venv/bin/python \
  scripts/submit_slurm.sh GOODRL 6 4 5.4 20000 change5 2024
```

Omit the final dataset argument to make the script pass the SLURM array task ID
as `--data_name`:

```bash
sbatch --export=ALL,PYTHON_BIN=/path/to/venv/bin/python \
  scripts/submit_slurm.sh GOODRL 6 4 5.4 20000 change5
```

Arguments are:

```text
METHOD VM_TYPES VMS_PER_TYPE ARRIVAL_RATE WF_NUM [RATE_DIST] [DATASET]
```

Both `RATE_DIST` and `DATASET` are optional. An empty or omitted `RATE_DIST`
leaves `--rate_dist` unset and therefore uses constant arrivals. To select a
dataset while keeping constant arrivals, pass an empty sixth argument, for
example `scripts/submit_slurm.sh HEFT 6 4 5.4 20000 "" 2024`. If `DATASET` is
absent, the script passes the SLURM array task ID through `--data_name`. If the
corresponding file does not exist, `int(DATASET)` is used as its generation
seed. Edit the resource directives for the local cluster policy.

## 📡 Optional W&B logging

Logging is disabled by default and no network connection is required for an evaluation:

```bash
python Step-1-HEFT.py --use_wandb false [arguments]
```

Enable logging with:

```bash
python Step-1-HEFT.py --use_wandb true --wandb_project DWS-Testbed [arguments]
```

## ✅ Reproducibility checklist

Record the following with each result:

1. Git commit and Python/package versions.
2. Driver/method and checkpoint identifier, if applicable.
3. `data_name`, `wf_num`, and `wf_size`.
4. VM pool: `vm_types × each_vm_type_num`.
5. Effective `rate_dist` (`constant` when omitted) and nominal `arr_rate`.
6. `algo_seed` and any cluster thread settings.
7. Raw `outputs/*.npy` file before aggregation.

## 📝 Citation

If this testbed or the GOODRL implementation supports your work, cite the GOODRL paper:

```bibtex
@inproceedings{yang2025graph,
  title     = {Graph Assisted Offline-Online Deep Reinforcement Learning for Dynamic Workflow Scheduling},
  author    = {Yang, Yifan and Chen, Gang and Ma, Hui and Zhang, Cong and Cao, Zhiguang and Zhang, Mengjie},
  booktitle = {International Conference on Learning Representations},
  year      = {2025}
}
```

## ⚖️ License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).
