# Bundled evaluation data

## Bundled instance streams

`instances/` contains three NumPy arrays with original shape `(1, 10000, 30)`:

- `validation_instance_2024.npy`: IDs 0–3; loaded in reverse order for historical compatibility.
- `validation_instance_2026.npy`: IDs 0–3; loaded in stored order.
- `validation_instance_12.npy`: IDs 0–11; loaded in stored order and used with `--wf_size all`.

The filename suffix is the random seed used to generate the array. The `2024`
and `2026` arrays use the equivalent of
`np.random.seed(seed); np.random.randint(0, 4, shape)`. The `12` array uses seed
12 and `np.random.randint(0, 12, shape)` so that it can select all 12 workflow
templates.

## Automatic generation

Dataset selection and generation use these rules:

- An existing `validation_instance_<suffix>.npy` selected with
  `--data_name <suffix>` is loaded directly.
- If that selected file is missing, `int(data_name)` is used as the dataset seed
  and the generated array is saved under the requested filename. `--algo_seed`
  does not determine the dataset.

Automatic generation takes the shape from `validation_instance_2026.npy` and
samples IDs uniformly from `{0, 1, 2, 3}`. The result is saved as:

```text
data/instances/validation_instance_<data_name>.npy
```

An existing same-name file is reused. These locally generated files are ignored
by Git. To use a bundled or previously generated file explicitly, pass its
suffix, for example `--data_name 2026` or `--data_name 42`.

Each ID selects one DAG template from `env/workflow_scheduling_v3/dax/`.

| ID | Workflow template |
|---:|---|
| 0 | CyberShake_30 |
| 1 | Montage_25 |
| 2 | Inspiral_30 |
| 3 | Sipht_30 |
| 4 | CyberShake_50 |
| 5 | Montage_50 |
| 6 | Inspiral_50 |
| 7 | Sipht_60 |
| 8 | CyberShake_100 |
| 9 | Montage_100 |
| 10 | Inspiral_100 |
| 11 | Sipht_100 |

## HEFT reference arrays

`heft_reference/` contains the per-workflow HEFT arrays used by
`utils/evaluation_utils.py` for an optional workload-normalized comparison. A
comparison is printed only when the method output and HEFT reference have
identical shapes.

The raw method output remains the source of truth. Do not substitute the normalized comparison percentage for mean flowtime without explicitly defining the metric.
