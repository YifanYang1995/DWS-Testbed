# Bundled pretrained artifacts

All evaluation-time saved parameters are grouped under this directory.

## ERL-DWS

`erldws/` contains complete PyTorch policy objects for VM pools `5×5` and `6×4`, with nominal training rates:

```text
0.001, 0.0015, 0.002, 0.0025, 0.003 workflows/second
```

The default evaluator uses `erldws/es_5_5_0.0015.pth`. These files require the compatible definitions in `policy/es_rl.py` and are intentionally loaded with `weights_only=False`. Only load trusted artifacts.

## GPHH

`gphh/` contains saved genetic-programming trees for VM pools `5×5` and `6×4` at the same five nominal rates. The default evaluator uses `gphh/5_5_0.0015.pkl`.

## GOODRL

`goodrl/actors/` and `goodrl/critics/` contain state dictionaries for:

```text
5_5_5.4
5_5_9
6_4_5.4
6_4_9
```

Only the actor is required by `Step-1-GOODRL.py`; critics are retained so the saved evaluation parameters remain complete. Select an actor with `--online_start_ac`, for example:

```bash
python Step-1-GOODRL.py --online_start_ac 6_4_9 [arguments]
```

Checkpoint identifiers describe training conditions. Evaluation VM-pool settings are controlled independently by `--vm_types` and `--each_vm_type_num`.
