# g1-locomotion-train

G1 (12-DoF) velocity-command walking policy trained with PPO on Isaac Gym preview-4,
based on [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
(upstream commit `276801e`) with [rsl_rl v1.0.2](https://github.com/leggedrobotics/rsl_rl) vendored.
See `NOTICE.md` for exact modifications and licenses.

## Layout

```
unitree_rl_gym/   upstream copy + [g1-baseline] tagged modifications (see NOTICE.md)
rsl_rl/           vendored rsl_rl v1.0.2 (unmodified)
gm_run_all.py     single entry: train -> acceptance eval -> policy export
INTERFACE.md      exact policy interface contract for deployment/evaluation
TRAINING_REPORT.md training runs, cost, reproduction steps
```

## Quick start (local, IsaacGym preview-4, python 3.8)

```bash
python gm_run_all.py --task=g1 --headless                    # full default training
python gm_run_all.py --task=g1 --headless --max_iterations=10  # smoke test
```

`gm_run_all.py` stages:
1. **Train** — PPO on `G1RoughCfg` (4096 envs by default). Checkpoints are mirrored to
   `logs/g1/exported_data/<run>/model_<it>.pt` (platform scan path).
2. **Acceptance eval** — fixed velocity commands, 10 s episodes, in-simulator
   lin-vel tracking RMSE + no-fall rate (protocol printed with the results).
3. **Export** — TorchScript (`policy_lstm_1.pt`, stateful LSTM module) and ONNX
   (`policy_lstm_1.onnx`, explicit `(obs,h0,c0)->(action,h,c)` signature), plus a
   torch-vs-jit-vs-onnx consistency check (<1e-5).

## Deploying the policy

Read `INTERFACE.md` — it fully specifies observation layout, action semantics, PD gains,
control rate, command ranges, initial pose, and checkpoint formats. The evaluation side
needs nothing beyond `INTERFACE.md` + the exported policy files.
