# g1-locomotion-train

G1 velocity-command walking policies trained with PPO on Isaac Gym preview-4,
based on [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
(upstream commit `276801e`) with [rsl_rl v1.0.2](https://github.com/leggedrobotics/rsl_rl) vendored.
See `NOTICE.md` for exact modifications and licenses.

## Layout

```
unitree_rl_gym/   upstream copy + [g1-baseline]/[g1-23dof] tagged modifications (see NOTICE.md)
rsl_rl/           vendored rsl_rl v1.0.2 (unmodified)
gm_run_all.py     single entry: train -> acceptance eval -> policy export
gm_diag.py        64-env diagnostics (termination causes, per-term reward decomposition)
INTERFACE.md      23-DoF contract-v0 policy interface (THE contract for evaluation)
INTERFACE_12DOF.md archived 12-DoF contract (predecessor attempt baseline)
TRAINING_REPORT.md training runs, incidents, cost, reproduction steps
```

## Tasks

| Task | Robot | Policy | Status |
|---|---|---|---|
| `g1_23dof` | G1 23-DoF (`g1_23dof_rev_1_0`), 78-dim obs in contract-v0 order, uniform PD 60/1.5, q_target clipped to limits | feedforward MLP 78-128-64-23 | **primary baseline** |
| `g1` | G1 12-DoF, 47-dim obs + gait clock | LSTM(64)+MLP[32] | legacy (predecessor attempt) |

## Quick start (local, IsaacGym preview-4, python 3.8)

```bash
python gm_run_all.py --task=g1_23dof --headless                    # full default training (5000 it)
python gm_run_all.py --task=g1_23dof --headless --max_iterations=30  # smoke test
python gm_run_all.py --task=g1 --headless                           # legacy 12-DoF task
python gm_diag.py --task=g1_23dof --headless --num_envs=64          # diagnostics
```

`gm_run_all.py` stages:
1. **Train** — PPO on `G1RoughCfg` (4096 envs by default). Checkpoints are mirrored to
   `logs/g1/exported_data/<run>/model_<it>.pt` (platform scan path).
2. **Acceptance eval** — 4096 fixed-command episodes × 10 s, in-simulator lin-vel
   tracking RMSE + no-fall rate (protocol printed with the results).
3. **Export** — TorchScript + ONNX. For `g1_23dof` (MLP): `policy_1.pt` /
   `policy_1.onnx` with a simple `obs->action` signature; for the legacy `g1` task
   (LSTM): `policy_lstm_1.*` with `(obs,h0,c0)->(action,h,c)`. Both include a
   torch-vs-jit-vs-onnx consistency check (<1e-5).

## Deploying the policy

Read `INTERFACE.md` (23-DoF primary contract; §0 confirms/corrects every contract-v0
assumption item). The evaluation side needs nothing beyond `INTERFACE.md` + the exported
policy files. `gm_run_all.py --load_ckpt=path1,path2` re-runs eval+export on existing
checkpoints.
