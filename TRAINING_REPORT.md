# TRAINING_REPORT.md — G1 12-DoF Velocity PPO Baseline

_Working draft — final numbers filled after training completes._

## 1. Framework decision

Two candidate routes were smoked on the remote 24G GPU (4090D):

| Route | Image | Stack | Smoke task | Result |
|---|---|---|---|---|
| (a) IsaacGym preview-4 + unitree_rl_gym | BJX00000001 / V000412 (`isaac-gym-v20`) | Py3.8 + torch 2.4.1 + vendored rsl_rl v1.0.2 | TASK_20261112_503/506 | _pending_ |
| (b) IsaacLab 2.2.1 official G1 velocity | BJX00000102 / V000418 (`isaac-lab2.2.1-v2`) | IsaacSim 5.0 + isaaclab | TASK_20261112_505 | _pending_ |

Decision criteria: fastest passing official-example smoke, complete G1 assets,
hour-scale training throughput. **Decision: _pending_** (rationale to be added).

## 2. Environment / task configuration

Base: upstream `G1RoughCfg` (unitree_rl_gym@276801e) — verified weights, defaults
kept. Changes over upstream (all tagged `[g1-baseline]`, see NOTICE.md):

| Change | Upstream | Ours | Reason |
|---|---|---|---|
| PD gain randomization | none | per-env multiplier U[0.9, 1.1] on Kp/Kd | basic domain rand (task spec) |
| action delay randomization | none | per-env {0,1,2} policy steps | basic domain rand (task spec) |
| friction / base mass / pushes | already upstream defaults (kept) | kept | — |
| numpy bridge | `.numpy().tolist()` | `.tolist()` | platform torch is numpy2-compiled, env numpy 1.x |

Observation 47-d, action 12-d, PPO LSTM(47→64)+MLP[32], upstream reward weights
(tracking 1.0/0.5, lin_vel_z −2.0, orientation −1.0, base_height −10.0,
dof_acc −2.5e-7, action_rate −0.01, dof_pos_limits −5.0, alive 0.15,
hip_pos −1.0, contact_no_vel −0.2, feet_swing_height −20.0, contact 0.18).
Full interface: see INTERFACE.md.

## 3. Cost ledger

| taskId | Purpose | GPU time | Cost (¥) | commit | Result |
|---|---|---|---|---|---|
| TASK_20261112_503 | route-a smoke #1 | ~1 min | ~0.10 | ecfc343 | failed (numpy bridge, fixed) |
| TASK_20261112_504 | duplicate of 503 (create retry) | 0 | 0 | — | deleted (never run) |
| TASK_20261112_505 | route-b smoke (IsaacLab) | _pending_ | _pending_ | — | _pending_ |
| TASK_20261112_506 | route-a smoke #2 | _pending_ | _pending_ | 07cda6a | _pending_ |
| _pending_ | full baseline training | ≤6 h | — | — | — |

Rate: 1×4090D 24G = ¥5.76/h (goodsId ESKU000001).

## 4. Reproduction

```bash
# remote (gradmotion), IsaacGym image BJX00000001 / V000412:
gm-run g1-locomotion-train/gm_run_all.py --task=g1 --headless --max_iterations=<N>
# smoke:
gm-run g1-locomotion-train/gm_run_all.py --task=g1 --headless --max_iterations=30
```

Local: `git clone https://github.com/yumxcode/g1-locomotion-train && python gm_run_all.py --task=g1 --headless`
(IsaacGym preview-4, python 3.8, torch ≥ 1.10).

## 5. Results (filled after training)

- Throughput: _pending_ env-steps/s (smoke-derived)
- Full run: _pending_ iterations, _pending_ wall time
- Acceptance: RMSE _pending_ m/s, no-fall _pending_% (_pending_/64 episodes)
- Export consistency: jit _pending_, onnx _pending_ (max abs diff, <1e-5 required)
