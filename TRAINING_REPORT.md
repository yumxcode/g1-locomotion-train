# TRAINING_REPORT.md — G1 12-DoF Velocity PPO Baseline

_Working draft — final numbers filled after training completes._

## 1. Framework decision

Two candidate routes were smoked on the remote 24G GPU (4090D):

| Route | Image | Stack | Smoke task | Result |
|---|---|---|---|---|
| (a) IsaacGym preview-4 + unitree_rl_gym | BJX00000001 / V000412 (`isaac-gym-v20`) | Py3.8 + torch 2.4.1 + vendored rsl_rl v1.0.2 | TASK_20261112_503/506/514/516 | **train 30 it OK (85k env-steps/s), ckpt upload OK**; platform-image fixes documented below |
| (b) IsaacLab 2.2.1 official G1 velocity | BJX00000102 / V000418 (`isaac-lab2.2.1-v2`) | IsaacSim 5.0 + isaaclab | TASK_20261112_505 | app launch OK, `Isaac-Velocity-Flat-G1-v0` created + reset OK; step failed on a one-line issue (numpy action vs torch); NOT retried |

**Decision: route (a).** Evidence:
1. Full chain (train → platform checkpoint upload) verified end-to-end; measured
   throughput 85,457 env-steps/s (4096 envs, 1.15 s/iter) → 10k iters ≈ 3.2 h, well within ≤6 h.
2. G1 assets complete: 12-DoF URDF + MuJoCo XML in-repo
   (`resources/robots/g1_description/`), upstream `deploy/deploy_mujoco` is a ready
   sim2sim reference for the evaluation side.
3. Route (b) reached env creation/reset but is a 29-DoF ManagerBased env (baseline
   spec is 12-DoF), has heavier cold-start and a far more complex observation/action
   contract — worse fit for the baseline and the INTERFACE.md hand-off. Smoke scripts
   remain in `yumxcode/g1-isaaclab-smoke`.
4. Route (a) smoke failures (503/506/514) were platform-image issues (torch 2.4.1
   py38 is numpy2-compiled while env numpy is 1.x → broken torch↔numpy bridge;
   inference-tensor inplace after rsl_rl rollout), fixed in-repo (commits 3303ab4,
   b575a59) — framework-independent fixes.

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
| TASK_20261112_503 | route-a smoke #1 | 63 s | 0.10 | ecfc343 | failed: numpy bridge in runner (fixed) |
| TASK_20261112_504 | duplicate of 503 (create retry) | 0 | 0 | — | deleted before run |
| TASK_20261112_505 | route-b smoke (IsaacLab 2.2.1) | 29 s | 0.05 | — | env created/reset OK, step needs torch action; not retried |
| TASK_20261112_506 | route-a smoke #2 | 57 s | 0.09 | 07cda6a | failed: numpy bridge in tensorboard add_scalar (fixed) |
| TASK_20261112_514 | route-a smoke #3 | 116 s | 0.19 | 3303ab4 | train+ckpt-upload OK (85k steps/s); eval failed: inference-tensor inplace (fixed) |
| TASK_20261112_516 | route-a smoke #4 (full chain) | _pending_ | _pending_ | b575a59 | _pending_ |
| _pending_ | full baseline training (~8000 it) | ≤2.7 h (budget ≤6 h) | ~15.5 | — | — |

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
