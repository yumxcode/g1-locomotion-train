# TRAINING_REPORT.md — G1 23-DoF Velocity PPO Baseline (contract-v0 aligned)

**Primary run:** `g1_23dof-baseline-v6-10000it` (TASK_20261113_124) · repo commit `28499a7` (docs @64b0252)
**Legacy:** the crashed predecessor attempt's 12-DoF baseline is preserved (§6).

## 1. Framework decision

Two candidate routes were smoked on the remote 24G GPU (4090D) — evidence from the
predecessor attempt's platform tasks, re-verified 2026-11-13 via `gm task info/logs`:

| Route | Image | Stack | Smoke tasks | Result |
|---|---|---|---|---|
| (a) IsaacGym preview-4 + unitree_rl_gym | BJX00000001 / V000412 (`isaac-gym-v20`) | Py3.8 + torch 2.4.1 + vendored rsl_rl v1.0.2 | TASK_20261112_503/506/514/516/518 | **train + ckpt upload OK end-to-end** (85k env-steps/s @ 12-DoF) |
| (b) IsaacLab 2.2.1 official G1 velocity | BJX00000102 / V000418 (`isaac-lab2.2.1-v2`) | IsaacSim 5.0 + IsaacLab 2.2.1 | TASK_20261112_505 | env created/reset OK; step needs torch action fix; 29-DoF ManagerBased env, heavier cold-start, complex obs/action contract |

**Decision: route (a).** Full chain verified end-to-end, G1 assets complete in-repo
(12/23/29-DoF URDF+MJCF), throughput hours-scale, and the unitree_rl_gym observation/
PD conventions are the exact source the evaluation side's contract v0 was derived from.
Route-b smoke scripts: `yumxcode/g1-isaaclab-smoke`.

## 2. Why a new 23-DoF task (repair scope)

The predecessor attempt delivered a 12-DoF baseline that PASSES the acceptance metrics
(TASK_20261112_557: RMSE 0.161 m/s, no-fall 99.1%, 4096 episodes) — but its interface
(47-dim obs with gait clock, 12 actions, LSTM, per-joint PD 100-150/2-4) is
dimensionally incompatible with the evaluation side's contract v0 (78-dim obs,
23 actions, 23-DoF robot `g1_23dof_rev_1_0`). A 12-DoF policy cannot actuate a 23-DoF
robot's arms/waist, so the evaluation side would have to lock 11 joints — beyond
"obs rescale/permute" adaptation. This rerun therefore trains a contract-v0-aligned
23-DoF baseline; the 12-DoF artifacts are kept as pipeline evidence + fallback.

## 3. Environment / task configuration (`g1_23dof`, all deltas tagged `[g1-23dof]`)

Base: upstream `G1RoughCfg` (unitree_rl_gym@276801e) with these changes:

| Change | Upstream (12dof) | Ours (23dof) | Reason |
|---|---|---|---|
| asset | `g1_12dof.urdf` | `g1_23dof_rev_1_0.urdf` | contract v0 robot (23 actuated joints, 32.107 kg, verified vs MJCF/menagerie) |
| observation | 47-dim [ang_vel, gravity, cmd, dof(12)×3, sin/cos phase] | **78-dim contract-v0 order** [cmd, ang_vel, gravity, dof(23)×3], no clock | exact contract match |
| policy | LSTM(64)+MLP[32] | **feedforward MLP [128, 64]** (rsl_rl `ActorCritic`) | contract assumes stateless policy |
| PD gains | hip100/knee150/ankle40, kd 2-4 | **uniform kp=60 / kd=1.5** (contract) | contract match |
| joint armature | 0 | **0.01 kg·m² per DOF** | explicit-PD stability on wrist/elbow (see §5 incident C) |
| action semantics | q_target = default + 0.25·a (no limit clip) | q_target = **clip**(default + 0.25·a, joint limits) | contract v0 action semantics |
| self-collisions | 0 (enabled; no arms in 12dof URDF) | **1 (disabled)** | arm-torso contact at zero pose (see §5 incident A) |
| gait rewards | contact 0.18 (phase-locked), feet_air_time 0 | contact 0 (clock unobserved), **feet_air_time 1.0** | free-form gait for MLP without clock |
| penalty rebalance | dof_acc −2.5e-7, dof_vel −1e-3, action_rate −0.01, alive 0.15 | dof_acc −1e-7, dof_vel −5e-4, action_rate −0.005, **alive 1.0** | 23-joint sums ≈ 2× penalty mass; per-step total must stay > 0 (see §5 incident B) |
| domain rand | friction U[0.1,1.25], mass ΔU[-1,3] kg, push 1.5 m/s @5 s | same + PD multiplier U[0.9,1.1], action delay {0,1,2} steps (kept from 12dof rerun) | task spec |

PPO (rsl_rl v1.0.2 defaults): adaptive lr (desired KL 0.01), γ 0.99, λ 0.95, clip 0.2,
entropy 0.01, 5 epochs, 4 minibatches, 24 steps/env, 4096 envs, episode 20 s.

### 3.5 Whole-table diff vs the proven-learnable 12-DoF config (mandatory vs free variables)

The 12-DoF task (upstream `G1RoughCfg`, reached RMSE 0.161 by ~2000 it) is the only
proven-learnable reference in this repo. Diff of every changed dimension, classified:

| Dimension | 12-DoF (learnable) | 23-DoF (ours) | Class |
|---|---|---|---|
| robot / DoF count | g1_12dof, 12 joints | g1_23dof_rev_1_0, 23 joints | **contract-mandatory** |
| observation | 47-dim [ang_vel, grav, cmd, dof×3, sin/cos **gait clock**] | 78-dim contract order, **no clock** | **contract-mandatory** |
| policy | LSTM(64)+MLP[32] | MLP [128, 64] | **contract-mandatory** (stateless) |
| PD | hip 100 / knee 150 / ankle 40, kd 2-4 | uniform 60 / 1.5 | **contract-mandatory** (v0 item 9) |
| action limit clip | none | clip to joint limits | **contract-mandatory** (v0 item 8) |
| armature | 0 (no arms) | 0.01 | **numerical necessity** (explicit-PD stability, §5-C) |
| self-collision | on (harmless w/o arms) | off | **necessity** (§5-A) |
| gait shaping | **phase-locked `contact` 0.18 with observed clock** | `feet_air_time` 1.0→2.0 (v6), command-gated | **free variable** — the 12-DoF's fast gait emergence is driven by the clock prior, which the contract forbids in obs; feet_air_time is its only contract-compatible equivalent |
| alive | 0.15 | 1.0 | **free variable** (23-joint penalty mass requires it, §5-B) |
| dof_acc / dof_vel / action_rate | −2.5e-7 / −1e-3 / −0.01 | −1e-7 / −5e-4 / −0.005 | **free variable** (halved to restore per-robot penalty mass) |
| tracking_lin_vel | 1.0 | 1.0 (v5 tried 1.5 → plateaued, reverted) | **free variable** |
| episode length / commands / PPO hparams | 20 s / ±1 @10 s / defaults | same | unchanged |

Mechanism conclusion: with the clock forbidden by the contract, gait must emerge from
feet_air_time + tracking alone; v6 (feet_air_time 2.0) is the single strongest
contract-compatible lever, empirically 2.87× faster to standing than v4 at it2000.

## 4. Runs, cost ledger, artifacts

Rate: 1×4090D 24G = ¥5.76/h (goodsId ESKU000001). All costs from `gm task info` runtime.

### This rerun (2026-11-13)

| taskId | Purpose | GPU time | Cost (¥) | commit | Result |
|---|---|---|---|---|---|
| TASK_20261113_005 | 23dof pipeline smoke v1 (30 it) | ~3 min | ~0.3 | 2aa8fc8 | full chain OK; exposed incident B (below) |
| TASK_20261113_006 | full baseline v1 (5000 it) | 767 s | 1.23 | 2aa8fc8 | FAILED: instant termination (incident A), stopped |
| TASK_20261113_007/008/009 | diagnostics (termination causes) | ~2-3 min each | ~0.9 | acc53c9/0943397 | isolated arm-torso self-collision (200-500 N) |
| TASK_20261113_010 | smoke v2 after selfcol fix | ~3 min | ~0.3 | b4853f9 | ep_len 2.5 → 26 ✅ |
| TASK_20261113_012 | full baseline v2 | 767 s | 1.23 | b4853f9 | FAILED: reward stall (incident B), stopped |
| TASK_20261113_016 | full baseline v3 (penalty rebalance) | ~13 min | ~1.3 | 41c9705 | FAILED: same stall — dof_vel chatter (incident C), stopped at gate |
| TASK_20261113_029/030/034/038 | diagnostics (reward decomposition) | ~2-3 min each | ~1.2 | f164291…d4f059a | isolated dof_vel −0.046/step chatter |
| TASK_20261113_041 | kd=5 experiment-diag | ~2 min | ~0.2 | 0e55686 | kd=5 WORSE (−0.096) → explicit-PD instability confirmed |
| TASK_20261113_042 | armature=0.01 mechanism gate | ~2 min | ~0.2 | 4f14510 | **PASS**: standing +0.023/step, chatter ×1/1000 |
| **TASK_20261113_043** | **full baseline v4 (5000 it)** | 4193 s | 6.72 | 4f14510 | finished; **no-fall 95.26% ✅ / RMSE 0.657 ❌**; jit 0.0 / onnx 7.6e-6 |
| TASK_20261113_095 | resume-from-5000 attempt | ~25 min | ~2.4 | 2629246 | FAILED: first PPO update destroyed policy (ep_len 678→46, std 1.69→2.4 entropy spiral) |
| TASK_20261113_096 | bisect: load-eval platform model_5000 | ~8 min | ~0.8 | 475c70d | **weights proven good** (RMSE 0.654 / no-fall 95.19%, exact reproduction) — resume path at fault |
| TASK_20261113_113 | full baseline v5 (tracking 1.5, 10000 it) | ~86 min | ~8.3 | ee166fa | FAILED: plateau ep_len≈64 (v4 same-iter ~300-678), stopped at it5050 |
| **TASK_20261113_124** | **full baseline v6 (feet_air_time 2.0, 10000 it) — PRIMARY** | 7896 s | 12.63 | 28499a7 | finished; **no-fall 99.90 % ✅ / RMSE 0.407 ❌**; jit 0.0 / onnx 5.7e-6; ep_len 910 (max 992), tracking plateau ~0.22-0.28 after it6500 |
| TASK_20261113_180 | nominal eval attempt 1 | 52 s | 0.08 | 3d37039^ | failed fast: make_env tuple bug (fixed) |
| TASK_20261113_182 | **supplementary nominal-domain eval of v6 model_10000** (domain rand OFF, same protocol) | ~2 min | ~0.2 | 3d37039 | RMSE **0.443**, no-fall **100.0 %** (supplementary only; preregistered numbers stay) |
| TASK_20261113_181 | full baseline v7 attempt (feet_air_time 3.0, 12000 it) | 1334 s | 2.14 | ecf7cfe | **killed by infra**: concurrent task 182 scheduled on same GPU (pit_9e74a116); trajectory healthy at it1500 (ep_len 127.6 = 2.1× v6 same-it) |
| TASK_20261113_209 | **full baseline v7b (feet_air_time 3.0, 12000 it, serial relaunch)** | _TBD_ | _TBD_ | ecf7cfe | audit round-1 lever (a); running |

### Predecessor attempt (2026-11-12, kept for the record / fallback)

| taskId | Purpose | Result |
|---|---|---|
| TASK_20261112_503-518 | route smokes a/b + pipeline fixes | route (a) chosen |
| TASK_20261112_519 | 12dof baseline 8000 it (collapsed @~2480, stopped @3229) | checkpoints 1750-3200 on platform |
| TASK_20261112_556/557 | 12dof eval+export (model_2000/2400) | **RMSE 0.161 / no-fall 99.15%** (model_2000), jit 2.6e-6 / onnx 2.4e-6 |

## 5. Incident log (root causes, all verified with 64-env diagnostics on-platform)

- **A. Instant termination (v1, 006):** at the zero arm pose G1's upper arm sits flush
  against the torso contour; with self-collision enabled, random-action arm offsets
  produced 200–500 N arm-torso contact on the termination body `torso_link` — every
  episode reset within ~2 policy steps (roll/pitch never exceeded; pelvis force 0 N).
  Fix: `self_collisions = 1` (the legged_gym framework default). Documented in
  INTERFACE.md §0/§8.
- **B. Reward-floor stall (v2/v3, 012/016):** 23-joint penalty sums (dof_acc, dof_vel,
  action_rate ≈ 2× the 12-DoF mass) made the per-step total negative; with
  `only_positive_rewards` each step clips to 0 → zero survival gradient → adaptive lr
  death spiral (2e-5), ep_len declined 26→15. Fix: halve those penalties + alive
  0.15→1.0. (Necessary but not sufficient — C remained.)
- **C. Explicit-PD joint chatter (v3→v4):** with kd=1.5 (contract) Isaac Gym's explicit
  200 Hz PD is underdamped-to-unstable on the small-inertia wrist/elbow joints
  (kd·dt/I ≈ 0.75): 14–20 rad/s RMS joint-velocity chatter → dof_vel penalty
  −0.046/step during quiet standing; raising kd to 5 made it WORSE (−0.096/step,
  kd·dt/I > 2 — the instability signature). Fix: `armature = 0.01` kg·m² per DOF
  (rotor inertia), kd stays at the contract 1.5. Verified: chatter ÷1000, standing
  per-step total +0.023 > 0. The 12-DoF task never hit A/B/C because it has no arms
  (bigger joint inertias, fewer joint-sum terms).

## 6. Results

**Primary baseline: v6 = TASK_20261113_124** (commit 28499a7, `g1_23dof` task,
feet_air_time 2.0, 10,000 iters, checkpoint `model_10000`).

- Throughput: **131,760 env-steps/s** (23 DoF, 4096 envs, self-collision off),
  0.75 s/iter — calibrates the algorithm-side anchor (55–110k estimate: 23-DoF
  actual is 132k, i.e. ABOVE the estimated band, 12-DoF was ~96k)
- Full run: 10,000 iters, **7,894 s (2.19 h) wall**, ¥12.63 (≈$1.8) — well within
  the ≤6 h GPU budget; total 23dof campaign ≈ ¥35 across all runs+diagnostics
- Learning curve (CSV `v6_task124_curves.csv`): ep_len 12 → 297 (it2000, 2.87× v4
  same-iter) → 712 (it3000) → 910 end (max 992); tracking reward plateau ~0.22–0.28
  after it6500 (converged, not under-trained)
- **Acceptance (in-sim, protocol in INTERFACE.md §11, 4096 episodes × 10 s):**
  - no-fall rate **99.90 %** ✅ (≥ 90 % required)
  - lin-vel tracking RMSE **0.407 m/s** ❌ (≤ 0.3 required, +36 % miss)
- **Export consistency:** TorchScript **0.0**, ONNX **5.7e-6** (both < 1e-5 ✅)
- Trajectory of configs: v1 fail (self-collision) → v2/v3 fail (reward floor / PD
  chatter) → v4 (5000 it): no-fall 95.3 % / RMSE 0.657 → v5 (tracking 1.5): plateau
  → **v6 (feet_air_time 2.0, 10k it): no-fall 99.9 % / RMSE 0.407**. The remaining
  RMSE gap is the converged capability of a minimal contract-constrained PPO baseline
  (no gait clock, uniform PD 60/1.5, free-form gait); see §3.5 for the mechanism
  analysis and the documented levers for the follow-up algorithm work (CES-PPO etc.).

## 7. Reproduction

```bash
# remote (gradmotion), IsaacGym image BJX00000001 / V000412:
gm-run g1-locomotion-train/gm_run_all.py --task=g1_23dof --headless --max_iterations=5000
# smoke (30 it) / diagnostics:
gm-run g1-locomotion-train/gm_run_all.py --task=g1_23dof --headless --max_iterations=30
gm-run g1-locomotion-train/gm_diag.py  --task=g1_23dof --headless --num_envs=64
# re-eval + re-export existing checkpoints:
gm-run g1-locomotion-train/gm_run_all.py --task=g1_23dof --headless \
    --load_ckpt=g1-locomotion-train/checkpoints/model_XXXX.pt
```

Local: `git clone https://github.com/yumxcode/g1-locomotion-train && python gm_run_all.py --task=g1_23dof --headless`
(IsaacGym preview-4, python 3.8, torch ≥1.10).

## 8. Extension points for future algorithm work (CES-PPO etc.)

- Reward terms: `legged_gym/envs/g1/g1_23dof_config.py::rewards.scales` (+ env-level
  `_reward_*` methods in `g1_23dof_env.py` / base `legged_robot.py`).
- Advantage estimation / PPO update: vendored `rsl_rl/rsl_rl/algorithms/ppo.py`
  (drop-in replacement class, selected via `runner.algorithm_class_name`).
- Runner/logging: `rsl_rl/rsl_rl/runners/on_policy_runner.py`; curves on the platform
  via `gm task data keys/get`.
The 23dof task keeps upstream conventions so such swaps stay one-file changes.
