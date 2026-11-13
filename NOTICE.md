# NOTICE

This repository is a training-side fork for the G1 velocity-command PPO baseline.
It bundles third-party open-source projects with minimal, clearly-marked modifications.

## 1. unitree_rl_gym (directory `unitree_rl_gym/`)

- Upstream: https://github.com/unitreerobotics/unitree_rl_gym
- Upstream commit: 276801e46c5d433564f24658bac64f254b7d2d4b (2025-07-25)
- License: BSD-3-Clause (see `unitree_rl_gym/LICENSE`, `unitree_rl_gym/legged_gym/LICENSE`)
- Local modifications (all tagged with `[g1-baseline]` in source):
  1. `legged_gym/envs/base/legged_robot.py`
     - `step()`: optional per-env action delay applied to torque targets (policy-issued
       action is still what enters the observation).
     - `_init_buffers()`: per-env multiplicative PD gain randomization
       (`randomize_gains`, `gain_multiplier_range`) and per-env action delay buffers
       (`randomize_action_delay`, `action_delay_steps_range`).
     - `reset_idx()`: action-delay history is zeroed on env reset.
  2. `legged_gym/envs/g1/g1_config.py`
     - `domain_rand`: enabled `randomize_gains = True` with multiplier range
       [0.9, 1.1] and `randomize_action_delay = True` with delay range [0, 2] policy
       steps (0-40 ms). Friction / base-mass / push randomization are upstream defaults.
- All other files are unmodified from the upstream commit above.

## 2. rsl_rl (directory `rsl_rl/`)

- Upstream: https://github.com/leggedrobotics/rsl_rl
- Version: v1.0.2 (tag), vendored unmodified.
- License: BSD-3-Clause (see `rsl_rl/LICENSE`)

## 3. Files added by this project

- `gm_run_all.py` — gradmotion platform entry: train -> in-sim acceptance eval ->
  TorchScript/ONNX export with consistency check.
- `README.md`, `INTERFACE.md`, `TRAINING_REPORT.md` — documentation.

## [g1-23dof] additions (contract-v0 baseline, 2026-11-13)

- `legged_gym/envs/g1/g1_23dof_{env,config}.py` — new task `g1_23dof`:
  23-DoF G1 (`g1_23dof_rev_1_0.urdf`), 78-dim observation in evaluation-contract-v0
  order `[cmd | ang_vel | gravity | dof_pos | dof_vel | prev_action]` (no gait clock),
  feedforward MLP policy, uniform PD Kp=60/Kd=1.5, q_target clipped to joint limits
  (contract-v0 action semantics). Reward deltas vs 12-DoF task: phase-locked `contact`
  disabled (clock unobserved), standard `feet_air_time` enabled instead.
- `gm_run_all.py` — export/eval paths extended to feedforward policies
  (`policy_1.pt` TorchScript, `obs->action` ONNX; LSTM path unchanged).
