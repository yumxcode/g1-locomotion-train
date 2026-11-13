# INTERFACE.md — G1 (12-DoF) Velocity-Command Walking Policy

**Version:** 1.0 (baseline PPO, experiment `g1`) · **Training repo:** `yumxcode/g1-locomotion-train` @ `07cda6a`
**Robot:** Unitree G1, 12 actuated DoF (waist/arms locked in the 12dof URDF variant)
**This document is the single contract for the evaluation side. No other source is needed.**

---

## 1. Policy summary

| Item | Value |
|---|---|
| Algorithm | PPO (rsl_rl v1.0.2), recurrent actor-critic |
| Actor | LSTM(47→64) → Linear(64→32) → ELU → Linear(32→12) |
| Input | `obs` — 47-dim proprioception (layout §3) |
| Output | `action` — 12-dim joint position offsets (§4) |
| Policy rate | 50 Hz (policy step = 20 ms) |
| Physics rate | 200 Hz (decimation = 4) |
| Action clip | ±100 (effectively inactive; policy output is bounded by training) |
| Observation clip | ±100 |

## 2. Control loop (per policy step, 50 Hz)

```
1. obs_t   <- build_observation()        # §3
2. a_t     <- policy(obs_t, (h, c))      # LSTM hidden state carried across steps
3. q_cmd   = a_t * 0.25 + q_default      # joint position targets, §4
4. for i in 1..4:                        # decimation = 4, sim dt = 5 ms
       tau = Kp * (q_cmd - q) - Kd * qd  # per-joint PD, §5
       clip tau to torque limits (§5)
       step physics 5 ms
```

PD gains are applied in the **joint-space of the training simulator** with perfect
state feedback; the evaluation side should use the same gains on the same joint
ordering.

## 3. Observation vector (47 dims, exact layout)

Order matters. Scales are already applied inside the vector described below
(i.e. the policy consumes the **scaled** values).

| Idx | Dim | Content | Scale applied | Unit (raw) |
|---|---|---|---|---|
| 0–2 | 3 | base angular velocity, **base frame** (x=pitch-rate? no — ω_x, ω_y, ω_z) | ×0.25 | rad/s |
| 3–5 | 3 | projected gravity vector, base frame | ×1 | — |
| 6–8 | 3 | velocity command (v_x, v_y, ω_z), **base frame** | v×2.0, ω×0.25 | m/s, m/s, rad/s |
| 9–20 | 12 | joint positions − default positions | ×1.0 | rad |
| 21–32 | 12 | joint velocities | ×0.05 | rad/s |
| 33–44 | 12 | previous policy action `a_{t-1}` (raw, unscaled) | ×1 | — |
| 45 | 1 | sin(2π·phase) | — | — |
| 46 | 1 | cos(2π·phase) | — | — |

- The whole vector is clipped to ±100 after optional observation noise
  (training adds noise; deployment must NOT add noise).
- **Gait clock**: `phase = (t_mod / 0.8 s) mod 1`, where `t_mod` is the elapsed
  time since the start of the current episode (policy steps × 20 ms).
  On episode start `phase = 0`. The left leg tracks `phase`, the right leg uses
  `(phase + 0.5) mod 1` internally (only the left-leg phase enters the
  observation as sin/cos).
- Joint order for indices 9–20, 21–32, 33–44 (also for actions): see §4.

## 4. Actions (12 dims)

`action ∈ R^12` (policy output, effectively in [−1, 1] after training but not
hard-clipped beyond ±100). Joint **position target**:

```
q_target[i] = action[i] * 0.25 + q_default[i]      # action_scale = 0.25
```

Joint order (= URDF revolute joint order used by the training simulator):

| i | Joint name | q_default [rad] | Kp [N·m/rad] | Kd [N·m·s/rad] | τ_max [N·m] | q_min [rad] | q_max [rad] |
|---|---|---|---|---|---|---|---|
| 0 | left_hip_pitch_joint | −0.1 | 100 | 2 | 88 | −2.5307 | 2.8798 |
| 1 | left_hip_roll_joint | 0.0 | 100 | 2 | 139 | −0.5236 | 2.9671 |
| 2 | left_hip_yaw_joint | 0.0 | 100 | 2 | 88 | −2.7576 | 2.7576 |
| 3 | left_knee_joint | 0.3 | 150 | 4 | 139 | −0.0873 | 2.8798 |
| 4 | left_ankle_pitch_joint | −0.2 | 40 | 2 | 50 | −0.8727 | 0.5236 |
| 5 | left_ankle_roll_joint | 0.0 | 40 | 2 | 50 | −0.2618 | 0.2618 |
| 6 | right_hip_pitch_joint | −0.1 | 100 | 2 | 88 | −2.5307 | 2.8798 |
| 7 | right_hip_roll_joint | 0.0 | 100 | 2 | 139 | −2.9671 | 0.5236 |
| 8 | right_hip_yaw_joint | 0.0 | 100 | 2 | 88 | −2.7576 | 2.7576 |
| 9 | right_knee_joint | 0.3 | 150 | 4 | 139 | −0.0873 | 2.8798 |
| 10 | right_ankle_pitch_joint | −0.2 | 40 | 2 | 50 | −0.8727 | 0.5236 |
| 11 | right_ankle_roll_joint | 0.0 | 40 | 2 | 50 | −0.2618 | 0.2618 |

(Values cross-checked at runtime in `interface_params.pt` produced by the
training job; the runtime dump is authoritative if any digit differs.)

## 5. PD control (200 Hz)

Per-joint gains in §4 (`stiffness`/`damping` columns). Torque is clipped to
`τ_max` per joint (§4). Control type: position (`control_type = 'P'`).

## 6. Commands

| Command | Range | Note |
|---|---|---|
| lin_vel_x (base frame) | [−1.0, 1.0] m/s | forward |
| lin_vel_y (base frame) | [−1.0, 1.0] m/s | lateral |
| ang_vel_yaw | [−1.0, 1.0] rad/s | yaw rate |

- Training resamples commands every 10 s; commands with |v_xy| < 0.2 m/s are
  zeroed at resample time (mixture with a stand-still mode).
- The evaluation side can hold commands fixed; the policy is a pure
  reactive controller on (obs → action), no command memory.
- The observation carries the command **scaled**: obs[6]=2.0·v_x,
  obs[7]=2.0·v_y, obs[8]=0.25·ω_z.

## 7. Initial state (episode start)

- Base position: (0, 0, 0.8) m, orientation: identity (z-up), zero velocity.
- Joint positions: default angles (§4) × U[0.5, 1.5] per joint (training);
  for deterministic evaluation use exactly q_default with zero velocity.
- LSTM memory: zeros at episode start (call `reset_memory()` on the jit module
  or pass h0=c0=0 for ONNX).

## 8. Domain randomization (training distribution — for reference)

| Parameter | Range |
|---|---|
| friction coefficient | U[0.1, 1.25] (per env, 64 buckets) |
| base mass addition | U[−1.0, +3.0] kg |
| PD gain multiplier (Kp, Kd) | U[0.9, 1.1] × nominal |
| action delay | {0, 1, 2} policy steps (0–40 ms), per env |
| push disturbance | base vel impulse up to 1.5 m/s every 5 s |
| observation noise | σ: ang_vel 0.2·0.25, gravity 0.05, dof_pos 0.01, dof_vel 1.5·0.05 |

The exported policy expects **nominal** parameters by default (randomization
only shaped training). For robustness testing sample from the table above.

## 9. Termination criteria used in training (definition of "fall")

An episode terminates (counts as a fall) when any of:
- contact force > 1 N on the pelvis body, or
- |roll| > 0.8 rad, or
- |pitch| > 1.0 rad.
Timeout at 20 s (training) / 10 s (evaluation protocol) is NOT a fall.

## 10. Checkpoint formats and loading

Three artifacts are produced per training run (all downloadable from the
platform task):

1. **`model_<it>.pt`** — rsl_rl `OnPolicyRunner` checkpoint:
   ```python
   ckpt = torch.load('model_5000.pt', map_location='cpu')
   # ckpt['model_state_dict']: actor_critic state dict (ActorCriticRecurrent)
   #   keys: memory_a.rnn.*, actor.0.*, actor.2.*  (+ std for training)
   # ckpt['optimizer_state_dict'], ckpt['iter'], ckpt['infos']
   ```
2. **`policy_lstm_1.pt`** — TorchScript module (upstream `PolicyExporterLSTM`):
   - stateful: hidden/cell states live inside the module as buffers
   - `out = module(obs_1x47)` → action 1×12 (batch dim = 1)
   - `module.reset_memory()` at every episode start
   - float32, CPU or GPU
3. **`policy_lstm_1.onnx`** (also wrapped as `policy_lstm_1_onnx.pt` = torch.save
   of the raw onnx bytes) — stateless single-step graph:
   - inputs: `obs` (1×47, float32), `h0` (1×1×64), `c0` (1×1×64)
   - outputs: `action` (1×12), `h` (1×1×64), `c` (1×1×64)
   - caller maintains (h, c) across steps; reset to zeros at episode start
   - opset 13

## 11. Export verification (filled after training)

| Check | Result |
|---|---|
| TorchScript vs torch (100-step random stream, max abs diff) | _pending_ |
| ONNX vs torch (same stream, max abs diff) | _pending_ |
| Acceptance: lin-vel RMSE ≤ 0.3 m/s (≥20 eps × 10 s) | _pending_ |
| Acceptance: no-fall rate ≥ 90 % (10 s episodes) | _pending_ |

## 12. MuJoCo assets

The upstream repo ships `unitree_rl_gym/resources/robots/g1_description/g1_12dof.xml`
(MuJoCo model matching the URDF). The default joint angles in §4 match the
`qpos` reference of that model (keyframe ordering may differ — map by joint name).
