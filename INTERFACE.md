# INTERFACE.md — G1 (23-DoF) Velocity-Command Walking Policy

**Version:** 1.0 (contract-v0 aligned baseline PPO, experiment `g1_23dof`) · **Training repo:** `yumxcode/g1-locomotion-train` @ `b4853f9`
**Robot:** Unitree G1 `g1_23dof_rev_1_0` — 23 actuated DoF (12 leg + 1 waist + 10 arm), total mass 32.107 kg
**This document is the single contract for the evaluation side. No other source is needed.**
(The legacy 12-DoF baseline from the crashed predecessor attempt is archived in `INTERFACE_12DOF.md`.)

---

## 0. Contract-v0 assumption check (for the evaluation side)

| # | v0 assumption | This contract | Status |
|---|---|---|---|
| 1 | obs 78 = [cmd(3) \| ang_vel(3) \| gravity(3) \| dof_pos−default(23) \| dof_vel(23) \| prev_action(23)] | identical layout & order (§3) | ✅ confirmed |
| 2 | cmd scale [2, 2, 0.25] | identical (§3) | ✅ confirmed |
| 3 | base_ang_vel ×0.25, base frame | identical (§3) | ✅ confirmed |
| 4 | projected_gravity = R^T·[0,0,−1], base frame, ×1 | identical (§3) | ✅ confirmed |
| 5 | (dof_pos − default) ×1.0 | identical (§3) | ✅ confirmed |
| 6 | dof_vel ×0.05 | identical (§3) | ✅ confirmed |
| 7 | prev action raw (unscaled) | identical (§3) | ✅ confirmed |
| 8 | action q_target = clip(default_q + 0.25·a, joint limits) | identical (§4); limits from URDF §4 table | ✅ confirmed |
| 9 | PD kp=60 / kd=1.5 (all joints) | kp=60 identical; **Kd = 5.0** (§5) | ⚠️ kd modified |
| 10 | torque limit = URDF effort 88/139/50/25 N·m | identical (§4 table, verified at runtime) | ✅ confirmed |
| 11 | sim dt 0.005, decimation 4 → policy 50 Hz | identical (§1) | ✅ confirmed |
| 12 | asset g1_23dof_rev_1_0, 32.107 kg | identical URDF/MJCF (§12) | ✅ confirmed |
| 13 | feedforward policy (no hidden state) | MLP 78→128→64→23, ELU (§1) | ✅ confirmed |
| 14 | (implicit) no obs/action clip beyond documented | obs & action clip ±100 (framework default, effectively inactive) (§1) | ✅ confirmed |
| 15 | (assumed) self-collisions in training sim | **training sim DISABLES self-collision** (§8 note) | ⚠️ only deviation |

Deviations from v0 (both documented in-place):
- **#9 Kd**: v0 assumed 1.5; the trained policy uses **Kd = 5.0** with Kp = 60.
  With kd=1.5 the explicit 200 Hz PD chatters at ~14 rad/s RMS joint velocity
  (diagnostic TASK_20261113_038) and even zero-action standing falls within
  ~16–30 policy steps. The evaluation side must apply Kd=5.0 (or its override
  mechanism) to reproduce the training dynamics.
- **#15**: the training simulator disables self-collision
because at the zero arm pose the upper arm sits flush against the torso contour and
arm-torso contact forces (200–500 N under random actions) would otherwise land on the
termination body. The evaluation side should either disable arm-torso collision or
expect benign contact events there; the trained policy keeps arms near the default
pose (see §8) so in practice the arms do not sweep the torso.

## 1. Policy summary

| Item | Value |
|---|---|
| Algorithm | PPO (rsl_rl v1.0.2), feedforward actor-critic |
| Actor | Linear(78→128) → ELU → Linear(128→64) → ELU → Linear(64→23) |
| Input | `obs` — 78-dim proprioception (layout §3) |
| Output | `action` — 23-dim joint position offsets (§4) |
| Policy rate | 50 Hz (policy step = 20 ms) |
| Physics rate | 200 Hz (decimation = 4, sim dt = 5 ms) |
| Action clip | ±100 (framework default; effectively inactive) |
| Observation clip | ±100 (framework default; effectively inactive) |

## 2. Control loop (per policy step, 50 Hz)

```
1. obs_t   <- build_observation()        # §3
2. a_t     <- policy(obs_t)              # stateless MLP, float32
3. q_cmd   = clip(a_t * 0.25 + q_default, q_min, q_max)   # §4 per-joint limits
4. for i in 1..4:                        # decimation = 4, sim dt = 5 ms
       tau = 60 * (q_cmd - q) - 1.5 * qd             # uniform PD, §5
       clip tau to per-joint torque limits (§4)
       step physics 5 ms
```

PD runs at physics rate (200 Hz) on the same target within a policy step, with perfect
state feedback. The evaluation side should apply the same gains on the same joint
ordering.

## 3. Observation vector (78 dims, exact layout)

Order matters. The policy consumes **scaled** values. Scales are applied before the
final ±100 clip. During training, uniform noise was added (§8); deployment must NOT
add noise.

| Idx | Dim | Content | Scale applied | Unit (raw) |
|---|---|---|---|---|
| 0–2 | 3 | velocity command (v_x, v_y, ω_z), **base frame** | v×2.0, ω×0.25 | m/s, m/s, rad/s |
| 3–5 | 3 | base angular velocity (ω_x, ω_y, ω_z), base frame | ×0.25 | rad/s |
| 6–8 | 3 | projected gravity (R^T·[0,0,−1]), base frame | ×1 | — |
| 9–31 | 23 | joint positions − default positions | ×1.0 | rad |
| 32–54 | 23 | joint velocities | ×0.05 | rad/s |
| 55–77 | 23 | previous policy action `a_{t-1}` (raw, unscaled) | ×1 | — |

- Joint order for indices 9–31, 32–54, 55–77 (also for actions): URDF order, §4.
- No gait clock, no hidden state: the policy is a pure reactive function of the 78-dim obs.

## 4. Actions (23 dims)

`action ∈ R^23` (policy output; effectively in [−1, 1] after training). Joint
**position target**:

```
q_target[i] = clip(action[i] * 0.25 + q_default[i], q_min[i], q_max[i])
```

Joint order = `g1_23dof_rev_1_0.urdf` revolute joint order (= Isaac Gym DOF order,
verified at runtime in `interface_params.pt`):

| i | Joint name | q_default [rad] | Kp | Kd | τ_max [N·m] | q_min [rad] | q_max [rad] |
|---|---|---|---|---|---|---|---|
| 0 | left_hip_pitch_joint | −0.1 | 60 | 1.5 | 88 | −2.5307 | 2.8798 |
| 1 | left_hip_roll_joint | 0.0 | 60 | 1.5 | 139 | −0.5236 | 2.9671 |
| 2 | left_hip_yaw_joint | 0.0 | 60 | 1.5 | 88 | −2.7576 | 2.7576 |
| 3 | left_knee_joint | 0.3 | 60 | 1.5 | 139 | −0.0873 | 2.8798 |
| 4 | left_ankle_pitch_joint | −0.2 | 60 | 1.5 | 50 | −0.8727 | 0.5236 |
| 5 | left_ankle_roll_joint | 0.0 | 60 | 1.5 | 50 | −0.2618 | 0.2618 |
| 6 | right_hip_pitch_joint | −0.1 | 60 | 1.5 | 88 | −2.5307 | 2.8798 |
| 7 | right_hip_roll_joint | 0.0 | 60 | 1.5 | 139 | −2.9671 | 0.5236 |
| 8 | right_hip_yaw_joint | 0.0 | 60 | 1.5 | 88 | −2.7576 | 2.7576 |
| 9 | right_knee_joint | 0.3 | 60 | 1.5 | 139 | −0.0873 | 2.8798 |
| 10 | right_ankle_pitch_joint | −0.2 | 60 | 1.5 | 50 | −0.8727 | 0.5236 |
| 11 | right_ankle_roll_joint | 0.0 | 60 | 1.5 | 50 | −0.2618 | 0.2618 |
| 12 | waist_yaw_joint | 0.0 | 60 | 1.5 | 88 | −2.618 | 2.618 |
| 13 | left_shoulder_pitch_joint | 0.0 | 60 | 1.5 | 25 | −3.0892 | 2.6704 |
| 14 | left_shoulder_roll_joint | 0.0 | 60 | 1.5 | 25 | −1.5882 | 2.2515 |
| 15 | left_shoulder_yaw_joint | 0.0 | 60 | 1.5 | 25 | −2.618 | 2.618 |
| 16 | left_elbow_joint | 0.0 | 60 | 1.5 | 25 | −1.0472 | 2.0944 |
| 17 | left_wrist_roll_joint | 0.0 | 60 | 1.5 | 25 | −1.9722 | 1.9722 |
| 18 | right_shoulder_pitch_joint | 0.0 | 60 | 1.5 | 25 | −3.0892 | 2.6704 |
| 19 | right_shoulder_roll_joint | 0.0 | 60 | 1.5 | 25 | −2.2515 | 1.5882 |
| 20 | right_shoulder_yaw_joint | 0.0 | 60 | 1.5 | 25 | −2.618 | 2.618 |
| 21 | right_elbow_joint | 0.0 | 60 | 1.5 | 25 | −1.0472 | 2.0944 |
| 22 | right_wrist_roll_joint | 0.0 | 60 | 1.5 | 25 | −1.9722 | 1.9722 |

(All values are cross-checked at runtime against the URDF by the training job and
dumped to `interface_params_<tag>.pt`; the runtime dump is authoritative if any digit
differs.)

## 5. PD control (200 Hz)

Uniform gains for **all 23 joints**: `Kp = 60 N·m/rad`, `Kd = 5.0 N·m·s/rad`
(v0 assumed Kd=1.5; raised because kd=1.5 is underdamped for the 32 kg G1 under
explicit 200 Hz PD — see §0 item 9).
Torque clipped per joint to `τ_max` (§4). Control type: position (`control_type = 'P'`).

## 6. Commands

| Command | Range | Note |
|---|---|---|
| lin_vel_x (base frame) | [−1.0, 1.0] m/s | forward |
| lin_vel_y (base frame) | [−1.0, 1.0] m/s | lateral |
| ang_vel_yaw | [−1.0, 1.0] rad/s | yaw rate |

- Training resamples commands every **10 s** (per env); commands with |v_xy| < 0.2 m/s
  are zeroed at resample (stand-still mixture).
- Training ran in **heading mode**: ω_z command is recomputed each step as
  `clip(0.5 · wrap(heading_cmd − yaw), −1, 1)` with heading_cmd resampled in
  [−π, π]. The policy only ever sees the ω_z value in obs[2]; for evaluation it is
  sufficient to feed a fixed ω_z directly.
- The observation carries the command **scaled**: obs[0]=2.0·v_x, obs[1]=2.0·v_y, obs[2]=0.25·ω_z.

## 7. Initial state (episode start)

- Base position: (0, 0, 0.8) m, orientation: identity (z-up), zero velocity (training
  randomizes initial base velocity ±0.5 m/s and joint angles = default × U[0.5, 1.5];
  for deterministic evaluation use exactly q_default with zero velocity).
- Policy is stateless — nothing to reset at episode start.

## 8. Domain randomization (training distribution — for reference)

| Parameter | Range |
|---|---|
| friction coefficient | U[0.1, 1.25] (per env, 64 buckets) |
| base mass addition | U[−1.0, +3.0] kg |
| PD gain multiplier (Kp, Kd) | U[0.9, 1.1] × nominal |
| action delay | {0, 1, 2} policy steps (0–40 ms), per env |
| push disturbance | base xy velocity impulse up to 1.5 m/s every 5 s |
| observation noise (σ) | ang_vel 0.05, gravity 0.05, dof_pos 0.01, dof_vel 0.075 (already incl. scales) |
| **self-collision** | **disabled** in the training sim (see §0 item 15) |

The exported policy expects **nominal** parameters by default. The policy keeps arms
near the default pose (small action-rate penalty shapes this), so the disabled
arm-torso self-collision rarely matters in deployment.

## 9. Termination criteria used in training (definition of "fall")

An episode terminates (counts as a fall) when any of:
- net contact force > 1 N on the pelvis or torso body, or
- |roll| > 0.8 rad, or
- |pitch| > 1.0 rad.
Timeout at 20 s (training) / 10 s (evaluation protocol) is NOT a fall.

## 10. Checkpoint formats and loading

Per training run the platform hosts (download via `gm task model list` → `policUrlDown`):

1. **`model_<it>.pt`** — rsl_rl `OnPolicyRunner` checkpoint (saved every 50 iters):
   ```python
   ckpt = torch.load('model_5000.pt', map_location='cpu')
   # ckpt['model_state_dict']: ActorCritic state dict
   #   keys: actor.0.*, actor.2.*, actor.4.*  (Linear 78-128, 128-64, 64-23)
   # ckpt['optimizer_state_dict'], ckpt['iter'], ckpt['infos']
   ```
2. **`policy_1.pt`** — TorchScript module (scripted `actor`, stateless):
   - `out = module(obs_1x78)` → action 1×23, float32, CPU/GPU
3. **`policy_1.onnx`** (also wrapped as `policy_1_onnx.pt` = torch.save of the raw
   onnx bytes) — single-step graph:
   - input `obs` (1×78 float32) → output `action` (1×23); opset 13
4. **`interface_params_<tag>.pt`** — runtime dump of dof names/order/limits/default
   angles/PD gains/command ranges (authoritative companion to this doc).
5. **`eval_results_<tag>.pt`** — acceptance metrics + per-episode RMSE + fall flags.

## 11. Acceptance measurement & export verification (filled after training)

Protocol (fixed, in the training simulator): 4096 parallel episodes × 10 s; per env
one fixed command drawn uniform vx,vy,wz ∈ [−1,1]³ with |v_xy| < 0.2 zeroed; commands
frozen for the episode; obs noise and pushes off; per-env training domain-rand samples
(friction/mass/PD/delay) kept; RMSE computed per episode over base-frame xy velocity
vs command, then averaged over episodes. Fall = non-timeout termination.

| Check | Result |
|---|---|
| lin-vel tracking RMSE (mean over episodes) | _TBD_ (target ≤ 0.3 m/s) |
| no-fall rate (10 s) | _TBD_ (target ≥ 90 %) |
| TorchScript vs torch (100-step random stream, max abs diff) | _TBD_ (< 1e-5) |
| ONNX vs torch (same stream, max abs diff) | _TBD_ (< 1e-5) |

## 12. Assets

Training asset: `unitree_rl_gym/resources/robots/g1_description/g1_23dof_rev_1_0.urdf`
(Isaac Gym, fixed joints collapsed). The matching MuJoCo model for evaluation is
`g1_23dof_rev_1_0.xml` in the same directory (identical link/joint set, masses and
limits; keyframe ordering differs — map by joint name). Total mass 32.107 kg.
