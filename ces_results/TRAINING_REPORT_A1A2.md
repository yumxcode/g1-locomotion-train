# TRAINING_REPORT_A1A2 — CES-PPO (Contact-Event Segmented PPO) on the frozen G1 v6 environment

**Date**: 2026-11-15 · **Task group**: grp-5e249ddcd7b23c29 (Phase 2 / plan step 6)
**Repo**: `yumxcode/g1-locomotion-train` branch `ces-ppo` @ `6dae7bd` (branched from v6 final `64b0252`; main-line v7 experiments untouched)
**Verdict (one line)**: CES-PPO was implemented exactly per the frozen SPEC and passes 24+9 unit/integration checks, but **A1 (event-segmented) training collapsed** (design-level failure, not an implementation bug): the CES reward structure suppressed gait emergence, after which entropy-driven exploration diverged. **A2 (randomized-boundary control) result: A2 (randomized-boundary control) collapsed as well, more slowly (RMSE 0.956, no-fall 0%, noise std 3.56, ep_len 26) — the collapse is driven by the CES reward structure shared by both arms, not by event-aligned grouping**. Neither arm approaches "not worse than A0".

---

## 1. What was implemented (SPEC → code mapping)

All six frozen-SPEC components, additively on top of the frozen v6 environment (no v6 file touched; new task `g1_23dof_ces` registered alongside `g1_23dof`):

| SPEC | Implementation |
|---|---|
| ① Event segmentation, c from sim contact sensor, L_min=5 greedy debounce | `g1_23dof_ces_env.py::_reward_ces_loc` — online greedy debounce (candidate kept iff ≥5 steps from last kept boundary; identical greedy verified against a per-column python reference in tests). Contact flag c = net contact z-force > 1 N — the SAME criterion/threshold/sensor as the v6 `feet_air_time` reward. Training-side only, never in the 78-dim obs. |
| ② Foot-local reward, w_loc=0.05 | `_reward_ces_loc` returns r_slip + r_imp + r_duty with scale `ces_loc=0.05` (same ×dt path as every v6 scale). r_slip = -Σᵢ1[cᵢ]‖v_foot,xyᵢ‖ (world-frame foot vel); r_imp = -Σᵢ1[switchᵢ]‖ΔFᵢ‖ (3-D net contact force jump, both feet); r_duty = -Σᵢ|dutyᵢ-0.62|, dutyᵢ = foot-i contact steps / segment length. **Implementation choice** (SPEC under-specified): r_duty is settled at the step where the segment close is detected and the PPO side re-spreads it evenly over the owning segment (`_respread_duty`), so segmented TD(λ) attributes the segment-level signal to the whole segment. Total reward mass conserved (integration test). |
| ③ Segmented advantage: intra-segment TD(λ=0.95) + hard bootstrap at boundaries | `ces.py::compute_ces_returns` — reverse recursion `A_t = δ_t + (1-done_t)·γλ·(1-cut_t)·A_{t+1}` with `cut_t = bdy[t+1]`: the boundary value enters only through δ (hard bootstrap V(s_{t+1})); the running advantage never crosses a boundary. γ=0.99, λ=0.95 (both inherited from v6). |
| ④ Whole-segment minibatch, global batch unchanged | `ces.py::segment_minibatch_order` — all segments shuffled once, greedily packed into 4 batches by cumulative step count; batch sizes within one segment length of 98304/4; segments never split (tested). Same single-shuffle-reused-across-epochs semantics as the vanilla generator. |
| ⑤ Per-segment advantage normalization | per-(env, segment) mean/std standardization (same +1e-8 guard as the vanilla global normalization). |
| ⑥ PPO clipped objective & all hyperparameters inherited | `ces_ppo.py::update` is a verbatim copy of the vendored PPO loop with only the generator exchanged; γ/λ/lr schedule/clip 0.2/entropy 0.01/5 epochs/4 minibatches all unchanged; seed=1 (v6's seed, paired comparison). |
| A2 control | `ces.py::shuffle_segment_boundaries` — per env, per episode: segment-length multiset preserved, order uniformly permuted, segments re-laid from the episode start; episode boundaries stay where physics put them (tested: multiset preserved, episode structure untouched, deterministic under fixed generator). Reward stream identical to A1 (A2 differs ONLY in where the algorithm cuts segments). |

PPO update itself: no clipping of the surrogate change, value clipping kept, KL-adaptive lr kept.

### Unit + integration validation (all PASS, runs on CPU torch, no IsaacGym needed)

`tests/test_ces.py` — **24/24 PASS**:
- debounce greedy vs per-column python reference (incl. chain case [0,3,6,12,15,25] → [0,6,12,25])
- hand-computed segmented-returns reference (γ=0.9, λ=0.5, T=3, boundary at t=2 — exact match)
- per-segment normalization matches hand calculation; segment mean = 0
- **degenerate equivalence**: no boundaries → returns bit-identical to vanilla rsl_rl GAE; advantages = per-env normalization (single segment per env) — the strictly finer grouping granularity, which is exactly the effect A2 controls for
- no advantage leak across an event boundary (later-segment reward perturbation leaves earlier-segment advantages untouched)
- A2 shuffle: per-(env,episode) length multiset preserved, boundaries actually move, no boundary at t=0, deterministic under fixed generator
- whole-segment minibatches: full coverage, no duplicates, each segment confined to one batch, members contiguous, sizes balanced

`tests/test_ces_integration.py` — **9/9 PASS**: full CESPPO iteration on CPU with a real MLP ActorCritic (act → add → compute_ces_returns(a1 & a2) → update): duty re-spread conserves reward mass (float32 1e-4), advantages finite in both modes, real PPO update runs (finite losses, parameters change, no NaN).

## 2. gm task ledger

| run | gm task | mode | iters | wall | est. cost* | account | status |
|---|---|---|---|---|---|---|---|
| smoke | TASK_20261114_238 | a1 200it | — | — | — | 49 | aborted at create (personal-storage mount incompatible with goods; deleted, recreated) |
| smoke | **TASK_20261114_239** | a1 200it | 200 | 5.9 min | ¥0.6 | 49 (ketin84997) | finished, healthy |
| A1 | **TASK_20261114_240** | a1 10000it | 10000 | 2.15 h (17:06–19:15) | ¥12.4 | 49 | finished — **collapsed** |
| A2 | **TASK_20261115_095** | a2 10000it | 10000 | 1.96 h (09:59–11:56) | ¥11.3 | 49 | finished — **collapsed** (milder than A1: noise 3.56 vs 10.2, ep_len 26 vs 14) |
| (stale) | TASK_20261114_237 | a1 200it | 0 | 0 | ¥0 | 48 (ridovi5230) | never started — account 48 balance insufficient (run rejected 1002056); rotated to 49 per lead authorization |

\* cost estimated from wall time × v6's measured rate (¥12.63 / 2.19 h ≈ ¥5.76/h); the gm API exposes no per-task cost field. GPU total ≈ 4.2 h (smoke 0.10 + A1 2.15 + A2 1.96), est. total ¥24.2 ≤ 5.5 h budget ✓. Seed = 1 for all runs (v6's seed).

## 3. Training dynamics

### 3.1 Smoke (200 it) — PASS on hard criteria
No crash, parameters NaN-CLEAN, eval+export pipeline OK, throughput 143k steps/s (→ ~1.9 h / 10000 it). ep_len@200 = 36.8 vs v6@200 = 36.9 (paired, healthy). Two monitored risks quantified and recorded:
- segment length median 5.0 (L_min=5 floor; contact noise dominates under a random policy) — below the smoke guideline of ≥10, judged acceptable to proceed because the hard criteria passed and segment length was expected to grow with gait formation (it never did — see below)
- r_imp magnitude: raw mean -54.4/step (≈ -0.054 after w_loc·dt) — the largest single negative reward component (alive +0.035, tracking +0.003 at that stage)

### 3.2 A1 — collapse (design problem)

Anchors (v6 in parentheses): it1000 ≈46 (46) → it1500 ≈49 (61) → **it2000-5000 plateau 32-48 (v6 explodes to 297 at it2000, 712 at it3000)** → it6500 28 → it7180 12.5 → it9999 **14.1, reward 0.00, noise std 10.15** (v6 ends at 969, reward 7.65, noise 1.69). Value loss collapses to ~1e-6 after it6500 (short-life episodes make returns near-constant — the critic trivially fits the collapsed regime). Segment length median stays at 5-6 throughout (gait never forms). lr stays in its normal band throughout — the collapse is NOT an lr runaway.

**Mechanism chain (evidence-backed)**:
1. **Gait emergence suppressed by the CES reward structure.** Measured at the A1 plateau: stepping would earn tracking +feet_air_time ≈ +0.018-0.02/step, while each contact switch pays w_loc·dt·‖ΔF‖ ≈ -0.17 (≈ -0.02/step amortized at v6's switch rate) — net stepping benefit ≈ 0, versus standing still which pays only -0.0008/step (duty penalty at duty=1.0). The incentive gradient that produced v6's it2000 gait explosion is gone. r_imp is the dominant term (raw -54.4/step mean vs r_slip -0.33, r_duty -0.11 at smoke).
2. **No gait ⇒ no strong advantage signal at the plateau.** v6's explosion provides sharply increasing alive/tracking returns that pin the policy distribution; A1 sits on a flat 40-step plateau under 5-second push disturbances.
3. **Entropy term (0.01, inherited from v6 where it is balanced by gait-phase signal) becomes dominant** → policy std grows monotonically from 0.92 (it5100) to 10.15 (it10000) → wilder contacts → shorter segments (median 6) → noisier per-segment normalization → weaker signal → further divergence. Ep_len settles at ~14.

**Design vs implementation**: classified as a **design problem** — the frozen SPEC components (chiefly the r_imp magnitude at w_loc=0.05 relative to the frozen v6 reward structure, interacting with the inherited entropy coefficient) are incompatible with gait emergence in this environment. Evidence that it is NOT an implementation bug: (a) 24+9 unit/integration validations pass, including exact hand-computed references and degenerate equivalence to vanilla GAE; (b) it0-1500 tracks v6 pairwise (46 vs 46 at it1000 — the CES code paths are exercised and healthy); (c) the collapse unfolds through a coherent incentive mechanism, not a numerical blow-up (no NaN, no lr runaway, value loss well-behaved); (d) v6 with the same seed/hyperparameters does not collapse. Per the debug playbook the single permitted retry variant is A2 itself (the randomized-boundary control), which also separates reward-driven vs grouping-driven collapse.

### 3.3 A2 — collapsed too, more slowly (reward-driven collapse confirmed)

A2 (randomized boundaries, same segment-length multiset, identical reward stream) followed the same failure mode with a slower timescale. Anchors (v6 / A1 in parentheses): it1000 39.7 (46 / 46) → it2000 41.4 (297 / ~40) → it3000 39.4 (712 / 32-48 plateau) → it5000 37.5 (859 / 44) → it7000 36.9 (888 / 12.5) → it9999 **26.3, reward 0.00, noise std 3.56** (969 / 14.1; 1.69 / 10.15). No gait explosion at any point; noise std grows monotonically from 0.27 (it1000) with the same entropy-driven divergence signature, reaching only 3.56 by it10000 (vs A1's 10.15). ces_loc component deepens from -0.073 (it3000) to -0.131 (final) — same incentive trap. Segment stats: median 5, imp raw mean -59.5/step (vs A1 -54.4) — the randomized grouping did not change the physics or the reward, only where the algorithm cut segments. NaN parameter check CLEAN; full 10000 iterations completed; eval+export ran at train end.

## 4. Same-protocol evaluation table (in-trainer, v6 eval protocol verbatim: fixed cmds uniform [-1,1]³ with |v_xy|<0.2 zeroed, 10 s episodes, per-env domain-rand kept, obs noise & pushes off, 4096 episodes)

| metric | A0 = v6 (TASK_20261113_124) | A1 (model_10000) | A2 (model_10000) |
|---|---|---|---|
| lin-vel RMSE (m/s, mean) | **0.407** | 0.921 | 0.956 |
| no-fall rate | **99.90 %** | 0 % (4096/4096 fell) | 0 % (4096/4096 fell) |
| train-end ep_len (steps) | 969 | 14.1 | 26.3 |
| train-end mean reward | 7.65 | 0.00 | 0.00 |
| per-step torque term (Episode/rew_torques) | -0.046 (active walking) | -0.005 (near-immobile) | -0.0015 (plateau@it4000) / -0.0066 (final) |
| per-step dof-acc term | -0.061 | -0.012 | -0.0041 (plateau@it4000) / -0.0225 (final) |
| segment length median (detected contact segments) | n/a (no CES records in v6) | 6 (final iters; 5 at smoke) | 5 (final iters; detected real contact segments, identical detector to A1) |
| noise std (final) | 1.69 | 10.15 | 3.56 |

Note on the low A1 torque/acc numbers: they reflect near-immobility of the collapsed policy, not efficiency.

## 5. Checkpoint export & forward-consistency (auto-run at train end)

| run | jit vs torch max abs diff | onnx vs torch max abs diff |
|---|---|---|
| A1 (model_10000) | **0.0 (PASS <1e-5)** | 4.58e-5 (> 1e-5) |
| A2 (model_10000) | **0.0 (PASS <1e-5)** | **9.54e-6 (PASS <1e-5)** |
| v6 reference (same export code path) | <1e-5 PASS | <1e-5 PASS |

A1's onnx deviation exceeds 1e-5: the collapsed policy (std 10) produces large action magnitudes where fp32 GEMM-order differences between onnxruntime and torch amplify to 4.6e-5. The TorchScript export is bit-identical. Reported as measured.

## 6. Conclusion vs A0 and honest diagnosis

- **A1 is strictly worse than A0 on every axis** (RMSE 0.921 vs 0.407; no-fall 0% vs 99.9%; ep_len 14 vs 969). The frozen-spec CES-PPO **failed to train** in this environment.
- - **A2 is also strictly worse than A0** (RMSE 0.956, no-fall 0%, ep_len 26) — the randomized-boundary control collapsed too.
- **Design-level diagnosis (what to change if CES is revisited)**: (1) r_imp must be rescaled/normalized (e.g. by body weight, or per-step force-rate rather than raw ΔF, or w_loc ≪ 0.05) so that stepping retains a positive net benefit; a quick calculation with the measured ΔF ≈ 170 N/switch says w_loc·dt·ΔF ≈ -0.17/switch vs +0.02/step tracking — the balance must be restored before any advantage-estimator effect can even be measured; (2) the duty target 0.62 penalizes standing (duty=1.0) by only -0.0008/step while r_imp penalizes the alternative (stepping) 20× more per switch — the local-reward vector points away from gait; (3) with gait suppressed, entropy 0.01 (tuned for v6's trajectory) diverges — any CES retry needs either a healthy stepping incentive or a reduced entropy/adaptive-std floor; (4) the segmentation/advantage machinery itself (debounce, segmented TD(λ), per-segment normalization, whole-segment minibatches, A2 shuffle) is implemented and validated — the failure was not in the estimator but in the reward that feeds it (A2 result further separates the two: A2 collapsed through the same mechanism with a milder trajectory — the collapse is dominated by the CES reward structure (r_imp magnitude at w_loc=0.05) shared by both arms; event-aligned segmentation contributed no measurable benefit and at most mildly accelerated the divergence (noise 10.2 vs 3.56 at it10000)).
- **Applicability**: conclusions hold for the frozen v6 environment/reward/PD/obs + this exact SPEC (w_loc=0.05, L_min=5, duty target 0.62, λ=0.95), seed=1, IsaacGym preview-4, 23-DoF G1. They do NOT rule out CES-style estimators per se — they rule out this frozen instantiation.

## 7. Artifacts

- `ces_results/a1_*.csv` — 10 curve keys × 10000 points (ep_len, reward, noise, lr, ces_loc/alive/tracking/feet_air_time components, value/surrogate losses)
- `ces_results/a1_final_summary.json` — A1 eval + export report (as printed by gm_run_all)
- `ces_results/a1_model_10000.pt` — A1 final checkpoint (475 KB)
- - `ces_results/a2_*.csv` — 12 curve keys × 10000 points
- `ces_results/a2_final_summary.json` — A2 eval + export report
- `ces_results/a2_model_10000.pt` — A2 final checkpoint
- repo branch `ces-ppo` @ `6dae7bd`: implementation + tests (`tests/test_ces.py`, `tests/test_ces_integration.py`)
- v6 anchors from `submit/v6_task124_curves.csv` (baseline deliverable)
