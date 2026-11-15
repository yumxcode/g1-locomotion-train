# EVAL_TABLE_ITER2 — A0 / A1′ / A2′ same-protocol in-trainer evaluation (estimator-only arms)

Protocol (identical for all arms, v6's stage2_eval verbatim): fixed commands uniform vx,vy,wz ∈ [-1,1]³ with |v_xy|<0.2 zeroed; base-frame xy velocity; per-episode RMSE over xy dims then mean over 4096 episodes; 10 s episodes; fall = early termination (non-timeout); per-env domain-rand samples kept; obs noise & pushes off.

Arms:
- **A0 = v6 baseline** (TASK_20261113_124, model_10000, account 48) — vanilla PPO, frozen v6 reward, 10000 iters, seed 1
- **A1′** (TASK_20261115_116, account 49) — **v6 reward verbatim (w_loc = 0, no `ces_loc` key in the effective reward table — verified on-platform)** + event-segmented advantage estimator package (debounce L_min=5, segment TD(λ=0.95) with hard boundary bootstrap, per-segment advantage normalization, whole-segment minibatches, global batch unchanged); 10000 iters budget, seed 1
- **A2′** (TASK_20261115_117, account 49) — identical to A1′ except segment boundaries randomly re-placed (per-(env,episode) length multiset preserved)

| metric | A0 = v6 | A1′ event-segmented | A2′ randomized-boundary |
|---|---|---|---|
| gm task | TASK_20261113_124 | TASK_20261115_116 | TASK_20261115_117 |
| seed | 1 | 1 | 1 |
| trained iters (budget 10000) | 10000 | **2500 (early-stopped)** | **2500 (early-stopped)** |
| it2500 early-stop gate | n/a | **ep_len 41.92 ≤ 50 → STOP ARM** (gate log archived) | **ep_len 38.94 ≤ 50 → STOP ARM** (gate log archived) |
| lin-vel tracking RMSE (m/s) | **0.407** | 0.803 | 0.817 |
| RMSE std over episodes | 0.251 | 0.244 | 0.243* |
| no-fall rate | **99.90 %** | 0 % (4096/4096 fell) | 0.02 % (4095/4096 fell) |
| train-end mean ep_len (steps, 1000 max) | 969 | 41.9 | 39.0 |
| train-end mean reward | 7.65 | 0.247 | 0.211 |
| policy noise std (final) | 1.69 | **0.637 (converging)** | **0.679 (converging)** |
| per-step torque term (Episode/rew_torques, last-50) | -0.046 (active walking) | -0.0014 (near-immobile) | -0.0015 (near-immobile) |
| per-step dof-acc term (last-50) | -0.061 | -0.0034 | -0.0039 |
| per-step alive term (last-50) | n/a (recorded in v6 ledger) | 0.0412 | 0.0403 |
| CES segment stats (last iter) | n/a (no CES records) | n=15240, mean 6.45, **median 5** (=L_min debounce), min 1, max 24 | n=15670, mean 6.27, **median 5**, min 1, max 24 |
| gait emergence | it1000: 46 → **it2000: 297** → it3000: 712 | **never** (plateau 36-42: it500 37.8 / it1000 40.3 / it2000 40.3 / it2499 41.9) | **never** (plateau 38-42: it500 38.7 / it1000 42.3 / it2000 40.0 / it2498 39.0) |
| jit vs torch max abs diff | 0.0 PASS | **0.0 PASS** | **0.0 PASS** |
| onnx vs torch max abs diff | 5.7e-6 PASS | **4.29e-6 PASS** | **4.29e-6 PASS** |

\* A2′ RMSE std from its gm_final_summary (0.2433). A0 numbers from the iteration-1 EVAL_TABLE / v6 TRAINING_REPORT ledger (same eval protocol).

**Reading.** With the foot-local reward fully removed (reward stream byte-identical to v6, the ONLY difference vs A0 being the advantage-estimator package), **both estimator arms still fail to reproduce v6's gait emergence**: v6 exploded to ~297 by it2000 while A1′/A2′ sat flat at 36-42 through it2500, triggering the pre-registered early-stop gate in BOTH arms. A1′ and A2′ are at the same level on every metric (RMSE 0.803 vs 0.817, no-fall 0% vs 0.02%, ep_len 41.9 vs 38.9, torque -0.0014 vs -0.0015): event-aligned segmentation is **indistinguishable from random segmentation**. Failure mode differs from iteration 1: noise std CONVERGES (0.64/0.68) instead of blowing up (10.2/3.56 there) — the estimator arms settle stably into a near-immobile local optimum rather than diverging. Export consistency <1e-5 holds on both arms for jit AND onnx.

Caveat (honest reporting): A1′/A2′ trained only 2500/10000 iters (pre-registered stop). Since v6's emergence is unambiguous by it2000 and both arms show zero upward trend over the last 1500 iters, a late explosion after it2500 is not plausible under this budget; per SPEC the arms were not run further.
