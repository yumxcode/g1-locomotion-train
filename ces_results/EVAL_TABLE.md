# EVAL_TABLE — A0/A1/A2 same-protocol in-trainer evaluation

Protocol (identical for all three runs, v6's stage2_eval verbatim): fixed commands uniform vx,vy,wz ∈ [-1,1]³ with |v_xy|<0.2 zeroed; base-frame xy velocity; per-episode RMSE over xy dims then mean over 4096 episodes; 10 s episodes; fall = early termination (non-timeout); per-env domain-rand samples kept; obs noise & pushes off.

| metric | A0 = v6 (TASK_20261113_124) | A1 event-segmented (TASK_20261114_240) | A2 randomized-boundary control (TASK_20261115_095) |
|---|---|---|---|
| lin-vel tracking RMSE (m/s) | **0.407** | 0.921 | 0.956 |
| RMSE std over episodes | 0.251 | 0.198 | 0.211 |
| no-fall rate | **99.90 %** | 0 % (4096/4096 fell) | 0 % (4096/4096 fell) |
| train-end ep_len (steps, 1000 max) | **969** | 14.1 | 26.3 |
| train-end mean reward | **7.65** | 0.00 | 0.00 |
| policy noise std (final) | 1.69 | 10.15 | 3.56 |
| per-step torque term (Episode/rew_torques) | -0.046 (active walking) | -0.005 (near-immobile) | -0.0066 (near-immobile) |
| per-step dof-acc term | -0.061 | -0.012 | -0.0225 |
| CES segment length median (detected contact segments) | n/a (v6 has no CES records) | 6 (smoke: 5) | 5 |
| CES reward components (raw means, smoke→final) | n/a | imp -54.4 → large; slip -0.33; duty -0.11 | imp -59.5; slip -0.41; duty -0.11 |
| gait emergence (v6 explodes at it2000) | it2000: 297 → it3000: 712 | **never** (plateau 32-48) | **never** (plateau 37-41) |
| jit vs torch max abs diff | 0.0 PASS | **0.0 PASS** | **0.0 PASS** |
| onnx vs torch max abs diff | 5.7e-6 PASS | 4.58e-5 (FAIL>1e-5) | **9.54e-6 PASS** |

\* v6 std computed from `deliverables/checkpoints/v6_23dof_final/eval_results_23dof_final.pt` (RMSE std 0.251, max 1.629); A1/A2 from their `*_final_summary.json`. v6 export numbers from the v6 TRAINING_REPORT ledger (TASK_20261113_124: jit 0.0 / onnx 5.7e-6).

**Reading**: both CES arms fail every behavioral metric by large margins; the low torque/acc numbers reflect near-immobility, not efficiency. A2 (randomized boundaries) mirrors A1's failure mode at a slower divergence rate — the failure is dominated by the CES reward structure shared by both arms (r_imp ≈ -0.17 per contact switch at w_loc=0.05·dt cancels the stepping benefit), not by event-aligned advantage grouping. Export consistency <1e-5 is met by TorchScript on both arms and by ONNX on A2; A1's ONNX deviation (4.58e-5) is reported as measured (collapsed policy with std 10 produces large action magnitudes where fp32 GEMM-order differences amplify).
