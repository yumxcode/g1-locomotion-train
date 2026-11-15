# CES repo update — branch `ces-ppo` @ 6dae7bd

Repo: https://github.com/yumxcode/g1-locomotion-train (branch `ces-ppo`, pushed; main line untouched — v7 experiments unaffected)

Branched from v6 final `64b0252`. **Only additive changes** — every v6 file is untouched; the frozen v6 environment/reward/PD/obs run unchanged under the original `g1_23dof` task. New/changed files:

| file | role |
|---|---|
| `rsl_rl/rsl_rl/utils/ces.py` | pure-torch core: debounce (L_min=5 greedy), segmented TD(λ) with hard boundary bootstrap, per-segment advantage normalization, A2 boundary shuffle (per-env per-episode length multiset preserved), whole-segment minibatch packing |
| `rsl_rl/rsl_rl/storage/ces_rollout_storage.py` | CES buffers, duty re-spread over owning segment (mass-conserving), compute_ces_returns, segment minibatch generator |
| `rsl_rl/rsl_rl/algorithms/ces_ppo.py` | CESPPO(ces_mode=a1\|a2); PPO update loop verbatim from vendored PPO with only the generator exchanged |
| `rsl_rl/rsl_rl/{algorithms,storage}/__init__.py`, `runners/on_policy_runner.py` | registrations + eval() namespace (one import line) |
| `unitree_rl_gym/legged_gym/envs/g1/g1_23dof_ces_env.py` | G1_23dofCESRobot: online debounced segmentation, r_slip/r_imp/r_duty reward (w_loc via scales.ces_loc=0.05), extras['ces'] export; obs/PD/other rewards untouched; contact flag = net contact z-force > 1 N (same criterion as v6 feet_air_time) |
| `unitree_rl_gym/legged_gym/envs/g1/g1_23dof_ces_config.py` | G1_23dofCESCfg = v6 frozen config + ces_loc scale only; PPO cfg swaps algorithm_class_name to CESPPO (all hyperparameters inherited: γ=0.99, λ=0.95, lr 1e-3 adaptive, clip 0.2, entropy 0.01, 5 epochs × 4 minibatches, 24 steps/env, seed=1) |
| `unitree_rl_gym/legged_gym/envs/__init__.py` | task registration `g1_23dof_ces` |
| `gm_run_all.py` | `--ces_mode=a1|a2` switch (pre-get_args extraction, same pattern as --load_ckpt) + CES diagnostics (segment stats, reward-component means, NaN check) |
| `tests/test_ces.py` | 24 unit tests — ALL PASS |
| `tests/test_ces_integration.py` | 9 integration tests (full CESPPO iteration on CPU) — ALL PASS |

Unit validation highlights: hand-computed segmented-returns reference match; degenerate equivalence to vanilla rsl_rl GAE when no boundaries; no advantage leak across boundaries; per-segment mean-zero; A2 multiset preservation; segments never split across minibatches; duty re-spread conserves reward mass; real PPO update runs without NaN.

Reproduce tests: `python tests/test_ces.py && python tests/test_ces_integration.py` (torch CPU env).
Reproduce training: `gm-run g1-locomotion-train/gm_run_all.py --task=g1_23dof_ces --headless --max_iterations=10000 --ces_mode=a1` (or a2).
