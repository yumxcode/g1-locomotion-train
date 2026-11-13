#!/usr/bin/env python
"""Nominal-domain acceptance eval (supplementary evidence, same fixed protocol).

Identical protocol to gm_run_all.stage2_eval (4096 episodes x 10 s, fixed commands
uniform [-1,1]^3 with |v_xy|<0.2 zeroed, per-episode xy RMSE -> mean, fall =
non-timeout termination) but with ALL domain randomization disabled (friction=1.0,
no mass offset, nominal PD gains, no action delay) — i.e. the deployment condition
declared in INTERFACE.md §8. Reported as SUPPLEMENTARY evidence alongside the
preregistered (domain-rand kept) numbers, never replacing them.

Run: gm-run g1-locomotion-train/gm_eval_nominal.py --task=g1_23dof --headless \
        --load_ckpt=g1-locomotion-train/checkpoints/model_10000_23dof.pt
"""
import copy
import os
import sys

ROOT = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'rsl_rl'))
sys.path.insert(1, os.path.join(ROOT, 'unitree_rl_gym'))

import isaacgym  # noqa: F401


def main():
    import gm_run_all  # noqa: F401  (also imports legged_gym etc.)
    from legged_gym.utils import get_args, task_registry

    load_ckpts = gm_run_all.parse_load_ckpts()
    if not load_ckpts:
        print('[nominal] ERROR: --load_ckpt=path required')
        sys.exit(2)
    sys.argv = [a for a in sys.argv if not a.startswith('--load_ckpt')]
    args = get_args()

    env_cfg, _ = task_registry.get_cfgs(args.task)
    cfg = copy.deepcopy(env_cfg)
    cfg.domain_rand.randomize_friction = False      # friction = terrain default 1.0
    cfg.domain_rand.randomize_base_mass = False     # nominal mass
    cfg.domain_rand.randomize_gains = False         # nominal PD 60/1.5
    cfg.domain_rand.randomize_action_delay = False  # no delay
    cfg.domain_rand.push_robots = False
    env = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    exp_name = train_cfg.runner.experiment_name
    print('[nominal] domain randomization DISABLED: friction/mass/PD/delay/push all nominal')

    summaries = {}
    for ck in load_ckpts:
        tag = os.path.splitext(os.path.basename(ck))[0].replace('model_', 'model')
        tag += '_nominal'
        print('[nominal] loading checkpoint:', ck, '-> tag', tag)
        ppo_runner.load(ck, load_optimizer=False)
        results, _ = gm_run_all.stage2_eval(args, env, ppo_runner, exp_name, tag=tag)
        summaries[tag] = results
    import json
    print('[nominal] FINAL SUMMARY:', json.dumps(summaries, indent=2))
    gm_run_all.settle_for_sdk_upload(60)


if __name__ == '__main__':
    main()
