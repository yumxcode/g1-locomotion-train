#!/usr/bin/env python
"""Diagnostic v3: per-term reward decomposition.

Two trajectories, per policy step we evaluate every reward function and print the
time-average of each term (scale already includes dt) plus the unclipped per-step
total. Goal: quantify which penalties eat the alive bonus during (a) quiet standing
(zero actions) and (b) a random policy (collapse trajectory), before the next full run.

Run: gm-run g1-locomotion-train/gm_diag.py --task=g1_23dof --headless --num_envs=64
"""
import os
import sys

ROOT = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'rsl_rl'))
sys.path.insert(1, os.path.join(ROOT, 'unitree_rl_gym'))

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa
from legged_gym.utils import get_args, task_registry


def decompose(env, actions_fn, label, n_steps=30):
    with torch.inference_mode():
        env.reset()
        sums = {name: 0.0 for name in env.reward_names}
        tot = 0.0
        steps_alive = []
        for t in range(n_steps):
            a = actions_fn(t)
            env.step(a)
            step_tot = 0.0
            for name, fn in zip(env.reward_names, env.reward_functions):
                v = float((fn() * env.reward_scales[name]).mean())
                sums[name] += v
                step_tot += v
            tot += step_tot
            steps_alive.append(float(env.episode_length_buf.mean()))
    print('[diag] ==== %s: per-term time-mean over %d steps (scale incl. dt) ====' % (label, n_steps))
    for name in sorted(sums, key=lambda k: sums[k]):
        print('[diag]   %-22s %+.5f' % (name, sums[name] / n_steps))
    print('[diag]   %-22s %+.5f   << unclipped per-step TOTAL' % ('TOTAL', tot / n_steps))
    print('[diag]   mean episode_length_buf over run: %.1f (last %.1f)' % (
        sum(steps_alive) / len(steps_alive), steps_alive[-1]))
    return tot / n_steps


def main():
    args = get_args()
    env, _ = task_registry.make_env(name=args.task, args=args)
    print('[diag] reward scales (post-dt):', {k: round(v, 8) for k, v in env.reward_scales.items()})
    print('[diag] only_positive_rewards:', env.cfg.rewards.only_positive_rewards)

    zero = lambda t: torch.zeros(env.num_envs, env.num_actions, device=env.device)
    torch.manual_seed(0)

    def rand(t):
        return torch.randn(env.num_envs, env.num_actions, device=env.device)

    a = decompose(env, zero, 'A: quiet standing (zero actions)')
    b = decompose(env, rand, 'B: random policy N(0,1)')
    print('[diag] VERDICT: standing per-step total %+.5f -> %s' % (
        a, 'PASS (>0)' if a > 0 else 'FAIL (<=0): alive bonus still dominated'))
    print('[diag] random per-step total %+.5f (expected negative during collapse)' % b)
    print('[diag] done')


if __name__ == '__main__':
    main()
