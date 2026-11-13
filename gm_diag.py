#!/usr/bin/env python
"""Diagnostic v2: replicate the training action path (random ~N(0,1)) and
classify termination causes per policy step.

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


def main():
    args = get_args()
    env, _ = task_registry.make_env(name=args.task, args=args)

    body_names = list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))
    print('[diag] termination bodies:', [body_names[i] for i in env.termination_contact_indices.tolist()])

    torch.manual_seed(0)
    n_termin = len(env.termination_contact_indices)
    with torch.inference_mode():
        obs = env.reset()
        std = 1.0
        # training-style random actions; also re-randomize via reset bookkeeping
        for t in range(10):
            actions = torch.randn(env.num_envs, env.num_actions, device=env.device) * std
            env.step(actions)
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            fn = torch.norm(env.contact_forces[:, :, :], dim=-1)  # [n, bodies]
            # termination cause decomposition (mirrors check_termination)
            term_contact = (fn[:, env.termination_contact_indices] > 1.0).any(dim=1)
            roll_bad = env.rpy[:, 0].abs() > 0.8
            pitch_bad = env.rpy[:, 1].abs() > 1.0
            newly = env.reset_buf & ~env.time_out_buf
            # among envs that just terminated, which cause fired
            n_term = int(newly.sum())
            c_contact = int((term_contact & newly).sum())
            c_roll = int((roll_bad & newly).sum())
            c_pitch = int((pitch_bad & newly).sum())
            # contact on each termination body among ALL envs
            term_forces = fn[:, env.termination_contact_indices]
            per_body = [float(term_forces[:, k].max()) for k in range(n_termin)]
            # biggest contact anywhere
            fn_max = fn.max(dim=0).values
            top = sorted(range(env.num_bodies), key=lambda i: -fn_max[i])[:5]
            print('[diag] step %d: newly_terminated %d (contact %d, roll %d, pitch %d) | term-body maxN %s | rpy max %.2f/%.2f | top: %s' % (
                t, n_term, c_contact, c_roll, c_pitch,
                ['%.1f' % x for x in per_body],
                float(env.rpy[:, 0].abs().max()), float(env.rpy[:, 1].abs().max()),
                ', '.join('%s %.0fN' % (body_names[i], float(fn_max[i])) for i in top)))
    print('[diag] done')


if __name__ == '__main__':
    main()
