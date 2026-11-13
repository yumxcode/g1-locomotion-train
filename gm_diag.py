#!/usr/bin/env python
"""Diagnostic: why does g1_23dof terminate at spawn?

Loads the env, resets, and steps zero actions, printing per-body contact
forces (top offenders), base pose stats and termination causes at each step.
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

    body_names = [env.gym.get_actor_rigid_body_name(env.envs[0], env.actor_handles[0], i)
                  for i in range(env.num_bodies)]
    print('[diag] num_bodies', env.num_bodies, 'num_dof', env.num_dof)
    print('[diag] bodies:', body_names)
    print('[diag] termination bodies:', [body_names[i] for i in env.termination_contact_indices.tolist()])
    print('[diag] feet:', [body_names[i] for i in env.feet_indices.tolist()])
    print('[diag] dof_names:', list(env.dof_names))
    print('[diag] p_gains[0]:', env.p_gains[0].tolist() if env.p_gains.dim() > 1 else env.p_gains.tolist())
    print('[diag] torque_limits:', env.torque_limits.tolist())

    with torch.inference_mode():
        obs = env.reset()
        # state right after reset, before any step
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        print('[diag] post-reset base z: mean %.3f min %.3f max %.3f' % (
            env.root_states[:, 2].mean(), env.root_states[:, 2].min(), env.root_states[:, 2].max()))
        print('[diag] post-reset dof_pos[0]:', [round(x, 3) for x in env.dof_pos[0].tolist()])

        zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        for t in range(6):
            env.step(zero)
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            fn = torch.norm(env.contact_forces[:, :, :], dim=-1)  # [n, bodies]
            fn_max = fn.max(dim=0).values
            top = sorted(range(env.num_bodies), key=lambda i: -fn_max[i])[:6]
            print('[diag] step %d: resets %d (timeout %d) | roll|%.2f pitch|%.2f | z mean %.3f | top contacts: %s' % (
                t, int(env.reset_buf.sum()), int(env.time_out_buf.sum()),
                float(env.rpy[:, 0].abs().max()), float(env.rpy[:, 1].abs().max()),
                float(env.root_states[:, 2].mean()),
                ', '.join('%s %.1fN' % (body_names[i], float(fn_max[i])) for i in top)))
    print('[diag] done')


if __name__ == '__main__':
    main()
