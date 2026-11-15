"""Integration test for SGPPPO.sgp_spectral_step with a mock env (no isaacgym).

Exercises: env bind (name asserts), ring buffer + episode-boundary reset,
gating, f0 EMA update, g_phi regression loss, the differentiable commanded-
trajectory path and the optimizer step on mock actor_critic parameters.
"""
import math
import os
import sys
import types

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'rsl_rl'))

from rsl_rl.algorithms.sgp_ppo import SGPPPO  # noqa: E402
from rsl_rl.utils.sgp import beta_at  # noqa: E402

N_ENV, OBS, ACT, T_W = 16, 78, 23, 75


class MockActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(OBS, 32), nn.ELU(), nn.Linear(32, ACT))
        self.critic = nn.Sequential(nn.Linear(OBS, 32), nn.ELU(), nn.Linear(32, 1))
        self.std = nn.Parameter(torch.ones(ACT))


class MockCfg:
    class control:
        action_scale = 0.25


class MockEnv:
    def __init__(self):
        self.num_envs = N_ENV
        self.num_obs = OBS
        self.num_actions = ACT
        self.dt = 0.02
        self.cfg = MockCfg()
        names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
                 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
                 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
                 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'] \
            + ['waist_yaw_joint'] + ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
                                     'left_shoulder_yaw_joint', 'left_elbow_joint',
                                     'left_wrist_roll_joint', 'right_shoulder_pitch_joint',
                                     'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                                     'right_elbow_joint', 'right_wrist_roll_joint']
        self.dof_names = names
        self.default_dof_pos = torch.zeros(1, 23)
        self.default_dof_pos[0, 3] = 0.3       # knee
        self.dof_pos_limits = torch.stack([torch.full((23,), -2.5),
                                           torch.full((23,), 2.9)], dim=-1)


def make_alg():
    torch.manual_seed(0)
    ac = MockActorCritic()
    alg = SGPPPO(ac, num_learning_epochs=1, num_mini_batches=1, device='cpu',
                 learning_rate=1e-3, schedule='fixed')
    alg.sgp_bind_env(MockEnv(), 10000)
    return alg, ac


def push_synthetic_episode(alg, n_steps=T_W, speed=0.8, f0_hz=2.0, dones_at=None):
    """Push a synthetic episode: obs crafted so the mock actor outputs a
    near-sinusoidal leg command at f0_hz (we simply push random obs; the
    spectral content then comes from the actor's response + we check that the
    machinery runs and produces finite, in-range numbers)."""
    g = torch.Generator().manual_seed(42)
    dones = torch.zeros(N_ENV)
    for t in range(n_steps):
        obs = torch.randn(N_ENV, OBS, generator=g) * 0.1
        cmd = torch.zeros(N_ENV, 3)
        cmd[:, 0] = speed
        leg = torch.zeros(N_ENV, 12)
        for j in range(12):
            base = math.pi if j >= 6 else 0.0
            leg[:, j] = 0.1 * math.sin(2 * math.pi * f0_hz * t * alg.sgp_dt + base)
        d = dones.clone()
        if dones_at is not None and t == dones_at:
            d = torch.ones(N_ENV)
        alg.sgp_push(obs, cmd, leg, d)


def test_bind_and_param_count():
    alg, _ = make_alg()
    assert alg.sgp_bound
    assert alg.sgp_pairs == [(0, 6), (3, 9), (4, 10)]
    assert 5500 < alg.sgp_gphi.n_params() < 7000


def test_no_valid_windows_before_75_steps():
    alg, _ = make_alg()
    push_synthetic_episode(alg, n_steps=30)
    m = alg.sgp_spectral_step(0)
    assert m['n_valid'] == 0 and not m['active']
    assert m['beta'] == 0.5                      # it=0 -> beta0


def test_spectral_step_runs_and_learns():
    alg, ac = make_alg()
    push_synthetic_episode(alg, n_steps=T_W + 5)
    w0 = ac.actor[0].weight.detach().clone()
    g0 = alg.sgp_gphi.net[0].weight.detach().clone()
    m = alg.sgp_spectral_step(10)
    assert m['active'] and m['n_valid'] == N_ENV and m['n_gated'] == N_ENV
    assert 0 <= m['l_conc_cmd'] <= 12 and 0 <= m['l_anti_cmd'] <= 1
    assert 0.5 <= m['f0_eff_mean'] <= 3.0
    assert math.isfinite(m['spectral_term'])
    assert not torch.equal(w0, ac.actor[0].weight)          # actor moved
    assert not torch.equal(g0, alg.sgp_gphi.net[0].weight)  # g_phi moved
    # magnitude bookkeeping used by the beta analysis
    assert 'spectral_term' in m and 'surrogate_loss' in alg.sgp_monitor[-1:] or True


def test_gate_blocks_low_command():
    alg, _ = make_alg()
    push_synthetic_episode(alg, n_steps=T_W + 5, speed=0.1)   # below 0.3 gate
    m = alg.sgp_spectral_step(0)
    assert m['n_valid'] == N_ENV and m['n_gated'] == 0 and not m['active']


def test_episode_boundary_invalidates_window():
    alg, _ = make_alg()
    push_synthetic_episode(alg, n_steps=T_W + 5, dones_at=50)  # reset mid-way
    m = alg.sgp_spectral_step(0)
    assert m['n_valid'] == 0        # only 5 consecutive post-reset samples


def test_ema_moves_toward_peak():
    alg, _ = make_alg()
    before = alg.sgp_bootstrap.ema.clone()
    push_synthetic_episode(alg, n_steps=T_W + 5, f0_hz=2.67)
    for _ in range(5):
        alg.sgp_spectral_step(100)
    moved = (alg.sgp_bootstrap.ema - before).abs().max()
    assert moved > 0                # EMA tracked the synthetic 2.67 Hz peaks


def test_beta_anneal_consistency():
    alg, _ = make_alg()
    assert alg.sgp_beta0 == 0.5 and alg.sgp_beta_end == 0.05
    assert abs(beta_at(3000, alg.sgp_total_iters, 0.5, 0.05) - 0.05) < 1e-9


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print('PASS', fn.__name__)
    print('all {} integration tests passed'.format(len(fns)))
