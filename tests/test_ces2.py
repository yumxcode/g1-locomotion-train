"""Iteration-2 (estimator-only CES, w_loc = 0) tests — CPU only, no IsaacGym.

The iteration-2 arm runs the segmented-advantage estimator package on the
frozen v6 reward, i.e. the CES duty stream is all-zero (no foot-local reward
is paid anywhere). These tests pin down the two algorithmic consequences:

  A. zero-credit no-op: with ces_duty_credit == 0 everywhere, _respread_duty
     must leave the reward buffer BIT-IDENTICAL (no re-spread of nothing).
  B. end-to-end estimator-only iteration, a1 AND a2: synthetic rollout with
     real boundaries but no duty rewards -> compute_ces_returns -> PPO update
     runs, advantages/losses finite, parameters change, no NaN.
  C. a2 boundary shuffle keeps the per-(env,episode) segment-length multiset
     under the estimator-only stream (re-check with zero credits).

(The reward-table semantics — ces2 has NO ces_loc key while v6 is otherwise
untouched — is verified on-platform: gm_run_all prints the effective reward
scales at startup of every CES run; see TRAINING_REPORT_ITER2.)

Run: python tests/test_ces2.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'rsl_rl'))

import torch

from rsl_rl.algorithms import CESPPO
from rsl_rl.modules import ActorCritic
from rsl_rl.utils.ces import shuffle_segment_boundaries

PASS, FAIL = 0, 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print('  PASS  {}'.format(name))
    else:
        FAIL += 1
        print('  FAIL  {}  {}'.format(name, detail))


def synth_rollout(alg, T, N, seed):
    """Fill the storage with a synthetic estimator-only rollout: real event
    boundaries, v6-style step rewards, ZERO duty credits (w_loc = 0)."""
    torch.manual_seed(seed)
    for t in range(T):
        obs = torch.randn(N, alg.storage.observations.shape[-1])
        act = alg.act(obs, obs)
        rew = torch.rand(N) * 0.1 + 0.05          # v6-style reward, NO duty term
        dones = torch.zeros(N)
        if t in (7, 19):
            dones[torch.randint(0, N, (1,)).item()] = 1.0
        bdy = torch.zeros(N, dtype=torch.bool)
        duty_kind = torch.zeros(N, dtype=torch.int64)
        if t in (6, 13, 20):
            sel = torch.rand(N) < 0.5
            bdy[sel] = True
            duty_kind[sel] = 0                    # event-closed, zero amount
        if (dones > 0).any():
            duty_kind[dones > 0] = 1              # done-closed, zero amount
        infos = {'time_outs': torch.zeros(N),
                 'ces': {'bdy': bdy, 'duty_credit': torch.zeros(N),
                         'duty_kind': duty_kind}}
        alg.process_env_step(rew, dones, infos)


T, N = 24, 8
OBS, ACT = 6, 2

# ---------------------------------------------------------------- A --------
print('[A] zero duty-credit => re-spread is a bit-identical no-op')
ac_a = ActorCritic(OBS, OBS, ACT, [16, 8], [16, 8], activation='elu',
                   init_noise_std=1.0)
alg_a = CESPPO(ac_a, num_learning_epochs=2, num_mini_batches=4, clip_param=0.2,
               gamma=0.99, lam=0.95, learning_rate=1e-3, schedule='fixed',
               device='cpu', ces_mode='a1')
alg_a.init_storage(N, T, [OBS], [OBS], [ACT])
synth_rollout(alg_a, T, N, seed=0)
storage = alg_a.storage
rewards_before = storage.rewards.clone()
storage._respread_duty()
check('A1 rewards unchanged (bit-identical)',
      torch.equal(rewards_before, storage.rewards),
      'max diff {}'.format(float((rewards_before - storage.rewards).abs().max())))

# ---------------------------------------------------------------- B --------
for mode in ('a1', 'a2'):
    print('[B] estimator-only end-to-end iteration, ces_mode={}'.format(mode))
    ac = ActorCritic(OBS, OBS, ACT, [16, 8], [16, 8], activation='elu',
                     init_noise_std=1.0)
    alg = CESPPO(ac, num_learning_epochs=2, num_mini_batches=4, clip_param=0.2,
                 gamma=0.99, lam=0.95, learning_rate=1e-3, schedule='fixed',
                 device='cpu', ces_mode=mode)
    alg.init_storage(N, T, [OBS], [OBS], [ACT])
    params0 = [p.detach().clone() for p in ac.parameters()]
    synth_rollout(alg, T, N, seed=1)
    alg.compute_returns(torch.randn(N, OBS))
    adv = alg.storage.advantages
    check('B[{}] advantages finite'.format(mode), bool(torch.isfinite(adv).all()))
    check('B[{}] advantages non-degenerate (std>0)'.format(mode),
          float(adv.std()) > 0, 'std {:.2e}'.format(float(adv.std())))
    n_minibatches = 0
    for _ in alg.storage.segment_mini_batch_generator(4, 2):
        n_minibatches += 1
    vl, sl = alg.update()
    check('B[{}] PPO update finite losses'.format(mode),
          vl == vl and sl == sl)  # NaN != NaN
    moved = max(float((p.detach() - p0).abs().max())
                for p, p0 in zip(ac.parameters(), params0))
    check('B[{}] parameters updated'.format(mode), moved > 0,
          'max move {:.2e}'.format(moved))
    check('B[{}] minibatch count preserved (2 epochs x 4)'.format(mode),
          n_minibatches == 8, 'got {}'.format(n_minibatches))
    check('B[{}] segment stats exported'.format(mode), bool(alg.storage.last_seg_stats))

# ---------------------------------------------------------------- C --------
print('[C] a2 shuffle keeps the per-(env,episode) length multiset (zero-credit stream)')
bdy = torch.zeros(T, N, dtype=torch.bool)
for n in range(N):
    for t in (3, 11, 18):
        bdy[t, n] = True
dones = torch.zeros(T, N)
dones[T - 1, :] = True
bdy2 = shuffle_segment_boundaries(bdy, dones)


def seg_len_multiset(b):
    out = []
    for n in range(b.shape[1]):
        starts = [0] + [t for t in range(b.shape[0]) if b[t, n]]
        ends = starts[1:] + [b.shape[0]]
        out.extend(e - s for s, e in zip(starts, ends))
    return sorted(out)


check('C1 length multiset preserved', seg_len_multiset(bdy) == seg_len_multiset(bdy2))
check('C2 shuffled grid differs from the event grid', not torch.equal(bdy, bdy2))

print()
print('test_ces2: {} passed, {} failed'.format(PASS, FAIL))
sys.exit(0 if FAIL == 0 else 1)
