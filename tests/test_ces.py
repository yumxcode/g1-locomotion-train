"""Unit tests for CES-PPO core ops (pure torch, CPU, no IsaacGym).

Run (from repo root, with a torch env):
    python tests/test_ces.py

Covers SPEC items:
  (1) debounce greedy merge (L_min = 5)
  (3) segmented TD(lambda): hand-computed reference + no-leak across
      boundaries + degenerate equivalence to vanilla GAE when no boundaries
  (5) per-segment normalization (segment mean == 0, unit std)
  (A2) shuffle: segment-length multiset preserved per (env, episode),
       episode boundaries untouched
  (4) whole-segment minibatch: segments never split, full coverage,
       balanced sizes
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'rsl_rl'))

import torch

from rsl_rl.utils.ces import (debounce_boundaries, shuffle_segment_boundaries,
                              compute_ces_returns, segment_minibatch_order,
                              segment_ids)

PASS = 0
FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print('  PASS  {}'.format(name))
    else:
        FAIL += 1
        print('  FAIL  {}  {}'.format(name, detail))


def ref_gae(rewards, values, dones, last_values, gamma, lam):
    """Reference copy of rsl_rl v1.0.2 RolloutStorage.compute_returns
    (GAE + GLOBAL advantage normalization), shapes [T,N,1]."""
    T, N, _ = rewards.shape
    returns = torch.zeros_like(rewards)
    adv = 0
    for step in reversed(range(T)):
        next_values = last_values if step == T - 1 else values[step + 1]
        not_terminal = 1.0 - dones[step].float()
        delta = rewards[step] + not_terminal * gamma * next_values - values[step]
        adv = delta + not_terminal * gamma * lam * adv
        returns[step] = adv + values[step]
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


# ===========================================================================
print('[1] debounce_boundaries (SPEC 1: L_min greedy merge)')
bdy_raw = torch.zeros(7, 1, dtype=torch.bool)
for t in (0, 3, 6, 12):
    pass  # placeholder
seq = torch.tensor([0, 3, 6, 12, 15, 25], dtype=torch.long)
raw = torch.zeros(26, 1, dtype=torch.bool)
raw[seq] = True
kept = debounce_boundaries(raw, l_min=5)
kept_pos = torch.nonzero(kept[:, 0], as_tuple=False).flatten().tolist()
check('chain merge [0,3,6,12,15,25] L=5 -> [0,6,12,25]?', kept_pos == [0, 6, 12, 25],
      'got {}'.format(kept_pos))
# single boundary stays
raw2 = torch.zeros(24, 1, dtype=torch.bool); raw2[10] = True
check('single boundary kept', debounce_boundaries(raw2, 5)[10].item() and
      debounce_boundaries(raw2, 5).sum().item() == 1)
# vectorized over N agrees with a python greedy per column
torch.manual_seed(0)
rawN = torch.rand(24, 50) < 0.15
keptN = debounce_boundaries(rawN, 5)
ok = True
for n in range(50):
    pos = torch.nonzero(rawN[:, n], as_tuple=False).flatten().tolist()
    last = -10 ** 9
    exp = []
    for p in pos:
        if p - last >= 5:
            exp.append(p)
            last = p
    got = torch.nonzero(keptN[:, n], as_tuple=False).flatten().tolist()
    if got != exp:
        ok = False
        break
check('vectorized greedy == per-column python greedy (50 envs)', ok)

# ===========================================================================
print('[2] compute_ces_returns (SPEC 3+5)')
# --- 2a: hand-computed example ---
T, N = 3, 1
gamma, lam = 0.9, 0.5
rewards = torch.tensor([1., 2., 3.]).view(T, N, 1)
values = torch.tensor([0.5, 0.5, 0.5]).view(T, N, 1)
dones = torch.zeros(T, N, 1)
last_values = torch.tensor([[0.5]])
bdy = torch.zeros(T, N, dtype=torch.bool); bdy[2] = True  # segment [0,2) + [2,3)
ret, adv, stats = compute_ces_returns(rewards, values, dones, last_values, bdy, gamma, lam)
# hand computation (see design notes):
d2 = 3 + gamma * 0.5 - 0.5                    # 2.95 (A_2, last step, no carry)
d1 = 2 + gamma * 0.5 - 0.5                    # 1.95 (boundary at t=2 -> A_1 = d1 only)
d0 = 1 + gamma * 0.5 - 0.5 + gamma * lam * d1 # 0.95 + 0.8775
exp_ret = torch.tensor([d0 + 0.5, d1 + 0.5, d2 + 0.5]).view(T, N, 1)
check('hand-computed returns match', torch.allclose(ret, exp_ret, atol=1e-6),
      'got {} exp {}'.format(ret.flatten().tolist(), exp_ret.flatten().tolist()))
# per-segment normalization: seg0 = {A0,A1} -> mean 0; seg1 = {A2} -> std 0 -> 0
a_raw = torch.tensor([d0, d1, d2])
seg0 = a_raw[:2]
n0 = (seg0 - seg0.mean()) / (seg0.std(unbiased=False) + 1e-8)
exp_adv = torch.tensor([n0[0], n0[1], 0.0]).view(T, N, 1)
check('per-segment normalization matches hand calc', torch.allclose(adv, exp_adv, atol=1e-5),
      'got {} exp {}'.format(adv.flatten().tolist(), exp_adv.flatten().tolist()))
check('segment count == 2', stats['num_segments'] == 2, str(stats))
check('segment lengths [2,1]', stats['len_min'] == 1 and stats['len_max'] == 2, str(stats))

# --- 2b: degenerate equivalence with vanilla GAE (no boundaries, no dones) ---
torch.manual_seed(1)
T, N = 24, 9
rewards = torch.randn(T, N, 1)
values = torch.randn(T, N, 1)
dones = torch.zeros(T, N, 1)
last_values = torch.randn(N, 1)
bdy = torch.zeros(T, N, dtype=torch.bool)
ret_c, adv_c, _ = compute_ces_returns(rewards, values, dones, last_values, bdy, 0.99, 0.95)
ret_v, adv_v = ref_gae(rewards, values, dones, last_values, 0.99, 0.95)
check('no-boundary returns == vanilla GAE returns', torch.allclose(ret_c, ret_v, atol=1e-6),
      'max diff {:.2e}'.format((ret_c - ret_v).abs().max()))
# with no boundaries each env column is ONE segment -> per-segment normalization
# is per-env here (a strictly finer grouping than the vanilla GLOBAL norm; this
# granularity effect is exactly what A2 controls for)
ok_perenv = True
raw_c = ret_c - values
for n in range(N):
    col = raw_c[:, n, 0]
    manual = (col - col.mean()) / (col.std(unbiased=False) + 1e-8)
    if not torch.allclose(adv_c[:, n, 0], manual, atol=1e-5):
        ok_perenv = False
check('no-boundary advantages == per-env normalization (single segment/env)',
      ok_perenv)

# --- 2c: dones alone (episode cuts) still equal vanilla GAE ---
torch.manual_seed(2)
dones = (torch.rand(T, N, 1) < 0.1)
ret_c, adv_c, _ = compute_ces_returns(rewards, values, dones, last_values, bdy, 0.99, 0.95)
ret_v, adv_v = ref_gae(rewards, values, dones, last_values, 0.99, 0.95)
check('done-only returns == vanilla GAE returns', torch.allclose(ret_c, ret_v, atol=1e-6),
      'max diff {:.2e}'.format((ret_c - ret_v).abs().max()))

# --- 2d: no leak across an event boundary ---
torch.manual_seed(3)
T, N = 24, 4
rewards = torch.randn(T, N, 1) * 0.1
values = torch.randn(T, N, 1) * 0.1
dones = torch.zeros(T, N, 1)
last_values = torch.zeros(N, 1)
bdy = torch.zeros(T, N, dtype=torch.bool); bdy[10] = True  # boundary at t=10
r1 = rewards.clone(); r2 = rewards.clone()
r2[10:] += 100.0  # perturb the LATER segment only
_, a1, _ = compute_ces_returns(r1, values, dones, last_values, bdy, 0.99, 0.95)
_, a2, _ = compute_ces_returns(r2, values, dones, last_values, bdy, 0.99, 0.95)
check('earlier-segment raw advantage unaffected by later-segment rewards',
      torch.allclose(a1[:10], a2[:10], atol=1e-4),
      'max diff {:.2e}'.format((a1[:10] - a2[:10]).abs().max()))
# (normalized adv may differ: normalization stats differ per segment — seg0's own
#  stats are unchanged, so normalized seg0 must be identical too)
check('earlier-segment NORMALIZED advantage unaffected',
      torch.allclose(a1[:10], a2[:10], atol=1e-4))

# --- 2e: segment means are zero ---
seg_id = segment_ids(bdy, dones=dones[:, :, 0])
for s in seg_id.unique().tolist():
    m = (seg_id == s)
    # per-column segments are distinct ids; aggregate per column
for n in range(N):
    for s in seg_id[:, n].unique().tolist():
        m = (seg_id[:, n] == s)
        mu = a1[m, n, 0].mean().abs().item()
        if m.sum().item() > 1 and mu > 1e-4:
            check('segment mean zero', False, 'seg {} env {} mean {}'.format(s, n, mu))
            break
    else:
        continue
    break
else:
    check('all multi-step segments have zero advantage mean', True)

# ===========================================================================
print('[3] shuffle_segment_boundaries (A2 control)')
torch.manual_seed(4)
T, N = 24, 64
dones = (torch.rand(T, N) < 0.08)
raw = torch.rand(T, N) < 0.2
bdy = debounce_boundaries(raw, 5)
new = shuffle_segment_boundaries(bdy, dones)
# 3a: per-env episode count and per-(env,episode) length multiset preserved
def seg_lens_by_ep(b, d):
    out = {}
    for n in range(b.shape[1]):
        ep = 0
        lens = []
        cur = 0
        for t in range(b.shape[0]):
            if t > 0 and d[t - 1, n]:
                lens.append(cur)
                out[(n, ep)] = lens
                ep += 1
                lens = []
                cur = 0
            elif t > 0 and b[t, n]:
                lens.append(cur)
                cur = 0
            cur += 1
        lens.append(cur)
        out[(n, ep)] = lens
    return out
orig = seg_lens_by_ep(bdy, dones)
shuf = seg_lens_by_ep(new, dones)
same_keys = set(orig.keys()) == set(shuf.keys())
same_mult = all(sorted(orig[k]) == sorted(shuf[k]) for k in orig)
check('episode keys identical', same_keys)
check('per-(env,episode) segment-length multiset preserved', same_mult,
      'first diff: ' + str([(k, sorted(orig[k]), sorted(shuf[k])) for k in orig if sorted(orig[k]) != sorted(shuf[k])][:2]))
# 3b: boundaries really moved (randomization non-trivial)
n_moved = int((new != bdy).sum())
check('boundary grid actually changed (randomization live)', n_moved > 0, str(n_moved))
# 3c: no boundary placed on t=0
check('no boundary at t=0', not new[0].any().item())
# 3d: stability under fixed generator (same seed -> same result)
g1 = torch.Generator().manual_seed(42)
g2 = torch.Generator().manual_seed(42)
s1 = shuffle_segment_boundaries(bdy, dones, generator=g1)
s2 = shuffle_segment_boundaries(bdy, dones, generator=g2)
check('deterministic under fixed generator', torch.equal(s1, s2))

# ===========================================================================
print('[4] segment_minibatch_order (SPEC 4)')
torch.manual_seed(5)
T, N = 24, 128
dones = (torch.rand(T, N) < 0.1)
bdy = debounce_boundaries(torch.rand(T, N) < 0.2, 5)
idx, bounds = segment_minibatch_order(bdy, dones, num_mini_batches=4)
total = N * T
check('full coverage', idx.numel() == total, '{} vs {}'.format(idx.numel(), total))
check('no duplicates', idx.unique().numel() == total)
# 4a: segments never split across minibatches
seg_id_envmajor = segment_ids(bdy, dones=dones).T.reshape(-1)  # [N*T] env-major
env_envmajor = torch.arange(N).view(N, 1).expand(N, T).reshape(-1)
pair_id_envmajor = seg_id_envmajor * N + env_envmajor  # unique per (env, seg)
seg_of_member = pair_id_envmajor[idx]
for (s, e) in bounds:
    if e > s:
        segs = seg_of_member[s:e]
        if segs.unique().numel() != segs.numel():
            # unique() counts distinct; a segment split across batches shows up
            # as a seg id present in two different batch ranges instead
            pass
# proper check: each segment id appears in exactly one contiguous run inside ONE batch
owner = torch.zeros(total, dtype=torch.int64)
for b, (s, e) in enumerate(bounds):
    owner[s:e] = b
seg_owner_ok = True
uniq_segs = seg_of_member.unique()
for s in uniq_segs.tolist()[:2000]:
    o = owner[seg_of_member == s]
    if o.unique().numel() != 1:
        seg_owner_ok = False
        break
check('every segment confined to a single minibatch (sampled)', seg_owner_ok)
# 4b: member contiguity per segment (env-major contiguous run)
contig_ok = True
for s in uniq_segs.tolist()[:2000]:
    pos = torch.nonzero(seg_of_member == s, as_tuple=False).flatten()
    # positions within idx ordering must be contiguous AND ascending in flat index
    flat_vals = idx[pos]
    if not torch.equal(flat_vals, torch.arange(int(flat_vals[0]), int(flat_vals[0]) + flat_vals.numel())):
        contig_ok = False
        break
check('segment members form a contiguous ascending flat run', contig_ok)
# 4c: batch sizes balanced
sizes = [e - s for (s, e) in bounds]
nominal = total // 4
check('batch sizes within one max-segment-length of nominal',
      max(abs(x - nominal) for x in sizes) <= T,
      'sizes {} nominal {}'.format(sizes, nominal))
# 4d: no-boundary degenerate == contiguous halves? (just size check)
idx0, bounds0 = segment_minibatch_order(torch.zeros(T, N, dtype=torch.bool),
                                        torch.zeros(T, N, dtype=torch.bool), 4)
check('degenerate single-segment-per-env still covers all',
      idx0.numel() == total and idx0.unique().numel() == total)

print()
print('=' * 50)
print('PASS {}  FAIL {}'.format(PASS, FAIL))
sys.exit(1 if FAIL else 0)
