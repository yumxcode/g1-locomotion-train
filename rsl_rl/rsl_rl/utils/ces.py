"""CES-PPO core tensor ops: contact-event segmentation utilities.

Pure-torch functions, no IsaacGym / legged_gym dependency (unit-testable on CPU).

Frozen SPEC mapping (see TRAINING_REPORT_A1A2.md):
  (1) Event segmentation: segment boundary at t iff any foot contact flag
      changed vs t-1 (c_t^i from the sim contact sensor, z-force > 1 N, same
      criterion as the v6 feet_air_time reward), debounced with L_min = 5,
      greedy: a candidate boundary closer than L_min steps to the last KEPT
      boundary is dropped ("merged").
  (3) Segmented advantage: within-segment TD(lambda) with hard bootstrap at
      segment boundaries:
          A_t = sum_{s=t}^{t_{k+1}-1} (gamma*lam)^(s-t) * delta_s
      i.e. the GAE recursion does not carry the running advantage across a
      boundary step (delta_s itself still bootstraps V(s_{s+1})).
  (4) Whole-segment minibatch grouping (global batch size unchanged).
  (5) Per-segment advantage normalization.
  A2 control: boundaries randomly re-placed with the SAME per-env, per-episode
      segment-length multiset (random permutation inside episodes); reward
      stream unchanged.

Boundary convention used everywhere below:
  bdy[t, n] = 1  <=>  t is the FIRST step of a new event segment, t > 0.
  t = 0 is always an implicit segment start (never stored).
  Episode resets (dones) are NOT stored in bdy; every consumer additionally
  treats the step after a done as a segment start, so no segment crosses an
  episode boundary. Segments are half-open [t_k, t_{k+1}).
"""

import torch


def segment_ids(bdy, dones=None):
    """Per-(t, n) integer segment ids from a boundary grid.

    Segment starts: t == 0, bdy[t, n] == 1, or (dones given) the step after a
    done. Ids are 0-based per env column. Returns int64 [T, N].
    """
    T = bdy.shape[0]
    seg_start = torch.zeros_like(bdy, dtype=torch.bool)
    seg_start[0] = True
    if T > 1:
        seg_start[1:] = bdy[1:].to(torch.bool)
        if dones is not None:
            if dones.dim() == 3:
                dones = dones.squeeze(-1)
            seg_start[1:] |= dones[:-1].to(torch.bool)
    return torch.cumsum(seg_start.to(torch.int64), dim=0) - 1


def debounce_boundaries(raw_bdy, l_min):
    """Greedy temporal debounce of a raw event-boundary grid.

    raw_bdy: [T, N] bool (any-foot contact change at t vs t-1). A candidate
    at t is KEPT iff its distance to the last KEPT boundary in the same
    column is >= l_min. Sequential over T (T is small, = 24 rollout steps),
    vectorized over N. Returns kept [T, N] bool.
    """
    T, N = raw_bdy.shape
    device = raw_bdy.device
    last_kept = torch.full((N,), -(10 ** 9), dtype=torch.int64, device=device)
    t_now = torch.arange(T, dtype=torch.int64, device=device)
    kept = torch.zeros_like(raw_bdy, dtype=torch.bool)
    for t in range(T):
        cand = raw_bdy[t] & ((t_now[t] - last_kept) >= l_min)
        kept[t] = cand
        last_kept = torch.where(cand, torch.full_like(last_kept, t), last_kept)
    return kept


def shuffle_segment_boundaries(bdy, dones, generator=None):
    """A2 control transform: random boundary placement, same length multiset.

    Per env and per episode (interval between done boundaries): the multiset
    of event-segment lengths is preserved, but their ORDER is uniformly
    permuted, and the permuted segments are laid out from the episode start.
    Episode boundaries stay exactly where physics put them. This destroys the
    alignment between segment boundaries and contact events while keeping the
    segmentation granularity identical (same lengths, same count).

    bdy:   [T, N] bool (debounced event boundaries)
    dones: [T, N] bool (episode termination flags)
    Returns new [T, N] bool event-boundary grid.
    """
    T, N = bdy.shape
    device = bdy.device
    dones = dones.to(torch.bool)

    # ---- episode structure -------------------------------------------------
    ep_start = torch.zeros(T, N, dtype=torch.bool, device=device)
    ep_start[0] = True
    if T > 1:
        ep_start[1:] = dones[:-1]
    ep_id = torch.cumsum(ep_start.to(torch.int64), dim=0) - 1          # [T,N]
    n_ep = int(ep_id.max().item()) + 1

    # ---- per-segment dense tables [N, K] ------------------------------------
    seg_start = bdy.to(torch.bool) | ep_start                           # [T,N]
    seg_id = torch.cumsum(seg_start.to(torch.int64), dim=0) - 1         # [T,N]
    K = int(seg_id.max().item()) + 1
    n_grid = torch.arange(N, device=device).view(1, N).expand(T, N)
    t_grid = torch.arange(T, device=device).view(T, 1).expand(T, N)

    key = (n_grid * K + seg_id).reshape(-1)
    seg_len = torch.bincount(key, minlength=N * K).reshape(N, K).to(torch.int64)
    valid = seg_len > 0

    seg_ep = torch.zeros(N, K, dtype=torch.int64, device=device)
    seg_ep[n_grid[seg_start], seg_id[seg_start]] = ep_id[seg_start]
    ep_start_t = torch.zeros(N, n_ep, dtype=torch.int64, device=device)
    ep_start_t[n_grid[ep_start], ep_id[ep_start]] = t_grid[ep_start]

    # ---- random permutation inside each (env, episode) ----------------------
    r = torch.rand(N, K, dtype=torch.float64, device=device, generator=generator)
    sort_key = torch.where(valid,
                           seg_ep.to(torch.float64) * 4.0 + r,
                           torch.full((N, K), float('inf'), device=device))
    order = torch.argsort(sort_key, dim=1)                              # [N,K]
    new_len = torch.gather(seg_len, 1, order)
    new_ep = torch.gather(seg_ep, 1, order)
    new_valid = torch.gather(valid, 1, order)

    # ---- re-lay segments from each episode's start --------------------------
    # row-cumulative exclusive sum, then subtract the row-cum value at the
    # first segment of each episode -> INTRA-episode offset of each segment
    row_cum_excl = torch.cumsum(new_len, dim=1) - new_len             # [N,K]
    first_mask = new_valid.clone()
    if K > 1:
        first_mask[:, 1:] = new_valid[:, 1:] & (new_ep[:, 1:] != new_ep[:, :-1])
    n_idx = torch.arange(N, device=device).view(N, 1).expand(N, K)
    ep_start_cum = torch.zeros(N, n_ep, dtype=torch.int64, device=device)
    ep_start_cum[n_idx[first_mask], new_ep[first_mask]] = row_cum_excl[first_mask]
    excl = row_cum_excl - ep_start_cum.gather(1, new_ep.clamp(min=0))  # intra-ep offset
    base = torch.gather(ep_start_t, 1, new_ep.clamp(min=0))
    new_head = base + excl                                              # [N,K] abs t of head

    new_bdy = torch.zeros(T, N, dtype=torch.bool, device=device)
    sel = new_valid & (excl > 0)      # first segment of an episode keeps no bdy flag
    pos = torch.nonzero(sel, as_tuple=False)                            # [S,2] (n, k)
    if pos.numel():
        new_bdy[new_head[pos[:, 0], pos[:, 1]], pos[:, 0]] = True
    return new_bdy


def compute_ces_returns(rewards, values, dones, last_values, bdy, gamma, lam):
    """Segmented TD(lambda) advantages with per-segment normalization.

    rewards/values/dones: [T, N, 1]; last_values: [N, 1]; bdy: [T, N] bool
    (event boundaries; dones cut segments as well).

    Recursion (reverse over T): with cut[t] = bdy[t+1] (t+1 starts a segment
    -> t is the last step of its segment):
        A_t = delta_t + (1-done_t) * gamma * lam * (1 - cut_t) * A_{t+1}
    i.e. the running advantage never crosses a boundary step — the boundary
    value enters only through delta (hard bootstrap V(s_{t+1})).

    Returns (returns, advantages, seg_stats):
        returns [T,N,1]     = A_raw + V (value targets, unnormalized)
        advantages [T,N,1]  per-segment standardized
        seg_stats           length statistics of the segments used
    """
    T, N, _ = rewards.shape
    device = rewards.device
    bdy = bdy.to(torch.bool)
    dones_f = dones.to(rewards.dtype)

    returns = torch.zeros_like(rewards)
    advantage = torch.zeros(N, 1, device=device)
    for t in reversed(range(T)):
        next_values = last_values if t == T - 1 else values[t + 1]
        not_terminal = 1.0 - dones_f[t]
        delta = rewards[t] + not_terminal * gamma * next_values - values[t]
        if t < T - 1:
            cut = bdy[t + 1].to(rewards.dtype).unsqueeze(1)
        else:
            cut = torch.zeros(N, 1, device=device)
        advantage = delta + not_terminal * gamma * lam * (1.0 - cut) * advantage
        returns[t] = advantage + values[t]

    advantages_raw = returns - values

    # ---- per-segment normalization -----------------------------------------
    seg_id = segment_ids(bdy, dones=dones)                              # [T,N]
    gid = seg_id + torch.arange(N, device=device).view(1, N) * (T + 1)
    M = N * (T + 1)
    flat_gid = gid.reshape(-1)
    flat_adv = advantages_raw.reshape(-1)
    ones = torch.ones_like(flat_adv)
    cnt = torch.zeros(M, device=device).index_add_(0, flat_gid, ones)
    mean = torch.zeros(M, device=device).index_add_(0, flat_gid, flat_adv) / cnt.clamp(min=1)
    msq = torch.zeros(M, device=device).index_add_(0, flat_gid, flat_adv * flat_adv) / cnt.clamp(min=1)
    std = (msq - mean * mean).clamp(min=0).sqrt()
    flat_norm = (flat_adv - mean[flat_gid]) / (std[flat_gid] + 1e-8)
    advantages = flat_norm.view(T, N, 1)

    lens = cnt[cnt > 0].to(torch.float64)
    seg_stats = {
        'num_segments': int((cnt > 0).sum().item()),
        'len_mean': float(lens.mean().item()) if lens.numel() else 0.0,
        'len_median': float(lens.median().item()) if lens.numel() else 0.0,
        'len_min': int(lens.min().item()) if lens.numel() else 0,
        'len_max': int(lens.max().item()) if lens.numel() else 0,
    }
    return returns, advantages, seg_stats


def segment_minibatch_order(bdy, dones, num_mini_batches, generator=None):
    """Whole-segment minibatch grouping with unchanged global batch size.

    All segments (event + episode cuts) are shuffled once and greedily packed
    into `num_mini_batches` groups by cumulative step count (a segment is
    never split; batch sizes stay within one segment length of
    (N*T)//num_mini_batches). Mirrors the vanilla generator's single
    randperm reused across epochs.

    Returns (member_index, bounds):
        member_index [N*T] long — flat indices into env-major [N, T] arrays
            (i.e. original (t, n) = (idx % T, idx // N... see note))
        bounds list of (start, end) into member_index, one per minibatch.

    NOTE on flattening: env-major means flat = n * T + t. Callers must
    flatten their tensors with .T.reshape(-1) or .permute(1,0).reshape(-1).
    """
    T, N = bdy.shape
    device = bdy.device
    seg_id = segment_ids(bdy, dones=dones)                              # [T,N]
    n_grid = torch.arange(N, device=device).view(1, N).expand(T, N)

    seg_key = (n_grid * T + seg_id)                                     # head pos keyable
    # head positions in env-major flat space
    seg_start = torch.zeros(T, N, dtype=torch.bool, device=device)
    seg_start[0] = True
    if T > 1:
        seg_start[1:] = bdy[1:].to(torch.bool) | dones[:-1].to(torch.bool)
    flat_start = seg_start.T.reshape(-1)                                # [N*T] env-major
    head_pos = torch.nonzero(flat_start, as_tuple=False).squeeze(1)     # [m]
    flat_seg = seg_id.T.reshape(-1)                                     # [N*T]
    flat_n = n_grid.T.reshape(-1)
    seg_key_head = flat_n[head_pos] * T + flat_seg[head_pos]
    m = head_pos.numel()

    # segment lengths via counting members by (env, seg)
    seg_slot = flat_n * T + flat_seg                                    # [N*T] unique per (env,seg); t collision ok — need unique slot!
    # NOTE: (n*T + seg_id) is unique per segment because seg_id < T.
    seg_len_all = torch.bincount(seg_slot, minlength=N * T)             # dense; only slots with members nonzero
    seg_len_head = seg_len_all[seg_key_head]                            # [m]

    perm = torch.randperm(m, device=device, generator=generator)
    len_sh = seg_len_head[perm]
    cum_excl = torch.cumsum(len_sh, 0) - len_sh
    batch_size = (N * T) // num_mini_batches
    batch_seg = torch.clamp(cum_excl // batch_size, max=num_mini_batches - 1)  # [m]

    # expand segments to member indices (vectorized)
    seg_of_member = torch.repeat_interleave(torch.arange(m, device=device), len_sh)
    head_sh = head_pos[perm]
    cum_excl_sh = cum_excl
    member_idx = (head_sh[seg_of_member]
                  + torch.arange(int(len_sh.sum().item()), device=device)
                  - torch.repeat_interleave(cum_excl_sh, len_sh))
    member_batch = torch.repeat_interleave(batch_seg, len_sh)

    counts = torch.bincount(member_batch, minlength=num_mini_batches)
    starts = torch.cat([torch.zeros(1, dtype=torch.int64, device=device),
                        torch.cumsum(counts, 0)])
    bounds = [(int(starts[b].item()), int(starts[b + 1].item()))
              for b in range(num_mini_batches)]
    return member_idx, bounds
