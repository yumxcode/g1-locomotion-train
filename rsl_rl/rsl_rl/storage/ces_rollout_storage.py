"""CES rollout storage: segmented-advantage GAE + whole-segment minibatches.

Extends the vendored rsl_rl v1.0.2 RolloutStorage (additively; the vanilla
path is untouched) with:
  - per-step CES records (debounced boundary indicator, settled duty credit)
  - compute_ces_returns(): duty re-spread over the owning segment, optional
    A2 boundary randomization, segmented TD(lambda) + per-segment
    normalization (SPEC 3/5, A2)
  - segment_mini_batch_generator(): whole-segment minibatch grouping with
    unchanged global batch size (SPEC 4)
"""

import torch

from .rollout_storage import RolloutStorage
from rsl_rl.utils.ces import (compute_ces_returns, segment_ids,
                              shuffle_segment_boundaries, segment_minibatch_order)


class CESRolloutStorage(RolloutStorage):

    def __init__(self, num_envs, num_transitions_per_env, obs_shape,
                 privileged_obs_shape, actions_shape, device='cpu'):
        super().__init__(num_envs, num_transitions_per_env, obs_shape,
                         privileged_obs_shape, actions_shape, device)
        T, N = num_transitions_per_env, num_envs
        self.ces_bdy = torch.zeros(T, N, dtype=torch.bool, device=device)
        self.ces_duty_credit = torch.zeros(T, N, device=device)
        self.ces_duty_kind = torch.zeros(T, N, dtype=torch.int64, device=device)
        self.last_seg_stats = {}
        self.last_bdy = None  # boundary grid used by the minibatch generator

    def add_ces(self, bdy, duty_credit, duty_kind):
        """Record the CES extras of the transition just added (call AFTER
        add_transitions incremented self.step)."""
        if self.step <= 0 or self.step > self.num_transitions_per_env:
            return
        i = self.step - 1
        self.ces_bdy[i] = bdy.to(torch.bool)
        self.ces_duty_credit[i] = duty_credit
        self.ces_duty_kind[i] = duty_kind

    # ------------------------------------------------------------------

    def _respread_duty(self):
        """Re-spread settled duty credits evenly over their owning segment.

        The env pays the segment duty at the settlement step (event boundary
        or done). Here we move that credit back onto every step of the
        segment that produced it, so segmented TD(lambda) attributes the
        segment-level signal to the whole segment. Credits whose segment
        lies entirely before this rollout (event at t=0) stay in place.
        """
        T, N = self.num_transitions_per_env, self.num_envs
        dones2d = self.dones.squeeze(-1).to(torch.bool)                # [T,N]
        seg_id = segment_ids(self.ces_bdy, dones=dones2d)              # [T,N]
        n_axis = torch.arange(N, device=self.device).view(1, N).expand(T, N)
        t_axis = torch.arange(T, device=self.device).view(T, 1).expand(T, N)
        gid = (n_axis * (T + 1) + seg_id)                              # [T,N]

        credit = self.ces_duty_credit                                  # [T,N]
        kind = self.ces_duty_kind
        has = credit != 0
        # target segment: done-settled -> the segment containing t;
        # event-settled -> the segment containing t-1 (skip t==0 events)
        prev_gid = torch.full_like(gid, -1)
        prev_gid[1:] = gid[:-1]
        tgt = torch.where(kind == 1, gid, prev_gid)
        valid = has & (tgt >= 0)

        # segment member counts
        M = N * (T + 1)
        flat_gid = gid.reshape(-1)
        cnt = torch.bincount(flat_gid[flat_gid >= 0], minlength=M)

        if valid.any():
            # remove the credit from the settlement step (in-place)
            self.rewards.squeeze(-1).sub_(credit * valid.to(credit.dtype))
            # add credit/len to every member of the target segment
            spread = torch.where(valid,
                                 credit / cnt[tgt.clamp(min=0)].clamp(min=1).to(credit.dtype),
                                 torch.zeros_like(credit))
            add = torch.zeros(M, device=self.device).index_add_(
                0, flat_gid, spread.reshape(-1))
            self.rewards.squeeze(-1).add_(add[gid])

    def compute_ces_returns(self, last_values, gamma, lam,
                            randomize_boundaries=False):
        """Duty re-spread -> (A2) boundary randomization -> segmented GAE +
        per-segment normalization. Stores returns/advantages and the boundary
        grid used (for the whole-segment minibatch generator)."""
        self._respread_duty()
        bdy = self.ces_bdy
        if randomize_boundaries:
            bdy = shuffle_segment_boundaries(bdy, self.dones.squeeze(-1))
        dones3 = self.dones.to(self.rewards.dtype)
        last_values = last_values.to(self.rewards.dtype).view(-1, 1)
        returns, advantages, stats = compute_ces_returns(
            self.rewards, self.values, dones3, last_values, bdy, gamma, lam)
        self.returns.copy_(returns)
        self.advantages.copy_(advantages)
        self.last_seg_stats = stats
        self.last_bdy = bdy

    # ------------------------------------------------------------------

    def segment_mini_batch_generator(self, num_mini_batches, num_epochs=5):
        """Same yield contract as the vanilla mini_batch_generator, but every
        minibatch is a union of WHOLE segments (global batch size unchanged).
        Feedforward policies only (the 23-DoF task is an MLP)."""
        T, N = self.num_transitions_per_env, self.num_envs
        bdy = self.last_bdy if self.last_bdy is not None else self.ces_bdy
        dones2d = self.dones.squeeze(-1).to(torch.bool)
        member_idx, bounds = segment_minibatch_order(bdy, dones2d, num_mini_batches)

        # env-major flattening (row r <-> (n = r // T, t = r % T))
        NT = N * T
        observations = self.observations.permute(1, 0, 2).reshape(NT, -1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.permute(1, 0, 2).reshape(NT, -1)
        else:
            critic_observations = observations
        actions = self.actions.permute(1, 0, 2).reshape(NT, -1)
        values = self.values.permute(1, 0, 2).reshape(NT, -1)
        returns = self.returns.permute(1, 0, 2).reshape(NT, -1)
        old_actions_log_prob = self.actions_log_prob.permute(1, 0, 2).reshape(NT, -1)
        advantages = self.advantages.permute(1, 0, 2).reshape(NT, -1)
        old_mu = self.mu.permute(1, 0, 2).reshape(NT, -1)
        old_sigma = self.sigma.permute(1, 0, 2).reshape(NT, -1)

        for _ in range(num_epochs):
            for (s, e) in bounds:
                idx = member_idx[s:e]
                yield (observations[idx], critic_observations[idx], actions[idx],
                       values[idx], advantages[idx], returns[idx],
                       old_actions_log_prob[idx], old_mu[idx], old_sigma[idx],
                       (None, None), None)
