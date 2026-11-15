# [sgp] SGPPPO: PPO + training-side spectral gait prior (SPEC_SGP_PPO v1.0).
#
# Design (training-side only; deployment interface bit-identical to A0):
#   L = L_PPO + beta(t) * (L_conc + lambda_a * L_anti)
#
#   L_PPO is the parent PPO update, UNTOUCHED (clipped surrogate + value +
#   entropy, 5 epochs x 4 minibatches). The spectral term is applied as ONE
#   extra gradient step per update on the same optimizer (one batched
#   torch.fft.rfft per update, SPEC: incremental cost <= 10%):
#
#   1. Ring buffer (per env, within-episode only) records obs / raw command /
#      leg-12 dof_pos at every rollout step (pushed by gm_run_all_sgp.py).
#   2. At update time: windows of T_w = 75 steps for envs with a >=75-step
#      episode run, gated to mean ||v_cmd_xy|| > 0.3 m/s (SPEC gating).
#   3. f0 for each window: self-bootstrap blend (1-w)*g_phi(c) + w*EMA[bucket]
#      with w ramping over early training ("initial value g_phi, early EMA
#      converges to the spectral peak"); the EMA tracks the argmax of the
#      ACTUAL dof_pos spectrum, g_phi is regressed onto the same peaks.
#   4. Gradient path: the actor mean is re-evaluated on the window's stored
#      observations; the COMMANDED leg trajectory
#         q_cmd = clamp(default_q + action_scale * a, dof limits)
#      (exactly the env's PD target map) is detrended + Hann-windowed + rFFT'd
#      and yields L_conc / L_anti. This is the only differentiable path from
#      the spectral objective to the policy (the physics itself is not
#      differentiable); the stiff PD (60/1.5) makes q_cmd's spectrum a tight
#      proxy for the realised dof_pos spectrum (both are monitored).
#   5. GRADIENT INJECTION: the spectral gradient beta*grad(L_spec)/(E*M) is
#      added to the accumulated gradient of each of the E*M PPO minibatch
#      steps, so every update applies exactly grad(L_PPO + beta*L_spec)
#      (SPEC combined objective; beta keeps a monotone effect under Adam).
#      g_phi trains on its own tiny Adam (peak regression; 1 step/update).
#
# Arms: s1 = full (w_conc=1, w_anti=1); s2 = concentration only (1, 0);
#       s3 = antiphase only (0, 1).

import os
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.utils.sgp import (ANTI_PAIRS, DEFAULT_T_W, GATE_VEL, LAMBDA_ANTI,
                              ANNEAL_FRAC, BETA0, BETA_END, F0Bootstrap, F0Head,
                              antiphase_loss, beta_at, concentration_loss,
                              spectral_peak_f0, windowed_fft, rfft_freqs)


class SGPPPO(PPO):
    def __init__(self, *args,
                 sgp_arm='s1', sgp_beta0=BETA0, sgp_beta_end=BETA_END,
                 sgp_lambda_anti=LAMBDA_ANTI, sgp_anneal_frac=ANNEAL_FRAC,
                 sgp_max_windows=512, sgp_window=DEFAULT_T_W,
                 sgp_gate_vel=GATE_VEL, sgp_f0_ema_alpha=0.05,
                 sgp_f0_w_ramp=1000, sgp_w_f0=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert sgp_arm in ('s1', 's2', 's3')
        self.sgp_arm = sgp_arm
        self.sgp_w_conc = 1.0 if sgp_arm in ('s1', 's2') else 0.0
        self.sgp_w_anti = 1.0 if sgp_arm in ('s1', 's3') else 0.0
        self.sgp_beta0 = float(sgp_beta0)
        self.sgp_beta_end = float(sgp_beta_end)
        self.sgp_lambda_anti = float(sgp_lambda_anti)
        self.sgp_anneal_frac = float(sgp_anneal_frac)
        self.sgp_max_windows = int(sgp_max_windows)
        self.sgp_gate_vel = float(sgp_gate_vel)
        self.sgp_w_f0 = float(sgp_w_f0)

        # f0 head + bootstrap (g_phi is NOT part of actor_critic: it never
        # touches the act path, so the deployed policy is bit-identical to A0;
        # it has its own tiny optimizer for the peak-regression objective)
        self.sgp_gphi = F0Head().to(self.device)
        self.sgp_gphi_opt = optim.Adam(self.sgp_gphi.parameters(), lr=1e-3)
        self.sgp_bootstrap = F0Bootstrap(alpha=sgp_f0_ema_alpha, w_ramp=sgp_f0_w_ramp,
                                         device=self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        self._spec_grads = []

        # env binding (set by gm_run_all_sgp.py before learn; stays None in
        # unit tests / pure-PPO usage -> spectral step is a no-op)
        self.sgp_bound = False
        self.sgp_total_iters = 10000
        self.sgp_t_w = int(sgp_window)
        self.sgp_dt = 0.02
        self.sgp_action_scale = 0.25
        self.sgp_default_leg = None      # (12,)
        self.sgp_leg_limits = None       # (12, 2)
        self.sgp_leg_idx = list(range(12))

        # ring buffer
        self._ring_obs = None            # (N, T, obs)
        self._ring_leg = None            # (N, T, 12)
        self._ring_cmd = None            # (N, T, 3)
        self._run_len = None             # (N,)
        self._head = 0

        # monitoring (one row per update; dumped to CSV by the entry script)
        self.sgp_monitor = []
        self.sgp_last = {}

    # ------------------------------------------------------------ env bind

    def sgp_bind_env(self, env, total_iters):
        leg_names = list(env.dof_names)[:12]
        expect = ['left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint',
                  'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint']
        for nm in expect:
            assert nm in leg_names, 'leg dof order mismatch: {} not in first 12'.format(nm)
        # verify the mirror pairs by name
        pairs = []
        for jl, jr in ANTI_PAIRS:
            assert env.dof_names[jl].replace('left_', '') == env.dof_names[jr].replace('right_', ''), \
                'anti pair mismatch: {} vs {}'.format(env.dof_names[jl], env.dof_names[jr])
            pairs.append((jl, jr))
        self.sgp_pairs = pairs
        self.sgp_leg_idx = list(range(12))
        self.sgp_default_leg = env.default_dof_pos[0, :12].detach().clone().float()
        self.sgp_leg_limits = env.dof_pos_limits[:12].detach().clone().float()
        self.sgp_action_scale = float(env.cfg.control.action_scale)
        self.sgp_dt = float(env.dt)
        self.sgp_total_iters = int(total_iters)
        self.sgp_bound = True
        n = env.num_envs
        self._ring_obs = torch.zeros(n, self.sgp_t_w, env.num_obs, device=self.device)
        self._ring_leg = torch.zeros(n, self.sgp_t_w, 12, device=self.device)
        self._ring_cmd = torch.zeros(n, self.sgp_t_w, 3, device=self.device)
        self._run_len = torch.zeros(n, dtype=torch.long, device=self.device)
        self._head = 0
        print('[sgp] bound to env: leg dofs {}..{}, pairs {}, dt {}, T_w {}, '
              'g_phi params {}'.format(env.dof_names[0], env.dof_names[11], pairs,
                                       self.sgp_dt, self.sgp_t_w, self.sgp_gphi.n_params()))
        print('[sgp] arm {} (w_conc={}, w_anti={}), beta {} -> {} over first {:.0%}, '
              'lambda_a {}'.format(self.sgp_arm, self.sgp_w_conc, self.sgp_w_anti,
                                   self.sgp_beta0, self.sgp_beta_end,
                                   self.sgp_anneal_frac, self.sgp_lambda_anti))

    # ------------------------------------------------------------ ring buffer

    def sgp_push(self, obs, cmd, leg_pos, dones):
        """Record one policy step per env. Call BEFORE env.step() in the rollout
        loop: obs/cmd/leg_pos must be the tensors the policy just acted on.
        dones are from the PREVIOUS env.step: a done at step i means the sample
        pushed at step i was the last of its episode, so the reset must apply
        BEFORE counting the next (post-reset) sample."""
        if not self.sgp_bound:
            return
        if dones is not None and dones.any():
            self._run_len[dones.view(-1) > 0] = 0
        h = self._head
        self._ring_obs[:, h] = obs
        self._ring_cmd[:, h] = cmd[:, :3]
        self._ring_leg[:, h] = leg_pos[:, self.sgp_leg_idx]
        self._head = (h + 1) % self.sgp_t_w
        self._run_len += 1

    # ------------------------------------------------------------ spectral step

    def sgp_spectral_step(self, it):
        """One batched-FFT spectral gradient step. Returns the monitor dict."""
        out = {'it': it, 'beta': beta_at(it, self.sgp_total_iters, self.sgp_beta0,
                                         self.sgp_beta_end, self.sgp_anneal_frac),
               'n_valid': 0, 'n_gated': 0, 'coverage': 0.0, 'active': False,
               'l_conc_cmd': None, 'l_anti_cmd': None, 'l_conc_pos': None,
               'l_anti_pos': None, 'f0_peak_mean': None, 'f0_eff_mean': None,
               'gphi_f0_mean': None, 'gphi_loss': None, 'spectral_term': 0.0,
               'grad_spec_norm': None, 'grad_ppo_norm': None}
        if not self.sgp_bound:
            return out
        valid = self._run_len >= self.sgp_t_w
        out['n_valid'] = int(valid.sum())
        if not valid.any():
            self.sgp_last = out
            return out
        idx = valid.nonzero(as_tuple=False).squeeze(-1)
        # gating on the FULL valid set (coverage = gated / valid, SPEC gate)
        h = self._head
        cols = torch.tensor([(h - 1 - i) % self.sgp_t_w for i in range(self.sgp_t_w)],
                            device=self.device).flip(0)          # oldest -> newest
        cmd_all = self._ring_cmd[idx.unsqueeze(1), cols.unsqueeze(0)]      # (V, T, 3)
        speed_all = cmd_all[..., :2].norm(dim=-1).mean(dim=-1)
        gate = speed_all > self.sgp_gate_vel
        out['n_gated'] = int(gate.sum())
        out['coverage'] = float(gate.float().mean())
        if not gate.any():
            self.sgp_last = out
            return out
        # gate FIRST, then subsample to sgp_max_windows (always up to the cap
        # of gated windows, so gating coverage in the loss is not diluted)
        idx = idx[gate]
        if idx.shape[0] > self.sgp_max_windows:
            idx = idx[torch.randperm(idx.shape[0], device=self.device)[:self.sgp_max_windows]]
        obs_w = self._ring_obs[idx.unsqueeze(1), cols.unsqueeze(0)]   # (S, T, obs)
        leg_w = self._ring_leg[idx.unsqueeze(1), cols.unsqueeze(0)]   # (S, T, 12)
        cmd_w = self._ring_cmd[idx.unsqueeze(1), cols.unsqueeze(0)]   # (S, T, 3)
        speed_w = cmd_w[..., :2].norm(dim=-1).mean(dim=-1)       # mean ||v_xy||

        # --- f0 bootstrap on the ACTUAL dof_pos spectrum (no grad)
        with torch.no_grad():
            _, power_pos = windowed_fft(leg_w)
            freqs = rfft_freqs(self.sgp_t_w, self.sgp_dt, device=self.device)
            f0_peak = spectral_peak_f0(power_pos, freqs)         # (S,)
            self.sgp_bootstrap.update(speed_w, f0_peak)
        cmd_mean = cmd_w.mean(dim=1)                             # (S, 3)
        gphi_f0 = self.sgp_gphi(cmd_mean)                        # (S,) [grad on purpose]
        with torch.no_grad():
            f0_eff = self.sgp_bootstrap.blend(it, gphi_f0.detach(), speed_w)
        gphi_loss = ((gphi_f0 - f0_peak) ** 2).mean()

        # --- differentiable path: actor mean on stored obs -> commanded q
        S, T, O = obs_w.shape
        a_mean = self.actor_critic.actor(obs_w.reshape(S * T, O)).reshape(S, T, -1)
        q_cmd = self.sgp_default_leg + self.sgp_action_scale * a_mean[..., :12]
        q_cmd = torch.clamp(q_cmd, self.sgp_leg_limits[:, 0], self.sgp_leg_limits[:, 1])

        q_c, pw_c = windowed_fft(q_cmd)                          # ONE batched rFFT
        l_conc_cmd, _ = concentration_loss(q_c, pw_c, freqs, f0_eff)
        l_anti_cmd = antiphase_loss(q_c, freqs, f0_eff, self.sgp_pairs)
        l_conc_cmd, l_anti_cmd = l_conc_cmd.mean(), l_anti_cmd.mean()

        # diagnostics: same losses on the ACTUAL dof_pos windows (eval口径)
        with torch.no_grad():
            q_p, pw_p = windowed_fft(leg_w)
            l_conc_pos = concentration_loss(q_p, pw_p, freqs, f0_eff)[0].mean()
            l_anti_pos = antiphase_loss(q_p, freqs, f0_eff, self.sgp_pairs).mean()

        spec_loss = (out['beta'] * (self.sgp_w_conc * l_conc_cmd +
                                    self.sgp_w_anti * self.sgp_lambda_anti * l_anti_cmd))
        if torch.isfinite(spec_loss) and torch.isfinite(gphi_loss):
            # [sgp] GRADIENT INJECTION design: the spectral term contributes its
            # gradient to every one of the E*M PPO optimizer steps, scaled by
            # 1/(E*M) -> each update applies exactly grad(L_PPO + beta*L_spec)
            # (the SPEC combined objective), and beta keeps a monotone effect
            # under Adam (a separate Adam step would unit-normalise the spectral
            # gradient and make beta nearly meaningless between 1e-4 and 0.5).
            actor_params = [p for p in self.actor_critic.parameters()]
            grads = torch.autograd.grad(spec_loss, actor_params,
                                        allow_unused=True, retain_graph=False)
            n_steps = max(1, self.num_learning_epochs * self.num_mini_batches)
            self._spec_grads = [(p, g * (1.0 / n_steps)) for p, g in zip(actor_params, grads)
                                if g is not None]
            gnorm = float(torch.sqrt(sum((g * g).sum() for g in grads
                                         if g is not None)))
            # g_phi: separate tiny optimizer, regressed onto the data peaks
            # (never touches the policy's act path)
            self.sgp_gphi_opt.zero_grad()
            (self.sgp_w_f0 * gphi_loss).backward()
            self.sgp_gphi_opt.step()
            out.update(active=True, l_conc_cmd=float(l_conc_cmd),
                       l_anti_cmd=float(l_anti_cmd), l_conc_pos=float(l_conc_pos),
                       l_anti_pos=float(l_anti_pos), f0_peak_mean=float(f0_peak.mean()),
                       f0_eff_mean=float(f0_eff.mean()),
                       gphi_f0_mean=float(gphi_f0.detach().mean()),
                       gphi_loss=float(gphi_loss), grad_spec_norm=gnorm,
                       spectral_term=float(spec_loss))
        else:  # NaN guard: never poison the run
            self._spec_grads = []
            print('[sgp] WARNING: non-finite spectral loss at it {} -> injection skipped'.format(it))
        self.sgp_last = out
        return out

    # ------------------------------------------------------------ update

    def update(self):
        """PPO.update verbatim + spectral gradient injection after each
        minibatch backward (see the design note in sgp_spectral_step)."""
        it = self.current_it if hasattr(self, 'current_it') else 0
        self.sgp_spectral_step(it)
        self._spec_grads = getattr(self, '_spec_grads', [])

        mean_value_loss = 0
        mean_surrogate_loss = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        first_ppo_grad_norm = None
        n_mb = 0
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            if n_mb == 0:
                first_ppo_grad_norm = float(torch.sqrt(sum(
                    (p.grad * p.grad).sum() for p in self.actor_critic.parameters()
                    if p.grad is not None)))
            # [sgp] inject the spectral gradient contribution for this step
            for p, g in self._spec_grads:
                if p.grad is not None:
                    p.grad += g
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            n_mb += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        self.sgp_last.update(grad_ppo_norm=first_ppo_grad_norm)
        self.sgp_monitor.append({**self.sgp_last,
                                 'value_loss': float(mean_value_loss),
                                 'surrogate_loss': float(mean_surrogate_loss)})
        return mean_value_loss, mean_surrogate_loss

    # ------------------------------------------------------------ io

    def sgp_monitor_rows(self):
        keys = ['it', 'beta', 'n_valid', 'n_gated', 'coverage', 'active',
                'l_conc_cmd', 'l_anti_cmd', 'l_conc_pos', 'l_anti_pos',
                'f0_peak_mean', 'f0_eff_mean', 'gphi_f0_mean', 'gphi_loss',
                'spectral_term', 'grad_spec_norm', 'grad_ppo_norm',
                'value_loss', 'surrogate_loss']
        rows = [','.join(keys)]
        for m in self.sgp_monitor:
            rows.append(','.join('' if m.get(k) is None else '{:.6g}'.format(m[k])
                                 if isinstance(m.get(k), float) else str(m.get(k))
                                 for k in keys))
        return '\n'.join(rows) + '\n'

    def sgp_state(self):
        return {'gphi': self.sgp_gphi.state_dict(),
                'bootstrap': self.sgp_bootstrap.state_dict(),
                'beta0': self.sgp_beta0, 'beta_end': self.sgp_beta_end,
                'arm': self.sgp_arm}
