"""G1 23-DoF env + CES (contact-event segmentation) training-side components.

Adds ONLY the frozen-SPEC CES pieces on top of the frozen v6 G1_23dofRobot
environment/reward/PD/observation (nothing in the v6 path is modified):

  (SPEC 2) foot-local reward `ces_loc`, weighted w_loc = 0.05 via the usual
           reward scales (multiplied by dt like every other scale):
             r_slip = -sum_i 1[c_i=1] * ||v_foot,xy_i||        (stance slip)
             r_imp  = -sum_i 1[switch_i] * ||F_i - F_{i-1}||   (impact)
             r_duty = -sum_i |duty_i - 0.62|  (per-segment duty factor,
                      duty_i = contact steps of foot i / segment length)
           r_slip / r_imp are per-step. r_duty describes a whole segment and
           is SETTLED at the step where the segment close is detected
           (event boundary -> next step; episode end -> the done step itself).
           The PPO side re-spreads the settled duty evenly back over the
           segment steps (credit assignment under segmented TD(lambda)).
  (SPEC 1) online greedy debounce of event boundaries with L_min = 5
           (a candidate closer than L_min steps to the last KEPT boundary is
           dropped). The debounced boundary indicator and duty settlement are
           exported per step through extras['ces'] (training side only):
             bdy        [N] bool  — this step IS the first step of a new segment
             duty_amount[N] float — |duty| penalty settled at this step (raw,
                                    i.e. before w_loc and dt scaling; the env
                                    reward term already contains it)
             duty_kind  [N] long  — 0 event-closed (prior segment), 1 done-
                                    closed (current segment)

Contact criterion c_t^i: net contact z-force > 1 N — identical to the v6
feet_air_time / feet_swing_height reward criterion (same sensor, same
threshold), training side only, NEVER enters the 78-dim observation.
"""

import torch

from legged_gym.envs.g1.g1_23dof_env import G1_23dofRobot

CES_L_MIN = 5          # debounce: min steps between kept boundaries
CES_DUTY_TARGET = 0.62  # target duty factor per foot


class G1_23dofCESRobot(G1_23dofRobot):

    def _init_buffers(self):
        super()._init_buffers()
        n = self.num_envs
        f = len(self.feet_indices)
        self.ces_prev_contact = torch.zeros(n, f, dtype=torch.bool, device=self.device)
        self.ces_prev_force = torch.zeros(n, f, 3, device=self.device)
        self.ces_last_bdy_t = torch.full((n,), -(10 ** 9), dtype=torch.long, device=self.device)
        self.ces_seg_len = torch.zeros(n, dtype=torch.long, device=self.device)
        self.ces_seg_contact_cnt = torch.zeros(n, f, dtype=torch.long, device=self.device)
        self.ces_seg_active = torch.zeros(n, dtype=torch.bool, device=self.device)
        # rolling diagnostics (means over steps since last print)
        self.ces_diag = {'slip': 0.0, 'imp': 0.0, 'duty': 0.0, 'bdy_rate': 0.0,
                         'switch_rate': 0.0, 'n': 0}

    def _ces_duty_penalty(self, cnt, ln):
        """duty penalty -sum_i |duty_i - 0.62| for closed segments.
        cnt [m, F] long contact counts, ln [m] long segment lengths."""
        duty = cnt.float() / ln.float().clamp(min=1).unsqueeze(1)
        return -(duty - CES_DUTY_TARGET).abs().sum(dim=1)

    def _reward_ces_loc(self):
        """Foot-local CES reward (SPEC 2) + online debounced segmentation
        (SPEC 1). Called once per policy step from compute_reward()."""
        c = self.contact_forces[:, self.feet_indices, 2] > 1.0          # [N,F]
        F3 = self.contact_forces[:, self.feet_indices, :3].clone()      # [N,F,3]
        v_xy = self.feet_vel[:, :, :2]                                  # [N,F,2] world

        # ---- r_slip: stance-phase sliding -------------------------------
        slip = -(c.float() * v_xy.norm(dim=-1)).sum(dim=1)              # [N]

        # ---- r_imp: impact at contact switches --------------------------
        dF = (F3 - self.ces_prev_force).norm(dim=-1)                    # [N,F]
        switch = c != self.ces_prev_contact                             # [N,F]
        imp = -(switch.float() * dF).sum(dim=1)                         # [N]

        # ---- event boundary + online debounce (SPEC 1) ------------------
        t_now = self.common_step_counter
        raw = switch.any(dim=1) & self.ces_seg_active
        debounced = raw & ((t_now - self.ces_last_bdy_t) >= CES_L_MIN)
        done = self.reset_buf.bool()                                    # [N]

        # ---- r_duty settlement at segment close -------------------------
        duty_amount = torch.zeros_like(slip)
        duty_kind = torch.full_like(slip, 0, dtype=torch.long)
        # (a) event-closed: the PREVIOUS segment (counts up to last step)
        evt = debounced & ~done
        if evt.any():
            duty_amount[evt] = self._ces_duty_penalty(
                self.ces_seg_contact_cnt[evt], self.ces_seg_len[evt])
            duty_kind[evt] = 0
        # (b) done-closed: the CURRENT segment (counts include this step)
        if done.any():
            cnt_now = self.ces_seg_contact_cnt[done] + c[done].long()
            len_now = self.ces_seg_len[done] + 1
            duty_amount[done] = self._ces_duty_penalty(cnt_now, len_now)
            duty_kind[done] = 1
        duty_out = duty_amount.clone()

        # ---- segment state update ----------------------------------------
        new_seg = (debounced & ~done) | ~self.ces_seg_active & ~done
        one = torch.ones_like(self.ces_seg_len)
        zero = torch.zeros_like(self.ces_seg_len)
        self.ces_seg_len = torch.where(done, zero,
                                       torch.where(new_seg, one,
                                                   self.ces_seg_len + 1))
        new_cnt = c.long()
        self.ces_seg_contact_cnt = torch.where(
            done.unsqueeze(1), torch.zeros_like(self.ces_seg_contact_cnt),
            torch.where(new_seg.unsqueeze(1), new_cnt,
                        self.ces_seg_contact_cnt + new_cnt))
        self.ces_seg_active = ~done
        self.ces_last_bdy_t = torch.where(new_seg & debounced,
                                          torch.full_like(self.ces_last_bdy_t, t_now),
                                          self.ces_last_bdy_t)
        self.ces_prev_contact = c.clone()
        self.ces_prev_force = F3

        # ---- export to the algorithm side (training only) ----------------
        # duty_credit = the exact amount already added into this step's env
        # reward (duty_amount * reward scale incl. dt); the PPO side re-spreads
        # it evenly over the owning segment steps.
        ces_scale = self.reward_scales.get('ces_loc', 0.0)
        self.extras['ces'] = {
            'bdy': debounced & ~done,       # [N] bool: this step starts a new segment
            'duty_amount': duty_amount,     # [N] raw penalty settled here
            'duty_credit': duty_amount * ces_scale,  # [N] as already paid in rew_buf
            'duty_kind': duty_kind,         # [N] 0 event / 1 done
        }

        # ---- rolling diagnostics -----------------------------------------
        d = self.ces_diag
        d['slip'] += float(slip.mean())
        d['imp'] += float(imp.mean())
        d['duty'] += float(duty_out.mean())
        d['bdy_rate'] += float((debounced & ~done).float().mean())
        d['switch_rate'] += float(switch.any(dim=1).float().mean())
        d['n'] += 1

        return slip + imp + duty_out

    def ces_diag_flush(self):
        """Return and reset the rolling diagnostic means."""
        d = self.ces_diag
        n = max(d.pop('n'), 1)
        out = {k: v / n for k, v in d.items()}
        d['n'] = 0
        return out
