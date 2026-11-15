# [sgp] Spectral Gait Prior core — SPEC_SGP_PPO v1.0 (exp_36a2c333, branch sgp-ppo)
#
# Pure-torch implementation, NO isaacgym import: used by
#   - rsl_rl/rsl_rl/algorithms/sgp_ppo.py  (training-side regularizer)
#   - gm_run_all_sgp.py stage-2 eval       (in-trainer spectral metrics)
#   - spectral_metrics.py (repo root)      (standalone eval CLI, reused by the
#                                           stage-3 MuJoCo sim2sim pipeline)
#
# SPEC mapping (all per SPEC_SGP_PPO v1.0):
#   - windows: T_w = 75 policy steps (1.5 s @ 50 Hz), linear detrend + Hann
#   - rFFT  : batched torch.fft.rfft along the time axis (ONE batched call)
#   - f0    : argmax_{f in [0.5,3] Hz} of sum_{j in legs} |Q^j(f)|^2
#             (self-bootstrap); initial value from the g_phi(c_hat) head
#             (~6.3k params), early-training EMA converges to the spectral peak
#   - L_conc: sum over the 12 leg joints of the out-of-harmonic-band energy
#             fraction; band H(f0) = union_{k=1..3} [k*f0*(1 +/- 0.15)]
#   - L_anti: mean over the 3 left/right mirror pairs (hipPitch / knee /
#             anklePitch) of (cos(dpsi(f0)) + 1)/2, target phase diff = pi
#   - gating: only windows with mean ||v_cmd_xy|| > 0.3 m/s are counted
#
# Frequency resolution note (honest): with T_w = 75 @ dt = 0.02 s the raw rFFT
# bin spacing is 0.667 Hz. The SPEC band H(f0) = k*f0*(1 +/- 0.15) is a
# CONTINUOUS-frequency definition; at 0.667 Hz resolution the Hann mainlobe
# (+/-1 bin) of an ideally periodic tone leaks ~33% of its energy outside the
# +/-15% band (ceiling ~0.67 for a perfect gait). The rFFT is therefore
# evaluated at 4x zero-padding (delta_f = 0.167 Hz): zero-padding is exact
# spectral interpolation (no new information), and it makes the discrete band
# mask match the continuous definition (an ideal tone now scores ~1.0).
# Identical in training and evaluation (same implementation).

import math
import torch
import torch.nn as nn

# ---------------------------------------------------------------- constants
DEFAULT_T_W = 75          # window length in policy steps (1.5 s)
DEFAULT_DT = 0.02         # policy step [s] (decimation 4 x sim dt 0.005)
F0_MIN, F0_MAX = 0.5, 3.0  # f0 search band [Hz]
HARMONIC_K = 3            # harmonics k = 1..3
HARMONIC_REL = 0.15       # band half-width, relative
GATE_VEL = 0.3            # ||v_cmd_xy|| gate threshold [m/s]
LAMBDA_ANTI = 0.5         # lambda_a in L = L_PPO + beta*(L_conc + lambda_a*L_anti)
BETA0, BETA_END = 0.5, 0.05
ANNEAL_FRAC = 0.30        # beta annealed over the first 30% of iterations
# left/right mirror pairs in the leg-12 block (dof 0..11):
#   0 left_hip_pitch  1 left_hip_roll  2 left_hip_yaw  3 left_knee
#   4 left_ankle_pitch 5 left_ankle_roll  6..11 right (same order)
ANTI_PAIRS = ((0, 6), (3, 9), (4, 10))  # hipPitch / knee / anklePitch
LEG_DIM = 12

# ---------------------------------------------------------------- windows


def hann(n, device=None, dtype=torch.float32):
    return torch.hann_window(n, device=device, dtype=dtype)


def detrend_linear(x):
    """Remove a linear trend along dim=-2. x: (B, T, J) -> zero-mean, zero-slope."""
    t = x.shape[-2]
    tc = torch.arange(t, device=x.device, dtype=x.dtype) - (t - 1) / 2.0
    tc = tc / tc.pow(2).sum().clamp(min=1e-12).sqrt()          # unit-norm t
    mean = x.mean(dim=-2, keepdim=True)
    slope = (x * tc.view(*([1] * (x.dim() - 2)), t, 1)).sum(dim=-2, keepdim=True)
    return x - mean - slope * tc.view(*([1] * (x.dim() - 2)), t, 1)


PAD_FACTOR = 4            # zero-padding factor: delta_f = 0.667/4 = 0.167 Hz


def windowed_fft(xw, pad=PAD_FACTOR):
    """(B, T, J) -> detrend + Hann + zero-padded rFFT -> (Q, power): (B, F', J).

    Q = complex rFFT coefficients; power = |Q|^2. n = pad*T samples -> bin
    spacing 1/(pad*T*dt) (exact spectral interpolation of the same windowed
    signal; see the resolution note above). Bin 0 (DC) is ~0 after the linear
    detrend; energy metrics below sum positive bins only."""
    t = xw.shape[-2]
    x = detrend_linear(xw)
    w = hann(t, device=x.device, dtype=x.dtype)
    w = w.view(*([1] * (x.dim() - 2)), -1, 1)
    x = x * w
    q = torch.fft.rfft(x, n=pad * t, dim=-2)
    return q, q.abs().pow(2)


def rfft_freqs(t_w, dt, device=None, pad=PAD_FACTOR):
    return torch.fft.rfftfreq(pad * t_w, d=dt, device=device)


# ---------------------------------------------------------------- f0


def spectral_peak_f0(power, freqs, f_lo=F0_MIN, f_hi=F0_MAX, eps=1e-12):
    """argmax over [f_lo, f_hi] of sum over joints of the window power.

    power: (B, F, J) -> f0: (B,) in Hz (bin-quantised)."""
    band = (freqs >= f_lo) & (freqs <= f_hi)
    pb = power[:, band, :]                                       # (B, Fb, J)
    tot = pb.sum(dim=-1).sum(dim=-1)                             # (B,) sum over f,J
    idx_local = pb.sum(dim=-1).argmax(dim=-1)                    # (B,)
    f0 = freqs[band][idx_local]
    # degenerate (all-zero window) -> return band centre so downstream math is safe
    f0 = torch.where(tot > eps, f0, torch.full_like(f0, 0.5 * (f_lo + f_hi)))
    return f0


def harmonic_band_mask(freqs, f0, k_max=HARMONIC_K, rel=HARMONIC_REL):
    """H(f0) = union_{k=1..k_max} [k*f0*(1-rel), k*f0*(1+rel)].

    freqs: (F,) f0: (B,) -> mask (B, F)."""
    f = freqs.view(1, -1)
    m = torch.zeros(f0.shape[0], freqs.shape[0], dtype=torch.bool, device=f0.device)
    for k in range(1, k_max + 1):
        c = (k * f0).view(-1, 1)
        m |= (f >= c * (1 - rel)) & (f <= c * (1 + rel))
    return m


def f0_bin_index(freqs, f0):
    """Nearest rFFT bin to each f0 (excluding DC). freqs: (F,), f0: (B,) -> (B,) long."""
    idx = (freqs.unsqueeze(0) - f0.unsqueeze(1)).abs().argmin(dim=-1)
    return idx.clamp(min=1)


# ---------------------------------------------------------------- losses


def concentration_loss(q, power, freqs, f0, eps=1e-12):
    """L_conc = sum_{j in legs} out-of-band energy fraction (SPEC, per window).

    Returns (l_conc: (B,), hr: (B,)) where hr = mean_j in-band fraction in
    [0, 1] (the reported 'harmonic ratio'; hr = 1 - l_conc / n_joints)."""
    mask = harmonic_band_mask(freqs, f0)                        # (B, F)
    m = mask.unsqueeze(-1)                                      # (B, F, 1)
    pos = freqs > 0                                             # exclude DC
    p = pos.view(1, -1, 1)
    e_in = (power * m * p).sum(dim=-2)                          # (B, J)
    e_tot = (power * p).sum(dim=-2) + eps                       # (B, J)
    frac_in = e_in / e_tot
    l_conc = (1.0 - frac_in).sum(dim=-1)                        # sum over 12 joints
    return l_conc, frac_in.mean(dim=-1)


def antiphase_loss(q, freqs, f0, pairs=ANTI_PAIRS):
    """L_anti = mean over mirror pairs of (cos(dpsi(f0)) + 1)/2 in [0, 1].

    0 <=> dpsi = pi (perfect antiphase), 1 <=> in phase."""
    b = f0_bin_index(freqs, f0)                                 # (B,)
    bb = b.view(-1, 1, 1).expand(-1, 1, q.shape[-1])
    ph = torch.angle(torch.gather(q, 1, bb).squeeze(1))         # (B, J) phase at f0 bin
    losses = []
    for jl, jr in pairs:
        dpsi = ph[:, jl] - ph[:, jr]
        losses.append((torch.cos(dpsi) + 1.0) / 2.0)
    return torch.stack(losses, dim=-1).mean(dim=-1)             # (B,)


# ---------------------------------------------------------------- g_phi head


class F0Head(nn.Module):
    """Command -> f0 map (SPEC: small head, ~6.3k params).

    Input: raw command [vx, vy, wz] (m/s, rad/s). Output: f0 in [0.5, 3] Hz.
    3 -> 76 -> 76 -> 1 = 6309 params. Initialised so the untrained head outputs
    ~1.4 Hz regardless of input (flat prior at the centre of the search band);
    it is then regressed online onto the data-driven spectral peaks."""

    def __init__(self, f0_init=1.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 76), nn.ELU(),
            nn.Linear(76, 76), nn.ELU(),
            nn.Linear(76, 1),
        )
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, nn.Linear):
                    m.weight.mul_(0.01)
            p = (f0_init - F0_MIN) / (F0_MAX - F0_MIN)
            self.net[-1].bias.fill_(math.log(p / (1 - p)))

    def forward(self, cmd):
        return F0_MIN + (F0_MAX - F0_MIN) * torch.sigmoid(self.net(cmd)).squeeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class F0Bootstrap:
    """f0 self-bootstrap state: per-speed-bucket EMA of the spectral peak +
    blend schedule from the g_phi prior to the EMA ('early training EMA
    converges to the spectral peak').

    Buckets on mean window command speed |v_xy|: [0.3,0.6) [0.6,0.9) [0.9,1.2)
    [1.2,inf) m/s. f0_eff(win) = (1-w)*g_phi(c) + w*EMA[bucket], w ramps 0->1
    linearly over the first w_ramp iterations."""

    EDGES = (0.3, 0.6, 0.9, 1.2)

    def __init__(self, alpha=0.05, w_ramp=1000, f0_init=1.4, device='cpu'):
        self.alpha = alpha
        self.w_ramp = w_ramp
        self.ema = torch.full((len(self.EDGES) + 1,), float(f0_init), device=device)

    @classmethod
    def bucket_of(cls, speed):
        """speed: (B,) -> bucket idx (B,) long."""
        idx = torch.zeros_like(speed, dtype=torch.long)
        for e in cls.EDGES:
            idx += (speed > e).to(torch.long)
        return idx.clamp(max=len(cls.EDGES))

    @torch.no_grad()
    def update(self, speed, f0_peak):
        """EMA-update buckets from this update's gated windows."""
        b = self.bucket_of(speed)
        for i in range(self.ema.shape[0]):
            sel = b == i
            if sel.any():
                m = f0_peak[sel].mean()
                self.ema[i] = self.alpha * m + (1 - self.alpha) * self.ema[i]

    def blend(self, it, g_phi_f0, speed):
        """Effective f0 per window: prior g_phi blended toward bucket EMA."""
        w = min(1.0, max(0.0, it / float(self.w_ramp))) if self.w_ramp > 0 else 1.0
        b = self.bucket_of(speed)
        return ((1.0 - w) * g_phi_f0 + w * self.ema[b]).clamp(F0_MIN, F0_MAX)

    def state_dict(self):
        return {'ema': self.ema.cpu(), 'alpha': self.alpha, 'w_ramp': self.w_ramp}


# ---------------------------------------------------------------- beta schedule


def beta_at(it, total_iters, beta0=BETA0, beta_end=BETA_END, anneal_frac=ANNEAL_FRAC):
    """Linear beta0 -> beta_end over the first anneal_frac of total iterations."""
    span = max(1.0, anneal_frac * total_iters)
    u = min(1.0, it / span)
    return beta0 + (beta_end - beta0) * u


# ---------------------------------------------------------------- eval metrics
# Same convention as the training losses, computed on RECORDED joint-position
# trajectories (dof_pos, not commanded) with the window self-bootstrap f0.
# Used by stage-2 eval and the stage-3 MuJoCo pipeline (identical call).


@torch.no_grad()
def gait_spectral_metrics(leg_pos, cmd_xy, dt=DEFAULT_DT, t_w=DEFAULT_T_W,
                          stride=25, gate_vel=GATE_VEL, pairs=ANTI_PAIRS):
    """Harmonic-ratio / antiphase metrics of a recorded leg-12 trajectory batch.

    leg_pos: (N, T, 12) joint positions [rad]; cmd_xy: (N, T, 2) or (N, 2)
    (constant per episode) velocity commands [m/s]. Windows of t_w steps with
    `stride`, within-episode only if episode boundaries are marked with NaN rows
    (a NaN row at t ends the running window). Gate: mean window ||v_cmd_xy|| >
    gate_vel (SPEC). f0 per window from the window's own spectral peak
    (self-bootstrap; same argmax as training).

    Returns dict with means and per-window series:
      harmonic_ratio in [0,1] (1 = energy concentrated in H(f0))
      antiphase_deg  in [0,1] (1 = perfect pi phase difference L/R)
      f0_hz mean, gate coverage (gated windows / valid windows), n_valid.
    """
    n, T, J = leg_pos.shape
    assert J == LEG_DIM, 'expected 12 leg joints'
    freqs = rfft_freqs(t_w, dt, device=leg_pos.device)
    if cmd_xy.dim() == 2:
        cmd_xy = cmd_xy.unsqueeze(1).expand(n, T, 2)
    # episode boundary = any NaN in the row
    boundary = torch.isnan(leg_pos).any(dim=-1) | torch.isnan(cmd_xy).any(dim=-1)

    hr_w, ap_w, f0_w = [], [], []
    n_windows = 0
    for s in range(0, max(0, T - t_w) + 1, stride):
        sl = slice(s, s + t_w)
        seg = leg_pos[:, sl, :]
        brk = boundary[:, sl].any(dim=-1)                        # (N,) window crosses boundary
        if (~brk).sum() == 0:
            continue
        seg = seg[~brk]
        cmd = cmd_xy[:, sl, :][~brk]
        speed = cmd.norm(dim=-1).mean(dim=-1)                    # (Nw,) mean ||v_xy||
        gate = speed > gate_vel
        n_windows += int(seg.shape[0])
        if gate.sum() == 0:
            continue
        seg = seg[gate]
        q, power = windowed_fft(seg)
        f0 = spectral_peak_f0(power, freqs)
        _, hr = concentration_loss(q, power, freqs, f0)
        ap = 1.0 - antiphase_loss(q, freqs, f0, pairs)
        hr_w.append(hr)
        ap_w.append(ap)
        f0_w.append(f0)
    if not hr_w:
        return {'harmonic_ratio': None, 'antiphase_deg': None, 'f0_hz': None,
                'gate_coverage': 0.0, 'n_valid_windows': 0, 'n_gated_windows': 0,
                'n_traj': int(n)}
    hr = torch.cat(hr_w)
    ap = torch.cat(ap_w)
    f0 = torch.cat(f0_w)
    return {
        'harmonic_ratio': float(hr.mean()),
        'antiphase_deg': float(ap.mean()),
        'f0_hz': float(f0.mean()),
        'gate_coverage': float(hr.shape[0] / max(1, n_windows)),
        'n_valid_windows': int(n_windows),
        'n_gated_windows': int(hr.shape[0]),
        'n_traj': int(n),
    }
