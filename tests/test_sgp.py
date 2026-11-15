"""Unit tests for the SGP spectral core (pure torch, CPU; no isaacgym).

Run:  python -m pytest tests/test_sgp.py -q   (or python tests/test_sgp.py)
"""
import os
import sys
import math

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'rsl_rl'))

from rsl_rl.utils.sgp import (F0Bootstrap, F0Head, antiphase_loss, beta_at,
                              concentration_loss, detrend_linear, f0_bin_index,
                              gait_spectral_metrics, harmonic_band_mask,
                              rfft_freqs, spectral_peak_f0, windowed_fft)

DT = 0.02
T_W = 75


def make_walk(f0_hz, amp=0.15, antiphase=True, jitter=0.0, seed=0, n=8):
    """Synthetic leg-12 trajectories: harmonics 1..3 of f0 on all joints;
    left/right block phase offset pi (antiphase) or 0 (in phase)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(T_W).float() * DT
    x = torch.zeros(n, T_W, 12)
    for k in (1, 2, 3):
        ph = 2 * math.pi * k * f0_hz * t
        for j in range(12):
            base = math.pi if (antiphase and j >= 6) else 0.0
            x[:, :, j] += (amp / k) * torch.sin(ph + base + 0.3 * (j % 6))
    if jitter:
        x += jitter * torch.randn(n, T_W, 12, generator=g)
    return x + 0.5 + 0.01 * t.view(1, -1, 1)  # add trend + offset: detrend must remove them


def cmd_batch(speed, n):
    v = torch.full((n,), float(speed))
    return torch.stack([v, torch.zeros_like(v)], dim=-1)


def test_detrend_removes_trend_and_offset():
    t = torch.arange(75).float() * 0.02
    x = (0.3 + 0.05 * t).unsqueeze(0).unsqueeze(-1).expand(2, 75, 3).clone()
    y = detrend_linear(x)
    assert y.abs().max() < 1e-5


def test_f0_peak_found_on_pure_tone():
    x = make_walk(2.0, antiphase=True)             # 2.0 Hz = bin 3 exactly
    _, power = windowed_fft(x)
    freqs = rfft_freqs(T_W, DT)
    f0 = spectral_peak_f0(power, freqs)
    assert (f0 - 2.0).abs().max() < 0.17           # within one padded bin


def test_l_conc_low_for_harmonic_walk_high_for_noise():
    freqs = rfft_freqs(T_W, DT)
    for name, x, expect_low in (('walk', make_walk(1.33), True),
                                ('noise', make_walk(1.33, jitter=5.0, amp=0.0), False)):
        q, power = windowed_fft(x)
        f0 = spectral_peak_f0(power, freqs)
        lc, hr = concentration_loss(q, power, freqs, f0)
        # resolution-bounded ceiling: an IDEAL tone at f0~1.3-2 Hz keeps only
        # ~0.4-0.5 of its Hann-windowed energy inside the +/-15% band (T_w=75);
        # broadband noise keeps ~0.02. Dynamic range ~20x is what discriminates.
        if expect_low:
            assert lc.mean() < 8.0, (name, float(lc.mean()))
            assert hr.mean() > 0.3, (name, float(hr.mean()))
        else:
            # self-bootstrap selects the band around the noise's own max bin,
            # so the floor is ~0.15 (selection effect); still well below walk
            assert hr.mean() < 0.25, (name, float(hr.mean()))


def test_antiphase_vs_inphase():
    freqs = rfft_freqs(T_W, DT)
    xa = make_walk(2.0, antiphase=True)
    xp = make_walk(2.0, antiphase=False)
    qa, pa = windowed_fft(xa)
    qp, pp = windowed_fft(xp)
    f0a = spectral_peak_f0(pa, freqs)
    f0p = spectral_peak_f0(pp, freqs)
    la = antiphase_loss(qa, freqs, f0a)
    lp = antiphase_loss(qp, freqs, f0p)
    assert la.mean() < 0.05, float(la.mean())
    assert lp.mean() > 0.95, float(lp.mean())


def test_harmonic_band_mask_shape():
    freqs = rfft_freqs(T_W, DT)                      # padded: 0.167 Hz steps
    f0 = torch.tensor([1.0, 2.0])
    m = harmonic_band_mask(freqs, f0)
    assert m.shape == (2, len(freqs))
    # f0=1.0: k=1 band [0.85,1.15] -> padded bins 1.0 only; k=2 [1.7,2.3] ->
    # bins 1.833,2.0,2.167; k=3 [2.55,3.45] -> 2.667..3.333
    def at(fr):
        return (freqs - fr).abs().argmin().item()
    assert m[0, at(1.0)] and m[0, at(1.833)] and m[0, at(2.0)] and m[0, at(2.167)] and m[0, at(3.0)]
    assert not m[0, at(0.667)] and not m[0, at(1.333)] and not m[0, at(2.5)]
    # f0=2.0: k=1 [1.7,2.3]; Hann mainlobe (+/-1 padded bin) stays in band
    assert m[1, at(1.833)] and m[1, at(2.0)] and m[1, at(2.167)]
    assert not m[1, at(1.667)] and not m[1, at(2.333)]


def test_f0_head_params_and_prior():
    h = F0Head()
    n = h.n_params()
    assert 5500 < n < 7000, n      # spec ~6.3k
    out = h(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.5, -1.0]]))
    assert out.shape == (2,)
    assert (out - 1.4).abs().max() < 0.05   # flat prior at init
    assert ((out >= 0.5) & (out <= 3.0)).all()


def test_f0_bootstrap_ema_and_blend():
    bs = F0Bootstrap(alpha=1.0, w_ramp=10, device='cpu')   # alpha=1: instant
    speed = torch.tensor([0.4, 0.7, 1.0, 1.5])
    f0 = torch.tensor([1.0, 2.0, 2.0, 2.67])
    bs.update(speed, f0)
    g = torch.tensor([1.4, 1.4, 1.4, 1.4])
    f = bs.blend(0, g, speed)          # w=0 -> pure g_phi
    assert (f - 1.4).abs().max() < 1e-6
    f = bs.blend(10, g, speed)         # w=1 -> pure EMA (per bucket)
    for i, s in enumerate((0.4, 0.7, 1.0, 1.5)):
        assert abs(f[i].item() - f0[i].item()) < 1e-6, (s, f[i], f0[i])


def test_beta_schedule():
    assert abs(beta_at(0, 10000) - 0.5) < 1e-9
    assert abs(beta_at(3000, 10000) - 0.05) < 1e-9
    assert abs(beta_at(10000, 10000) - 0.05) < 1e-9
    mid = beta_at(1500, 10000)
    assert abs(mid - 0.275) < 1e-9


def test_eval_metrics_on_synthetic_walk():
    n, T = 16, 500
    t = torch.arange(T).float() * DT
    leg = torch.zeros(n, T, 12)
    for j in range(12):
        base = math.pi if j >= 6 else 0.0
        leg[:, :, j] = 0.1 * torch.sin(2 * math.pi * 2.0 * t + base + 0.2 * (j % 6)) \
            + 0.03 * torch.sin(2 * math.pi * 4.0 * t + base)
    cmd = torch.zeros(n, 2)
    cmd[:, 0] = 0.8
    m = gait_spectral_metrics(leg, cmd, dt=DT, stride=25)
    assert m['harmonic_ratio'] > 0.4, m
    assert m['antiphase_deg'] > 0.9, m
    assert abs(m['f0_hz'] - 2.0) < 0.4, m
    # gated out
    m0 = gait_spectral_metrics(leg, torch.zeros(n, 2), dt=DT, stride=25)
    assert m0['harmonic_ratio'] is None and m0['n_gated_windows'] == 0


def test_eval_metrics_respect_nan_boundary():
    n, T = 8, 300
    t = torch.arange(T).float() * DT
    leg = torch.zeros(n, T, 12)
    for j in range(12):
        base = math.pi if j >= 6 else 0.0
        leg[:, :, j] = 0.1 * torch.sin(2 * math.pi * 1.33 * t + base)
    leg[:4, 150:, :] = float('nan')       # half the envs end at t=150
    cmd = torch.zeros(n, 2)
    cmd[:, 0] = 0.5
    m = gait_spectral_metrics(leg, cmd, dt=DT, stride=25)
    assert m['n_gated_windows'] > 0
    assert m['harmonic_ratio'] is not None


def test_gradient_flows_to_policy():
    torch.manual_seed(0)
    actor = torch.nn.Sequential(torch.nn.Linear(78, 32), torch.nn.ELU(),
                                torch.nn.Linear(32, 23))
    obs = torch.randn(4, T_W, 78, requires_grad=False)
    a = actor(obs.reshape(-1, 78)).reshape(4, T_W, 23)
    default = torch.zeros(12)
    lim = torch.tensor([[-2.5, 2.9]] * 12)
    q = torch.clamp(default + 0.25 * a[..., :12], lim[:, 0], lim[:, 1])
    q, pw = windowed_fft(q)
    freqs = rfft_freqs(T_W, DT)
    f0 = spectral_peak_f0(pw, freqs).detach()
    lc, _ = concentration_loss(q, pw, freqs, f0)
    loss = lc.mean()
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for p in actor.parameters() if p.grad is not None)
    assert gnorm > 0


def test_f0_bin_index_excludes_dc():
    freqs = rfft_freqs(T_W, DT)
    idx = f0_bin_index(freqs, torch.tensor([0.1, 0.55, 2.1]))
    assert (idx >= 1).all()
    assert idx[0] == 1 and idx[1] == 3 and idx[2] == 13   # nearest padded bins


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print('PASS', fn.__name__)
    print('all {} tests passed'.format(len(fns)))
