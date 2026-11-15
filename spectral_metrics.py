#!/usr/bin/env python
"""Standalone spectral gait metrics (harmonic ratio / antiphase degree).

SAME implementation / convention as the SGP-PPO training objective
(rsl_rl.utils.sgp): T_w = 75-step (1.5 s) windows, linear detrend + Hann,
batched rFFT, f0 self-bootstrap = argmax over [0.5, 3] Hz of the summed leg
spectrum, harmonic band H(f0) = union_{k=1..3} k*f0*(1+/-0.15), gate
mean ||v_cmd_xy|| > 0.3 m/s, antiphase over the hipPitch/knee/anklePitch
left/right pairs. Reused verbatim by the stage-3 MuJoCo sim2sim pipeline so
trainer and cross-simulator numbers share one definition.

Input formats (choose one):
  --npz FILE.npz    arrays: leg_pos (N, T, 12) [rad], cmd_xy (N, 2)|(N, T, 2)
                    [m/s]; optional dt (scalar, default 0.02)
  --pt FILE.pt      dict/tensor payload with the same keys (torch tensors)

Episode boundaries inside a trajectory: mark the step row with NaN in any
leg/cmd channel; windows crossing a boundary are dropped (within-episode only,
same as the training ring buffer).

Output: JSON to stdout (and --out FILE.json).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'rsl_rl'))

import torch  # noqa: E402
from rsl_rl.utils.sgp import gait_spectral_metrics  # noqa: E402


def load_input(path):
    if path.endswith('.npz'):
        import numpy as np
        z = np.load(path)
        leg = torch.as_tensor(z['leg_pos']).float()
        cmd = torch.as_tensor(z['cmd_xy']).float()
        dt = float(z['dt']) if 'dt' in z.files else 0.02
    elif path.endswith('.pt'):
        z = torch.load(path, map_location='cpu')
        if isinstance(z, dict) and 'leg_pos' in z:
            leg = z['leg_pos'].float()
            cmd = z['cmd_xy'].float()
            dt = float(z.get('dt', 0.02))
        else:
            raise SystemExit('pt payload must be a dict with leg_pos/cmd_xy')
    else:
        raise SystemExit('unsupported input: {}'.format(path))
    return leg, cmd, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz')
    ap.add_argument('--pt')
    ap.add_argument('--stride', type=int, default=25)
    ap.add_argument('--out')
    a = ap.parse_args()
    path = a.npz or a.pt
    if not path:
        raise SystemExit('need --npz or --pt')
    leg, cmd, dt = load_input(path)
    m = gait_spectral_metrics(leg, cmd, dt=dt, stride=a.stride)
    m['input'] = path
    m['note'] = ('harmonic_ratio in [0,1] (1 = leg energy inside H(f0)); '
                 'antiphase_deg in [0,1] (1 = perfect pi L/R phase); '
                 'f0 from window self-bootstrap argmax [0.5,3] Hz '
                 '(rFFT at 4x zero-padding, bin resolution 0.167 Hz @ T_w=75, dt=0.02)')
    txt = json.dumps(m, indent=2)
    print(txt)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(txt)


if __name__ == '__main__':
    main()
