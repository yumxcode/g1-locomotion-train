#!/usr/bin/env python
"""Resume training from a platform-mounted checkpoint (gradmotion resume tasks).

The platform mounts the source checkpoint into `checkPointMountPath` (set in the
task JSON; typically `<repo>/ckpt_in/`). This script:
  1. locates the mounted *.pt (searches a few candidate dirs),
  2. copies it to logs/<exp>/resumed/model_<it>.pt (legged_gym resume layout),
  3. delegates to gm_run_all.main() with --resume --load_run=resumed --checkpoint=<it>.

Extra iterations are given via --max_iterations (rsl_rl counts from the loaded iter).
Usage: gm-run g1-locomotion-train/gm_resume.py --task=g1_23dof --headless --max_iterations=4000
"""
import glob
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.realpath(__file__))


def find_mounted_ckpt():
    candidates = [
        os.path.join(ROOT, 'ckpt_in'),
        os.path.join(os.getcwd(), 'ckpt_in'),
        '/workspace/ckpt_in',
        os.path.join(os.getcwd(), 'g1-locomotion-train', 'ckpt_in'),
        '/workspace',
    ]
    best = None
    for d in candidates:
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, '*.pt')), key=os.path.getmtime):
            m = re.match(r'model_(\d+)', os.path.basename(p))
            if m:
                best = (p, int(m.group(1)))
    return best


def main():
    ck = find_mounted_ckpt()
    if ck is None:
        print('[gm-resume] ERROR: no mounted model_<it>*.pt found in candidate dirs')
        sys.exit(3)
    src, it = ck
    print('[gm-resume] mounted checkpoint:', src, '-> iter', it)
    exp = 'g1_23dof'
    import isaacgym  # noqa: F401,E402  must precede legged_gym torch usage
    from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402
    targets = [
        os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', exp, 'resumed'),  # rsl_rl resume reads here
        os.path.join(os.getcwd(), 'logs', exp, 'resumed'),
    ]
    dst = None
    for run_dir in targets:
        os.makedirs(run_dir, exist_ok=True)
        dst = os.path.join(run_dir, 'model_%d.pt' % it)
        shutil.copy2(src, dst)
        print('[gm-resume] staged to', dst)

    # delegate to gm_run_all with resume args
    sys.path.insert(0, ROOT)
    import gm_run_all  # noqa: E402  (re-uses its stages after training)
    extra = ['--resume', '--load_run=resumed', '--checkpoint=%d' % it]
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if not a.startswith('--load_run') and not a.startswith('--checkpoint=')] + extra
    print('[gm-resume] argv:', sys.argv)
    gm_run_all.main()


if __name__ == '__main__':
    main()
