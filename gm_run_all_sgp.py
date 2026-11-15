#!/usr/bin/env python
"""gm platform single entry: SGP-PPO train -> in-sim acceptance eval (+spectral
metrics + energy) -> policy export. SPEC_SGP_PPO v1.0 (branch sgp-ppo).

Run on gradmotion (IsaacGym preview-4 image, Py3.8 + torch 2.4.1):
    gm-run gm_run_all_sgp.py --task=g1_23dof_sgp --headless \
        --sgp_arm=s1 --run_name=s1

Stages
  1. train   : SGPPPO = frozen v6 PPO + training-side spectral gait prior
               (L = L_PPO + beta(t)*(L_conc + lambda_a*L_anti)); environment,
               reward, PD, obs, network, seed identical to A0 (deployment
               interface bit-identical; the prior only shapes training).
               it2500 early-stop gate (ep_len <= 50 floor, v6 reference
               it2000 ~ 297) as in the CES protocol.
  2. eval    : v6 acceptance protocol (4096 fixed-command episodes, 10 s, DR
               kept, noise/pushes off) + fixed eval seed (all arms share the
               same command set) + energy (mean |tau*qdot| power) + spectral
               gait metrics (harmonic ratio / antiphase degree, the SAME
               windowed-FFT implementation as the training objective).
  3. export  : identical to v6 (TorchScript + ONNX, consistency < 1e-5).

Extra CLI (parsed before get_args):
    --sgp_arm=s1|s2|s3     arm selector (s1 full, s2 conc only, s3 anti only)
    --sgp_beta0=0.5        beta start (annealed linearly to beta_end over the
                           first 30% of iterations, SPEC default 0.5 -> 0.05)
    --sgp_beta_end=0.05
    --sgp_check_at=2500    early-stop gate iteration
    --sgp_floor=50.0       early-stop ep_len floor
    --sgp_eval_seed=1234   fixed eval command seed (cross-arm comparability)
Artifacts (platform-scanned):
    logs/{exp}/exported_data/{run}/model_N.pt            checkpoints
    logs/{exp}/exported_data/{run}/policy_1.pt/.onnx/... exports
    logs/{exp}/gm_play/eval_results.pt                   acceptance metrics
    logs/{exp}/gm_play/sgp_monitor.pt                    per-update SGP monitor
    output/final_g1_policy.pt                            final checkpoint
"""
import os
import sys
import json
import shutil
import time
import statistics
from collections import deque

ROOT = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'rsl_rl'))
sys.path.insert(1, os.path.join(ROOT, 'unitree_rl_gym'))

import isaacgym  # noqa: F401  must be imported before legged_gym torch usage
import torch
import numpy as np

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR
from gm_run_all import mirror_paths, stage3_export, settle_for_sdk_upload  # noqa: E402  reuse v6 stages
from rsl_rl.utils.sgp import gait_spectral_metrics  # noqa: E402


# ---------------------------------------------------------------- CLI (pre get_args)

def parse_sgp_args():
    """Extract --sgp_* flags (legged_gym's get_args aborts on unknown args)."""
    spec = {
        'sgp_arm': ('s1', str, ('s1', 's2', 's3')),
        'sgp_beta0': (0.5, float, None),
        'sgp_beta_end': (0.05, float, None),
        'sgp_check_at': (2500, int, None),
        'sgp_floor': (50.0, float, None),
        'sgp_eval_seed': (1234, int, None),
    }
    out = {k: d for k, (d, _, _) in spec.items()}
    for a in sys.argv[1:]:
        if not a.startswith('--sgp_'):
            continue
        key, _, val = a[2:].partition('=')
        if key not in spec or not val:
            print('[gm] bad --sgp_* argument:', a)
            sys.exit(2)
        _, cast, allowed = spec[key]
        try:
            out[key] = cast(val)
        except ValueError:
            print('[gm] bad value for', key, ':', val)
            sys.exit(2)
        if allowed and out[key] not in allowed:
            print('[gm] {} must be one of {}'.format(key, allowed))
            sys.exit(2)
    sys.argv = [a for a in sys.argv if not a.startswith('--sgp_')]
    return out


def parse_load_ckpts():
    out = []
    for a in sys.argv[1:]:
        if a.startswith('--load_ckpt='):
            out.extend(p for p in a[len('--load_ckpt='):].split(',') if p)
        elif a == '--load_ckpt':
            print('[gm] --load_ckpt requires =path (comma separated)')
            sys.exit(2)
    sys.argv = [a for a in sys.argv if not a.startswith('--load_ckpt')]
    return out


# ---------------------------------------------------------------- learn loop

def learn_sgp(runner, num_learning_iterations, init_at_random_ep_len=True,
              check_at=2500, floor=50.0):
    """runner.learn verbatim + (a) per-step sgp_push, (b) SGP tensorboard
    curves, (c) it2500 early-stop gate (same gate as the CES protocol)."""
    if runner.log_dir is not None and runner.writer is None:
        from torch.utils.tensorboard import SummaryWriter
        runner.writer = SummaryWriter(log_dir=runner.log_dir, flush_secs=10)
    if init_at_random_ep_len:
        runner.env.episode_length_buf = torch.randint_like(
            runner.env.episode_length_buf, high=int(runner.env.max_episode_length))
    obs = runner.env.get_observations()
    privileged_obs = runner.env.get_privileged_observations()
    critic_obs = privileged_obs if privileged_obs is not None else obs
    obs, critic_obs = obs.to(runner.device), critic_obs.to(runner.device)
    runner.alg.actor_critic.train()

    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(runner.env.num_envs, dtype=torch.float, device=runner.device)
    cur_episode_length = torch.zeros(runner.env.num_envs, dtype=torch.float, device=runner.device)

    stopped = False
    ep_len_at_check = None
    prev_dones = None
    tot_iter = runner.current_learning_iteration + num_learning_iterations
    for it in range(runner.current_learning_iteration, tot_iter):
        start = time.time()
        with torch.inference_mode():
            for i in range(runner.num_steps_per_env):
                actions = runner.alg.act(obs, critic_obs)
                # [sgp] record the step the policy is acting on (obs/cmd/dof_pos
                # at time t; prev_dones mark the previous sample as last-of-episode)
                runner.alg.sgp_push(obs, runner.env.commands[:, :3],
                                    runner.env.dof_pos[:, :12], prev_dones)
                obs, privileged_obs, rewards, dones, infos = runner.env.step(actions)
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = (obs.to(runner.device), critic_obs.to(runner.device),
                                                   rewards.to(runner.device), dones.to(runner.device))
                runner.alg.process_env_step(rewards, dones, infos)
                prev_dones = dones

                if runner.log_dir is not None:
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            start = stop
            runner.alg.compute_returns(critic_obs)

        runner.alg.current_it = it
        mean_value_loss, mean_surrogate_loss = runner.alg.update()
        stop = time.time()
        learn_time = stop - start
        if runner.log_dir is not None:
            runner.log({'ep_infos': ep_infos, 'it': it, 'mean_value_loss': mean_value_loss,
                        'mean_surrogate_loss': mean_surrogate_loss,
                        'collection_time': collection_time, 'learn_time': learn_time,
                        'rewbuffer': rewbuffer, 'lenbuffer': lenbuffer,
                        'num_learning_iterations': num_learning_iterations})
            m = runner.alg.sgp_last
            w = runner.writer
            for k in ('l_conc_cmd', 'l_anti_cmd', 'l_conc_pos', 'l_anti_pos', 'beta',
                      'f0_peak_mean', 'f0_eff_mean', 'gphi_f0_mean', 'gphi_loss',
                      'spectral_term', 'coverage'):
                if m.get(k) is not None:
                    w.add_scalar('Spectral/' + k, float(m[k]), it)
            w.add_scalar('Spectral/n_gated', float(m.get('n_gated') or 0), it)
            if it % 50 == 0 and m.get('active'):
                print('[sgp] it {} beta {:.4f} | L_conc(cmd/pos) {:.3f}/{:.3f} '
                      'L_anti(cmd/pos) {:.3f}/{:.3f} | f0 peak/eff/gphi {:.2f}/{:.2f}/{:.2f} Hz '
                      '| gated {}/{} (cov {:.2f}) | spec_term {:.4g} vs surrogate {:.4g}'.format(
                          it, m['beta'], m['l_conc_cmd'], m['l_conc_pos'], m['l_anti_cmd'],
                          m['l_anti_pos'], m['f0_peak_mean'], m['f0_eff_mean'],
                          m['gphi_f0_mean'], m['n_gated'], m['n_valid'], m['coverage'],
                          m['spectral_term'], mean_surrogate_loss))
        if it % runner.save_interval == 0:
            runner.save(os.path.join(runner.log_dir, 'model_{}.pt'.format(it)))
        ep_infos.clear()

        if (it + 1) == check_at:
            ep_len_at_check = float(statistics.mean(lenbuffer)) if len(lenbuffer) else -1.0
            verdict = ep_len_at_check <= floor
            print('[gm] EARLY-STOP GATE iter {}: mean ep_len (last {} eps) = {:.1f} '
                  '(v6 reference: it2000 ~297) floor={} -> {}'.format(
                      it + 1, len(lenbuffer), ep_len_at_check, floor,
                      'STOP ARM' if verdict else 'CONTINUE'))
            if verdict:
                stopped = True
                runner.save(os.path.join(runner.log_dir, 'model_{}.pt'.format(it + 1)))
                print('[gm] EARLY-STOPPED at iter {} (checkpoint saved; eval/export proceed)'.format(it + 1))
                break

    if stopped:
        runner.current_learning_iteration = check_at
    else:
        runner.current_learning_iteration += num_learning_iterations
        runner.save(os.path.join(runner.log_dir, 'model_{}.pt'.format(runner.current_learning_iteration)))
    return {'stopped': stopped, 'check_iter': check_at, 'floor': floor,
            'ep_len_at_check': ep_len_at_check}


# ---------------------------------------------------------------- stage 1

def stage1_train(args, sgp):
    print('=' * 70)
    print('STAGE 1: SGP-PPO TRAINING [arm={} beta0={} beta_end={}]'.format(
        sgp['sgp_arm'], sgp['sgp_beta0'], sgp['sgp_beta_end']))
    print('=' * 70)
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    assert ppo_runner.alg.__class__.__name__ == 'SGPPPO', 'task must build SGPPPO (task=g1_23dof_sgp)'
    alg = ppo_runner.alg
    alg.sgp_arm = sgp['sgp_arm']
    alg.sgp_w_conc = 1.0 if sgp['sgp_arm'] in ('s1', 's2') else 0.0
    alg.sgp_w_anti = 1.0 if sgp['sgp_arm'] in ('s1', 's3') else 0.0
    alg.sgp_beta0 = float(sgp['sgp_beta0'])
    alg.sgp_beta_end = float(sgp['sgp_beta_end'])
    n_iters = train_cfg.runner.max_iterations
    alg.sgp_bind_env(env, n_iters)
    print('[gm] effective reward scales:', json.dumps(
        {k: float(v) for k, v in sorted(env.reward_scales.items())}))
    print('[gm] seed =', train_cfg.seed, 'num_envs =', env.num_envs,
          'max_iterations =', n_iters)

    exp_name = train_cfg.runner.experiment_name
    run_dir_name = os.path.basename(ppo_runner.log_dir)
    targets = mirror_paths(exp_name, run_dir_name)
    orig_save = ppo_runner.save

    def save_and_mirror(path, infos=None):
        orig_save(path, infos)
        for t in targets:
            os.makedirs(t, exist_ok=True)
            shutil.copy2(path, os.path.join(t, os.path.basename(path)))

    ppo_runner.save = save_and_mirror
    t0 = os.times().elapsed
    early_stop = learn_sgp(ppo_runner, n_iters, init_at_random_ep_len=True,
                           check_at=int(sgp['sgp_check_at']), floor=float(sgp['sgp_floor']))
    dt = os.times().elapsed - t0
    print('[gm] training wall time: {:.1f} s for {} iterations ({:.2f} s/it)'.format(
        dt, n_iters, dt / max(1, n_iters)))
    print('[gm] throughput: {:.0f} env-steps/s'.format(n_iters * 24 * env.num_envs / max(dt, 1e-9)))

    # SGP monitor dump (CSV bytes inside .pt: platform scans .pt only)
    csv_text = alg.sgp_monitor_rows()
    mon = {'monitor': alg.sgp_monitor, 'csv': csv_text, 'early_stop': early_stop,
           'sgp_state': {k: (v.cpu() if hasattr(v, 'cpu') else v)
                         for k, v in alg.sgp_bootstrap.state_dict().items()},
           'arm': alg.sgp_arm, 'beta0': alg.sgp_beta0, 'beta_end': alg.sgp_beta_end}
    out_top = os.path.join(os.getcwd(), 'output')
    for base in [os.getcwd(), LEGGED_GYM_ROOT_DIR, ROOT, out_top]:
        d = os.path.join(base, 'logs', exp_name, 'gm_play') if base != out_top else out_top
        os.makedirs(d, exist_ok=True)
        torch.save(mon, os.path.join(d, 'sgp_monitor.pt'))
        with open(os.path.join(d, 'sgp_monitor.csv'), 'w') as f:
            f.write(csv_text)
    last = alg.sgp_monitor[-50:]
    act = [m for m in last if m.get('active')]
    if act:
        print('[gm] SGP monitor tail (last {} active updates): l_conc_cmd {:.3f} '
              'l_anti_cmd {:.3f} f0_peak {:.2f} Hz coverage {:.2f}'.format(
                  len(act),
                  statistics.mean(m['l_conc_cmd'] for m in act),
                  statistics.mean(m['l_anti_cmd'] for m in act),
                  statistics.mean(m['f0_peak_mean'] for m in act),
                  statistics.mean(m['coverage'] for m in act)))
    bad = [n for n, p in alg.actor_critic.named_parameters() if torch.isnan(p).any().item()]
    print('[gm] NaN parameter check:', 'CLEAN' if not bad else 'NaN in {}'.format(bad))
    return env, ppo_runner, train_cfg, run_dir_name, early_stop, alg


# ---------------------------------------------------------------- stage 2 (v6 + spectral + energy)

def stage2_eval_sgp(args, env, ppo_runner, exp_name, tag='', eval_seed=1234):
    print('=' * 70)
    print('STAGE 2: ACCEPTANCE EVALUATION (v6 protocol + spectral metrics + energy)')
    print('=' * 70)
    actor_critic = ppo_runner.alg.actor_critic
    actor_critic.eval()
    actor_critic.to(env.device)

    n = env.num_envs
    device = env.device
    env.cfg.domain_rand.push_robots = False
    env.add_noise = False
    env.max_episode_length = int(round(10.0 / env.dt))
    env.cfg.commands.heading_command = False
    env.cfg.commands.resampling_time = 1e9

    with torch.inference_mode():
        # [sgp] FIXED eval seed: every arm (and S0) evaluates on the SAME 4096
        # commands (protocol otherwise identical to v6 stage-2)
        torch.manual_seed(int(eval_seed))
        vx = torch.empty(n).uniform_(-1.0, 1.0)
        vy = torch.empty(n).uniform_(-1.0, 1.0)
        wz = torch.empty(n).uniform_(-1.0, 1.0)
        small = (vx ** 2 + vy ** 2).sqrt() < 0.2
        vx[small] = 0.0
        vy[small] = 0.0

        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        env.commands[:, 0] = vx.to(device)
        env.commands[:, 1] = vy.to(device)
        env.commands[:, 2] = wz.to(device)
        cmd_xy = torch.stack([vx, vy], dim=-1).to(device)      # (N, 2) constant per env

        T = env.max_episode_length
        err_sq = torch.zeros(n, device=device)
        cnt = torch.zeros(n, device=device)
        fall = torch.zeros(n, dtype=torch.bool, device=device)
        done_first = torch.zeros(n, dtype=torch.bool, device=device)
        ep_steps = torch.zeros(n, device=device)
        work = torch.zeros(n, device=device)                   # sum |tau*qdot| dt [J]
        leg_traj = torch.full((n, T, 12), float('nan'), device=device)
        dt = env.dt

        for t in range(T):
            actions = actor_critic.act_inference(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            active = ~done_first
            if active.any():
                idx = active.nonzero(as_tuple=False).squeeze(-1)
                cmd = env.commands[idx, :2]
                vel = env.base_lin_vel[idx, :2]
                err_sq[idx] += ((cmd - vel) ** 2).sum(dim=1)
                cnt[idx] += 2.0
                ep_steps[idx] += 1
                # energy: mechanical work magnitude sum_j |tau_j * omega_j| * dt
                tau = env.torques[idx, :12]
                omega = env.dof_vel[idx, :12]
                work[idx] += (tau * omega).abs().sum(dim=1) * dt
                leg_traj[idx, t, :] = env.dof_pos[idx, :12]
            newly = env.reset_buf & active
            fall |= newly & (~env.time_out_buf)
            done_first |= env.reset_buf

        rmse_env = (err_sq / cnt.clamp(min=1.0)).sqrt()
        no_fall = float((~fall).float().mean())
        power_mean = work / (ep_steps * dt).clamp(min=dt)      # mean |work| rate [W]

        # spectral gait metrics on the RECORDED dof_pos trajectories (same
        # windowed-FFT implementation as training; NaN rows = episode boundary)
        spec = gait_spectral_metrics(leg_traj, cmd_xy, dt=dt, stride=25)

        results = {
            'num_episodes': int(n),
            'episode_length_s': 10.0,
            'rmse_lin_vel_xy_mean': float(rmse_env.mean()),
            'rmse_lin_vel_xy_std': float(rmse_env.std()),
            'rmse_lin_vel_xy_max': float(rmse_env.max()),
            'no_fall_rate': no_fall,
            'num_falls': int(fall.sum()),
            'mean_episode_steps_active': float(ep_steps.mean()),
            'energy_mean_power_W': float(power_mean.mean()),
            'energy_mean_power_W_std': float(power_mean.std()),
            'energy_total_J_all_envs': float(work.sum()),
            'spectral_harmonic_ratio': spec['harmonic_ratio'],
            'spectral_antiphase_deg': spec['antiphase_deg'],
            'spectral_f0_hz_mean': spec['f0_hz'],
            'spectral_gate_coverage': spec['gate_coverage'],
            'spectral_n_gated_windows': spec['n_gated_windows'],
            'spectral_n_valid_windows': spec['n_valid_windows'],
            'pass_rmse_le_0.3': bool(float(rmse_env.mean()) <= 0.3),
            'pass_no_fall_ge_0.9': bool(no_fall >= 0.9),
            'eval_seed': int(eval_seed),
            'protocol': ('v6 protocol + fixed eval seed {s}: fixed cmds uniform vx,vy,wz in '
                         '[-1,1]^3 with |v_xy|<0.2 zeroed; base-frame xy; per-episode RMSE '
                         'then mean; fall = early termination within 10 s; training DR kept; '
                         'obs noise & pushes off; energy = sum_j |tau*qdot|*dt / active_time; '
                         'spectral = T_w 75 (1.5 s) detrend+Hann+rFFT windows, stride 25, '
                         'gate mean||v_cmd_xy||>0.3, f0 self-bootstrap argmax [0.5,3] Hz, '
                         'harmonic ratio = in-band(H(f0)) fraction, antiphase = (1-cos(dpsi))/2 '
                         'mean over hipPitch/knee/anklePitch pairs').format(s=int(eval_seed)),
            'torch_version': torch.__version__,
        }
        print(json.dumps(results, indent=2))

        payload = {'results': results, 'rmse_per_episode': rmse_env.cpu(),
                   'fall_flags': fall.cpu(), 'power_per_episode': power_mean.cpu(),
                   'cmd_xy': cmd_xy.cpu(), 'spectral_windows': spec}
        saved = []
        suffix = ('_' + tag) if tag else ''
        out_top = os.path.join(os.getcwd(), 'output')
        for base in [os.getcwd(), LEGGED_GYM_ROOT_DIR, ROOT, out_top]:
            d = os.path.join(base, 'logs', exp_name, 'gm_play') if base != out_top else out_top
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, 'eval_results{}.pt'.format(suffix))
            torch.save(payload, p)
            saved.append(p)
        print('[gm] eval results saved:', saved)
        return results


# ---------------------------------------------------------------- main

def main():
    load_ckpts = parse_load_ckpts()
    sgp = parse_sgp_args()
    args = get_args()

    if load_ckpts:
        print('=' * 70)
        print('LOAD-AND-EVAL MODE:', load_ckpts, 'sgp args:', sgp)
        print('=' * 70)
        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
        exp_name = train_cfg.runner.experiment_name
        run_dir_name = os.path.basename(ppo_runner.log_dir)
        summaries = {}
        for ck in load_ckpts:
            tag = os.path.splitext(os.path.basename(ck))[0].replace('model_', 'model')
            print('[gm] loading checkpoint:', ck, '-> tag', tag)
            ppo_runner.load(ck, load_optimizer=False)
            results = stage2_eval_sgp(args, env, ppo_runner, exp_name, tag=tag,
                                      eval_seed=sgp['sgp_eval_seed'])
            report = stage3_export(args, env, ppo_runner, run_dir_name, exp_name, tag=tag)
            summaries[tag] = {'eval': results, 'export': report}
        with open(os.path.join(os.getcwd(), 'gm_final_summary.json'), 'w') as f:
            json.dump(summaries, f, indent=2)
        print('[gm] FINAL SUMMARY:', json.dumps(summaries, indent=2))
        settle_for_sdk_upload()
        return

    env, ppo_runner, train_cfg, run_dir_name, early_stop, alg = stage1_train(args, sgp)
    exp_name = train_cfg.runner.experiment_name
    results = stage2_eval_sgp(args, env, ppo_runner, exp_name,
                              eval_seed=sgp['sgp_eval_seed'])
    report = stage3_export(args, env, ppo_runner, run_dir_name, exp_name)
    final = {'eval': results, 'export': report, 'early_stop': early_stop,
             'sgp': {'arm': alg.sgp_arm, 'beta0': alg.sgp_beta0,
                     'beta_end': alg.sgp_beta_end, 'seed': train_cfg.seed,
                     'max_iterations': train_cfg.runner.max_iterations,
                     'n_monitor_updates': len(alg.sgp_monitor)}}
    print('[gm] FINAL SUMMARY:', json.dumps(final, indent=2))
    with open(os.path.join(os.getcwd(), 'gm_final_summary.json'), 'w') as f:
        json.dump(final, f, indent=2)
    settle_for_sdk_upload()


if __name__ == '__main__':
    main()
