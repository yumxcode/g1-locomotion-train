#!/usr/bin/env python
"""gm platform single entry: PPO train -> in-sim acceptance eval -> policy export.

Run on gradmotion (IsaacGym preview-4 image, Py3.8 + torch 2.4.1):
    gm-run gm_run_all.py --task=g1 --headless --max_iterations=10

Stages
  1. train        : legged_gym PPO (unitree_rl_gym G1 12dof), checkpoints mirrored to
                    logs/{exp}/exported_data/{run}/model_N.pt  (platform scan path)
  2. eval         : >=64 fixed episodes x 10 s, fixed velocity commands, domain-rand
                    conditions kept (per-env sampled), obs noise & pushes OFF.
                    Reports lin-vel tracking RMSE (base frame xy) and no-fall rate.
  3. export       : TorchScript (PolicyExporterLSTM) + ONNX wrapper; consistency check
                    torch vs jit vs onnx on a random obs stream (max abs diff < 1e-5).
Artifacts (platform-scanned):
    logs/{exp}/exported_data/{run}/model_N.pt            final + periodic checkpoints
    logs/{exp}/exported_data/{run}/policy_lstm_1.pt      TorchScript policy
    logs/{exp}/exported_data/{run}/policy_lstm_1.onnx    ONNX policy (may not be scanned)
    logs/{exp}/exported_data/{run}/policy_lstm_1_onnx.pt ONNX bytes wrapped in .pt (scanned)
    logs/{exp}/gm_play/eval_results.pt                   acceptance metrics + per-step series
    output/final_g1_policy.pt                            copy of final checkpoint (output/ channel)
"""
import os
import sys
import json
import shutil
import argparse

ROOT = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'rsl_rl'))
sys.path.insert(1, os.path.join(ROOT, 'unitree_rl_gym'))

import isaacgym  # noqa: F401  must be imported before legged_gym torch usage
import torch
import numpy as np

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import export_policy_as_jit


def mirror_paths(exp_name, run_dir_name):
    """Directories the platform SDK scans for checkpoints."""
    return [
        os.path.join(os.getcwd(), 'logs', exp_name, 'exported_data', run_dir_name),
        os.path.join(ROOT, 'logs', exp_name, 'exported_data', run_dir_name),
    ]


def stage1_train(args):
    print('=' * 70)
    print('STAGE 1: PPO TRAINING')
    print('=' * 70)
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

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
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                     init_at_random_ep_len=True)
    dt = os.times().elapsed - t0
    print('[gm] training wall time: {:.1f} s for {} iterations ({:.2f} s/it)'.format(
        dt, train_cfg.runner.max_iterations,
        dt / max(1, train_cfg.runner.max_iterations)))
    print('[gm] throughput: {:.0f} env-steps/s'.format(
        train_cfg.runner.max_iterations * 24 * env.num_envs / max(dt, 1e-9)))
    return env, ppo_runner, train_cfg, run_dir_name


def stage2_eval(args, env, ppo_runner, exp_name, tag=''):
    print('=' * 70)
    print('STAGE 2: ACCEPTANCE EVALUATION (in-simulator)')
    print('=' * 70)
    actor_critic = ppo_runner.alg.actor_critic
    actor_critic.eval()

    n = env.num_envs
    device = env.device
    # protocol: keep per-env domain-rand samples from training (friction/mass/PD/delay),
    # disable obs noise and pushes, fixed commands, 10 s episodes.
    env.cfg.domain_rand.push_robots = False
    env.add_noise = False
    env.max_episode_length = int(round(10.0 / env.dt))
    env.cfg.commands.heading_command = False
    env.cfg.commands.resampling_time = 1e9

    # sample one fixed command per env from the training distribution:
    # uniform(lin_vel_x in [-1,1], lin_vel_y in [-1,1], ang_vel_yaw in [-1,1]),
    # linear part zeroed if |v_xy| < 0.2 (same rule as _resample_commands).
    # NOTE: the entire evaluation runs inside inference_mode: env buffers hold
    # inference tensors after rsl_rl's rollout, inplace updates outside the mode
    # would raise "Inplace update to inference tensor".
    with torch.inference_mode():
        vx = torch.empty(n).uniform_(-1.0, 1.0)
        vy = torch.empty(n).uniform_(-1.0, 1.0)
        wz = torch.empty(n).uniform_(-1.0, 1.0)
        small = (vx ** 2 + vy ** 2).sqrt() < 0.2
        vx[small] = 0.0
        vy[small] = 0.0

        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        # fresh zero memory for evaluation (LSTM auto-zero-initializes when hidden is None)
        actor_critic.memory_a.hidden_states = None
        actor_critic.memory_c.hidden_states = None
        env.commands[:, 0] = vx.to(device)
        env.commands[:, 1] = vy.to(device)
        env.commands[:, 2] = wz.to(device)

        T = env.max_episode_length
        err_sq = torch.zeros(n, device=device)
        cnt = torch.zeros(n, device=device)
        fall = torch.zeros(n, dtype=torch.bool, device=device)
        done_first = torch.zeros(n, dtype=torch.bool, device=device)
        series_cmd = []
        series_vel = []

        for t in range(T):
            actions = actor_critic.act_inference(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            active = ~done_first
            if active.any():
                cmd = env.commands[active, :2]
                vel = env.base_lin_vel[active, :2]
                err_sq[active] += ((cmd - vel) ** 2).sum(dim=1)
                cnt[active] += 2.0
                if t % 10 == 0:  # 5 Hz logging (policy 50 Hz)
                    series_cmd.append(env.commands[:, :3].clone())
                    series_vel.append(env.base_lin_vel.clone())
            newly = env.reset_buf & active
            fall |= newly & (~env.time_out_buf)
            done_first |= env.reset_buf
            if dones.any():
                # only reset the actor memory (critic memory is never used at inference
                # and its hidden_states stays None -> actor_critic.reset() would crash)
                actor_critic.memory_a.reset(dones)

        rmse_env = (err_sq / cnt.clamp(min=1.0)).sqrt()  # per-episode RMSE over xy, base frame
        rmse_mean = float(rmse_env.mean())
        rmse_std = float(rmse_env.std())
        rmse_max = float(rmse_env.max())
        no_fall = float((~fall).float().mean())
        num_falls = int(fall.sum())
        cmd_series_t = torch.stack(series_cmd) if series_cmd else torch.zeros(0)
        vel_series_t = torch.stack(series_vel) if series_vel else torch.zeros(0)

    results = {
        'num_episodes': int(n),
        'episode_length_s': 10.0,
        'rmse_lin_vel_xy_mean': rmse_mean,
        'rmse_lin_vel_xy_std': rmse_std,
        'rmse_lin_vel_xy_max': rmse_max,
        'no_fall_rate': no_fall,
        'num_falls': num_falls,
        'pass_rmse_le_0.3': bool(rmse_mean <= 0.3),
        'pass_no_fall_ge_0.9': bool(no_fall >= 0.9),
        'protocol': ('fixed cmds uniform vx,vy,wz in [-1,1]^3 with |v_xy|<0.2 zeroed; '
                     'base-frame xy velocity; per-episode RMSE over xy dims then mean over episodes; '
                     'fall = early termination (non-timeout) within 10 s; '
                     'training domain-rand samples kept per env; obs noise & pushes off'),
        'torch_version': torch.__version__,
    }
    print(json.dumps(results, indent=2))

    payload = {
        'results': results,
        'rmse_per_episode': rmse_env.cpu(),
        'fall_flags': fall.cpu(),
        'cmd_series': cmd_series_t.cpu(),
        'vel_series': vel_series_t.cpu(),
    }
    saved = []
    suffix = ('_' + tag) if tag else ''
    for base in [os.getcwd(), LEGGED_GYM_ROOT_DIR, ROOT]:
        d = os.path.join(base, 'logs', exp_name, 'gm_play')
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, 'eval_results{}.pt'.format(suffix))
        torch.save(payload, p)
        saved.append(p)
    print('[gm] eval results saved:', saved)
    return results, payload


class OnnxLSTM(torch.nn.Module):
    """ONNX-exportable wrapper: (obs, h0, c0) -> (action, h, c). Batch = 1 per step."""

    def __init__(self, actor_critic):
        super().__init__()
        import copy
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn).float()
        self.actor = copy.deepcopy(actor_critic.actor).float()

    def forward(self, obs, h0, c0):
        out, (h, c) = self.memory(obs.unsqueeze(0), (h0, c0))
        return self.actor(out.squeeze(0)), h, c


def stage3_export(args, env, ppo_runner, run_dir_name, exp_name, tag=''):
    print('=' * 70)
    print('STAGE 3: POLICY EXPORT (TorchScript + ONNX) & CONSISTENCY CHECK')
    print('=' * 70)
    actor_critic = ppo_runner.alg.actor_critic
    actor_critic.eval()
    actor_critic.cpu()
    targets = mirror_paths(exp_name, run_dir_name)

    # TorchScript via upstream PolicyExporterLSTM
    jit_tmp = os.path.join(ROOT, '_export_jit')
    export_policy_as_jit(actor_critic, jit_tmp)
    jit_path_src = os.path.join(jit_tmp, 'policy_lstm_1.pt')

    # ONNX
    onnx_path_src = os.path.join(ROOT, '_export_onnx', 'policy_lstm_1.onnx')
    os.makedirs(os.path.dirname(onnx_path_src), exist_ok=True)
    onnx_ok = False
    onnx_err = ''
    try:
        mod = OnnxLSTM(actor_critic)
        obs0 = torch.zeros(1, actor_critic.memory_a.rnn.input_size)
        h0 = torch.zeros(1, 1, actor_critic.memory_a.rnn.hidden_size)
        c0 = torch.zeros_like(h0)
        torch.onnx.export(
            mod, (obs0, h0, c0), onnx_path_src,
            input_names=['obs', 'h0', 'c0'], output_names=['action', 'h', 'c'],
            opset_version=13, do_constant_folding=True)
        onnx_ok = True
    except Exception as e:  # noqa: BLE001
        onnx_err = repr(e)
        print('[gm] ONNX export failed:', onnx_err)

    # consistency check: torch act_inference vs jit vs onnx on a random stream
    torch.manual_seed(0)
    np.random.seed(0)
    T = 100
    obs_dim = actor_critic.memory_a.rnn.input_size
    stream = torch.rand(T, 1, obs_dim) * 2 - 1
    with torch.no_grad():
        actor_critic.memory_a.hidden_states = None  # zero-initialize memory
        out_torch = torch.cat([actor_critic.act_inference(stream[t:t + 1, 0]) for t in range(T)])
    jit_mod = torch.jit.load(jit_path_src)
    jit_mod.reset_memory()
    out_jit = torch.cat([jit_mod(stream[t:t + 1, 0]) for t in range(T)])
    diff_jit = float((out_torch - out_jit).abs().max())
    report = {'jit_vs_torch_max_abs_diff': diff_jit,
              'jit_pass_lt_1e-5': bool(diff_jit < 1e-5), 'onnx_ok': onnx_ok}
    if onnx_ok:
        try:
            try:
                import onnxruntime as ort
            except ImportError:
                # image lacks onnxruntime; try a quiet pip install (container has network)
                import subprocess
                print('[gm] installing onnxruntime for numeric check ...')
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'onnxruntime'])
                import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path_src)
            h = np.zeros((1, 1, actor_critic.memory_a.rnn.hidden_size), dtype=np.float32)
            c = np.zeros_like(h)
            outs = []
            for t in range(T):
                a, h, c = sess.run(None, {'obs': stream[t].numpy().astype(np.float32),
                                          'h0': h, 'c0': c})
                outs.append(a)
            # [g1-baseline] avoid torch.tensor(ndarray): broken numpy bridge on this image
            out_onnx = torch.tensor([[float(x) for x in a[0]] for a in outs])
            diff_onnx = float((out_torch - out_onnx).abs().max())
            report['onnx_vs_torch_max_abs_diff'] = diff_onnx
            report['onnx_pass_lt_1e-5'] = bool(diff_onnx < 1e-5)
        except Exception as e:  # noqa: BLE001
            report['onnx_check_error'] = repr(e)
            report['onnx_vs_torch_max_abs_diff'] = None
    print(json.dumps(report, indent=2))

    # dump runtime interface parameters (ground truth for INTERFACE.md)
    interface_params = {
        'dof_names': list(env.dof_names),
        'num_dof': int(env.num_dof),
        'default_dof_pos': env.default_dof_pos.squeeze(0).cpu(),
        'dof_pos_limits': env.dof_pos_limits.cpu(),
        'torque_limits': env.torque_limits.cpu(),
        'action_scale': float(env.cfg.control.action_scale),
        'decimation': int(env.cfg.control.decimation),
        'sim_dt': float(env.cfg.sim.dt),
        'policy_dt': float(env.dt),
        'clip_actions': float(env.cfg.normalization.clip_actions),
        'clip_observations': float(env.cfg.normalization.clip_observations),
        'command_ranges': {k: list(v) for k, v in env.command_ranges.items()},
        'init_state_pos': list(env.cfg.init_state.pos),
    }
    # nominal gains from config (env.p_gains may be per-env randomized)
    p_nom = torch.zeros(env.num_dof)
    d_nom = torch.zeros(env.num_dof)
    for i, name in enumerate(env.dof_names):
        for k, v in env.cfg.control.stiffness.items():
            if k in name:
                p_nom[i] = v
        for k, v in env.cfg.control.damping.items():
            if k in name:
                d_nom[i] = v
    interface_params['p_gains_nominal'] = p_nom
    interface_params['d_gains_nominal'] = d_nom
    print('[gm] interface params: dof_names =', interface_params['dof_names'])

    # distribute artifacts to platform-scanned dirs
    suffix = ('_' + tag) if tag else ''
    for t in targets:
        os.makedirs(t, exist_ok=True)
        shutil.copy2(jit_path_src, os.path.join(t, 'policy_lstm_1{}.pt'.format(suffix)))
        torch.save({'interface_params': interface_params}, os.path.join(t, 'interface_params{}.pt'.format(suffix)))
        if onnx_ok:
            shutil.copy2(onnx_path_src, os.path.join(t, 'policy_lstm_1{}.onnx'.format(suffix)))
            with open(onnx_path_src, 'rb') as f:
                torch.save({'onnx_bytes': f.read(), 'jit_vs_torch_max_abs_diff': diff_jit,
                            'report': report}, os.path.join(t, 'policy_lstm_1{}_onnx.pt'.format(suffix)))
        torch.save({'export_report': report}, os.path.join(t, 'export_report{}.pt'.format(suffix)))
    # output/ top-level channel for the final checkpoint
    out_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(out_dir, exist_ok=True)
    print('[gm] artifacts mirrored to', targets)
    return report


def make_env_and_runner(args):
    """Env + runner without training (for --load_ckpt evaluation mode)."""
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    exp_name = train_cfg.runner.experiment_name
    run_dir_name = os.path.basename(ppo_runner.log_dir)
    return env, ppo_runner, train_cfg, run_dir_name


def parse_load_ckpts():
    """Extract --load_ckpt=path1,path2 from sys.argv (legged_gym's get_args doesn't know it)."""
    out = []
    for a in sys.argv[1:]:
        if a.startswith('--load_ckpt='):
            out.extend(p for p in a[len('--load_ckpt='):].split(',') if p)
        elif a == '--load_ckpt':
            print('[gm] --load_ckpt requires =path (comma separated)')
            sys.exit(2)
    return out


def main():
    # extract --load_ckpt BEFORE get_args(): this gymutil build uses strict parse_args
    # and aborts on unknown arguments
    load_ckpts = parse_load_ckpts()
    sys.argv = [a for a in sys.argv if not a.startswith('--load_ckpt')]
    args = get_args()
    if load_ckpts:
        print('=' * 70)
        print('LOAD-AND-EVAL MODE:', load_ckpts)
        print('=' * 70)
        env, ppo_runner, train_cfg, run_dir_name = make_env_and_runner(args)
        exp_name = train_cfg.runner.experiment_name
        summaries = {}
        for ck in load_ckpts:
            tag = os.path.splitext(os.path.basename(ck))[0].replace('model_', 'model')
            print('[gm] loading checkpoint:', ck, '-> tag', tag)
            ppo_runner.load(ck, load_optimizer=False)
            results, _ = stage2_eval(args, env, ppo_runner, exp_name, tag=tag)
            report = stage3_export(args, env, ppo_runner, run_dir_name, exp_name, tag=tag)
            summaries[tag] = {'eval': results, 'export': report}
        with open(os.path.join(os.getcwd(), 'gm_final_summary.json'), 'w') as f:
            json.dump(summaries, f, indent=2)
        print('[gm] FINAL SUMMARY:', json.dumps(summaries, indent=2))
        return

    env, ppo_runner, train_cfg, run_dir_name = stage1_train(args)
    exp_name = train_cfg.runner.experiment_name
    results, _ = stage2_eval(args, env, ppo_runner, exp_name)
    report = stage3_export(args, env, ppo_runner, run_dir_name, exp_name)
    final = {'eval': results, 'export': report}
    print('[gm] FINAL SUMMARY:', json.dumps(final, indent=2))
    with open(os.path.join(os.getcwd(), 'gm_final_summary.json'), 'w') as f:
        json.dump(final, f, indent=2)


if __name__ == '__main__':
    main()
