from legged_gym.envs.g1.g1_23dof_config import G1_23dofCfg, G1_23dofCfgPPO


class G1_23dofSGPCfg(G1_23dofCfg):
    """[sgp] SGP-PPO task: environment BIT-IDENTICAL to the frozen v6 task
    (reward / PD / obs / domain-rand / asset all inherited unchanged). The
    spectral gait prior is a training-side loss only (SGPPPO); the deployed
    policy interface is identical to A0. Separate experiment_name so the
    platform log/scan dirs stay disjoint per arm."""
    pass  # no overrides: anything env-side would violate the freeze


class G1_23dofSGPCfgPPO(G1_23dofCfgPPO):
    class runner(G1_23dofCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "SGPPPO"     # [sgp] PPO + spectral gait prior
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1_23dof_sgp'

    class algorithm(G1_23dofCfgPPO.algorithm):
        entropy_coef = 0.01
        # ---- SPEC_SGP_PPO v1.0 defaults (arm S1); CLI overrides in gm_run_all_sgp.py
        sgp_arm = 's1'
        sgp_beta0 = 0.5
        sgp_beta_end = 0.05
        sgp_lambda_anti = 0.5
        sgp_anneal_frac = 0.30
        sgp_max_windows = 512
        sgp_window = 75
        sgp_gate_vel = 0.3
        sgp_f0_ema_alpha = 0.05
        sgp_f0_w_ramp = 1000
        sgp_w_f0 = 1.0
