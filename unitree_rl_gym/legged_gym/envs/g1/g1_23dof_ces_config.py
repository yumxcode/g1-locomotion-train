"""CES-PPO task config: frozen v6 environment + CES components ONLY.

Everything inherits from G1_23dofCfg (the frozen v6 baseline config at this
commit: feet_air_time = 2.0, tracking 1.0, uniform PD 60/1.5, 78-dim obs,
no gait clock). The ONLY additions are:
  - rewards.scales.ces_loc = 0.05   (SPEC 2: w_loc)
  - runner.algorithm_class_name = 'CESPPO' (segmented advantage estimator;
    all PPO hyperparameters gamma=0.99 / lam=0.95 / 5 epochs / 4 minibatches /
    lr 1e-3 adaptive / clip 0.2 / entropy 0.01 unchanged from v6)
  - seed stays 1 (v6's training seed, paired comparison)

ces_mode ('a1' | 'a2') is a runtime switch on the ALGORITHM side, set by
gm_run_all.py --ces_mode: a1 = event boundaries (treatment), a2 = randomly
re-placed boundaries with the same segment-length multiset (control). The
environment/reward stream is identical in both modes.
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.g1.g1_23dof_config import G1_23dofCfg, G1_23dofCfgPPO


class G1_23dofCESCfg(G1_23dofCfg):
    """v6 frozen env/reward/PD/obs + the CES foot-local reward term."""

    class rewards(G1_23dofCfg.rewards):
        class scales(G1_23dofCfg.rewards.scales):
            # SPEC 2: w_loc applied to (r_slip + r_imp + r_duty); same dt
            # scaling path as every other reward scale.
            ces_loc = 0.05


class G1_23dofCESCfgPPO(G1_23dofCfgPPO):
    class runner(G1_23dofCfgPPO.runner):
        algorithm_class_name = 'CESPPO'   # [ces] segmented advantage PPO
        experiment_name = 'g1_23dof_ces'
