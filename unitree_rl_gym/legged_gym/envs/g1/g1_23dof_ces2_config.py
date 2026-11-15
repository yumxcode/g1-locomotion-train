"""CES-PPO iteration-2 task config: v6 VERBATIM + estimator-only CES.

Everything inherits from G1_23dofCfg (the frozen v6 baseline config): env,
reward scales (NO ces_loc entry — the foot-local reward is fully removed,
w_loc = 0 by absence), PD, observations, domain randomization, seed = 1
(v6's training seed, paired comparison). The ONLY difference vs v6 is the
runner algorithm: CESPPO (the segmented-advantage estimator package).

ces_mode ('a1' | 'a2') stays a runtime switch on the algorithm side, set by
gm_run_all.py --ces_mode: a1 = event boundaries (treatment), a2 = randomly
re-placed boundaries with the same segment-length multiset (control). The
environment/reward stream is identical in both modes and identical to v6.
"""

from legged_gym.envs.g1.g1_23dof_config import G1_23dofCfg, G1_23dofCfgPPO


class G1_23dofCES2Cfg(G1_23dofCfg):
    """v6 frozen env/reward/PD/obs, byte-identical (estimator-only CES).

    Intentionally EMPTY: the v6 reward terms apply verbatim. w_loc = 0 is
    realized by the ABSENCE of the ces_loc scale, not by a 0.0 entry (the
    base class pops zero scales, which would also skip the segmentation
    export — see g1_23dof_ces2_env.py).
    """


class G1_23dofCES2CfgPPO(G1_23dofCfgPPO):
    class runner(G1_23dofCfgPPO.runner):
        algorithm_class_name = 'CESPPO'   # [ces2] segmented advantage PPO
        experiment_name = 'g1_23dof_ces2'
