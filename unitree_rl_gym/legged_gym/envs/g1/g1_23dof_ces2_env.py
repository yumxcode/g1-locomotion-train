"""G1 23-DoF CES iteration-2 env: v6 reward VERBATIM + estimator-only CES.

SPEC v1.1 (iteration 2): the foot-local reward is fully removed (w_loc = 0,
implemented by NOT having a ces_loc entry in the reward scales table at all —
the reward path is the frozen v6 one, byte-for-byte) while the CES estimator
side keeps running: online debounced segmentation + extras['ces'] export
happen right after compute_reward(), i.e. at exactly the same point of
post_physics_step() where the iteration-1 reward function used to be called
(same reset_buf / contact_forces / feet_vel tensor states).
"""

from legged_gym.envs.g1.g1_23dof_ces_env import G1_23dofCESRobot


class G1_23dofCES2Robot(G1_23dofCESRobot):
    """v6 rewards verbatim + CES segmentation export only (no reward effect).

    In this class reward_scales has no 'ces_loc' key, so the base class's
    _prepare_reward_function never builds the foot-local reward — but it ALSO
    never calls the segmentation code. We therefore re-attach the segmentation
    step to compute_reward() (after the v6 terms are computed), which keeps the
    estimator stream identical to iteration 1 while the reward stream is
    identical to v6.
    """

    def compute_reward(self):
        """v6 reward functions verbatim (no ces_loc scale), then the CES
        segmentation bookkeeping + extras['ces'] export — reward-free."""
        super().compute_reward()
        self._ces_step()
