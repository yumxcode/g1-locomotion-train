"""G1 23-DoF velocity-command env, aligned with the evaluation-side contract v0.

Contract v0 alignment ([g1-23dof] task):
  - observation, 78 dims, exact order:
      [0:3]   velocity command (vx, vy, wz) * [2.0, 2.0, 0.25]
      [3:6]   base angular velocity (base frame) * 0.25
      [6:9]   projected gravity (base frame)
      [9:32]  dof_pos - default_dof_pos   (23 joints, URDF order)
      [32:55] dof_vel * 0.05              (23)
      [55:78] previous policy action, raw (23)
  - no gait-clock phase observation (free-form gait shaped by
    feet_air_time / feet_swing_height / contact_no_vel rewards)
  - action semantics: q_target = clip(default_q + 0.25 * a, joint limits)
"""

from legged_gym.envs.g1.g1_env import G1Robot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch


class G1_23dofRobot(G1Robot):

    def _get_noise_scale_vec(self, cfg):
        """Noise vector matching the 78-dim contract-v0 observation layout."""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        na = self.num_actions
        noise_vec[:3] = 0.  # commands (no noise)
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:9 + na] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9 + na:9 + 2 * na] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9 + 2 * na:9 + 3 * na] = 0.  # previous actions (no noise)
        return noise_vec

    def compute_observations(self):
        """Contract-v0 order: [cmd | ang_vel | gravity | dof_pos | dof_vel | prev_action]."""
        self.obs_buf = torch.cat((self.commands[:, :3] * self.commands_scale,
                                  self.base_ang_vel * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions
                                  ), dim=-1)
        self.privileged_obs_buf = None
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _compute_torques(self, actions):
        """Position PD with contract-v0 action semantics:
        q_target = clip(default_q + action_scale * a, dof limits), torque clipped to URDF effort."""
        actions_scaled = actions * self.cfg.control.action_scale
        q_target = actions_scaled + self.default_dof_pos
        # [g1-23dof] clip position targets to joint limits (contract v0)
        q_target = torch.clamp(q_target, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
        torques = self.p_gains * (q_target - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
