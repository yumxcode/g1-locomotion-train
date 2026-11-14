"""CES-PPO: contact-event-segmented PPO on the frozen v6 hyperparameters.

SPEC mapping:
  (1)+(2) the env provides debounced event boundaries and the foot-local
          reward through extras['ces'] (see g1_23dof_ces_env.py)
  (3)     compute_returns -> CESRolloutStorage.compute_ces_returns:
          duty re-spread, [A2] boundary randomization, segmented TD(lambda)
          with hard bootstrap + per-segment normalization
  (4)     update -> CESRolloutStorage.segment_mini_batch_generator
  (6)     the PPO update itself (clipped objective, value clipping, KL-adaptive
          lr, entropy bonus) is IDENTICAL to the vendored PPO — the loop body
          below is a verbatim copy with only the generator call exchanged.

ces_mode:
  'a1' — event boundaries (treatment)
  'a2' — randomly re-placed boundaries with the same segment-length multiset
         (scientific control; reward stream identical to a1)
"""

import torch
import torch.nn as nn

from .ppo import PPO
from ..storage.ces_rollout_storage import CESRolloutStorage


class CESPPO(PPO):

    def __init__(self, *args, ces_mode='a1', **kwargs):
        super().__init__(*args, **kwargs)
        assert ces_mode in ('a1', 'a2'), ces_mode
        self.ces_mode = ces_mode

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, action_shape):
        self.storage = CESRolloutStorage(num_envs, num_transitions_per_env,
                                         actor_obs_shape, critic_obs_shape,
                                         action_shape, self.device)

    def process_env_step(self, rewards, dones, infos):
        ces = infos.get('ces') if isinstance(infos, dict) else None
        super().process_env_step(rewards, dones, infos)
        if ces is not None:
            self.storage.add_ces(ces['bdy'], ces['duty_credit'], ces['duty_kind'])

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_ces_returns(last_values, self.gamma, self.lam,
                                         randomize_boundaries=(self.ces_mode == 'a2'))

    def update(self):
        assert not self.actor_critic.is_recurrent, \
            'CESPPO minibatch grouping is implemented for feedforward policies'
        mean_value_loss = 0
        mean_surrogate_loss = 0
        generator = self.storage.segment_mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, \
            advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
