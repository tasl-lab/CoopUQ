from collections import deque
import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check

class R_MAPPO_Lagrangian():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self.cost_limit = args.cost_limit
        self.lagrangian_multiplier = torch.tensor(args.init_lagrangian_multiplier,
                                                 device=device,
                                                 requires_grad=True)
        self.lambda_lr = args.lambda_lr
        self.min_lambda = args.min_lambda
        self.max_lambda = args.max_lambda
        self.lambda_optimizer = torch.optim.Adam([self.lagrangian_multiplier], lr=self.lambda_lr)
        self.episode_length = args.episode_length
        self.use_cost_adv = getattr(args, 'use_cost_adv', False)

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
        self.cost_value_normalizer = self.value_normalizer

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch, normalizer=None):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        normalizer = normalizer if normalizer is not None else self.value_normalizer
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            normalizer.update(return_batch)
            error_clipped = normalizer.normalize(return_batch) - value_pred_clipped
            error_original = normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        if len(sample) == 17:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, rnn_states_cost_critic_batch, actions_batch, \
            value_preds_batch, cost_value_preds_batch, return_batch, cost_return_batch, mean_cost_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, cost_adv_targ, available_actions_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, rnn_states_cost_critic_batch, actions_batch, \
            value_preds_batch, cost_value_preds_batch, return_batch, cost_return_batch, mean_cost_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, cost_adv_targ, available_actions_batch, _ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        cost_adv_targ = check(cost_adv_targ if self.use_cost_adv else adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        cost_value_preds_batch = check(cost_value_preds_batch).to(**self.tpdv)
        cost_return_batch = check(cost_return_batch).to(**self.tpdv)
        mean_cost_batch = check(mean_cost_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        values, cost_values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch,
                                                                              rnn_states_batch,
                                                                              rnn_states_critic_batch,
                                                                              rnn_states_cost_critic_batch,
                                                                              actions_batch,
                                                                              masks_batch,
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        cost_surrogate = mean_cost_batch.detach().mean()

        penalty = torch.clamp(self.lagrangian_multiplier, self.min_lambda, self.max_lambda)
        penalty = check(penalty).to(**self.tpdv)

        adv_targ = (adv_targ - penalty * cost_adv_targ) / (1 + penalty)

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        cost_value_loss = self.cal_value_loss(cost_values, cost_value_preds_batch, cost_return_batch, active_masks_batch,
                                              normalizer=self.cost_value_normalizer)

        self.policy.cost_critic_optimizer.zero_grad()
        (cost_value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            cost_critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.cost_critic.parameters(), self.max_grad_norm)
        else:
            cost_critic_grad_norm = get_gard_norm(self.policy.cost_critic.parameters())

        self.policy.cost_critic_optimizer.step()


        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, cost_value_loss, cost_critic_grad_norm, penalty.item(), cost_surrogate.item()

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
            cost_advantages = buffer.cost_returns[:-1] - self.cost_value_normalizer.denormalize(buffer.cost_value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
            cost_advantages = buffer.cost_returns[:-1] - buffer.cost_value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        cost_advantages_copy = cost_advantages.copy()
        cost_advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_cost_advantages = np.nanmean(cost_advantages_copy)
        std_cost_advantages = np.nanstd(cost_advantages_copy)
        cost_advantages = (cost_advantages - mean_cost_advantages) / (std_cost_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['cost_value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['cost_critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['lagrangian_multiplier'] = 0
        train_info['cost_surrogate'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, cost_advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, \
                 cost_value_loss, cost_critic_grad_norm, lagrangian_value, cost_surrogate \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['cost_value_loss'] += cost_value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['cost_critic_grad_norm'] += cost_critic_grad_norm
                train_info['ratio'] += imp_weights.mean().detach()

                train_info['lagrangian_multiplier'] += lagrangian_value
                train_info['cost_surrogate'] += cost_surrogate


        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        mean_episode_cost = float(np.mean(buffer.costs)) * self.episode_length
        train_info['mean_step_cost'] = float(np.mean(buffer.costs))
        self.lambda_optimizer.zero_grad()
        lambda_loss = -self.lagrangian_multiplier * (mean_episode_cost - self.cost_limit)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        with torch.no_grad():
            self.lagrangian_multiplier.clamp_(self.min_lambda, self.max_lambda)
        train_info['mean_episode_cost'] = mean_episode_cost
        train_info['lagrangian_multiplier'] = self.lagrangian_multiplier.item()

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.cost_critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.cost_critic.eval()
