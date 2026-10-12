import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner_cost import CRunner
from onpolicy.utils.ogm_util import transform_point_clouds, transform_lidar_to_ogm, shannon_entropy
from predictor.scripts.model import warp_features_translate
from collections import deque


def _t2n(x):
    return x.detach().cpu().numpy()


class UnityCRunner(CRunner):
    """Runner class to perform training, evaluation. and data collection in the Unity simulator. See parent class for details."""
    def __init__(self, config):
        super(UnityCRunner, self).__init__(config)

    def run(self):
        self.log_file = open(f"training_{self.all_args.experiment_name}_seed_{self.all_args.seed}.txt", "a")
        self.warmup()

        total_start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, cost_values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, rnn_states_cost_critic = self.collect(step)

                obs, rewards, dones, infos = self.envs.step(actions_env)

                ogm_obs, lidar, vec_obs, human_obs = self.prepare_obs(obs)

                ogm_pred, ogm_pred_entropy, costs = self.predict_occupancy(lidar, vec_obs)
                obs = np.concatenate((ogm_pred.reshape(self.n_rollout_threads, self.num_agents, -1), vec_obs.reshape(self.n_rollout_threads, self.num_agents, -1)), axis=-1)

                data = obs, rewards, costs, dones, infos, values, cost_values, actions, action_log_probs, rnn_states, rnn_states_critic, rnn_states_cost_critic

                self.insert(data)

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int((total_num_steps) / (end - total_start))))
                log_message = ("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                           .format(self.all_args.scenario_name,
                                   self.algorithm_name,
                                   self.experiment_name,
                                   episode,
                                   episodes,
                                   total_num_steps,
                                   self.num_env_steps,
                                   int((total_num_steps) / (end - total_start))))
                self.log_file.write(log_message)
                self.log_file.flush()
                env_infos = {}
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["average_episode_costs"] = np.mean(self.buffer.costs) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                print("average episode costs is {}".format(train_infos["average_episode_costs"]))
                avg_reward_message = "average episode rewards is {}\n".format(
                np.mean(self.buffer.rewards) * self.episode_length)
                avg_reward_message += "average episode costs is {}\n".format(
                np.mean(self.buffer.costs) * self.episode_length
                )
                self.log_file.write(avg_reward_message)
                self.log_file.flush()
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        self.log_file.close()
        self.eval_log_file.close()

    def warmup(self):
        obs = self.envs.reset()

        ogm_obs, lidar, vec_obs, human_obs = self.prepare_obs(obs)

        ogm_pred, ogm_pred_entropy, costs = self.predict_occupancy(lidar, vec_obs)
        obs = np.concatenate((ogm_pred.reshape(self.n_rollout_threads, self.num_agents, -1), vec_obs.reshape(self.n_rollout_threads, self.num_agents, -1)), axis=-1)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    def prepare_obs(self, obs):

        n_rollout_threads, num_agents, vis_seq, _ = obs.shape
        h_ogm, w_ogm = self.gt_ogm_size
        c_ogm = 1
        n_points = self.n_points

        ogm_obs = obs[:,:,:,:c_ogm*h_ogm*w_ogm].reshape(n_rollout_threads, num_agents, vis_seq, h_ogm, w_ogm)
        ogm_obs = np.rot90(ogm_obs, k=1, axes=(-2, -1))
        ogm_obs = (ogm_obs != 0).astype(np.uint8)
        lidar = obs[:,:,:,c_ogm*h_ogm*w_ogm:c_ogm*h_ogm*w_ogm+n_points].reshape(n_rollout_threads, num_agents, vis_seq, -1)
        n_vec = 6 * num_agents    # position and goal of the robot itself and of every other robot
        vec_obs = obs[:,:,:,c_ogm*h_ogm*w_ogm+n_points:c_ogm*h_ogm*w_ogm+n_vec+n_points].reshape(n_rollout_threads, num_agents, vis_seq, -1)
        human_obs = obs[:,:,:,c_ogm*h_ogm*w_ogm+n_vec+n_points:].reshape(n_rollout_threads, num_agents, vis_seq, -1)

        return ogm_obs, lidar, vec_obs, human_obs

    def predict_occupancy(self, lidar, vec_obs, is_eval=False):
        """Cooperative forecast for every robot, plus the per-step cost when constrained."""
        ogm_obs = []
        ogm_pred_entropy = []

        batch_size = self.n_eval_rollout_threads if is_eval else self.n_rollout_threads

        self._middle_fusion_prediction(lidar, vec_obs, batch_size,
                                       ogm_obs, ogm_pred_entropy)

        ogm_obs = np.array(np.stack(ogm_obs, axis=1))

        if self.uncertainty_type == 'entropy':
            ogm_pred_entropy = np.array(np.stack(ogm_pred_entropy, axis=1))
        if self.constrained:
            pred_max_intrusion_costs = np.zeros((batch_size, self.num_agents))
            entropy_max_intrusion_costs = np.zeros((batch_size, self.num_agents))

            for agent_id in range(self.num_agents):
                batch_min_pred_distances = np.full(batch_size, np.inf)
                batch_min_entropy_distances = np.full(batch_size, np.inf)

                cost_horizon = getattr(self.all_args, 'cost_horizon', 0) or (self.vis_seq - 2)
                for t in range(min(cost_horizon, ogm_obs.shape[2])):
                    ogm_batch = ogm_obs[:, agent_id, t]
                    ogm_batch_pred = ogm_batch[:, 0]
                    ogm_batch_pred_entropy = ogm_batch[:, 1]

                    occupied_mask = (ogm_batch_pred > self.occupied_probability_threshold) & (ogm_batch_pred < 1) & self.center_mask
                    entropy_mask = (ogm_batch_pred_entropy > self.uq_threshold) & self.center_mask

                    occupied_distances = np.where(occupied_mask, self.dist_from_center, np.inf)
                    entropy_distances = np.where(entropy_mask, self.dist_from_center, np.inf)

                    timestep_min_pred_distances = np.minimum(occupied_distances.min(axis=(1, 2)), self.center_radius)
                    timestep_min_entropy_distances = np.minimum(entropy_distances.min(axis=(1, 2)), self.center_radius)

                    batch_min_pred_distances = np.minimum(batch_min_pred_distances, timestep_min_pred_distances)
                    batch_min_entropy_distances = np.minimum(batch_min_entropy_distances, timestep_min_entropy_distances)

                pred_intrusion_depths = np.maximum(0, self.center_radius - batch_min_pred_distances)
                entropy_intrusion_depths = np.maximum(0, self.center_radius - batch_min_entropy_distances)

                pred_max_intrusion_costs[:, agent_id] = pred_intrusion_depths * self.time_step * self.discomfort_factor
                entropy_max_intrusion_costs[:, agent_id] = entropy_intrusion_depths * self.time_step * self.discomfort_factor

            pred_max_intrusion_costs = pred_max_intrusion_costs.reshape(batch_size, self.num_agents, -1)
            entropy_max_intrusion_costs = entropy_max_intrusion_costs.reshape(batch_size, self.num_agents, -1)

            occ_cost_weight = getattr(self.all_args, 'occ_cost_weight', 1.0)
            return ogm_obs, ogm_pred_entropy, occ_cost_weight * pred_max_intrusion_costs + self.entropy_cost_weight * entropy_max_intrusion_costs
        else:
            return ogm_obs, ogm_pred_entropy, None

    def _middle_fusion_prediction(self, lidar, vec_obs, batch_size,
                                  ogm_obs, ogm_pred_entropy):
        """Feature-level fusion path: per-agent ConvLSTM encode -> warp to ego frame ->
        MiddleFusion -> VAE+decoder. Mutates `ogm_obs` and `ogm_pred_entropy` in place.
        """
        per_agent_ogm = []
        per_agent_ogm_t = []
        for agent_id in range(self.num_agents):
            ax = vec_obs[:, agent_id, :, 2]
            ay = vec_obs[:, agent_id, :, 0]
            a_motion_t2d = np.stack([ax[:, [self.vis_seq - 1]] - ax,
                                     ay[:, [self.vis_seq - 1]] - ay], axis=-1)
            fl = np.zeros((batch_size, self.vis_seq, self.n_points, self.num_agents))
            fl[:, :, :, agent_id] = lidar[:, agent_id, :]
            fl[:, :, :, agent_id] = transform_point_clouds(fl[:, :, :, agent_id], a_motion_t2d)
            agent_ogm = transform_lidar_to_ogm(fl, map_size=self.lidar_map_size,
                                               resolution=self.resolution)
            per_agent_ogm.append(agent_ogm)
            per_agent_ogm_t.append(
                torch.tensor(agent_ogm, dtype=torch.float32, device=self.device)
            )

        per_agent_h = []
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                h_enc = self.prediction_model.encode_temporal(per_agent_ogm_t[agent_id])
                per_agent_h.append(h_enc)

        for ego_id in range(self.num_agents):
            ego_xs = vec_obs[:, ego_id, -1, 2]
            ego_ys = vec_obs[:, ego_id, -1, 0]
            agent_order = [ego_id] + [a for a in range(self.num_agents) if a != ego_id]
            warped_h = []
            warped_x = []
            with torch.no_grad():
                for agent_id in agent_order:
                    cur_ogm = per_agent_ogm_t[agent_id][:, -1:]
                    if agent_id == ego_id:
                        warped_h.append(per_agent_h[agent_id])
                        warped_x.append(cur_ogm)
                        continue
                    ax = vec_obs[:, agent_id, -1, 2]
                    ay = vec_obs[:, agent_id, -1, 0]
                    tx_pix = torch.tensor((ax - ego_xs) / self.resolution,
                                          dtype=torch.float32, device=self.device)
                    ty_pix = torch.tensor((ay - ego_ys) / self.resolution,
                                          dtype=torch.float32, device=self.device)
                    warped_h.append(
                        self.prediction_model.align_feature(
                            per_agent_h[agent_id], per_agent_h[ego_id], tx_pix, ty_pix)
                    )
                    warped_x.append(
                        warp_features_translate(cur_ogm, tx_pix, ty_pix)
                    )

                fused = self.prediction_model.fuse(
                    torch.stack(warped_h, dim=0))
                fused_x = torch.stack(warped_x, dim=0).amax(dim=0)
                n_samples = max(1, int(getattr(self.all_args, 'num_pred_samples', 1)))
                prediction_samples = torch.stack(
                    [self.prediction_model.decode_from_fused(fused, fused_x)[0]
                     for _ in range(n_samples)], dim=0)
                prediction = prediction_samples.mean(dim=0)

            ego_ogm_pred = prediction.detach().cpu().numpy()
            ego_ogm_pred = np.concatenate(
                (per_agent_ogm[ego_id][:, -1, None], ego_ogm_pred), axis=1
            )

            if self.uncertainty_type == 'entropy':
                ego_ogm_pred_entropy = shannon_entropy(prediction_samples).detach().cpu().numpy()
                ego_ogm_pred_entropy /= np.log(2)
                ego_ogm_pred_entropy = np.concatenate(
                    (np.zeros_like(ego_ogm_pred_entropy[:, :1]), ego_ogm_pred_entropy), axis=1
                )
                ogm_pred_entropy.append(ego_ogm_pred_entropy)
                ego_ogm_pred = np.stack([ego_ogm_pred, ego_ogm_pred_entropy], axis=2)

            ogm_obs.append(ego_ogm_pred)

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, cost_values, action, action_log_prob, rnn_states, rnn_states_critic, rnn_states_cost_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.rnn_states_cost_critic[step]),
                            np.concatenate(self.buffer.masks[step]))

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        cost_values = np.array(np.split(_t2n(cost_values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        rnn_states_cost_critic = np.array(np.split(_t2n(rnn_states_cost_critic), self.n_rollout_threads))

        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            actions_env=action.view(self.n_rollout_threads, self.num_agents, -1)

        return values, cost_values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, rnn_states_cost_critic

    def insert(self, data):
        obs, rewards, costs, dones, infos, values, cost_values, actions, action_log_probs, rnn_states, rnn_states_critic, rnn_states_cost_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        rnn_states_cost_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_cost_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, rnn_states_cost_critic, actions, action_log_probs, values, cost_values, rewards, costs, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):

        count = 0

        success_count = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        obstacle_collision_count = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        npc_collision_count = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        robot_collision_count = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        navigation_time_single = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        navigation_time_total = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        navigation_time_step_single = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        navigation_time_step_total = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        path_length_single = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        path_length_total = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        sharp_turn_count = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        danger_steps_episode = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        total_steps_episode  = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        sd_dist_sum_episode  = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        sd_danger_cnt_episode = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        itr_sum = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        itr_episode_cnt = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        sd_sum = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        sd_episode_cnt = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        future_k = 5
        intrude_radius = 0.5

        robot_pos_past = [[deque(maxlen=future_k+1) for _ in range(self.num_agents)] for _ in range(self.n_eval_rollout_threads)]
        human_pos_past = [deque(maxlen=future_k+1) for _ in range(self.n_eval_rollout_threads)]

        previous_velocity = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        velocity_changes_single = np.zeros((self.n_eval_rollout_threads, self.num_agents))
        velocity_changes_total = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        timeout_count = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        end_counts = np.zeros((self.n_eval_rollout_threads, self.num_agents))


        previous_pos = np.zeros((self.n_eval_rollout_threads, self.num_agents, 3))

        hold_k = int(self.all_args.eval_hold_steps)
        hold = np.full((self.n_eval_rollout_threads, self.num_agents), hold_k)

        # --eval_traj_path: per-step positions of every thread, drawn by scripts/render_2d.py
        traj_path = self.all_args.eval_traj_path
        traj = {'robot': [], 'goal': [], 'human': [], 'human_yaw': [], 'done': [], 'reward': []}

        eval_obs = self.eval_envs.reset()
        ogm_obs, lidar, vec_obs, human_obs = self.prepare_obs(eval_obs)

        ogm_pred, ogm_pred_entropy, costs = self.predict_occupancy(lidar, vec_obs, is_eval=True)
        eval_obs = np.concatenate((ogm_pred.reshape(self.n_eval_rollout_threads, self.num_agents, -1), vec_obs.reshape(self.n_eval_rollout_threads, self.num_agents, -1)), axis=-1)

        for eval_episode in range(self.all_args.eval_episodes):

            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for eval_step in range(self.episode_length):
                count += 1
                self.trainer.prep_rollout()
                eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

                if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[0].shape):
                        eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                        if i == 0:
                            eval_actions_env = eval_uc_actions_env
                        else:
                            eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
                elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                    eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
                else:
                    eval_actions_env = eval_action.view(self.n_eval_rollout_threads, self.num_agents, -1)

                if hold_k > 0:
                    eval_actions_env = eval_actions_env.clone()
                    eval_actions_env[torch.from_numpy(hold > 0).to(eval_actions_env.device)] = 0.0
                    hold = np.maximum(hold - 1, 0)
                eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
                if hold_k > 0:
                    hold[np.asarray(eval_dones).reshape(hold.shape).astype(bool)] = hold_k
                ogm_obs, lidar, vec_obs, human_obs = self.prepare_obs(eval_obs)
                if traj_path:
                    # planar positions as (x, z); pedestrian yaw in degrees (0 = +z, clockwise towards +x)
                    peds = human_obs[:, 0, -1].reshape(self.n_eval_rollout_threads, -1, 9)
                    traj['robot'].append(vec_obs[:, :, -1, [0, 2]].astype(np.float32))
                    traj['goal'].append(vec_obs[:, :, -1, [3, 5]].astype(np.float32))
                    traj['human'].append(peds[:, :, [0, 2]].astype(np.float32))
                    traj['human_yaw'].append(peds[:, :, 4].astype(np.float32))
                    traj['done'].append(np.asarray(eval_dones).reshape(self.n_eval_rollout_threads, -1).astype(bool))
                    traj['reward'].append(np.asarray(eval_rewards).reshape(self.n_eval_rollout_threads, -1).astype(np.float32))

                ogm_pred, ogm_pred_entropy, costs = self.predict_occupancy(lidar, vec_obs, is_eval=True)
                eval_obs = np.concatenate((ogm_pred.reshape(self.n_eval_rollout_threads, self.num_agents, -1), vec_obs.reshape(self.n_eval_rollout_threads, self.num_agents, -1)), axis=-1)

                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

                for rollout in range(self.n_eval_rollout_threads):
                    human_pos_past[rollout].append(human_obs[rollout, 0, -1, :].reshape(-1, 9)[:, [0, 2]])
                    for agent_id in range(self.num_agents):

                        if eval_actions_env[rollout,agent_id, 1] >= 0.8 or eval_actions_env[rollout,agent_id, 1] <= -0.8:
                                sharp_turn_count[rollout, agent_id] += 1

                        current_pos = eval_obs[rollout, agent_id, -self.num_agents*6:-self.num_agents*6+3]

                        total_steps_episode[rollout, agent_id] += 1

                        robot_pos_past[rollout][agent_id].append(current_pos[[0, 2]])

                        danger = False
                        min_dist = None

                        if not danger and len(robot_pos_past[rollout][agent_id]) == future_k + 1:

                            for prev_idx in range(future_k+1):
                                dists = np.linalg.norm(human_pos_past[rollout][prev_idx] - robot_pos_past[rollout][agent_id][0], axis=1)
                                if min_dist is None or dists.min() < min_dist:
                                    min_dist = dists.min()
                            if min_dist < intrude_radius:
                                danger = True

                        if danger:

                            danger_steps_episode[rollout, agent_id] += 1
                            sd_dist_sum_episode[rollout, agent_id] += float(min_dist)
                            sd_danger_cnt_episode[rollout, agent_id] += 1

                        if not eval_dones[rollout, agent_id] and previous_pos[rollout, agent_id].any():
                            navigation_time_single[rollout][agent_id] += 0.2
                            navigation_time_step_single[rollout][agent_id] += 1
                            path_length_single[rollout][agent_id] += np.linalg.norm(current_pos - previous_pos[rollout, agent_id])

                            velocity = np.linalg.norm(current_pos - previous_pos[rollout, agent_id]) / 0.2

                            if previous_velocity[rollout, agent_id] > 0:
                                delta_v = abs(velocity - previous_velocity[rollout, agent_id])
                                velocity_changes_single[rollout, agent_id] += delta_v

                            previous_velocity[rollout, agent_id] = velocity

                        previous_pos[rollout, agent_id] = current_pos

                        if eval_dones[rollout, agent_id]:
                            end_counts[rollout, agent_id] += 1

                            velocity_changes_total[rollout, agent_id] += velocity_changes_single[rollout, agent_id]

                            if eval_rewards[rollout, agent_id] == 10.0:
                                success_count[rollout, agent_id] += 1
                                navigation_time_total[rollout, agent_id] += navigation_time_single[rollout][agent_id]
                                navigation_time_step_total[rollout, agent_id] += navigation_time_step_single[rollout][agent_id]
                                path_length_total[rollout, agent_id] += path_length_single[rollout][agent_id]

                            elif eval_rewards[rollout, agent_id] == -20:
                                obstacle_collision_count[rollout, agent_id] += 1
                            elif eval_rewards[rollout, agent_id] == -15:
                                npc_collision_count[rollout, agent_id] += 1
                            elif eval_rewards[rollout, agent_id] == -10:
                                robot_collision_count[rollout, agent_id] += 1
                            elif eval_rewards[rollout, agent_id] == 0.0:
                                timeout_count[rollout, agent_id] += 1

                            if total_steps_episode[rollout, agent_id] > 0:
                                ep_itr = danger_steps_episode[rollout, agent_id] / total_steps_episode[rollout, agent_id]
                                itr_sum[rollout, agent_id] += ep_itr
                                itr_episode_cnt[rollout, agent_id] += 1

                            if sd_danger_cnt_episode[rollout, agent_id] > 0:
                                ep_sd = sd_dist_sum_episode[rollout, agent_id] / sd_danger_cnt_episode[rollout, agent_id]
                                sd_sum[rollout, agent_id] += ep_sd
                                sd_episode_cnt[rollout, agent_id] += 1

                            danger_steps_episode[rollout, agent_id] = 0
                            total_steps_episode[rollout, agent_id] = 0
                            sd_dist_sum_episode[rollout, agent_id] = 0
                            sd_danger_cnt_episode[rollout, agent_id] = 0

                            robot_pos_past[rollout][agent_id].clear()

                            navigation_time_step_single[rollout][agent_id] = 0
                            navigation_time_single[rollout][agent_id] = 0
                            path_length_single[rollout][agent_id] = 0
                            velocity_changes_single[rollout, agent_id] = 0
                            previous_pos[rollout, agent_id] = np.zeros(3)
                            previous_velocity[rollout, agent_id] = 0

        log_message = f"total_num_steps: {total_num_steps}\n"
        log_message += f'success_count: {success_count}, obstacle_collision_count: {obstacle_collision_count}, npc_collision_count: {npc_collision_count}, robot_collision_count: {robot_collision_count}, timeout_count: {timeout_count}\n'
        log_message += f"end counts: {end_counts}\n"
        end_counts[end_counts == 0] += 1
        success_rate = np.mean(success_count/end_counts, axis=0)
        obstacle_collision_rate = np.mean(obstacle_collision_count/end_counts, axis=0)
        npc_collision_rate = np.mean(npc_collision_count/end_counts, axis=0)
        robot_collision_rate = np.mean(robot_collision_count/end_counts, axis=0)
        timeout_rate = np.mean(timeout_count/end_counts, axis=0)

        navigation_time = np.mean(navigation_time_total/success_count, axis=0)
        navigation_time_step = np.mean(navigation_time_step_total/success_count, axis=0)
        path_length = np.mean(path_length_total/success_count, axis=0)
        velocity_change = np.mean(velocity_changes_total/end_counts, axis=0)
        sharp_turn_rate = np.mean(sharp_turn_count/count, axis=0)

        itr_episode_cnt_safe = np.where(itr_episode_cnt == 0, 1, itr_episode_cnt)
        sd_episode_cnt_safe  = np.where(sd_episode_cnt == 0, 1, sd_episode_cnt)

        itr_per_thread = itr_sum / itr_episode_cnt_safe
        sd_per_thread  = sd_sum / sd_episode_cnt_safe

        itr_final = np.mean(itr_per_thread, axis=0) * 100.0
        sd_final  = np.mean(sd_per_thread, axis=0)

        for agent_id in range(self.num_agents):
            log_message += (
                f"Agent {agent_id} - SR: {np.round(success_rate[agent_id], 2)}, "
                f"obstacle_CR: {np.round(obstacle_collision_rate[agent_id], 2)}, "
                f"npc_CR: {np.round(npc_collision_rate[agent_id], 2)}, "
                f"robot_CR: {np.round(robot_collision_rate[agent_id], 2)}, "
                f"TR: {np.round(timeout_rate[agent_id], 2)}, "
                f"ITR: {np.round(itr_final[agent_id], 2)}, "
                f"SD: {np.round(sd_final[agent_id], 2)}\n"
            )
        for agent_id in range(self.num_agents):
            log_message += (
                f"Agent {agent_id} - ANT: {np.round(navigation_time[agent_id],2)}, "
                f"ANT_step: {np.round(navigation_time_step[agent_id],2)}, "
                f"PL: {np.round(path_length[agent_id],2)}, "
                f"STR: {np.round(sharp_turn_rate[agent_id],2)}, "
                f"AA: {np.round(velocity_change[agent_id],2)}\n "
            )
        self.eval_log_file.write(log_message)
        self.eval_log_file.flush()
        if traj_path:
            np.savez_compressed(traj_path, scene=np.array(self.all_args.eval_scenes),
                                **{k: np.stack(v, axis=1) for k, v in traj.items()})
