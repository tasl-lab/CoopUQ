try:
    import wandb          # only needed with --use_wandb
except ImportError:
    wandb = None
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer_cost import SharedReplayBufferCost

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class CRunner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.human_num = self.all_args.human_num
        self.max_steps = self.all_args.max_steps

        self.vis_seq = self.all_args.vis_seq
        self.gt_ogm_size = self.all_args.gt_ogm_size
        self.use_prediction = self.all_args.use_prediction
        self.uncertainty_type = self.all_args.uncertainty_type
        self.constrained = self.all_args.constrained
        self.prediction_model = config['prediction_model']

        self.resolution = self.all_args.resolution
        self.lidar_map_size = self.all_args.lidar_map_size
        self.n_points = self.all_args.n_points

        self.cost_limit = self.all_args.cost_limit
        self.init_lagrangian_multiplier = self.all_args.init_lagrangian_multiplier
        self.lambda_lr = self.all_args.lambda_lr
        self.min_lambda = self.all_args.min_lambda
        self.max_lambda = self.all_args.max_lambda

        self.center_radius = self.all_args.center_radius
        self.occupied_probability_threshold = self.all_args.occupied_probability_threshold
        self.time_step = self.all_args.time_step
        self.discomfort_factor = self.all_args.discomfort_factor
        self.center_y = self.lidar_map_size // 2
        self.center_x = self.lidar_map_size // 2
        y_indices, x_indices = np.ogrid[:self.lidar_map_size, :self.lidar_map_size]
        self.dist_from_center = np.sqrt((y_indices - self.center_y)**2 + (x_indices - self.center_x)**2)
        self.center_mask = self.dist_from_center <= self.center_radius
        self.entropy_cost_weight = self.all_args.entropy_cost_weight
        self.uq_threshold = self.all_args.uq_threshold

        self.eval_log_file = open(f"evaluating_{self.all_args.experiment_name}_seed_{self.all_args.seed}.txt", "a")

        self.experiment_name = self.all_args.experiment_name

        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        from onpolicy.algorithms.r_mappo.r_mappo_lag import R_MAPPO_Lagrangian as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_lag import R_MAPPOPolicy_Lagrangian as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        self.policy = Policy(self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], device = self.device)

        if self.model_dir is not None:
            self.restore(self.model_dir)

        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

        if self.model_dir is not None:
            self._restore_trainer_state(self.model_dir)

        self.buffer = SharedReplayBufferCost(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values, next_cost_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                    np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                    np.concatenate(self.buffer.rnn_states_cost_critic[-1]),
                                                    np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        next_cost_values = np.array(np.split(_t2n(next_cost_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, next_cost_values, self.trainer.value_normalizer,
                                    getattr(self.trainer, 'cost_value_normalizer', None))

    def train(self, update_actor=True):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer, update_actor=update_actor)
        self.buffer.after_update()
        return train_infos

    def save(self, episode=0):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
        policy_cost_critic = self.trainer.policy.cost_critic
        torch.save(policy_cost_critic.state_dict(), str(self.save_dir) + "/cost_critic.pt")
        self._save_trainer_state(str(self.save_dir))

    def _save_trainer_state(self, out_dir):
        """Extra training state needed to resume without a transient: the return
        normalisers' running statistics and the Lagrange multiplier."""
        state = {}
        vn = getattr(self.trainer, 'value_normalizer', None)
        cvn = getattr(self.trainer, 'cost_value_normalizer', None)
        if isinstance(vn, torch.nn.Module):
            state['value_normalizer'] = vn.state_dict()
        if isinstance(cvn, torch.nn.Module) and cvn is not vn:
            state['cost_value_normalizer'] = cvn.state_dict()
        lam = getattr(self.trainer, 'lagrangian_multiplier', None)
        if lam is not None:
            state['lagrangian_multiplier'] = float(lam.detach().cpu())
        torch.save(state, out_dir + "/trainer_state.pt")

    def _restore_trainer_state(self, model_dir):
        path = str(model_dir) + '/trainer_state.pt'
        if not os.path.exists(path):
            return
        state = torch.load(path, map_location='cpu')
        vn = getattr(self.trainer, 'value_normalizer', None)
        cvn = getattr(self.trainer, 'cost_value_normalizer', None)
        if 'value_normalizer' in state and isinstance(vn, torch.nn.Module):
            vn.load_state_dict(state['value_normalizer'])
        if 'cost_value_normalizer' in state and isinstance(cvn, torch.nn.Module) and cvn is not vn:
            cvn.load_state_dict(state['cost_value_normalizer'])
        print('restored trainer state (return normalisers) from', path)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt', map_location=self.device)
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        # the critics are only used for training; evaluation needs the actor alone
        if os.path.exists(str(self.model_dir) + '/critic.pt'):
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt', map_location=self.device)
            self.policy.critic.load_state_dict(policy_critic_state_dict)
        if os.path.exists(str(self.model_dir) + '/cost_critic.pt'):
            policy_cost_critic_state_dict = torch.load(str(self.model_dir) + '/cost_critic.pt', map_location=self.device)
            self.policy.cost_critic.load_state_dict(policy_cost_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
