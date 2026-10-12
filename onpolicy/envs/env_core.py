import numpy as np

from onpolicy.envs.unity_env_wrapper import UnityEnvWrapper

class EnvCore(object):
    def __init__(self, config):
        self.env = UnityEnvWrapper(
            unity_env_path=config.unity_env_path,
            base_port=config.base_port,
            no_graphics=config.no_graphics,
            continuous_action=config.continuous_action,
            rank=config.rank,
            vis_seq=config.vis_seq,
            num_agents=config.num_agents,
            # occupancy crop, LiDAR, 6 values per robot, 9 per pedestrian slot
            obs_size=(config.lidar_map_size ** 2 + config.n_points + 6 * config.num_agents + 9 * config.pad_humans
                      if getattr(config, 'pad_humans', 0) else 0)
        )
        self.agent_num = config.num_agents
        self.action_dim = 2
        if config.uncertainty_type == "entropy":
            self.obs_dim = (config.lidar_map_size*config.lidar_map_size*2 + config.ego_vec_size + config.other_vec_size) * config.vis_seq + config.lidar_map_size*config.lidar_map_size*2
        elif config.use_prediction:
            self.obs_dim = (config.lidar_map_size*config.lidar_map_size + config.ego_vec_size + config.other_vec_size) * config.vis_seq + config.lidar_map_size*config.lidar_map_size
        else:
            self.obs_dim = (config.lidar_map_size*config.lidar_map_size + config.ego_vec_size + config.other_vec_size) * config.vis_seq

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        actions=np.array(actions.cpu())
        return self.env.step(actions)

    def close(self):
        self.env.close()
