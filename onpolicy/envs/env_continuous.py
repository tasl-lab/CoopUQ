import gym
from gym import spaces
import numpy as np
from onpolicy.envs.env_core import EnvCore


class ContinuousActionEnv(object):
    """
    Wrapper for continuous action environment.
    """

    def __init__(self, config):
        self.env = EnvCore(config)
        self.num_agent = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim

        self.signal_action_dim = self.env.action_dim

        self.discrete_action_input = False

        self.movable = True

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        total_action_space = []
        self.max_speed = config.max_speed

        for agent in range(self.num_agent):
            u_action_space = spaces.Box(
                low=-self.max_speed,
                high=+self.max_speed,
                shape=(self.signal_action_dim,),
                dtype=np.float32,
            )

            if self.movable:
                total_action_space.append(u_action_space)

            self.action_space.append(total_action_space[0])

            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.signal_obs_dim,),
                    dtype=np.float32,
                )
            )

        self.share_observation_space = [
            spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32
            )
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        """
        Input actions dimension assumption:
        # actions shape = (5, 2, 5)
        # 5 threads of environment, there are 2 agents inside, and each agent's action is a 5-dimensional one_hot encoding
        """
        results = self.env.step(actions)
        obs, rews, dones, infos = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        self.env.close()

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass
