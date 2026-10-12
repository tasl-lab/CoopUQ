from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
from collections import deque
from onpolicy.envs.lazy_frame import LazyFrames

class UnityEnvWrapper:
    def __init__(self, unity_env_path, base_port, no_graphics, continuous_action=True, rank=0, vis_seq=3, num_agents=3, obs_size=0):
        self.unity_env = UnityEnvironment(file_name=unity_env_path, base_port=base_port+rank, no_graphics=no_graphics)
        self.unity_env.reset()
        self.group_name = list(self.unity_env.behavior_specs.keys())[0]
        self.group_spec = self.unity_env.behavior_specs[self.group_name]
        self.continuous_action = continuous_action
        self.vis_seq = vis_seq
        self.num_agents = num_agents
        # Builds with different numbers of pedestrians have observations of different widths. With
        # obs_size > 0 they are padded to that width; the padded pedestrian slots are placed far away.
        self.obs_size = obs_size

        self.action_space = self.group_spec.action_spec

        self.observation_space = self.group_spec.observation_specs[0]

        self.lz4_compress = True
        self.frames = {i: deque(maxlen=self.vis_seq) for i in range(self.num_agents)}

    def _pad(self, obs):
        if self.obs_size <= len(obs):
            return obs
        return np.concatenate([obs, np.full(self.obs_size - len(obs), 1e4, dtype=obs.dtype)])

    def reset(self):
        self.unity_env.reset()
        decision_steps, _ = self.unity_env.get_steps(self.group_name)
        observations = []

        for agent_id in decision_steps.agent_id:
            obs = self._pad(decision_steps[agent_id].obs[0])
            for _ in range(self.vis_seq):
                self.frames[agent_id].append(obs)

        for agent_id in decision_steps.agent_id:
            observations.append(LazyFrames(list(self.frames[agent_id]), self.lz4_compress))

        return observations

    def step(self, actions):
        if self.continuous_action:
            action_tuple = ActionTuple(continuous=actions)
        else:
            action_tuple = ActionTuple(discrete=actions)

        self.unity_env.set_actions(self.group_name, action_tuple)
        self.unity_env.step()

        decision_steps, terminal_steps = self.unity_env.get_steps(self.group_name)

        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        processed_agent_ids = set()
        for agent_id in terminal_steps.agent_id:
            self.frames[agent_id].append(self._pad(terminal_steps[agent_id].obs[0]))
            observations[agent_id] = LazyFrames(list(self.frames[agent_id]), self.lz4_compress)
            reward = terminal_steps[agent_id].reward
            rewards[agent_id] = [reward]
            dones[agent_id] = True
            processed_agent_ids.add(agent_id)

        for agent_id in decision_steps.agent_id:
            if agent_id not in processed_agent_ids:
                self.frames[agent_id].append(self._pad(decision_steps[agent_id].obs[0]))
                observations[agent_id] = LazyFrames(list(self.frames[agent_id]), self.lz4_compress)
                rewards[agent_id] = [decision_steps[agent_id].reward]
                dones[agent_id] = False
                infos[agent_id] = {}

        observations = [observations[key] for key in sorted(observations.keys())]
        rewards = [rewards[key] for key in sorted(rewards.keys())]
        dones = [dones[key] for key in sorted(dones.keys())]
        infos = [infos[key] for key in sorted(infos.keys())]

        return [observations, rewards, dones, infos]

    def close(self):
        self.unity_env.close()
