from typing import List, Dict

import numpy as np
from gymnasium import Space

from xuance.common import MARL_OffPolicyBuffer_RNN, space2shape


class VBC_Buffer(MARL_OffPolicyBuffer_RNN):
    def __init__(self,
                 agent_keys: List[str],
                 state_space: Dict[str, Space] = None,
                 obs_space: Dict[str, Dict[str, Space]] = None,
                 act_space: Dict[str, Dict[str, Space]] = None,
                 n_envs: int = 1,
                 buffer_size: int = 1,
                 batch_size: int = 1,
                 max_episode_steps: int = 1,
                 **kwargs):
        self.msg_shape = {k: (kwargs['dim_message'],) for k in agent_keys}
        super(VBC_Buffer, self).__init__(agent_keys, state_space, obs_space, act_space,
                                    n_envs, buffer_size, batch_size, max_episode_steps, **kwargs)

    def clear(self):
        self.data = {
            'obs': {k: np.zeros((self.buffer_size, self.max_eps_len + 1) + self.obs_shape[k], dtype=np.float32)
                    for k in self.agent_keys},
            'actions': {k: np.zeros((self.buffer_size, self.max_eps_len) + self.act_shape[k], dtype=np.float32)
                        for k in self.agent_keys},
            'message': {k: np.zeros((self.buffer_size, self.max_eps_len + 1) + self.msg_shape[k], np.float32)
                        for k in self.agent_keys},
            'rewards': {k: np.zeros((self.buffer_size, self.max_eps_len), dtype=np.float32) for k in self.agent_keys},
            'terminals': {k: np.zeros((self.buffer_size, self.max_eps_len), dtype=np.bool_) for k in self.agent_keys},
            'agent_mask': {k: np.zeros((self.buffer_size, self.max_eps_len), dtype=np.bool_) for k in self.agent_keys},
            'filled': np.zeros((self.buffer_size, self.max_eps_len), dtype=np.bool_),
        }

        if self.store_global_state:
            state_shape = (self.buffer_size, self.max_eps_len + 1) + space2shape(self.state_space)
            self.data.update({'state': np.zeros(state_shape, dtype=np.float32)})
        if self.use_actions_mask:
            self.data.update({
                'avail_actions': {k: np.zeros((self.buffer_size, self.max_eps_len + 1) + self.avail_actions_shape[k],
                                              dtype=np.bool_) for k in self.agent_keys}})
        self.ptr, self.size = 0, 0

    def clear_episodes(self):
        self.episode_data = {
            'obs': {k: np.zeros((self.n_envs, self.max_eps_len + 1) + self.obs_shape[k], dtype=np.float32)
                    for k in self.agent_keys},
            'actions': {k: np.zeros((self.n_envs, self.max_eps_len) + self.act_shape[k], dtype=np.float32)
                        for k in self.agent_keys},
            'message': {k: np.zeros((self.n_envs, self.max_eps_len + 1) + self.msg_shape[k], np.float32)
                        for k in self.agent_keys},
            'rewards': {k: np.zeros((self.n_envs, self.max_eps_len), dtype=np.float32) for k in self.agent_keys},
            'terminals': {k: np.zeros((self.n_envs, self.max_eps_len), dtype=np.bool_) for k in self.agent_keys},
            'agent_mask': {k: np.zeros((self.n_envs, self.max_eps_len), dtype=np.bool_) for k in self.agent_keys},
            'filled': np.zeros((self.n_envs, self.max_eps_len), dtype=np.bool_),
        }

        if self.store_global_state:
            state_shape = (self.n_envs, self.max_eps_len + 1) + space2shape(self.state_space)
            self.episode_data.update({'state': np.zeros(state_shape, dtype=np.float32)})
        if self.use_actions_mask:
            self.episode_data.update({
                "avail_actions": {k: np.zeros((self.n_envs, self.max_eps_len + 1) + self.avail_actions_shape[k],
                                              dtype=np.bool_) for k in self.agent_keys}})

    def finish_path(self, i_env, **terminal_data):
        env_step = terminal_data['episode_step']
        # Store terminal data into self.episode_data.
        if self.store_global_state:
            self.episode_data['state'][i_env, env_step] = terminal_data['state']
        for agt_key in self.agent_keys:
            self.episode_data['obs'][agt_key][i_env, env_step] = terminal_data['obs'][agt_key]
            self.episode_data['message'][agt_key][i_env, env_step] = terminal_data['message'][agt_key]
            if self.use_actions_mask:
                self.episode_data['avail_actions'][agt_key][i_env, env_step] = terminal_data['avail_actions'][agt_key]
        # Store the episode data of ith env into self.data.
        self.store_episodes(i_env)