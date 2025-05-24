from copy import deepcopy
from operator import itemgetter
from typing import List, Optional, Union

import numpy as np
import torch
from argparse import Namespace

from torch.nn import ModuleDict

from examples.vbc.vbc_buffer import VBC_Buffer
from examples.vbc.vbc_comm import VBC_Comm
from examples.vbc.vbc_learner import VBC_Learner
from examples.vbc.vbc_policy import VBC_Policy
from xuance.environment import DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv
from xuance.torch import Module, REGISTRY_Learners
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy, QMIX_mixer
from xuance.torch.agents import OffPolicyMARLAgents


class VBC_Agents(OffPolicyMARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecMultiAgentEnv, SubprocVecMultiAgentEnv]):
        super(VBC_Agents, self).__init__(config, envs)
        self.state_space = envs.state_space
        self.use_global_state = True

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.e_greedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

        self.communicator = self._build_communicator(config=self.config)
        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy)

    def _build_communicator(self, config: Namespace) -> Module:
        # build communicator
        communicator = ModuleDict()
        for key in self.model_keys:
            hidden_shape = self.config.recurrent_hidden_size
            action_dim = self.envs.action_space[key].n
            input_communicator = dict(
                hidden_sizes=hidden_shape,
                action_dim=action_dim,
                threshold_2=self.config.threshold_2,
                model_keys=self.model_keys,
                n_agents=self.n_agents,
                device=self.device)
            communicator[key] = VBC_Comm(**input_communicator)
        return communicator


    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]

        # build communicator
        communicator = self.communicator

        # build representations
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        dim_state = self.state_space.shape[-1]
        mixer = QMIX_mixer(dim_state, self.config.hidden_dim_mixing_net,
                           self.config.hidden_dim_hyper_net, self.n_agents, self.device)
        if self.config.policy == "VBC_Policy":
            REGISTRY_Policy["VBC_Policy"] = VBC_Policy
            policy = REGISTRY_Policy["VBC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                mixer=mixer, hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=self.device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None, communicator=communicator, threshold_1=self.config.threshold_1)
        else:
            raise AttributeError(f"VBC_Policy currently does not support the policy named {self.config.policy}.")

        return policy

    def _build_memory(self):
        """Build replay buffer for models training
        """
        if self.use_actions_mask:
            avail_actions_shape = {key: (self.action_space[key].n,) for key in self.agent_keys}
        else:
            avail_actions_shape = None
        input_buffer = dict(agent_keys=self.agent_keys,
                            state_space=self.state_space if self.use_global_state else None,
                            obs_space=self.observation_space,
                            act_space=self.action_space,
                            n_envs=self.n_envs,
                            buffer_size=self.buffer_size,
                            batch_size=self.batch_size,
                            avail_actions_shape=avail_actions_shape,
                            use_actions_mask=self.use_actions_mask,
                            max_episode_steps=self.episode_length,
                            dim_message=self.config.dim_message)
        Buffer = VBC_Buffer
        return Buffer(**input_buffer)

    def _build_learner(self, *args):
        REGISTRY_Learners["VBC_Learner"] = VBC_Learner
        return REGISTRY_Learners["VBC_Learner"](*args)

    def action(self,
               obs_dict: List[dict],
               avail_actions_dict: Optional[List[dict]] = None,
               rnn_hidden: Optional[dict] = None,
               test_mode: Optional[bool] = False,
               message: Optional[dict] = None,
               **kwargs):
        batch_size = len(obs_dict)
        obs_input, agents_id, avail_actions_input = self._build_inputs(obs_dict, avail_actions_dict)
        message_input = {k: message[k].transpose(1, 0, 2) for k in self.model_keys}
        hidden_state, actions, _ = self.policy(observation=obs_input,
                                               agent_ids=agents_id,
                                               avail_actions=avail_actions_input,
                                               rnn_hidden=rnn_hidden,
                                               message_input=message_input)

        if self.use_parameter_sharing:
            key = self.agent_keys[0]
            actions_out = actions[key].reshape([batch_size, self.n_agents]).cpu().detach().numpy()
            actions_dict = [{k: actions_out[e, i] for i, k in enumerate(self.agent_keys)} for e in range(batch_size)]
        else:
            actions_out = {k: actions[k].reshape(batch_size).cpu().detach().numpy() for k in self.agent_keys}
            actions_dict = [{k: actions_out[k][i] for k in self.agent_keys} for i in range(batch_size)]

        if not test_mode:  # get random actions
            actions_dict = self.exploration(batch_size, actions_dict, avail_actions_dict)
        message = {k: hidden_state[k][0].detach().cpu().numpy() for k in self.model_keys}
        return {"hidden_state": hidden_state, "actions": actions_dict, "message": message}

    def init_message_item(self, i_env: int, message: dict) -> dict:
        for key in self.model_keys:
            message[key][:, i_env, :] = 0
        return message

    def store_experience(self, obs_dict, avail_actions, actions_dict, obs_next_dict, avail_actions_next,
                         rewards_dict, terminals_dict, info, message=None, **kwargs):
        experience_data = {
            'obs': {k: np.array([data[k] for data in obs_dict]) for k in self.agent_keys},
            'actions': {k: np.array([data[k] for data in actions_dict]) for k in self.agent_keys},
            'message': message,
            'obs_next': {k: np.array([data[k] for data in obs_next_dict]) for k in self.agent_keys},
            'rewards': {k: np.array([data[k] for data in rewards_dict]) for k in self.agent_keys},
            'terminals': {k: np.array([data[k] for data in terminals_dict]) for k in self.agent_keys},
            'agent_mask': {k: np.array([data['agent_mask'][k] for data in info]) for k in self.agent_keys},
        }
        if self.use_rnn:
            experience_data['episode_steps'] = np.array([data['episode_step'] - 1 for data in info])
        if self.use_global_state:
            experience_data['state'] = np.array(kwargs['state'])
            experience_data['state_next'] = np.array(kwargs['next_state'])
        if self.use_actions_mask:
            experience_data['avail_actions'] = {k: np.array([data[k] for data in avail_actions])
                                                for k in self.agent_keys}
            experience_data['avail_actions_next'] = {k: np.array([data[k] for data in avail_actions_next])
                                                     for k in self.agent_keys}
        self.memory.store(**experience_data)


    def run_episodes(self, env_fn=None, n_episodes: int = 1, test_mode: bool = False):
        envs = self.envs if env_fn is None else env_fn()
        num_envs = envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        episode_count, scores, best_score = 0, [], -np.inf
        obs_dict, info = envs.reset()
        state = envs.buf_state.copy() if self.use_global_state else None
        avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                images = envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)
        else:
            if self.use_rnn:
                self.memory.clear_episodes()
        rnn_hidden = self.init_rnn_hidden(num_envs)
        message = {k: self.communicator[k].init_message(num_envs) for k in self.model_keys}

        while episode_count < n_episodes:
            step_info = {}
            policy_out = self.action(obs_dict=obs_dict,
                                     avail_actions_dict=avail_actions,
                                     rnn_hidden=rnn_hidden,
                                     test_mode=test_mode,
                                     message=message)
            rnn_hidden, actions_dict, message = policy_out['hidden_state'], policy_out['actions'], policy_out['message']
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = envs.step(actions_dict)
            next_state = envs.buf_state.copy() if self.use_global_state else None
            next_avail_actions = envs.buf_avail_actions if self.use_actions_mask else None
            if test_mode:
                if self.config.render_mode == "rgb_array" and self.render:
                    images = envs.render(self.config.render_mode)
                    for idx, img in enumerate(images):
                        videos[idx].append(img)
            else:
                self.store_experience(obs_dict, avail_actions, actions_dict, next_obs_dict, next_avail_actions,
                                      rewards_dict, terminated_dict, info, message,
                                      **{'state': state, 'next_state': next_state})
            obs_dict = deepcopy(next_obs_dict)
            if self.use_global_state:
                state = deepcopy(next_state)
            if self.use_actions_mask:
                avail_actions = deepcopy(next_avail_actions)

            for i in range(num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    episode_count += 1
                    obs_dict[i] = info[i]["reset_obs"]
                    envs.buf_obs[i] = info[i]["reset_obs"]
                    message = self.init_message_item(i_env=i, message=message)
                    if self.use_global_state:
                        state = info[i]["reset_state"]
                        self.envs.buf_state[i] = info[i]["reset_state"]
                    if self.use_actions_mask:
                        avail_actions[i] = info[i]["reset_avail_actions"]
                        envs.buf_avail_actions[i] = info[i]["reset_avail_actions"]
                    if self.use_rnn:
                        rnn_hidden = self.init_hidden_item(i_env=i, rnn_hidden=rnn_hidden)
                        if not test_mode:
                            # Todo store message[i] with dim=(64,)
                            store_msg = {k: message[k][:, i, :].flatten() for k in self.model_keys}
                            terminal_data = {'obs': next_obs_dict[i],
                                             'message': store_msg,
                                             'episode_step': info[i]['episode_step']}
                            if self.use_global_state:
                                terminal_data['state'] = next_state[i]
                            if self.use_actions_mask:
                                terminal_data['avail_actions'] = next_avail_actions[i]
                            self.memory.finish_path(i, **terminal_data)
                    episode_score = float(np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"])))
                    scores.append(episode_score)
                    if test_mode:
                        if best_score < episode_score:
                            best_score = episode_score
                            episode_videos = videos[i].copy()
                        if self.config.test_mode:
                            print("Episode: %d, Score: %.2f" % (episode_count, episode_score))
                    else:
                        if self.use_wandb:
                            step_info["Train-Results/Episode-Steps/env-%d" % i] = info[i]["episode_step"]
                            step_info["Train-Results/Episode-Rewards/env-%d" % i] = info[i]["episode_score"]
                        else:
                            step_info["Train-Results/Episode-Steps"] = {"env-%d" % i: info[i]["episode_step"]}
                            step_info["Train-Results/Episode-Rewards"] = {
                                "env-%d" % i: np.mean(itemgetter(*self.agent_keys)(info[i]["episode_score"]))}
                        self.current_step += info[i]["episode_step"]
                        self.log_infos(step_info, self.current_step)
                        self._update_explore_factor()

        if test_mode:
            if self.config.render_mode == "rgb_array" and self.render:
                # time, height, width, channel -> time, channel, height, width
                videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
                self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

            if self.config.test_mode:
                print("Best Score: %.2f" % best_score)

            test_info = {
                "Test-Results/Episode-Rewards": np.mean(scores),
                "Test-Results/Episode-Rewards-Std": np.std(scores),
            }

            self.log_infos(test_info, self.current_step)
            if env_fn is not None:
                envs.close()
        return scores
