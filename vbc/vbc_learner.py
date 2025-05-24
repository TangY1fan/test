from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
from xuance.torch.learners import LearnerMAS
from xuance.common import List
from argparse import Namespace
from operator import itemgetter


class VBC_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module):
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        super(VBC_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters_model, config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.config.running_steps)
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}

    def build_training_data(self, sample: Optional[dict],
                            use_parameter_sharing: Optional[bool] = False,
                            use_actions_mask: Optional[bool] = False,
                            use_global_state: Optional[bool] = False):
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        state, avail_actions, filled = None, None, None
        obs_next, state_next, avail_actions_next = None, None, None
        IDs = None
        obs = {k: Tensor(sample['obs'][k]).to(self.device) for k in self.agent_keys}
        actions = {k: Tensor(sample['actions'][k]).to(self.device) for k in self.agent_keys}
        message = {k: Tensor(sample['message'][k]).to(self.device) for k in self.agent_keys}
        rewards = {k: Tensor(sample['rewards'][k]).to(self.device) for k in self.agent_keys}
        terminals = {k: Tensor(sample['terminals'][k]).float().to(self.device) for k in self.agent_keys}
        agent_mask = {k: Tensor(sample['agent_mask'][k]).float().to(self.device) for k in self.agent_keys}
        if not self.use_rnn:
            obs_next = {k: Tensor(sample['obs_next'][k]).to(self.device) for k in self.agent_keys}
        if use_actions_mask:
            avail_actions = {k: Tensor(sample['avail_actions'][k]).float().to(self.device) for k in self.agent_keys}
            if not self.use_rnn:
                avail_actions_next = {k: Tensor(sample['avail_actions_next'][k]).float().to(self.device) for k in self.model_keys}

        if use_global_state:
            state = Tensor(sample['state']).to(self.device)
            if not self.use_rnn:
                state_next = Tensor(sample['state_next']).to(self.device)

        if self.use_rnn:
            filled = Tensor(sample['filled']).float().to(self.device)

        sample_Tensor = {
            'batch_size': batch_size,
            'state': state,
            'state_next': state_next,
            'obs': obs,
            'actions': actions,
            'message': message,
            'obs_next': obs_next,
            'rewards': rewards,
            'terminals': terminals,
            'agent_mask': agent_mask,
            'avail_actions': avail_actions,
            'avail_actions_next': avail_actions_next,
            'agent_ids': IDs,
            'filled': filled,
            'seq_length': seq_length,
        }
        return sample_Tensor

    def update(self, sample):
        pass

    def update_rnn(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        seq_len = sample['sequence_length']
        state = sample_Tensor['state']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        message = sample_Tensor['message']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        filled = sample_Tensor['filled'].reshape([-1, 1])
        IDs = sample_Tensor['agent_ids']


        bs_rnn = batch_size
        rewards_tot = torch.stack(itemgetter(*self.agent_keys)(rewards), dim=1).mean(dim=1).reshape(-1, 1)
        terminals_tot = torch.stack(itemgetter(*self.agent_keys)(terminals), dim=1).all(1).reshape([-1, 1]).float()

        # calculate the individual Q values.
        rnn_hidden = {k: self.policy.representation[k].init_hidden(bs_rnn) for k in self.model_keys}

        _, actions_greedy, q_eval = self.policy(obs, agent_ids=IDs, avail_actions=avail_actions, rnn_hidden=rnn_hidden, message_input=message)

        target_rnn_hidden = {k: self.policy.target_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        _, q_next_seq = self.policy.Qtarget(obs, agent_ids=IDs, rnn_hidden=target_rnn_hidden, message_input=message)

        q_eval_a, q_next, q_next_a = {}, {}, {}
        for key in self.model_keys:
            message[key] = message[key][:, :-1]
            q_eval_a[key] = q_eval[key][:, :-1].gather(-1, actions[key].long().unsqueeze(-1)).reshape(bs_rnn, seq_len)
            q_next[key] = q_next_seq[key][:, 1:]

            if self.use_actions_mask:
                q_next[key][avail_actions[key][:, 1:] == 0] = -1e10

            if self.config.double_q:
                act_next = {k: actions_greedy[k].unsqueeze(-1)[:, 1:] for k in self.model_keys}
                q_next_a[key] = q_next[key].gather(-1, act_next[key].long().detach()).reshape(bs_rnn, seq_len)
            else:
                q_next_a[key] = q_next[key].max(dim=-1, keepdim=True).values.reshape(bs_rnn, seq_len)

            q_eval_a[key] = q_eval_a[key] * agent_mask[key]
            q_next_a[key] = q_next_a[key] * agent_mask[key]

            if self.use_parameter_sharing:
                q_eval_a[key] = q_eval_a[key].reshape(batch_size, self.n_agents, seq_len).transpose(1, 2).reshape(-1, self.n_agents)
                q_next_a[key] = q_next_a[key].reshape(batch_size, self.n_agents, seq_len).transpose(1, 2).reshape(-1, self.n_agents)
            else:
                q_eval_a[key] = q_eval_a[key].reshape(-1, 1)
                q_next_a[key] = q_next_a[key].reshape(-1, 1)

        # Todo return std_message
        msg_eval_std = self.policy.message_std(message_input=message)

        # calculate the total Q values.
        q_tot_eval = self.policy.Q_tot(q_eval_a, state[:, :-1].reshape([batch_size * seq_len, -1]))
        q_tot_next = self.policy.Qtarget_tot(q_next_a, state[:, 1:].reshape([batch_size * seq_len, -1]))
        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        # calculate the loss function
        td_errors = (q_tot_eval - q_tot_target.detach()) * filled
        loss_std = msg_eval_std.sum() / filled.sum()
        loss_Q = (td_errors ** 2).sum() / filled.sum()
        loss = loss_Q + 4.0 * loss_std
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters_model, self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "loss_Q": loss_Q.item(),
            "loss_std": loss_std.item(),
            "predictQ": q_tot_eval.mean().item()
        })

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        return info
