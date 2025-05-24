from copy import deepcopy
from types import ModuleType
from typing import Optional, Dict, Sequence, Callable, Union, List

import torch
from gymnasium.spaces import Discrete
from torch import Tensor
from torch.nn import ModuleDict
from torch.nn.parallel import DistributedDataParallel

from xuance.torch.policies import VDN_mixer
from xuance.torch.policies.deterministic_marl import BasicQnetwork


class VBC_Policy(BasicQnetwork):

    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation: ModuleDict,
                 communicator: ModuleDict,
                 mixer: Optional[VDN_mixer] = None,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(VBC_Policy, self).__init__(action_space, n_agents, representation, hidden_size,
                                             normalize, initialize, activation, device, use_distributed_training,
                                             **kwargs)
        self.eval_Qtot = mixer
        self.target_Qtot = deepcopy(self.eval_Qtot)
        self.communicator = communicator
        self.threshold_1 = kwargs['threshold_1']
        if self.distributed_training:
            self.eval_Qtot = DistributedDataParallel(module=self.eval_Qtot, device_ids=[self.rank])

    @property
    def parameters_model(self):
        parameters_model = list(self.eval_Qtot.parameters()) + list(self.representation.parameters()) + list(
            self.eval_Qhead.parameters()) + list(self.communicator.parameters())
        return parameters_model

    def forward(self, observation: Dict[str, Tensor], agent_ids: Tensor = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None,
                 message_input: dict = None):
        rnn_hidden_new, argmax_action, evalQ, std_eval_msg = {}, {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}

        for key in agent_list:
            if self.use_rnn:
                outputs = self.representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            if self.use_parameter_sharing:
                q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
            else:
                q_inputs = outputs['state']

            msg_eval = self.communicator[key](message_input)
            evalQ[key] = self.eval_Qhead[key](q_inputs)
            # Todo if abs(max_1 - max_2) > threshold_1 then evalQ[key] += msg_eval
            diff = torch.abs(torch.diff(torch.topk(evalQ[key], k=2, dim=-1)[0], dim=-1))
            evalQ[key] = torch.where(diff > self.threshold_1, evalQ[key]+msg_eval, evalQ[key])

            if avail_actions is not None:
                evalQ_detach = evalQ[key].clone().detach()
                evalQ_detach[avail_actions[key] == 0] = -1e10
                argmax_action[key] = evalQ_detach.argmax(dim=-1, keepdim=False)
            else:
                argmax_action[key] = evalQ[key].argmax(dim=-1, keepdim=False)

        return rnn_hidden_new, argmax_action, evalQ

    def Qtarget(self, observation: Dict[str, Tensor], agent_ids: Dict[str, Tensor],
                agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None,
                message_input: dict = None):
        rnn_hidden_new, q_target = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            outputs = self.target_representation[key](observation[key], *rnn_hidden[key])
            rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            if self.use_parameter_sharing:
                q_inputs = torch.concat([outputs['state'], agent_ids], dim=-1)
            else:
                q_inputs = outputs['state']

            msg_eval = self.communicator[key](message_input)
            q_target[key] = self.target_Qhead[key](q_inputs)
            diff = torch.abs(torch.diff(torch.topk(q_target[key], k=2, dim=-1)[0], dim=-1))
            q_target[key] = torch.where(diff > 1.0, q_target[key] + msg_eval, q_target[key])

        return rnn_hidden_new, q_target

    def Q_tot(self, individual_values: Dict[str, Tensor], states: Optional[Tensor] = None):
        if self.use_parameter_sharing:
            individual_inputs = individual_values[self.model_keys[0]].reshape([-1, self.n_agents, 1])
        else:
            individual_inputs = torch.concat([individual_values[k] for k in self.model_keys],
                                             dim=-1).reshape([-1, self.n_agents, 1])
        evalQ_tot = self.eval_Qtot(individual_inputs, states)
        return evalQ_tot

    def Qtarget_tot(self,
                    individual_values: Dict[str, Tensor],
                    states: Optional[Tensor] = None):
        if self.use_parameter_sharing:
            individual_inputs = individual_values[self.model_keys[0]].reshape([-1, self.n_agents, 1])
        else:
            individual_inputs = torch.concat([individual_values[k] for k in self.model_keys],
                                             dim=-1).reshape([-1, self.n_agents, 1])
        q_target_tot = self.target_Qtot(individual_inputs, states)
        return q_target_tot

    def message_std(self, message_input: dict) -> Dict[str, Tensor]:
        msg_eval_std = {k: self.communicator[k].get_msg_std(message_input) for k in self.model_keys}
        msg_eval_std = torch.stack(list(msg_eval_std.values()))
        msg_eval_std = msg_eval_std.sum(dim=0)
        msg_eval_std= msg_eval_std.reshape(-1, 1)
        return msg_eval_std

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qtot.parameters(), self.target_Qtot.parameters()):
            tp.data.copy_(ep)