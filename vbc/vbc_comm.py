from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn import Module


class VBC_Comm(Module):
    def __init__(self,
                 hidden_sizes: int,
                 action_dim: int,
                 threshold_2: int,
                 model_keys: list,
                 device: Optional[Union[str, int, torch.device]] = None,
                 n_agents: int = 1,
                 **kwargs: object):
        super().__init__()

        self.device = device
        self.action_dim = action_dim
        self.threshold_2 = threshold_2
        self.hidden_size = hidden_sizes
        self.model_keys = model_keys
        self.n_agents = n_agents
        self.message_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_dim),
        ).to(self.device)

    def init_message(self, num_envs: int):
        message = torch.zeros((1, num_envs, self.hidden_size))
        return message.numpy()

    def get_msg_std(self, message: dict):
        std_eval = {k: self.message_encoder(message[k]).std(dim=-1, keepdim=True) for k in self.model_keys}
        std_eval = torch.stack(list(std_eval.values()))
        std_eval = std_eval.sum(dim=0) / self.n_agents
        return std_eval

    def forward(self, msg: dict) -> torch.Tensor:
        msg_eval = {}
        for key in self.model_keys:
            msg_eval[key] = self.message_encoder(torch.as_tensor(msg[key], dtype=torch.float32, device=self.device))
            std = msg_eval[key].std(dim=-1, keepdim=True)
            mask = std < self.threshold_2
            msg_eval[key]= torch.where(mask.expand_as(msg_eval[key]), torch.zeros_like(msg_eval[key]), msg_eval[key])
        x = torch.stack(list(msg_eval.values()))
        x = x.sum(dim=0) / self.n_agents
        return x