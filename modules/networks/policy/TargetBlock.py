import math

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal


def state_similarity(state, goal, mask=None, threshold=0.01):
    if mask is None:
        mask = torch.ones_like(state)
    return torch.mean(torch.abs(state[mask] - goal[mask])) < threshold


# Hierarchical Reinforcement Learning Block
class HRLBlock(nn.Module):

    def __init__(self, state_size, action_size, n_embed=400, device="cuda", action_type="discrete"):
        super(HRLBlock, self).__init__()
        self.n_embed = n_embed
        self.action_type = action_type
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.model = nn.ModuleDict(dict(
            target=nn.Sequential(
                nn.Linear(state_size, self.n_embed),
                nn.ReLU(),
                nn.Linear(self.n_embed, self.n_embed),
                nn.ReLU(),
                nn.Linear(self.n_embed, state_size * 2),
                nn.Tanh(),
            ),
            sigmoid=nn.Sigmoid(),
            tanh=nn.Tanh(),
            mu=nn.Linear(n_embed, action_size),
            var=nn.Linear(n_embed, action_size),
            generator=nn.Sequential(
                nn.Linear(state_size * 2, self.n_embed),
                nn.ReLU(),
                nn.Linear(self.n_embed, self.n_embed),
                nn.ReLU(),
                nn.Linear(self.n_embed, self.n_embed),
            ),
        ))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.entropy_beta = 0.01
        self.to(self.device)

    def forward(self, data):
        states, target_states = data
        combined = torch.cat((states, target_states), dim=-1)
        x = self.model.generator(combined)

        mean = self.model.tanh(self.model.mu(x))
        var = self.model.sigmoid(self.model.var(x)) + 1e-6
        std = torch.sqrt(var)

        dist = Normal(mean, std)
        sample = dist.rsample()
        actions = torch.clamp(sample, min=-1.0, max=1.0)

        log_prob = dist.log_prob(sample)

        if self.action_type == "discrete":
            actions = nn.Tanh(actions)
        else:
            actions = nn.Softmax(actions)
        return actions

    def backward(self, data):
        states, target_states, next_states = data

        # Loss is defined as the new state target distance after taking action A
        # minus the old distance - Optimises for minimal distance
        loss = state_similarity(next_states, target_states) \
               - state_similarity(states, target_states)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
