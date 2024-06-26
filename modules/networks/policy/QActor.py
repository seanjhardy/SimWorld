import math

import torch
from torch import optim
from torch.distributions import Normal

from modules.networks.policy.actor import Actor


class QActor(Actor):
    def __init__(self, state_size, action_size, n_embed):
        super().__init__(state_size, action_size, n_embed)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, state):
        x = self.model.mish(self.model.fc1(state))
        x = self.model.mish(self.model.fc2(x))
        mean = self.model.tanh(self.model.mu(x))
        var = self.model.softplus(self.model.var(x)) + 1e-6
        return mean, var

    def backward(self, latents, imagine):
        mean, var = self(latents)
        std = torch.sqrt(var)
        dist = Normal(mean, std)
        sample = dist.rsample()
        action = torch.clamp(sample, min=-1.0, max=1.0)

        log_prob = dist.log_prob(sample)

        entropy_loss = self.entropy_beta * (-(torch.log(2 * math.pi * var) + 1) / 2).mean()



