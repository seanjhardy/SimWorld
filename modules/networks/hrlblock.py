import math

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.nn import functional as F


def state_similarity(state, goal, mask=None, threshold=0.01):
    if mask is None:
        mask = np.ones_like(state)
    return np.mean(np.abs(state[mask] - goal[mask])) < threshold


# Hierarchical Reinforcement Learning Block
class HRLBlock(nn.Module):

    def __init__(self, state_size, action_size, device="cuda"):
        super(HRLBlock, self).__init__()
        self.n_embed = 400
        self.device = device

        self.model = nn.ModuleDict(dict(
            # Actor model proposes actions given the current state
            actor=Actor(state_size, action_size, self.n_embed),
            # Critic model provides a score for the current state + action pair
            critic=Critic(state_size, action_size, self.n_embed)
        ))
        self.q_value = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.gamma = 0.99
        self.entropy_beta = 1e-4
        self.to(self.device)

    def forward(self, state):
        actions, _, _, _ = self._forward(state)
        return actions

    def _forward(self, state):
        # Generate proposed action
        actions, log_prob, var = self.model.actor(state)
        # Evaluate action score in current state
        q_values = self.model.critic(state, actions.detach())
        self.q_value = q_values
        return actions, q_values, log_prob, var

    def train(self, data):
        states, actions, rewards, next_states = data

        with torch.no_grad():
            next_actions, next_q_values, _, _ = self._forward(next_states)

        self.optimizer.zero_grad()

        actions, q_values, log_probs, var = self._forward(states)

        delta = rewards + self.gamma * next_q_values.detach()

        critic_loss = ((delta - q_values) ** 2).mean()
        advantage = delta - q_values.detach()
        actor_loss = -(advantage * log_probs).mean()
        entropy_loss = self.entropy_beta * (-(torch.log(2 * math.pi * var) + 1) / 2).mean()

        loss = critic_loss + actor_loss + entropy_loss
        loss.backward()
        self.optimizer.step()


class Actor(nn.Module):
    def __init__(self, state_size, action_size, n_embed):
        super(Actor, self).__init__()
        self.model = nn.ModuleDict(dict(
            fc1=nn.Linear(state_size, n_embed),
            fc2=nn.Linear(n_embed, n_embed),
            mu=nn.Linear(n_embed, action_size),
            var=nn.Linear(n_embed, action_size),
            relu=nn.ReLU(),
            tanh=nn.Tanh(),
            softplus=nn.Softplus(),
            sigmoid=nn.Sigmoid(),
        ))

    def forward(self, state):
        x = self.model.relu(self.model.fc1(state))
        x = self.model.relu(self.model.fc2(x))
        mean = self.model.tanh(self.model.mu(x))
        var = self.model.sigmoid(self.model.var(x)) + 1e-6
        std = torch.sqrt(var)

        dist = Normal(mean, std)
        sample = dist.rsample()
        action = torch.clamp(sample, min=-1.0, max=1.0)

        log_prob = dist.log_prob(sample)
        #log_prob = - ((mean - sample) ** 2) / (2*var) \
        #           - torch.log(torch.sqrt(2 * math.pi * var))
        return action, log_prob, var


class Critic(nn.Module):
    def __init__(self, state_size, action_size, n_embed):
        super(Critic, self).__init__()
        self.model = nn.ModuleDict(dict(
            fc1=nn.Linear(state_size + action_size, n_embed),
            fc2=nn.Linear(n_embed, n_embed),
            fc3=nn.Linear(n_embed, 1),
            relu=nn.ReLU(),
        ))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.model.relu(self.model.fc1(x))
        x = self.model.relu(self.model.fc2(x))
        return self.model.fc3(x)
