import math

import numpy as np
import torch
from torch import nn, optim


# Hierarchical Reinforcement Learning Block
class HRLBlock(nn.Module):

    def __init__(self, state_size, action_size):
        super(HRLBlock, self).__init__()
        self.n_embed = 400

        self.model = nn.ModuleDict(dict(
            # Actor model proposes actions given the current state
            actor=Actor(state_size, action_size, self.n_embed),
            # Critic model provides a score for the current state + action pair
            critic=Critic(state_size, action_size, self.n_embed)
        ))
        self.prev_action = None
        self.prev_q = None
        self.actor_optimizer = optim.Adam(self.model.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.model.critic.parameters(), lr=0.001)
        self.gamma = 0.99
        self.replay_buffer = np.array((1000, state_size + action_size + 1 + state_size), dtype=np.float32)

    def forward(self, state, reward):
        if self.prev_q is not None:
            with torch.no_grad():
                action = self.model.actor(state)
                q_value = self.model.critic(state, action)
            advantage = reward + self.gamma * q_value.detach() - self.prev_q
            actor_loss = -torch.log(self.prev_action) * advantage
            critic_loss = nn.MSELoss(advantage)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # Generate proposed action
        action = self.model.actor(state)

        # Evaluate action score in current state
        q_value = self.model.critic(state, action)

        self.prev_action = action
        self.prev_q = q_value

        return action, q_value

class Actor(nn.Module):
    def __init__(self, state_size, action_size, n_embed):
        super(Actor, self).__init__()
        self.model = nn.ModuleDict(dict(
            fc1=nn.Linear(state_size, n_embed),
            fc2=nn.Linear(n_embed, n_embed),
            fc3=nn.Linear(n_embed, action_size),
            relu=nn.ReLU(),
            tanh=nn.Tanh()
        ))

    def forward(self, state):
        x = self.model.relu(self.model.fc1(state))
        x = self.model.relu(self.model.fc2(x))
        return self.tanh(self.model.fc3(x))


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
        x = torch.cat([state, action], dim=1)
        x = self.model.relu(self.model.fc1(x))
        x = self.model.relu(self.model.fc2(x))
        return self.model.fc3(x)
