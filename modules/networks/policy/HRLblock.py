import math

import numpy as np
import torch
from torch import nn, optim
from modules.networks.policy.ACBlock import Critic, Actor


def check_goal(state, goal, mask=None, threshold=0.01):
    if mask is None:
        mask = torch.ones_like(state)
    return torch.mean(torch.abs(state[mask] - goal[mask])) < threshold


# Hierarchical Reinforcement Learning Block
class HRLLayer(nn.Module):

    def __init__(self, state_size, action_size, n_embed=400, device="cuda", action_type="discrete"):
        super(HRLLayer, self).__init__()
        self.n_embed = n_embed
        self.action_type = action_type
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.model = nn.ModuleDict(dict(
            actor=Actor(state_size * 2, action_size, self.n_embed),
            critic=Critic(state_size * 2 + action_size, action_size, self.n_embed),
        ))
        self.actor_optimizer = optim.Adam(self.model.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.model.critic.parameters(), lr=0.001)

        self.entropy_beta = 0.1
        self.to(self.device)

    def forward(self, data):
        states, targets = data
        combined = torch.cat((states, targets), dim=-1)
        return self.model.actor(combined)

    def backward(self, data, imagine):
        states, actions, rewards, next_states, target_states = data

        next_actions, _, _ = self((next_states, target_states))
        next_combined = torch.cat((next_states, target_states), dim=-1)

        target_q = self.model.critic(next_combined, next_actions.detach()).detach()

        delta = rewards + self.gamma * target_q.detach()

        combined = torch.cat((states, target_states), dim=-1)
        q_values = self.model.critic(combined, actions)
        critic_loss = ((delta - q_values) ** 2).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        advantage = delta - q_values.detach()

        actions, log_probs, var = self((states, target_states))
        action_loss = -(advantage.unsqueeze(-1) * log_probs).mean()
        entropy_loss = self.entropy_beta * (-(torch.log(2 * math.pi * var) + 1) / 2).mean()

        actor_loss = action_loss + entropy_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()


class HRLBlock(nn.Module):
    def __init__(self, state_size, action_size, n_embed=400, n_layers=3, device="cuda", action_type="discrete"):
        super(HRLBlock, self).__init__()
        self.n_embed = n_embed
        self.n_layers = n_layers
        self.action_type = action_type
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.goals = []
        self.current_layer = 0

        self.models = nn.ModuleDict(dict(
            hrl_layer_0=HRLLayer(state_size, action_size, n_embed)
        ))

        for i in range(n_layers - 1):
            self.models.add_module(f"hrl_layer_{i + 1}",
                                    HRLLayer(state_size, state_size, n_embed))
    #def forward(self):

    def run_layer(self, state, goal, is_subgoal_test):
        self.goals[self.current_layer] = goal
        for _ in range(8):
            # if this is a subgoal test, then next/lower level goal has to be a subgoal test
            is_next_subgoal_test = is_subgoal_test

            action = self.models[f"hrl_layer_{self.current_layer}"](state, goal)
            if self.current_layer > 0:
                if np.random.random_sample() < 0.3:
                    is_next_subgoal_test = True

                #if is_next_subgoal_test and not check_goal(action, state):
                #    self.


