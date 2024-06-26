import math
import torch
from torch import nn, optim

from modules.networks.policy.actor import Actor
from modules.networks.policy.critic import Critic


class ACBlock(nn.Module):

    def __init__(self, state_size, action_size, n_embed, device="cuda", actor_only=False):
        super(ACBlock, self).__init__()
        self.n_embed = n_embed
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.actor_only = actor_only

        self.model = nn.ModuleDict(dict(
            # Actor model proposes actions given the current state
            actor=Actor(state_size, action_size, self.n_embed),
        ))
        if not actor_only:
            # Critic model provides a score for the current state + action pair
            self.model.add_module("critic",
                                  Critic(state_size, action_size, self.n_embed))

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = 0.99
        self.entropy_beta = 0.01
        self.to(self.device)

    def forward(self, state):
        actions, _, _, _ = self._forward(state)
        return actions.detach().cpu().numpy()

    def _forward(self, state):
        # Generate proposed action
        actions, log_prob, var = self.model.actor(state)
        q_values = None
        if not self.actor_only:
            # Evaluate action score in current state
            q_values = self.model.critic(state, actions.detach())
        return actions, q_values, log_prob, var

    def backward(self, data):
        if self.actor_only:
            states, actions, q_values, rewards, next_states, next_q_values = data
            actions, _, log_probs, var = self._forward(states)
        else:
            states, actions, rewards, next_states = data
            with torch.no_grad():
                next_actions, next_q_values, _, _ = self._forward(next_states)
            actions, q_values, log_probs, var = self._forward(states)

        delta = rewards + self.gamma * next_q_values.detach()
        critic_loss = ((delta - q_values) ** 2).mean()
        advantage = delta - q_values.detach()
        actor_loss = -(advantage.unsqueeze(-1) * log_probs).mean()
        entropy_loss = self.entropy_beta * (-(torch.log(2 * math.pi * var) + 1) / 2).mean()

        loss = actor_loss + entropy_loss

        if not self.actor_only:
            loss += critic_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

