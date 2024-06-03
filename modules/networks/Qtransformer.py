import math
import inspect
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.networks.transformer import Transformer


class QTransformer(Transformer):

    def __init__(self, config, gamma=0.99):
        super().__init__(config)
        self.gamma = gamma
        self.to(config.device_type)

    def backwards(self, logits, targets):
        rewards = targets[:, :, -1]  # Reward in format of (b, t)
        q_values = logits[:, :, -1]  # Predicted Q values (b, t)

        cumulative_reward = logits[:, [-1], -1].detach()  # Final q value (b, t)
        target_q_values = torch.zeros_like(q_values)
        # Compute the target Q values in reverse order
        for t in reversed(range(targets.size(1) - 1)):
            cumulative_reward = rewards[:, t] + self.gamma * cumulative_reward
            target_q_values[:, t] = cumulative_reward

        critic_loss = ((target_q_values - q_values) ** 2).mean()
        reconstruction_loss = F.l1_loss(logits[:, :, :-1], targets[:, :, :-1])
        loss = reconstruction_loss + critic_loss / targets.size(2)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss
