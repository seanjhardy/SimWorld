import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.networks.attention.transformer import Transformer, init_weights


class QTransformer(Transformer):

    def __init__(self, config, gamma=0.5):
        super().__init__(config)
        self.gamma = gamma
        self.to(config.device_type)
        self.model.add_module("q_head",
                              nn.Linear(config.n_embed, 1, bias=config.bias))
        self.to(config.device_type)
        self.apply(init_weights)
        self.configure_optimizers(config)

    def forward(self, idx, targets=None, rewards=None):
        latent = self._forward(idx)

        if targets is None:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.model.lm_head(latent[:, [-1], :])  # note: using list [-1] to preserve the time dim
            q_values = self.model.q_head(latent[:, [-1], :])
            loss = None
        else:
            # if we are given some desired targets also calculate the loss
            logits, q_values, loss = self.backward(latent, targets, rewards)

        return logits.cpu().detach().numpy()[0], \
               q_values.cpu().detach().numpy()[0, :, 0], \
               loss

    def backward(self, latents, targets, rewards):
        logits = self.model.lm_head(latents)
        q_values = self.model.q_head(latents)  # Predicted Q values (b, t, 1)

        cumulative_reward = q_values[:, -1, 0].detach()  # Final q value (b, t)
        target_q_values = torch.zeros_like(q_values)

        # Compute the target Q values in reverse order
        for t in reversed(range(rewards.size(1))):
            cumulative_reward = rewards[:, t] + self.gamma * cumulative_reward
            target_q_values[:, t, 0] = cumulative_reward

        critic_loss = ((target_q_values - q_values) ** 2).mean() * 0.00001

        reconstruction_loss = F.mse_loss(logits, targets)
        loss = reconstruction_loss + critic_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return logits, q_values, loss
