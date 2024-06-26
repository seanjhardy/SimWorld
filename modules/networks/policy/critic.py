import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, state_size, action_size, n_embed):
        super(Critic, self).__init__()
        self.model = nn.ModuleDict(dict(
            fc1=nn.Linear(state_size + action_size, n_embed),
            fc2=nn.Linear(n_embed, n_embed),
            fc3=nn.Linear(n_embed, 1),
            mish=nn.Mish(),
        ))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.model.mish(self.model.fc1(x))
        x = self.model.mish(self.model.fc2(x))
        return self.model.fc3(x)
