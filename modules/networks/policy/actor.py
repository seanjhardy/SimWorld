import torch
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, state_size, action_size, n_embed):
        super(Actor, self).__init__()
        self.model = nn.ModuleDict(dict(
            fc1=nn.Linear(state_size, n_embed),
            fc2=nn.Linear(n_embed, n_embed),
            mu=nn.Linear(n_embed, action_size),
            var=nn.Linear(n_embed, action_size),
            mish=nn.Mish(),
            tanh=nn.Tanh(),
            softplus=nn.Softplus(),
            sigmoid=nn.Sigmoid(),
        ))

    def forward(self, state):
        x = self.model.mish(self.model.fc1(state))
        x = self.model.mish(self.model.fc2(x))
        mean = self.model.tanh(self.model.mu(x))
        var = self.model.softplus(self.model.var(x)) + 1e-6
        std = torch.sqrt(var)

        dist = Normal(mean, std)
        sample = dist.rsample()
        action = torch.clamp(sample, min=-1.0, max=1.0)

        log_prob = dist.log_prob(sample)
        return action, log_prob, var
