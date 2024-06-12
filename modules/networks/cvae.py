import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# Define the autoencoder architecture
class CVAE(nn.Module):
    def __init__(self, latent_dim, device="cuda"):
        super().__init__()
        self.model = nn.ModuleDict(dict(
            encoder=nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=(1, 5), stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(1216, 512),
            ),
            fc_mu=nn.Linear(512, latent_dim),
            fc_log_var=nn.Linear(512, latent_dim),
            decoder=nn.Sequential(
                nn.Linear(latent_dim, 1216),
                nn.Unflatten(1, (16, 1, 76)),
                nn.ConvTranspose2d(16, 8, kernel_size=(1, 5),
                                   stride=1, padding=0, output_padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(1, 3),
                                   stride=1, padding=(0, 1), output_padding=0),
                nn.Sigmoid()
            )
        ))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.model.encoder(x)

        mu = self.model.fc_mu(x)
        log_var = self.model.fc_log_var(x)

        latent = self.reparameterize(mu, log_var)

        reconstruction = self.model.decoder(latent)
        return latent, reconstruction, mu, log_var

    def encode(self, x):
        x = self.model.encoder(x)

        mu = self.model.fc_mu(x)
        log_var = self.model.fc_log_var(x)

        latent = self.reparameterize(mu, log_var)
        return latent.cpu().detach().numpy()

    def decode(self, latent):
        reconstruction = self.model.decoder(latent)
        return reconstruction.cpu().detach().numpy().reshape(latent.shape[0], -1)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def kl_divergence(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()


    def backward(self, x):
        latent, reconstruction, mu, log_var = self(x)
        reconstruction_loss_l2 = F.mse_loss(reconstruction, x)
        #kl_loss = 0.00001 * self.kl_divergence(mu, log_var)
        temporal_consistency_loss = 0.05 * torch.mean(torch.abs(reconstruction[1:] - reconstruction[:-1]))
        loss = reconstruction_loss_l2 + temporal_consistency_loss # + kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return latent.cpu().detach().numpy(), \
               reconstruction.cpu().detach().numpy().reshape(x.shape[0], -1), \
               loss.item()
