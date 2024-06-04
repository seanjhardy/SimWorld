import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Define the autoencoder architecture
class CVAE(nn.Module):
    def __init__(self, latent_dim, device="cuda"):
        super().__init__()
        self.model = nn.ModuleDict(dict(
            encoder=nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=(3, 1), stride=1, padding=0, dtype=torch.half),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=(3, 1), stride=2, padding=0),
                nn.ReLU(),
            ),
            fc1=nn.Linear(64, 128),
            fc_mu=nn.Linear(128, latent_dim),
            fc_log_var=nn.Linear(128, latent_dim),
            fc2=nn.Linear(latent_dim, 64),
            decoder=nn.Sequential(
                nn.ConvTranspose2d(16, 8, kernel_size=(3, 1),
                                   stride=2, padding=0, output_padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(8, 3, kernel_size=(3, 1),
                                   stride=1, padding=0, output_padding=0),
                nn.Sigmoid()
            )
        ))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.to(device)

    def forward(self, x):
        print(x.shape)
        x = self.model.encoder(x)

        x = self.model.fc(x)
        mu = self.model.fc_mu(x)
        log_var = self.model.fc_log_var(x)

        latent = self.reparameterize(mu, log_var)

        x = self.model.fc2(latent)
        x = self.model.decoder(x)
        return latent, x

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def backwards(self, x):
        latent, reconstruction = self(x)
        loss = nn.MSELoss(reconstruction, x)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return latent.cpu().detach().numpy(), \
               reconstruction.cpu().detach().numpy(), \
               loss.item()
