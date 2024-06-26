import torch
from torch import nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

import numpy as np

from modules.networks.vision.scae.models.autoencoder import ImageAutoencoder
from dataclasses import dataclass, field


@dataclass
class SCAEConfig:
    canvas_size: list[int] = field(default_factory=lambda: [1, 80])
    batch_size: int = 1
    lr: float = 0.01
    gamma: float = 1
    template_size: list[int] = field(default_factory=lambda: [1, 16])
    n_part_caps: int = 16
    d_part_pose: int = 6
    d_part_features: int = 32
    n_channels: int = 3
    n_obj_caps: int = 10
    n_obj_caps_params: int = 30
    colorize_templates: bool = True
    use_alpha_channel: bool = False
    template_nonlin: str = 'relu1'
    color_nonlin: str = 'relu1'
    prior_within_example_sparsity_weight: float = 1.0
    prior_between_example_sparsity_weight: float = 1.0
    posterior_within_example_sparsity_weight: float = 10.0
    posterior_between_example_sparsity_weight: float = 10.0
    device_type: str = "cuda"


class SCAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output = np.zeros(1)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.model = ImageAutoencoder(config=config).to(self.device)
        self.eps = 1e-2 / float(config.batch_size) ** 2
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, eps=self.eps)

        self.scheduler = ExponentialLR(self.optimizer, gamma=config.gamma)

    def forward(self, image):
        self.model.eval()
        with torch.no_grad():
            res, latent = self.model(image.to(self.device), None, device=self.device)
        return res, latent

    def backward(self, x):
        self.model.train()
        self.optimizer.zero_grad()

        # Train
        res = self.model(x, None, device=self.device)
        loss = self.model.loss(res)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        with torch.no_grad():
            output = res.new_pdf.cpu().numpy()
            output = output.squeeze()
            output = np.repeat(output[:, :, np.newaxis], 3, axis=2)
            output = np.swapaxes(output, 0, 1)
            self.output = output
        return self.output