import math
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from modules.networks.attention.multiHeadAttention import MultiHeadAttention
from modules.networks.classic.block import Block
from modules.networks.classic.layerNorm import LayerNorm
from modules.networks.classic.utils import init_weights
from modules.networks.vision.glom.utils import PatchEmbed, attention, patches2image_batch
from modules.networks.vision.stereo.stereoAE import StereoAE


class GlomLayer(nn.Module):
    def __init__(self, n_levels, level_dim, **kwargs):
        super().__init__()
        self.bottom_up_net = Block(level_dim, **kwargs, bias=True,
                                   with_attention=True)
        self.top_down_net = Block(level_dim, **kwargs, bias=True,
                                  with_attention=True)
        self.current_net = attention
        self.norm = nn.LayerNorm(level_dim)

    def forward(self, x):
        """
        Inputs:
            x: tensor of shape [B, N, NL, DL]
        Outputs:
            z: tensor of shape [B, N, NL, DL]
        """
        B, N, NL, DL = x.shape
        z = []
        for l in range(NL):
            x_lev_cur = x[:, :, l]
            x_lev_prev = x[:, :, l - 1] if l > 0 else torch.zeros_like(x_lev_cur)  # [B, N, DL]
            x_lev_next = x[:, :, l + 1] if l < (NL - 1) else torch.zeros_like(x_lev_cur)  # [B, N, DL]

            z_lev_prev = self.bottom_up_net(x_lev_prev)  # [B, N, DL]
            z_lev_next = self.top_down_net(x_lev_next) # [B, N, DL]
            z_lev_cur = z_lev_prev + z_lev_next  # [B, N, DL]
            z_lev_cur += self.current_net(x_lev_cur)# + x_lev_cur

            z += [self.norm(z_lev_cur)]
        z = torch.stack(z, 2)  # [B, N, NL, DL]
        return z


class Glom(nn.Module):
    """
    Glom Backbone.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, n_head=6, n_layers=3, n_levels=5, n_embed=1280,
                 mlp_ratio=4, device="cuda"):
        super().__init__()
        assert n_embed % n_levels == 0, 'Features dimension must be divisible by number of levels!'
        self.n_levels = n_levels
        self.n_embed = n_embed
        self.device = device
        self.level_dim = self.n_embed // self.n_levels
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=self.n_embed)
        self.layers = nn.ModuleList([
            GlomLayer(n_levels, self.level_dim, mlp_ratio=mlp_ratio, n_head=n_head) for _ in range(n_layers)
        ])
        self.pos_emb = nn.Parameter(torch.randn(self.patch_embed.num_patches, self.n_embed))
        self.pos_lev_emb = nn.Parameter(torch.randn(n_levels, self.level_dim))

    def forward(self, x):
        """
        Inputs:
            x: tensor of shape [B, C, H, W]
        Outputs:
            x: tensor of shape [B, N, NL, DL]
        """
        _, C, H, W = x.shape
        x = self.patch_embed(x)  # [B, N, D]
        B, N, D = x.shape
        x += self.pos_emb.unsqueeze(0)
        x = x.reshape(B, -1, self.n_levels, self.level_dim)  # [B, N, NL, DL]
        x += self.pos_lev_emb.unsqueeze(0).unsqueeze(1)

        for layer in self.layers:
            x = layer(x)
        return x  # [B, N, NL, DL]


@dataclass
class GlomAEConfig:
    img_size: Union[int, tuple[int, int]] = field(default_factory=lambda: (1, 80))
    patch_size: Union[int, tuple[int, int]] = field(default_factory=lambda: (1, 16))
    stereoscopic: bool = False,
    in_chans: int = 3
    n_layers: int = 3
    n_levels: int = 4
    n_head: int = 6
    n_embed: int = 1280
    lr: float = 0.01
    wd: float = 0.0001
    betas: tuple[float, float] = (0.9, 0.95)
    bias: bool = False


# Glom autoencoder
class GlomAE(nn.Module):
    def __init__(self, config: GlomAEConfig, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.model = nn.ModuleDict(dict(
            glom=Glom(img_size=config.img_size,
                      patch_size=config.patch_size,
                      in_chans=config.in_chans,
                      n_layers=config.n_layers,
                      n_head=config.n_head,
                      n_levels=config.n_levels, n_embed=config.n_embed,
                      device=device),
            rec_head=nn.Linear(config.n_embed, config.in_chans * (np.prod(config.patch_size)))
        ))
        if config.stereoscopic:
            self.model.add_module("stereo", StereoAE(config))
        self.optimizer = Adam(self.model.parameters(),
                              lr=config.lr,
                              weight_decay=config.wd,
                              betas=config.betas)
        self.to(device)
        self.apply(init_weights)

    def decode(self, x):
        B, N, D = x.shape
        PS = self.config.patch_size  # patch_size
        x = self.model.rec_head(x)  # [B, N, C * PS * PS]
        x = x.reshape(B, N, self.config.in_chans, PS[0], PS[1]).transpose(1, 2)
        x = patches2image_batch(x, self.config.img_size)
        return x

    def forward(self, x, to_numpy=False):
        if self.config.stereoscopic:
            left, right = torch.split(x, x.size(3) // 2, dim=3)
            left_latent = self.forward_single(left)
            left_rec = self.decode(left_latent)

            right_latent = self.forward_single(right)
            right_rec = self.decode(right_latent)
            latent, combined_loss, left_rec2, right_rec2 = self.model.stereo.backward(left_latent.detach(),
                                                                                      right_latent.detach())
            rec = torch.cat([left_rec, right_rec])  # B, C, H, W
        else:
            latent = self.forward_single(x)
            rec = self.decode(latent)

        rec = torch.swapaxes(rec, 1, 3)  # B, W, H , C
        rec = rec.reshape(x.size(0), -1)

        if to_numpy:
            latent = latent.cpu().detach().numpy()
            rec = rec.cpu().detach().numpy()
        return latent, rec

    def forward_single(self, x):
        """
        Inputs:
            x: tensor of shape [B, C, H, W]
        Outputs:
            x: tensor of shape [B, N, NL, DL]
        """
        B, C, H, W = x.shape
        x = self.model.glom(x)  # [B, N, NL, DL]
        N = x.size(1)
        latent = x.reshape(B, N, self.config.n_embed)  # [B, N, D]
        return latent

    def backward(self, x):
        self.optimizer.zero_grad(set_to_none=True)

        if self.config.stereoscopic:
            left, right = torch.split(x, x.size(3) // 2, dim=3)
            left_latent = self.forward_single(left)
            left_rec = self.decode(left_latent)
            left_loss = F.mse_loss(left_rec, left)

            right_latent = self.forward_single(right)
            right_rec = self.decode(right_latent)
            right_loss = F.mse_loss(right_rec, right)

            if (left_loss + right_loss).item() < 0.01:
                latent, combined_loss, left_rec, right_rec = self.model.stereo.backward(left_latent.detach(),
                                                                                        right_latent.detach())
                left_rec = self.decode(left_rec)
                right_rec = self.decode(right_rec)
            else:
                latent = left_latent
            loss = left_loss + right_loss
            rec = torch.cat([left_rec, right_rec])  # B, C, H, W
        else:

            latent = self.forward_single(x)
            rec = self.decode(latent)

            loss = F.mse_loss(rec, x)
        loss.backward()
        self.optimizer.step()

        rec = torch.swapaxes(rec, 1, 3)  # B, W, H , C
        rec = rec.reshape(x.size(0), -1)

        return latent.detach().cpu().numpy(), \
               rec.detach().cpu().numpy(), \
               loss.item()
