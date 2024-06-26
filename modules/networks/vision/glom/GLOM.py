import math
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from modules.networks.attention.crossAttention import CrossAttention
from modules.networks.classic.block import Block


def patches2image_batch(x, output_size):
    B, C, N_PATCH, H_PATCH, W_PATCH = x.shape
    x = x.reshape(B, C, N_PATCH, -1).transpose(2, 3).reshape(B, -1, N_PATCH)  # [B, C, prod(PATCH_SIZE)]
    x = F.fold(x, output_size, kernel_size=(H_PATCH, W_PATCH), stride=(H_PATCH, W_PATCH))  # [B, C, H, W]
    return x  # [B, C, H, W]


def attention(x):
    """
    Inputs:
        x: tensor of shape [B, N, D]
    Outputs:
        x: tensor of shape [B, N, D]
    """
    att = torch.bmm(x, x.transpose(1, 2))
    att = F.softmax(att, 1)
    x = torch.bmm(att, x)
    return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=(1, 16), in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BDHW -> BND
        x = self.norm(x)
        return x


class GlomLayer(nn.Module):
    def __init__(self, n_levels, level_dim, **kwargs):
        super().__init__()
        self.bottom_up_net = Block(level_dim, **kwargs)
        self.top_down_net = Block(level_dim, **kwargs)
        self.current_net = attention
        self.norm = nn.LayerNorm(level_dim)

    def forward(self, x, pos_lev_emb):
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
            z_lev_next = self.top_down_net(x_lev_next) + \
                         pos_lev_emb[l].unsqueeze(0).unsqueeze(0).repeat(B, N, 1)  # [B, N, DL]
            z_lev_cur = self.current_net(x_lev_cur) + z_lev_prev + z_lev_next  # [B, N, DL]
            z += [self.norm(z_lev_cur)]
        z = torch.stack(z, 2)  # [B, N, NL, DL]
        return z


class Glom(nn.Module):
    """
    Glom Backbone.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, n_layers=3, n_levels=5, n_embed=1280,
                 use_pos_emb=True, mlp_ratio=4, device="cuda"):
        super().__init__()
        assert n_embed % n_levels == 0, 'Features dimension must be divisible by number of levels!'
        self.n_levels = n_levels
        self.n_embed = n_embed
        self.device = device
        self.use_pos_emb = use_pos_emb
        self.level_dim = self.n_embed // self.n_levels
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=self.n_embed)
        self.pos_lev_emb = nn.Parameter(torch.randn(n_levels, self.level_dim))
        self.layers = nn.ModuleList([
            GlomLayer(n_levels, self.level_dim, mlp_ratio=mlp_ratio) for _ in range(n_layers)
        ])
        if self.use_pos_emb:
            self.pos_emb = nn.Embedding(self.patch_embed.num_patches, self.n_embed)

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
        if self.use_pos_emb:
            pos = torch.arange(0, N, dtype=torch.int32, device=self.device)
            x += self.pos_emb(pos)
        x = x.reshape(B, -1, self.n_levels, self.level_dim)  # [B, N, NL, DL]
        for layer in self.layers:
            x = layer(x, self.pos_lev_emb)
        return x  # [B, N, NL, DL]


@dataclass
class GlomAEConfig:
    img_size: Union[int, tuple[int, int]] = field(default_factory=lambda: (1, 80))
    patch_size: Union[int, tuple[int, int]] = field(default_factory=lambda: (1, 16))
    stereoscopic: bool = False,
    in_chans: int = 3
    n_layers: int = 3
    n_levels: int = 4
    n_embed: int = 1280
    lr: float = 0.01
    wd: float = 0.0001
    betas: tuple[float, float] = (0.9, 0.95)
    bias: bool = False


class StereoAE(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.models = nn.ModuleDict(dict(
            cross_attention=CrossAttention(n_embed, n_head=3),
            left_decoder=nn.Linear(n_embed, n_embed),
            right_decoder=nn.Linear(n_embed, n_embed),
        ))

    def forward(self, left, right):
        x = self.models.cross_attention(left, right)
        left_rec = self.models.left_decoder(x)
        right_rec = self.models.right_decoder(x)
        return x, left_rec, right_rec

    def backward(self, left, right):
        x, left_rec, right_rec = self(left, right)

        loss = F.mse_loss(left_rec, left) + F.mse_loss(right_rec, right)
        return x, loss

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
                      n_levels=config.n_levels, n_embed=config.n_embed,
                      use_pos_emb=True, device=device),
            rec_head=nn.Linear(config.n_embed, config.in_chans * (np.prod(config.patch_size)))
        ))
        if config.stereoscopic:
            self.model.add_module("stereo",
                                  StereoAE(n_embed=config.n_embed))
        self.optimizer = Adam(self.model.parameters(),
                              lr=config.lr,
                              weight_decay=config.wd,
                              betas=config.betas)
        self.to(device)

    def forward(self, x):
        """
        Inputs:
            x: tensor of shape [B, C, H, W]
        Outputs:
            x: tensor of shape [B, N, NL, DL]
        """
        B, C, H, W = x.shape
        PS = self.config.patch_size  # patch_size
        x = self.model.glom(x)  # [B, N, NL, DL]
        N = x.size(1)
        x = x.reshape(B, N, self.config.n_embed)  # [B, N, D]
        B, N, D = x.shape
        latent = x
        x = self.model.rec_head(x)  # [B, N, C * PS * PS]
        x = x.reshape(B, N, C, PS[0], PS[1]).transpose(1, 2)
        x = patches2image_batch(x, (H, W))
        return latent, x

    def backward(self, x):
        if self.config.stereoscopic:
            left, right = torch.split(x, int(x.size(3) / 2), dim=3)
            left_latent, left_rec = self(left)
            right_latent, right_rec = self(left)

            latent, combined_loss = self.model.stereo.backward(left.detach(), right.detach())
            left_loss = F.mse_loss(left_rec, left)
            right_loss = F.mse_loss(right_rec, right)

            loss = left_loss + right_loss + combined_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        else:
            latent, rec = self(x)

            loss = F.mse_loss(rec, x)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return latent, \
               rec.detach().cpu().numpy().reshape(x.shape[0], -1), \
               loss
