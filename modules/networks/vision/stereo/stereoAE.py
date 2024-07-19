from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from modules.networks.attention.multiHeadAttention import MultiHeadAttention
from modules.networks.classic.block import Block
from modules.networks.classic.layerNorm import LayerNorm


class StereoAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.n_embed
        self.model = nn.ModuleDict(dict(
            cross_attention=MultiHeadAttention(self.n_embed, n_head=1),
            norm=LayerNorm(self.n_embed, True),
            left_decoder=Block(self.n_embed, n_head=6, bias=True),
            right_decoder=Block(self.n_embed, n_head=6, bias=True,),
        ))
        self.optimizer = Adam(self.model.parameters(),
                              lr=config.lr,
                              weight_decay=config.wd,
                              betas=config.betas)

    def forward(self, left, right):
        x = left + self.model.cross_attention(left, right)
        x = self.model.norm(x)
        left_rec = self.model.left_decoder(x)
        right_rec = self.model.right_decoder(x)
        return x, left_rec, right_rec

    def backward(self, left, right):
        self.optimizer.zero_grad(set_to_none=True)
        latent, left_rec, right_rec = self(left, right)

        loss = F.mse_loss(left_rec, left) + F.mse_loss(right_rec, right)
        loss.backward()
        self.optimizer.step()

        return latent, loss, left_rec, right_rec
