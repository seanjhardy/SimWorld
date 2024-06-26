from torch import nn

from modules.networks.attention.selfAttention import SelfAttention
from modules.networks.classic.MLP import MLP
from modules.networks.classic.layerNorm import LayerNorm


class Block(nn.Module):
    def __init__(self, dim, n_head=1, mlp_ratio=4.,
                 bias=True, drop=0., attn_drop=0.,
                 causal=False, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = LayerNorm(dim, bias)
        self.attn = SelfAttention(dim, n_head=n_head, bias=bias,
                                  attn_dropout=attn_drop, dropout=drop, causal=causal)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = LayerNorm(dim, bias)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, n_embed=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
