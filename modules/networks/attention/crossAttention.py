import math

import torch
from torch import nn
from torch.nn import functional as F


class CrossAttention(nn.Module):
    def __init__(self, n_embed, n_inputs=0, n_head=6,
                 bias=True, dropout=0., attn_dropout=0.):
        super().__init__()
        assert n_embed % n_head == 0
        # query projection for the first sequence
        self.q_proj = nn.Linear(n_embed, n_embed, bias=bias)
        # key and value projections for the second sequence
        self.kv_proj = nn.Linear(n_embed, 2 * n_embed, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embed = n_embed
        self.dropout = dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, context):
        B, T, C = x.size()  # sequence length, embedding dimensionality (n_embed)
        _, S, _ = context.size()  # context sequence length

        # calculate query for the input sequence
        q = self.q_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # calculate key and value for the context sequence
        k, v = self.kv_proj(context).split(self.n_embed, dim=2)
        k = k.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, S, hs)
        v = v.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, S, hs)

        # cross-attention; Attend: (B, nh, T, hs) x (B, nh, hs, S) -> (B, nh, T, S)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=self.dropout if self.training else 0,
                                                                 is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, S) x (B, nh, S, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
