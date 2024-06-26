import math

import torch
from torch import nn
from torch.nn import functional as F


class CrossAttention(nn.Module):
    def __init__(self, n_embed, n_head=6, bias=True,
                 dropout=0., attn_dropout=0.):
        super().__init__()
        assert n_embed % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.q_attn = nn.Linear(n_embed, n_embed, bias=bias)
        self.k_attn = nn.Linear(n_embed, n_embed, bias=bias)
        self.q_attn = nn.Linear(n_embed, n_embed, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embed = n_embed
        self.dropout = dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size()  # sequence length, embedding dimensionality (n_embed)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_attn(x).split(self.n_embed, dim=2)
        k = self.k_attn(x).split(self.n_embed, dim=2)
        v = self.v_attn(x).split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=self.dropout if self.training else 0,
                                                                 is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
