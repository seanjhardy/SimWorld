import math

import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_embed, n_inputs=0, n_head=6,
                 causal=False, bias=True,
                 dropout=0., attn_dropout=0., allow_flash=True):
        super().__init__()
        assert n_embed % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embed = n_embed
        self.dropout = dropout
        self.causal = causal

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = allow_flash and hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(n_inputs, n_inputs))
                                 .view(1, 1, n_inputs, n_inputs))

    def forward(self, x):
        B, T, C = x.size()  # sequence length, embedding dimensionality (n_embed)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=self.dropout if self.training else 0,
                                                                 is_causal=self.causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
