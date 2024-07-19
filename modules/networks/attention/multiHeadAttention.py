import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_head=3):
        super(MultiHeadAttention, self).__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        self.W_Q = nn.Linear(n_embed, n_embed)
        self.W_K = nn.Linear(n_embed, n_embed)
        self.W_V = nn.Linear(n_embed, n_embed)
        self.concat = nn.Linear(n_embed, n_embed)

    def forward(self, x, context):
        B, T, C = x.size()  # sequence length, embedding dimensionality (n_embed)
        _, S, _ = context.size()  # context sequence length

        # 1）linear projection [batch_size, seq_len, d_model] ->  [batch_size, n_heads, seq_len, d_k/d_v]
        Q = (
            self.W_Q(x).view(B, -1, self.n_head, self.n_embed // self.n_head).transpose(1, 2)
        )  # Q: [batch_size, n_heads, len_q, d_k]
        K = (
            self.W_K(context).view(B, -1, self.n_head, self.n_embed // self.n_head).transpose(1, 2)
        )  # K: [batch_size, n_heads, len_k, d_k]
        V = (
            self.W_V(context).view(B, -1, self.n_head, self.n_embed // self.n_head).transpose(1, 2)
        )  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        context = F.scaled_dot_product_attention(Q, K, V, attn_mask=None,
                                                 dropout_p=0,
                                                 is_causal=False)

        context = torch.cat(
            [context[:, i, :, :] for i in range(context.size(1))], dim=-1
        )
        y = self.concat(context)  # [batch_size, len_q, d_model]
        print(y)
        return y
