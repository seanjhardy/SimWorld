import math

import numpy as np
import torch


def generate_attention_mask(rows, cols):
    attn_mask = torch.ones(rows, cols, dtype=torch.bool).diag(diagonal=0)
    """row_indices = torch.arange(rows).reshape(-1, 1)
    col_indices = torch.arange(cols)
    result = (row_indices * col_indices) / (rows * cols)
    result = result < (1 - torch.rand((rows, cols)) * 0.1)"""
    result = attn_mask
    result = torch.where(result == 0, -torch.inf, 0)
    return result.to("cuda", non_blocking=True)


def cosine_embedding(embed_range, length, values=None):
    excess = length - math.floor(length/2) * 2
    pe_length = length - excess

    if values is None:
        values = np.arange(0, embed_range)
    values = np.expand_dims(values, 1)

    pe = np.zeros((values.shape[0], pe_length))
    div_term = np.exp((np.arange(0, pe_length, 2, dtype=np.float32) *
                       -(4 / pe_length)))
    pe[:, 0::2] = np.sin(values * div_term)
    pe[:, 1::2] = np.cos(values * div_term)
    pe = np.pad(pe, (0, excess))
    return pe
