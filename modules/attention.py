"""
Core attention building blocks for the fusion decoder: standard scaled
dot-product attention (FullAttention) and the multi-head wrapper that
projects Q/K/V and merges heads back together (AttentionLayer).
"""
from math import sqrt

import torch.nn as nn
import torch

from utils.masking import TriangularCausalMask


class FullAttention(nn.Module):
    """Standard scaled dot-product attention, with optional causal masking.

    NOTE: `factor` (this project's model_config.attention_scaling_factor)
    and `tau`/`delta` are accepted here for interface compatibility but are
    never referenced in forward() below — they have no effect on this
    implementation's behavior. Likely carried over from a different
    attention variant (e.g. ProbSparse-style attention, where `factor`
    controls the sampling amount).
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        batch_size, query_len, num_heads, head_dim = queries.shape
        scale = self.scale or 1. / sqrt(head_dim)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(batch_size, query_len, device=queries.device)
            scores.masked_fill_(attn_mask.mask, float('-inf'))

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values)

        return (out.contiguous(), attn) if self.output_attention else (out.contiguous(), None)


class AttentionLayer(nn.Module):
    """Projects queries/keys/values into multiple heads, runs the given
    attention mechanism, then merges heads back to d_model."""

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        batch_size, query_len, _ = queries.shape
        _, key_len, _ = keys.shape
        num_heads = self.n_heads

        queries = self.query_projection(queries).view(batch_size, query_len, num_heads, -1)
        keys = self.key_projection(keys).view(batch_size, key_len, num_heads, -1)
        values = self.value_projection(values).view(batch_size, key_len, num_heads, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        out = out.view(batch_size, query_len, -1)
        return self.out_projection(out), attn