"""
CrossAttentionLayer — Cross-Attention (paper Component ②), reworked to
attend over a learned text-prototype bank (see llm_block.py) instead of a
projected vocabulary.

Queries come from the time-series patches (d_model).
Keys/Values come from the text-prototype bank (d_llm).

A GLU gate sits right after the multi-head attention output: it projects
to twice the width, then GLU(dim=-1) splits that in half and computes
a * sigmoid(b), letting the network learn to suppress noisy cross-attention
output before it reaches the frozen LLM.
"""
from math import sqrt

import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

        # (d_keys*n_heads) -> (2 * d_keys*n_heads) -> GLU -> (d_keys*n_heads)
        self.glu = nn.Sequential(
            nn.Linear(d_keys * n_heads, 2 * d_keys * n_heads),
            nn.GLU(dim=-1),
        )

    def forward(self, target_embedding, source_embedding, value_embedding):
        # target_embedding: (batch, num_patches, d_model)  — time-series patches
        # source_embedding: (num_prototypes, d_llm)         — prototype keys
        # value_embedding:  (num_prototypes, d_llm)         — prototype values
        B, T, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, T, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self._reprogram(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, T, -1)
        out = self.glu(out)  # gated bottleneck before the frozen LLM sees it
        return self.out_projection(out)

    def _reprogram(self, target_embedding, source_embedding, value_embedding):
        # Equations (3) and (4) from the paper
        _, _, _, E = target_embedding.shape
        scale = 1. / sqrt(E)

        # how much each patch attends to each prototype
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))

        # weighted sum of prototype values
        return torch.einsum("bhls,she->blhe", attn, value_embedding)