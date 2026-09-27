"""
AdaptiveGatedFusion — (M4) Adaptive Gated Fusion, improved-architecture
addition, used inside the fusion decoder's DecoderLayer (see decoder.py).

The decoder layer produces two branches from the same input: a
self-attention branch (the target series attending to its own history)
and a cross-attention branch (that representation attending to the
covariate encoding). Rather than combining them with a fixed additive
residual, this module learns a per-position, per-channel gate — a
highway/GRU-style update gate — that decides how much of the covariate
(cross-attention) signal to blend in versus how much of the target's own
self-attended representation to keep. The gate is conditioned on both
branches jointly, so the network can lean on covariate information when
it's informative and suppress it when it isn't, instead of always
weighting the two equally.
"""
import torch
import torch.nn as nn


class AdaptiveGatedFusion(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.gate_projection = nn.Linear(2 * d_model, d_model)

    def forward(self, self_attention_branch, cross_attention_branch):
        gate = torch.sigmoid(
            self.gate_projection(torch.cat([self_attention_branch, cross_attention_branch], dim=-1)))
        return gate * cross_attention_branch + (1 - gate) * self_attention_branch