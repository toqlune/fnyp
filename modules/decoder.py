"""
Fusion decoder (paper Component ⑤): a stack of self-attention +
cross-attention decoder layers, each followed by a feed-forward block.

Each DecoderLayer produces two branches from the same input: a
self-attention branch (target series attending to its own history) and a
cross-attention branch (that representation attending to the covariate
encoding). The two branches are combined via (M4) Adaptive Gated Fusion
(see modules.adaptive_gated_fusion.AdaptiveGatedFusion) — a learned gate
that decides how much of the covariate signal to blend in, rather than a
fixed additive residual — right before that step's Add & Norm.
"""
import torch.nn as nn
import torch.nn.functional as F

from modules.adaptive_gated_fusion import AdaptiveGatedFusion


class DecoderLayer(nn.Module):

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.adaptive_gated_fusion = AdaptiveGatedFusion(d_model)  # (M4)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)  # after the self-attention residual
        self.norm2 = nn.LayerNorm(d_model)  # after the M4 gated-fusion residual
        self.norm3 = nn.LayerNorm(d_model)  # after the feed-forward residual
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # self-attention on the decoder's own (causal) sequence
        self_attention_out = self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x + self_attention_out)

        # cross-attention to the covariate encoder's output
        cross_attention_out = self.dropout(
            self.cross_attention(x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0])

        # (M4) Adaptive Gated Fusion: blend the self-attended representation
        # with the cross-attention (covariate) branch via a learned gate,
        # instead of a plain additive residual
        x = self.norm2(self.adaptive_gated_fusion(x, cross_attention_out))

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):

    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x