"""
Embedding building blocks used by the fusion decoder (DataEmbedding) and
the LLM encoder (PatchEmbedding).

configs.embed is 'timeF' everywhere in this pipeline, so DataEmbedding only
needs the continuous TimeFeatureEmbedding branch, not the categorical
TemporalEmbedding/FixedEmbedding lookup-table branch — those have been
dropped. Add them back if you want to experiment with 'fixed'/'learned'
time encodings.
"""
import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional encoding (not learned)."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """Projects raw feature values to d_model via a circular 1D convolution."""

    def __init__(self, c_in, d_model):
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model, kernel_size=3,
            padding=1, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.token_conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.token_conv(x.float().permute(0, 2, 1)).transpose(1, 2)


class TimeFeatureEmbedding(nn.Module):
    """Projects continuous calendar features (from utils.timefeatures) to
    d_model via a single linear layer."""

    FREQ_TO_DIM = {'h': 4, 't': 5, 's': 6, '15min': 5, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}

    def __init__(self, d_model, freq='h'):
        super().__init__()
        self.embed = nn.Linear(self.FREQ_TO_DIM[freq], d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    """Combines value, positional, and time-feature embeddings — used for
    the fusion decoder's input (paper Component ⑤'s "Data Embedding" step)."""

    def __init__(self, c_in, d_model, embed_type='timeF', freq='h', dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model, freq)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x).to(x.device)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class ReplicationPad1d(nn.Module):
    """Pads a sequence by repeating its last value, so patching never
    drops a final partial window."""

    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        replicate = x[:, :, -1:].repeat(1, 1, self.padding[-1])
        return torch.cat([x, replicate], dim=-1)


class PatchEmbedding(nn.Module):
    """Splits a time series into overlapping patches and embeds each one —
    the input stage of LLMBlock, ahead of cross-attention."""

    def __init__(self, d_model, patch_len, stride, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_layer = ReplicationPad1d((0, stride))
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, n_vars, seq_len)
        n_vars = x.shape[1]
        x = self.padding_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: (batch, n_vars, patch_num, patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)
        return self.dropout(x), n_vars