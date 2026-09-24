"""
RevIN — Reversible Instance Normalization.

Normalizes each target series per-instance (per batch sample, per channel)
before it enters the model, and reverses that exact transform on the
model's output. This absorbs distribution shift between training and
inference windows, since each window is centered/scaled using its own
statistics rather than one global statistic fixed at training time.

The mean/stdev are computed fresh on every forward pass ('norm' mode) and
cached so the exact inverse can be applied later in the same pass ('denorm'
mode) — this only works as a live nn.Module, not a static preprocessing step.
"""
import torch
import torch.nn as nn


class RevIN(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            # learnable per-channel scale/shift applied after normalization
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        raise NotImplementedError(f"RevIN mode '{mode}' is not supported")

    def _get_statistics(self, x):
        # one mean/stdev per (batch, channel), reduced over the seq_len dim
        dims = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dims, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dims, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps ** 2)
        return x * self.stdev + self.mean