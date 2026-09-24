"""
Causal attention mask for the fusion decoder's self-attention.

The original masking.py/mask_method.py pair also contained a large set of
missing-data mask generators (MCAR, MAR, MNAR, RDO, block-missing) for an
imputation task this project doesn't use, plus a ProbMask class needed only
by Informer's ProbAttention (also out of scope). None of that is reachable
from the forecasting pipeline, so it's dropped here rather than carried
forward as dead weight.
"""
import torch


class TriangularCausalMask:
    """Upper-triangular boolean mask: True marks positions a query is not
    allowed to attend to (everything strictly after itself in the sequence)."""

    def __init__(self, B, L, device="cpu"):
        mask_shape = (B, 1, L, L)
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask