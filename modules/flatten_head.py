"""
FlattenHead — Output Projection (paper Component ⑥).

Flattens the frozen LLM's patch-wise output and projects it to the
forecast horizon length.
"""
import torch.nn as nn


class FlattenHead(nn.Module):

    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)