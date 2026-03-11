"""
Baseline DMN
"""

from __future__ import annotations
import torch, torch.nn as nn
from .components import TemporalBlock
from .config import MODEL


class BaselineDMN(nn.Module):
    def __init__(self, input_dim: int, num_assets: int, cfg = None):
        super().__init__()
        if cfg is None:
            cfg = MODEL
        self.cfg = cfg
        self.encoder = TemporalBlock(input_dim, cfg["hidden_dim"], num_assets, cfg["dropout"])
        self.head = nn.Linear(cfg["hidden_dim"], 1)

    def forward(self, x: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
        """
        x   : [B, T, F]
        sid : [B]
        returns positions [B, T] ∈ (-1, 1)
        """
        h = self.encoder(x, sid)                     # [B,T,H]
        return torch.tanh(self.head(h)).squeeze(-1)  # [B,T]
