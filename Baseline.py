"""
Baseline DMN (Deep Momentum Network) — paper §3.1 / Eq. 7.

This is the paper's *actual* baseline: a single TemporalBlock (VSN + LSTM +
skip connections + per-asset embedding init) followed by tanh(Linear(·)).
It is NOT a raw multi-layer LSTM; the paper's baseline already uses the full
sequence representation Ξ(·,·) as the model g(·).

The position is produced at every time-step so that the Sharpe loss can use
the warm-up masking (ignore first l_s steps).

A single nn.Embedding is created here and passed into TemporalBlock so that
every sub-component (VSN, LSTM init, SideInfoFFN) shares one "Embedding(s)"
— matching the paper's notation where no subscript distinguishes embeddings.
"""

from __future__ import annotations
import torch, torch.nn as nn
from .components import TemporalBlock
from .config import MODEL


class BaselineDMN(nn.Module):
    def __init__(self, input_dim: int, num_assets: int, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = MODEL
        self.cfg = cfg
        # one shared entity embedding for the entire model
        self.emb = nn.Embedding(num_assets, cfg["hidden_dim"])
        self.encoder = TemporalBlock(input_dim, cfg["hidden_dim"], num_assets, cfg["dropout"], self.emb)
        self.head = nn.Linear(cfg["hidden_dim"], 1)

    def forward(self, x: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
        """
        x   : [B, T, F]
        sid : [B]
        returns positions [B, T] ∈ (-1, 1)
        """
        h = self.encoder(x, sid)                     # [B,T,H]
        return torch.tanh(self.head(h)).squeeze(-1)  # [B,T]
