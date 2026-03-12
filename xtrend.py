"""
X-Trend "Sharpe" variant  (paper §3, Fig. 4).

Architecture:
  Encoder:
    query_block   Ξ_query   – encodes target sequence
    key_block     Ξ_key     – encodes context features  (NO next-day returns)
    value_block   Ξ_value   – encodes context features + next-day returns
  Self-attention  Eq. 17    – over context value hidden states
  Cross-attention Eq. 18    – target query → context keys / updated values
  Decoder         Eq. 19    – fuses cross-attention output with target features
  Position head             – tanh(Linear(decoder_out))

This is the plain Sharpe-loss version (no joint MLE / QRE).
"""

from __future__ import annotations
import torch, torch.nn as nn
from .components import TemporalBlock, DecoderBlock
from .config import ModelConfig


class XTrend(nn.Module):
    def __init__(self, input_dim: int, num_assets: int, cfg: ModelConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg
        H = cfg.hidden_dim

        # ── encoder blocks (separate params for Q / K / V) ────────────────
        self.query_block = TemporalBlock(input_dim, H, num_assets, cfg.dropout)
        self.key_block   = TemporalBlock(input_dim, H, num_assets, cfg.dropout)
        self.value_block = TemporalBlock(input_dim + 1, H, num_assets, cfg.dropout)  # +1 for returns

        # ── self-attention over context values  (Eq. 17) ──────────────────
        self.self_attn = nn.MultiheadAttention(H, cfg.num_heads, dropout=0.0, batch_first=True)
        self.self_ffn  = nn.Sequential(nn.Linear(H, H), nn.ELU(), nn.Linear(H, H))
        self.self_norm = nn.LayerNorm(H)

        # ── cross-attention: target → context  (Eq. 18) ──────────────────
        self.cross_attn = nn.MultiheadAttention(H, cfg.num_heads, dropout=0.0, batch_first=True)
        self.cross_ffn  = nn.Sequential(nn.Linear(H, H), nn.ELU(), nn.Linear(H, H))
        self.cross_norm = nn.LayerNorm(H)

        # ── decoder + position head ───────────────────────────────────────
        self.decoder = DecoderBlock(input_dim, H, num_assets, cfg.dropout)
        self.head    = nn.Linear(H, 1)

    def forward(
        self,
        target_x:  torch.Tensor,   # [B, lt, F]
        target_id: torch.Tensor,   # [B]
        ctx_x:     torch.Tensor,   # [B, C, lc, F]
        ctx_y:     torch.Tensor,   # [B, C, lc]
        ctx_id:    torch.Tensor,   # [B, C]
        return_attn: bool = False,
    ):
        B, C, lc, F = ctx_x.shape
        H = self.cfg.hidden_dim

        # query
        q = self.query_block(target_x, target_id)                        # [B,lt,H]

        # flatten contexts for encoding
        cx_flat  = ctx_x.reshape(B * C, lc, F)
        cid_flat = ctx_id.reshape(B * C)

        # keys – context features only, take final hidden state ("F" mode)
        k = self.key_block(cx_flat, cid_flat)[:, -1].reshape(B, C, H)   # [B,C,H]

        # values – context features + returns
        cv_in = torch.cat([ctx_x, ctx_y.unsqueeze(-1)], dim=-1)         # [B,C,lc,F+1]
        cv_flat = cv_in.reshape(B * C, lc, F + 1)
        v = self.value_block(cv_flat, cid_flat)[:, -1].reshape(B, C, H) # [B,C,H]

        # self-attention over context (Eq. 17)
        v2, sa_w = self.self_attn(v, v, v, need_weights=return_attn)
        v2 = self.self_norm(v2 + self.self_ffn(v2))

        # cross-attention (Eq. 18)
        y, ca_w = self.cross_attn(q, k, v2, need_weights=return_attn)
        y = self.cross_norm(self.cross_ffn(y))

        # decoder
        dec = self.decoder(target_x, target_id, y)                       # [B,lt,H]
        positions = torch.tanh(self.head(dec)).squeeze(-1)               # [B,lt]

        out = {"positions": positions}
        if return_attn:
            out["self_attn"] = sa_w
            out["cross_attn"] = ca_w
        return out
