import torch
import torch.nn as nn

from .x_trend import XTrend
from .components import CrossSectionBlock

# Extension 1: Cross-Extension
class XTrendCS(XTrend):
    def __init__(self, input_dim, num_assets, cfg=None):
        super().__init__(input_dim, num_assets, cfg)
        hid = self.cfg["hidden_dim"]

        # Slightly stronger regularization on the CS branch.
        cs_dropout = max(self.cfg["dropout"] * 2.0, 0.2)

        self.cs_block = CrossSectionBlock(
            hid, self.cfg["num_heads"], cs_dropout
        )

        # Residual adapter:
        # enc_y = reg_y + alpha * cs_proj(cs_y)
        # Zero-init means epoch 0 == plain X-Trend exactly.
        self.cs_proj = nn.Linear(hid, hid)
        nn.init.zeros_(self.cs_proj.weight)
        nn.init.zeros_(self.cs_proj.bias)

        # Per-feature, per-timestep gate controlling how much CS info to use.
        self.gate = nn.Linear(2 * hid, hid)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)   # sigmoid(-2) ≈ 0.12

    def encode_peers(self, peer_x, peer_id):
        B, N, T, F = peer_x.shape
        x_flat = peer_x.reshape(B * N, T, F)
        id_flat = peer_id.reshape(B * N)
        h_flat = self.query_encoder(x_flat, id_flat)
        return h_flat.reshape(B, N, T, -1)

    def forward(self, target_x, target_id, ctx_x, ctx_y, ctx_id,
                peer_x=None, peer_id=None, peer_mask=None):
        q = self.query_encoder(target_x, target_id)                   # [B, T, H]

        # Historical CPD branch (unchanged)
        k, v = self.encode_contexts(ctx_x, ctx_y, ctx_id)
        v_prime = self.ctx_ffn(self.ctx_self_attn(v))
        reg_attn = self.cross_attn(q, k, v_prime)
        reg_y = self.post_cross_norm(self.post_cross_ffn(reg_attn))   # [B, T, H]

        # Cross-sectional branch as a residual improvement to reg_y
        if peer_x is not None and peer_id is not None:
            peer_h = self.encode_peers(peer_x, peer_id)               # [B, N, T, H]
            cs_y = self.cs_block(q, peer_h, peer_mask)                # [B, T, H]

            delta = self.cs_proj(cs_y)                                # zero at init
            alpha = torch.sigmoid(self.gate(torch.cat([reg_y, cs_y], dim=-1)))
            enc_y = reg_y + alpha * delta                             # == reg_y at init
        else:
            enc_y = reg_y

        dec_out = self.decoder(target_x, target_id, enc_y)
        return torch.tanh(self.head(dec_out)).squeeze(-1)
