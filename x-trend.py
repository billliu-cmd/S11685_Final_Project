# X-Trend Model based on Page 8 from https://arxiv.org/pdf/2310.10500

import torch, torch.nn as nn
import torch.nn.functional as F
from .components import (TemporalBlock, DecoderBlock, SelfAttention, CrossAttention, SideInfoFFN)
from .config import MODEL

class XTrend(nn.Module):
    def __init__(self, input_dim, num_assets, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = MODEL
        self.cfg = cfg
        hid = cfg["hidden_dim"]
      
        # Embedding
        self.emb = nn.Embedding(num_assets, hid)

        # Encoder
        self.encoder = TemporalBlock(input_dim, hid, num_assets,cfg["dropout"], self.emb)
        
        # Self-Attention
        self.ctx_self_attn = SelfAttention(hid, cfg["num_heads"], cfg["dropout"])

        # Context FFN
        self.ctx_ffn = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))

        # Cross-Attention
        self.cross_attn = CrossAttention(hid, cfg["num_heads"], cfg["dropout"])

        # Post-Attention/Pre-Decoder
        self.post_cross_ffn = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.post_cross_norm = nn.LayerNorm(hid)

        # Decoder
        self.drop = nn.Dropout(cfg["dropout"])
        self.decoder = DecoderBlock(input_dim + 1, hid, num_assets, drop, self.emb)
        
        # Final Stage
        self.head = nn.Linear(hid, 1)

    def encode_context(self, ctx_x, ctx_id):
        B, C, lc, F = ctx_x.shape
        x_flat  = ctx_x.reshape(B * C, lc, F)
        id_flat = ctx_id.reshape(B * C)
        h_flat  = self.encoder(x_flat, id_flat)
        h_pool  = h_flat[:, -1, :]
        self.drop = nn.Dropout(cfg["dropout"])
        return h_pool.reshape(B, C, -1)
        
    def forward(self, target_x, target_id, ctx_x, ctx_y, ctx_id):
        ctx_h = self.encode_contexts(ctx_x, ctx_id)
        k = ctx_h                                        # K: raw encoder output
        v = self.ctx_ffn(self.ctx_self_attn(ctx_h))       # V: self-attn → FFN
        q = self.encoder(target_x, target_id)             # Q: target encoder
        cross_out = self.cross_attn(q, k, v)

        cross_out = self.post_cross_norm(self.drop(self.post_cross_ffn(cross_out)) + cross_out)

        dec_in = torch.cat([target_x, target_y.unsqueeze(-1)], dim=-1)
        dec_out = self.decoder(dec_in, target_id, cross_out)

        return torch.tanh(self.head(dec_out)).squeeze(-1)
      
      
      
      
      
        

  
      
