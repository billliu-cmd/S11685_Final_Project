# X-Trend Model based on Paper

import torch, torch.nn as nn
from .components import TemporalBlock, DecoderBlock, CrossAttention
from .config import MODEL, NUM_FEATURES

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

      # Decoder
      self.decoder = DecoderBlock(input_dim, hid, num_assets,cfg["dropout"], self.emb)

      # Cross-Attention
      self.cross_attn = CrossAttention(hid, cfg["num_heads"], cfg["dropout"])

      # Position Head
      self.head = nn.Linear(hid, 1)

    def forward(self, target_x, target_id, ctx_x, ctx_y, ctx_id):
      enc_out = self.encoder(target_x, target_id)
      ctx_h = self.encode_contexts(ctx_x, ctx_id)
      cross_out = self.cross_attn(enc_out, ctx_h)
      dec_out = self.decoder(target_x, target_id, cross_out)

      return torch.tanh(self.head(dec_out)).squeeze(-1)
      
      
      
      
      
        

  
      
