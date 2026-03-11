"""
Building blocks from paper
  SideInfoFFN   – Eq. 12   : fuse hidden state with asset embedding
  VSN           – Eq. 13   : variable selection network
  TemporalBlock – Eq. 14a-d: VSN → LSTM → skip + LayerNorm → FFN + skip
  DecoderBlock  – Eq. 19a-d: same idea but concatenates encoder output y_t
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


# Eq. 12: FFN(h_t, s) = Linear3( ELU( Linear1(h_t) + Linear2(Emb(s)) ) )
class SideInfoFFN(nn.Module):
    def __init__(self, in_dim: int, hid: int, n_assets: int, dropout: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(n_assets, hid)
        self.lin_1 = nn.Linear(in_dim, hid)
        self.lin_2 = nn.Linear(hid, hid)
        self.lin_3  = nn.Linear(hid, hid)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
        e = self.emb(sid)
        if x.dim() == 3:
            e = e.unsqueeze(1)
        return self.lin_3(self.drop(F.elu(self.lin_1(x) + self.lin_2(e))))


# Eq. 13: per-feature FFN weighted by softmax importance 
class VSN(nn.Module):
    def __init__(self, in_dim: int, hid: int, n_assets: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        # weight gate
        self.emb     = nn.Embedding(n_assets, hid)
        self.w_x     = nn.Linear(in_dim, hid)
        self.w_s     = nn.Linear(hid, hid)
        self.w_out   = nn.Linear(hid, in_dim)
        # per-feature transforms
        self.feat_fn = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hid), nn.ELU(), nn.Linear(hid, hid)) # FNN without side info per feature
            for _ in range(in_dim)
        ])
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hid)

    def forward(self, x: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
        e = self.emb(sid).unsqueeze(1)                         # [B,1,H]
        w = torch.softmax(self.w_out(F.elu(self.w_x(x) + self.w_s(e))), dim=-1)  # [B,T,F]
        parts = torch.stack([fn(x[..., j:j+1]) for j, fn in enumerate(self.feat_fn)], dim=2)  # [B,T,F,H]
        return self.norm(self.drop((w.unsqueeze(-1) * parts).sum(2)))              # [B,T,H]


# Eq. 14: Encoder temporal block 
class TemporalBlock(nn.Module):
    def __init__(self, in_dim: int, hid: int, n_assets: int, dropout: float = 0.3):
        super().__init__()
        self.vsn  = VSN(in_dim, hid, n_assets, dropout)
        # per-asset LSTM init (h0, c0) = (FFN3(Emb(s)), FFN4(Emb(s)))
        self.emb  = nn.Embedding(n_assets, hid)
        self.h0   = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.c0   = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.lstm = nn.LSTM(hid, hid, num_layers=1, batch_first=True)
        self.norm1 = nn.LayerNorm(hid)
        self.ffn   = SideInfoFFN(hid, hid, n_assets, dropout)
        self.norm2 = nn.LayerNorm(hid)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
        x_p = self.vsn(x, sid)                                 # [B,T,H]
        e  = self.emb(sid)                                     # [B,H]
        h = self.h0(e).unsqueeze(0)                           # [1,B,H]
        c = self.c0(e).unsqueeze(0)
        h, _ = self.lstm(x_p, (h, c))                        # [B,T,H]
        a = self.norm1(h + x_p)                                 # skip
        return self.norm2(self.drop(self.ffn(a, sid)) + a)     # skip


# Eq. 19: Decoder block 
class DecoderBlock(nn.Module):
    def __init__(self, in_dim: int, hid: int, n_assets: int, dropout: float = 0.3):
        super().__init__()
        self.vsn  = VSN(in_dim, hid, n_assets, dropout)
        self.fnn1 = nn.Sequential(nn.Linear(2 * hid, hid), nn.ELU(), nn.Linear(hid, hid),)
        self.pre_norm = nn.LayerNorm(hid)
        self.emb  = nn.Embedding(n_assets, hid)
        self.h0   = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.c0   = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.lstm = nn.LSTM(hid, hid, num_layers=1, batch_first=True)
        self.norm1 = nn.LayerNorm(hid)
        self.ffn2   = SideInfoFFN(hid, hid, n_assets, dropout)
        self.norm2 = nn.LayerNorm(hid)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sid: torch.Tensor, enc_out: torch.Tensor):
        v = self.vsn(x, sid)                                   # [B,T,H]
        x_p = self.pre_norm(self.fnn1(torch.cat([v, enc_out], -1)))
        e = self.emb(sid)
        h = self.h0(e).unsqueeze(0)
        c = self.c0(e).unsqueeze(0)
        h, _ = self.lstm(x_p, (h, c))
        a = self.norm1(h + x_p)
        return self.norm2(self.drop(self.ffn2(a, sid)) + a)
