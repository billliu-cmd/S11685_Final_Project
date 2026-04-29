"""
Shared building blocks (paper §3.1):
  SideInfoFFN   – Eq. 12   : fuse hidden state with asset embedding
  VSN           – Eq. 13   : variable selection network
  TemporalBlock – Eq. 14   : VSN → LSTM → skip + LayerNorm → FFN + skip
  DecoderBlock  – Eq. 19   : same idea but concatenates encoder output y_t
  Self-Attention - Eq. 17  : among contexts
  Cross-Attention - Eq. 18 : targets attending contexts
  Cross-Section : inspired by structure in https://arxiv.org/pdf/2105.10019
"""

from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


# ── Eq. 12: FFN(h_t, s) = Linear3( ELU( Linear1(h_t) + Linear2(Emb(s)) ) ) ──
class SideInfoFFN(nn.Module):
    def __init__(self, in_dim: int, hid: int, n_assets: int, dropout: float = 0.0, emb: nn.Embedding = None):
        super().__init__()
        self.emb = emb if emb is not None else nn.Embedding(n_assets, hid)
        self.lin_1 = nn.Linear(in_dim, hid)
        self.lin_2 = nn.Linear(hid, hid)
        self.lin_3   = nn.Linear(hid, hid)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
        e = self.emb(sid)
        if x.dim() == 3:
            e = e.unsqueeze(1)
        return self.lin_3(self.drop(F.elu(self.lin_1(x) + self.lin_2(e))))


# ── Eq. 13: per-feature FFN weighted by softmax importance ────────────────────
class VSN(nn.Module):
    def __init__(self, in_dim: int, hid: int, n_assets: int, dropout: float = 0.0, emb: nn.Embedding = None):
        super().__init__()
        self.in_dim = in_dim
        # weight gate
        self.emb     = emb if emb is not None else nn.Embedding(n_assets, hid)
        self.w_x     = nn.Linear(in_dim, hid)
        self.w_s     = nn.Linear(hid, hid)
        self.w_out   = nn.Linear(hid, in_dim)
        # per-feature transforms
        self.feat_fn = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hid), nn.ELU(), nn.Linear(hid, hid))
            for _ in range(in_dim)
        ])
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hid)

    def forward(self, x: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
        e = self.emb(sid).unsqueeze(1)                         # [B,1,H]
        w = torch.softmax(self.w_out(F.elu(self.w_x(x) + self.w_s(e))), dim=-1)  # [B,T,F]
        parts = torch.stack([fn(x[..., j:j+1]) for j, fn in enumerate(self.feat_fn)], dim=2)  # [B,T,F,H]
        return self.norm(self.drop((w.unsqueeze(-1) * parts).sum(2)))              # [B,T,H]


# ── Eq. 14: encoder temporal block ─────────────────────────────────────────
class TemporalBlock(nn.Module):
    def __init__(self, in_dim: int, hid: int, n_assets: int, dropout: float = 0.3, emb: nn.Embedding = None):
        super().__init__()
        self.emb  = emb if emb is not None else nn.Embedding(n_assets, hid)
        self.vsn  = VSN(in_dim, hid, n_assets, dropout, self.emb)
        # per-asset LSTM init  (h0, c0) = (FFN3(Emb(s)), FFN4(Emb(s)))
        self.h0   = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.c0   = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.lstm = nn.LSTM(hid, hid, num_layers=1, batch_first=True)
        self.norm1 = nn.LayerNorm(hid)
        self.ffn   = SideInfoFFN(hid, hid, n_assets, dropout, self.emb)
        self.norm2 = nn.LayerNorm(hid)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
        x0 = self.vsn(x, sid)                                 # [B,T,H]
        e  = self.emb(sid)                                     # [B,H]
        h0 = self.h0(e).unsqueeze(0)                           # [1,B,H]
        c0 = self.c0(e).unsqueeze(0)
        h, _ = self.lstm(x0, (h0, c0))                        # [B,T,H]
        a = self.norm1(h + x0)                                 # skip
        return self.norm2(self.drop(self.ffn(a, sid)) + a)     # skip


# ── Eq. 19: decoder block (fuses encoder output y_t) ──────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_dim: int, hid: int, n_assets: int, dropout: float = 0.3, emb: nn.Embedding = None):
        super().__init__()
        self.emb  = emb if emb is not None else nn.Embedding(n_assets, hid)
        self.vsn  = VSN(in_dim, hid, n_assets, dropout, self.emb)
        self.fuse = nn.Sequential(nn.Linear(2 * hid, hid), nn.ELU())
        self.pre_norm = nn.LayerNorm(hid)
      
        self.h0   = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.c0   = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.lstm = nn.LSTM(hid, hid, num_layers=1, batch_first=True)
        self.norm1 = nn.LayerNorm(hid)
        self.ffn   = SideInfoFFN(hid, hid, n_assets, dropout, self.emb)
        self.norm2 = nn.LayerNorm(hid)
        self.drop  = nn.Dropout(dropout)
      

    def forward(self, x: torch.Tensor, sid: torch.Tensor, enc_out: torch.Tensor):
        v = self.vsn(x, sid)
        f = self.pre_norm(self.fuse(torch.cat([v, enc_out], -1)))
        e = self.emb(sid)
        h0, c0 = self.h0(e).unsqueeze(0), self.c0(e).unsqueeze(0)
        h, _ = self.lstm(f, (h0, c0))
        a = self.norm1(h + f)
        return self.norm2(self.drop(self.ffn(a, sid)) + a)
      
# ── self & cross-attention ─────────
class SelfAttention(nn.Module):
    def __init__(self, hid, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hid, num_heads, dropout, batch_first=True)

    def forward(self, context_h):
        out, _ = self.attn(query=context_h, key=context_h, value=context_h)
        return out

class CrossAttention(nn.Module):
    def __init__(self, hid, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hid, num_heads, dropout, batch_first=True)

    def forward(self, query, key, value):
        out, _ = self.attn(query=query, key=key, value=value)
        return out

# Cross-Section Block (Self-Attention + Cross-Attention)
class CrossSectionBlock(nn.Module):
    """
    Time-synchronous peer attention:
    for each timestep t, target[t] attends only to peers at the same t.
    """
    def __init__(self, hid, num_heads=4, dropout=0.1):
        super().__init__()
        self.peer_self_attn = nn.MultiheadAttention(hid, num_heads, dropout, batch_first=True)
        self.peer_attn_norm = nn.LayerNorm(hid)
        self.peer_ffn = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.peer_ffn_norm = nn.LayerNorm(hid)

        self.target_cross_attn = nn.MultiheadAttention(hid, num_heads, dropout, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(hid)
        self.cross_ffn = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.cross_ffn_norm = nn.LayerNorm(hid)

        self.drop = nn.Dropout(dropout)

    def forward(self, target_h, peer_h, peer_mask=None):
        # target_h: [B, T, H]
        # peer_h:   [B, N, T, H]
        B, N, T, H = peer_h.shape

        if peer_mask is None:
            peer_mask = torch.ones(B, N, dtype=torch.bool, device=peer_h.device)

        # Time-synchronous relative peer tokens: peer[t] - target[t]
        target_exp = target_h.unsqueeze(1)                           # [B, 1, T, H]
        peer_rel = peer_h - target_exp                               # [B, N, T, H]

        # Flatten time so attention happens independently at each t.
        peer_bt = peer_rel.permute(0, 2, 1, 3).reshape(B * T, N, H)  # [B*T, N, H]
        target_bt = target_h.reshape(B * T, 1, H)                    # [B*T, 1, H]

        key_padding_mask = (~peer_mask).unsqueeze(1).expand(B, T, N).reshape(B * T, N)

        peer_attn, _ = self.peer_self_attn(
            peer_bt, peer_bt, peer_bt,
            key_padding_mask=key_padding_mask,
        )
        peer_ctx = self.peer_attn_norm(peer_bt + self.drop(peer_attn))
        peer_ctx = self.peer_ffn_norm(peer_ctx + self.drop(self.peer_ffn(peer_ctx)))

        cross_out, _ = self.target_cross_attn(
            target_bt, peer_ctx, peer_ctx,
            key_padding_mask=key_padding_mask,
        )
        cross_ctx = self.cross_attn_norm(target_bt + self.drop(cross_out))
        cross_ctx = self.cross_ffn_norm(cross_ctx + self.drop(self.cross_ffn(cross_ctx)))

        return cross_ctx.squeeze(1).reshape(B, T, H)

# "Legacy lag block used by earlier exploratory models; final A2/A3/A4 experiments useLagAwarePeerBlock in lag_blocks.py
class LeadLagBlock(nn.Module):
    """
    Build lagged peer tokens and let the current target sequence attend to them.
    This is additive on top of the simple cross-sectional approach.
    """
    def __init__(self, hid, lag_steps=(1, 5, 21), num_heads=4, dropout=0.1,
             include_delta_tokens=False, lag_topk_mask=None):

        super().__init__()
        self.lag_steps = tuple(int(l) for l in lag_steps if int(l) > 0)
        if not self.lag_steps:
            raise ValueError("lag_steps must contain at least one positive lag")

        self.include_delta_tokens = include_delta_tokens
        self.lag_emb = nn.Embedding(len(self.lag_steps), hid)
        if include_delta_tokens:
            self.token_type_emb = nn.Embedding(2, hid)   # 0=relative, 1=delta
        if lag_topk_mask is not None:
            self.register_buffer("lag_topk_mask", lag_topk_mask.bool())
        else:
            self.lag_topk_mask = None

        self.token_self_attn = nn.MultiheadAttention(hid, num_heads, dropout, batch_first=True)
        self.token_attn_norm = nn.LayerNorm(hid)
        self.token_ffn = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.token_ffn_norm = nn.LayerNorm(hid)

        self.target_cross_attn = nn.MultiheadAttention(hid, num_heads, dropout, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(hid)
        self.cross_ffn = nn.Sequential(nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid))
        self.cross_ffn_norm = nn.LayerNorm(hid)

        self.drop = nn.Dropout(dropout)

    def forward(self, target_h, target_id, peer_h, peer_id, peer_mask=None):
        B, N, T, H = peer_h.shape
        if peer_mask is None:
            peer_mask = torch.ones(B, N, dtype=torch.bool, device=peer_h.device)

        tokens, masks = [], []
        target_expand = target_id.unsqueeze(1).expand_as(peer_id)   # [B, N]

        for lag_idx, lag in enumerate(self.lag_steps):
            if lag >= T:
                continue

            peer_lag = peer_h[:, :, T - 1 - lag, :]                    # [B, N, H]
            target_lag = target_h[:, T - 1 - lag, :].unsqueeze(1)      # [B, 1, H]

            lag_peer_mask = peer_mask
            if self.lag_topk_mask is not None:
                allowed = self.lag_topk_mask[lag_idx][peer_id, target_expand]  # [B, N]
                lag_peer_mask = lag_peer_mask & allowed

            tok_rel = peer_lag - target_lag
            tok_rel = tok_rel + self.lag_emb.weight[lag_idx].view(1, 1, H)
            if self.include_delta_tokens:
                tok_rel = tok_rel + self.token_type_emb.weight[0].view(1, 1, H)
            tokens.append(tok_rel)
            masks.append(lag_peer_mask)

            if self.include_delta_tokens:
                peer_now = peer_h[:, :, -1, :]
                tok_delta = peer_now - peer_lag
                tok_delta = tok_delta + self.lag_emb.weight[lag_idx].view(1, 1, H)
                tok_delta = tok_delta + self.token_type_emb.weight[1].view(1, 1, H)
                tokens.append(tok_delta)
                masks.append(lag_peer_mask)

        if not tokens:
            return torch.zeros_like(target_h)

        token_x = torch.cat(tokens, dim=1)      # [B, M, H]
        token_mask = torch.cat(masks, dim=1)    # [B, M]
        key_padding_mask = ~token_mask

        token_attn, _ = self.token_self_attn(
            token_x, token_x, token_x,
            key_padding_mask=key_padding_mask,
        )
        token_ctx = self.token_attn_norm(token_x + self.drop(token_attn))
        token_ctx = self.token_ffn_norm(token_ctx + self.drop(self.token_ffn(token_ctx)))

        cross_out, _ = self.target_cross_attn(
            target_h, token_ctx, token_ctx,
            key_padding_mask=key_padding_mask,
        )
        cross_ctx = self.cross_attn_norm(target_h + self.drop(cross_out))
        cross_ctx = self.cross_ffn_norm(cross_ctx + self.drop(self.cross_ffn(cross_ctx)))
        return cross_ctx


  
