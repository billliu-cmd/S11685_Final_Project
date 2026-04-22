from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LagAwarePeerBlock(nn.Module):
    """
    Custom lag-aware peer attention over (peer, lag) candidates.

    This block supports:
    - A2: lag-specific Bennett-style soft bias via rank_strength
    - A3: lag-specific hard top-k ranking mask
    - A4: same as A3 plus delta-valued V path
    """

    def __init__(
        self,
        hid: int,
        dropout: float = 0.1,
        lag_set: Iterable[int] = (1, 5, 21),
        top_k: int = 3,
        rank_strength: Optional[torch.Tensor] = None,
        rank_topk_mask: Optional[torch.Tensor] = None,
        use_bennett: bool = False,
        alpha_init: float = 0.1,
        use_delta_value: bool = False,
    ) -> None:
        super().__init__()
        self.hid = hid
        self.lag_set = tuple(int(x) for x in lag_set)
        if any(l < 1 for l in self.lag_set):
            raise ValueError("lag_set entries must be >= 1")
        self.L = len(self.lag_set)
        self.top_k = int(top_k)
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")

        self.W_Q = nn.Linear(hid, hid)
        self.W_K = nn.Linear(hid, hid)
        self.W_V = nn.Linear(hid, hid)

        self.drop = nn.Dropout(dropout)
        self.out_ffn = nn.Sequential(
            nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, hid),
        )
        self.norm = nn.LayerNorm(hid)

        self.register_buffer(
            "lag_tensor", torch.tensor(self.lag_set, dtype=torch.long),
            persistent=False,
        )

        if rank_strength is not None:
            S = torch.as_tensor(rank_strength, dtype=torch.float32)
            if S.ndim != 3:
                raise ValueError(f"rank_strength must be [L,N,N]; got {tuple(S.shape)}")
            if S.shape[0] != self.L or S.shape[1] != S.shape[2]:
                raise ValueError(
                    f"rank_strength must have shape [L,N,N] with L={self.L}; got {tuple(S.shape)}"
                )
            self.register_buffer("log_rank_strength", torch.log(S.clamp_min(1e-6)))
        else:
            self.register_buffer("log_rank_strength", torch.zeros(0), persistent=False)

        if rank_topk_mask is not None:
            M = torch.as_tensor(rank_topk_mask, dtype=torch.bool)
            if M.ndim != 3:
                raise ValueError(f"rank_topk_mask must be [L,N,N]; got {tuple(M.shape)}")
            if M.shape[0] != self.L or M.shape[1] != M.shape[2]:
                raise ValueError(
                    f"rank_topk_mask must have shape [L,N,N] with L={self.L}; got {tuple(M.shape)}"
                )
            self.register_buffer("rank_topk_mask", M)
        else:
            self.register_buffer("rank_topk_mask", torch.zeros(0, dtype=torch.bool), persistent=False)

        self.use_bennett = bool(use_bennett) and self.log_rank_strength.numel() > 0
        if self.use_bennett:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_parameter("alpha", None)

        self.use_rank_mask = self.rank_topk_mask.numel() > 0
        self.use_delta_value = bool(use_delta_value)

    def forward(
        self,
        target_h: torch.Tensor,
        peer_h: torch.Tensor,
        peer_mask: Optional[torch.Tensor] = None,
        target_id: Optional[torch.Tensor] = None,
        peer_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, T, H = peer_h.shape
        if target_h.shape != (B, T, H):
            raise ValueError(
                f"target_h shape {tuple(target_h.shape)} does not match "
                f"peer_h-derived (B, T, H)=({B}, {T}, {H})"
            )
        L = self.L
        device = peer_h.device
        dtype = peer_h.dtype

        lag_t = self.lag_tensor.to(device=device)
        t_arange = torch.arange(T, device=device)
        t_idx = t_arange.unsqueeze(0) - lag_t.unsqueeze(1)           # [L, T]
        valid_time = t_idx >= 0
        t_idx_c = t_idx.clamp_min(0)

        peer_h_lag = peer_h[:, :, t_idx_c, :]                        # [B, N, L, T, H]
        peer_h_lag = peer_h_lag.permute(0, 3, 1, 2, 4).contiguous()  # [B, T, N, L, H]

        Q = self.W_Q(target_h)                                       # [B, T, H]
        K = self.W_K(peer_h_lag)                                     # [B, T, N, L, H]

        if self.use_delta_value:
            peer_h_now = peer_h.unsqueeze(3).expand(-1, -1, -1, L, -1)   # [B,N,T,L,H]
            peer_h_now = peer_h_now.permute(0, 2, 1, 3, 4).contiguous()   # [B,T,N,L,H]
            V_src = peer_h_now - peer_h_lag
        else:
            V_src = peer_h_lag
        V = self.W_V(V_src)

        logits = torch.einsum("bth,btnlh->btnl", Q, K) / math.sqrt(H)

        if self.use_bennett and target_id is not None and peer_id is not None:
            tgt_b = target_id.unsqueeze(1).expand(-1, N)            # [B, N]
            prior = torch.stack(
                [self.log_rank_strength[l_idx][peer_id, tgt_b] for l_idx in range(L)],
                dim=-1,
            ).to(dtype)                                             # [B, N, L]
            logits = logits + self.alpha.to(dtype) * prior.unsqueeze(1)

        mask_t = valid_time.permute(1, 0).unsqueeze(0).unsqueeze(2)  # [1, T, 1, L]
        logits = logits.masked_fill(~mask_t, float("-inf"))

        if peer_mask is not None:
            mp = peer_mask.to(torch.bool).unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            logits = logits.masked_fill(~mp, float("-inf"))

        if self.use_rank_mask and target_id is not None and peer_id is not None:
            tgt_b = target_id.unsqueeze(1).expand(-1, N)            # [B, N]
            rank_mask = torch.stack(
                [self.rank_topk_mask[l_idx][peer_id, tgt_b] for l_idx in range(L)],
                dim=-1,
            )                                                       # [B, N, L]
            logits = logits.masked_fill(~rank_mask.unsqueeze(1), float("-inf"))

        NL = N * L
        k = min(self.top_k, NL)
        logits_flat = logits.reshape(B, T, NL)
        topk_vals, topk_idx = logits_flat.topk(k, dim=-1)

        all_inf = torch.isinf(topk_vals).all(dim=-1, keepdim=True)
        topk_safe = topk_vals.masked_fill(torch.isinf(topk_vals), -1e9)
        weights = F.softmax(topk_safe, dim=-1)
        weights = weights * (~all_inf).to(weights.dtype)
        weights = self.drop(weights)

        V_flat = V.reshape(B, T, NL, H)
        idx_h = topk_idx.unsqueeze(-1).expand(-1, -1, -1, H)
        V_sel = torch.gather(V_flat, 2, idx_h)
        ll_y = (weights.unsqueeze(-1) * V_sel).sum(dim=2)

        ll_y = self.norm(ll_y + self.drop(self.out_ffn(ll_y)))
        return ll_y
