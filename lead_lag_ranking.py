from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import torch


def _corr_to_strength(rho: float, n_obs: int, eps: float = 1e-12) -> float:
    if not np.isfinite(rho):
        return 0.0
    denom = max(1.0 - float(rho) ** 2, eps)
    t_abs = abs(float(rho)) * math.sqrt(max(n_obs - 2, 1) / denom)
    return float(math.log1p(t_abs))


def build_lag_ranking_artifact(
    panel: pd.DataFrame,
    train_d,
    tk2id: dict,
    lags: Iterable[int] = (1, 5, 21),
    top_k: int = 5,
    min_obs: int = 252,
):
    """
    Build train-only lag-specific peer rankings using target_return.

    Returns matrices indexed as [peer_id, target_id].
    """
    lags = tuple(int(l) for l in lags)
    tickers = [tk for tk, _ in sorted(tk2id.items(), key=lambda kv: kv[1])]
    n = len(tickers)

    ret = (
        panel.pivot(index="date", columns="ticker", values="target_return")
        .reindex(columns=tickers)
        .sort_index()
    )
    ret_train = ret.loc[ret.index.intersection(pd.DatetimeIndex(train_d))].copy()

    signed_corr = {}
    strength = {}
    topk_mask = {}
    topk_lists = {}
    obs_count = {}

    for lag in lags:
        corr_mat = np.zeros((n, n), dtype=np.float32)
        strength_mat = np.zeros((n, n), dtype=np.float32)
        obs_mat = np.zeros((n, n), dtype=np.int32)

        for peer_id, peer_tk in enumerate(tickers):
            x = ret_train[peer_tk].shift(lag).to_numpy()

            for target_id, target_tk in enumerate(tickers):
                if peer_id == target_id:
                    continue

                y = ret_train[target_tk].to_numpy()
                mask = np.isfinite(x) & np.isfinite(y)
                n_obs = int(mask.sum())
                obs_mat[peer_id, target_id] = n_obs
                if n_obs < min_obs:
                    continue

                rho = np.corrcoef(x[mask], y[mask])[0, 1]
                if not np.isfinite(rho):
                    continue

                corr_mat[peer_id, target_id] = float(rho)
                strength_mat[peer_id, target_id] = _corr_to_strength(rho, n_obs)

        mask_mat = np.zeros((n, n), dtype=bool)
        topk_for_lag = {}

        for target_id, target_tk in enumerate(tickers):
            order = np.argsort(-strength_mat[:, target_id])
            keep = []
            for peer_id in order:
                if peer_id == target_id:
                    continue
                if strength_mat[peer_id, target_id] <= 0:
                    continue
                keep.append(peer_id)
                if len(keep) >= top_k:
                    break

            mask_mat[keep, target_id] = True
            topk_for_lag[target_tk] = [tickers[i] for i in keep]

        signed_corr[lag] = corr_mat
        strength[lag] = strength_mat
        topk_mask[lag] = mask_mat
        topk_lists[lag] = topk_for_lag
        obs_count[lag] = obs_mat

    return {
        "lags": lags,
        "tickers": tickers,
        "tk2id": dict(tk2id),
        "top_k": int(top_k),
        "min_obs": int(min_obs),
        "score_name": "target_return_lag_tstat_log1p_v1",
        "signed_corr": signed_corr,
        "strength": strength,
        "topk_mask": topk_mask,
        "topk_lists": topk_lists,
        "obs_count": obs_count,
    }


def artifact_to_lag_topk_mask_tensor(artifact: dict, lag_order: Iterable[int]) -> torch.Tensor:
    lag_order = tuple(int(l) for l in lag_order)
    return torch.from_numpy(
        np.stack([artifact["topk_mask"][lag] for lag in lag_order], axis=0)
    ).bool()


def artifact_to_lag_strength_tensor(artifact: dict, lag_order: Iterable[int]) -> torch.Tensor:
    lag_order = tuple(int(l) for l in lag_order)
    return torch.from_numpy(
        np.stack([artifact["strength"][lag] for lag in lag_order], axis=0)
    ).float()
