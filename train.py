from __future__ import annotations
import copy, math
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from tqdm.auto import tqdm

from .config import TRAIN, MODEL
# ═══════════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════════
def sharpe_loss_tc(pos: torch.Tensor, ret: torch.Tensor,
                warmup: int = 63, eps: float = 1e-9) -> torch.Tensor:
    """Negative annualised Sharpe with warm-up masking."""
    if warmup > 0:
        pos, ret = pos[:, warmup:], ret[:, warmup:]
    raw = pos * ret
    turnover = torch.cat([pos[:, :1].abs(),(pos[:, 1:] - pos[:, :-1]).abs()], dim=1)
    net = raw - (cost_bps / 10_000) * turnover
    mu = net.mean()
    var = (net.pow(2).mean() - mu.pow(2)).clamp_min(eps)
    return -math.sqrt(252.0) * mu / var.sqrt()

# ═══════════════════════════════════════════════════════════════════════════════
# Performance Metrics
# ═══════════════════════════════════════════════════════════════════════════════
def annualised_sharpe(daily: pd.Series, eps=1e-9) -> float:
    x = daily.dropna().values.astype(np.float64)
    if len(x) == 0:
        return np.nan
    mu = x.mean()
    var = max((x**2).mean() - mu**2, eps)
    return float(np.sqrt(252) * mu / np.sqrt(var))


def max_drawdown(daily: pd.Series) -> float:
    eq = (1 + daily.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())

# ═══════════════════════════════════════════════════════════════════════════════
# Baseline Step
# ═══════════════════════════════════════════════════════════════════════════════
def _baseline_step(model, batch, device, warmup):
    x   = batch["x"].to(device)
    y   = batch["y"].to(device)       # scalar per sample
    sid = batch["sid"].to(device)
    pos = model(x, sid)               # [B,T]

    # baseline dataset gives scalar y; expand to match warm-up slicing
    # The target_return at the endpoint of each window is replicated
    # across time-steps so the Sharpe loss operates on the full sequence.
    y_seq = y.unsqueeze(1).expand_as(pos)
    loss = sharpe_loss(pos, y_seq, warmup)
    return loss, pos, y, batch["date"], batch["ticker"]

def train_epoch(model, loader, optim, device, warmup, max_gn, step_fn, scheduler=None):
    model.train()
    total_loss, n = 0.0, 0
    for batch in tqdm(loader, leave=False, desc="train"):
        loss, *_ = step_fn(model, batch, device, warmup)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if max_gn:
            nn.utils.clip_grad_norm_(model.parameters(), max_gn)
        optim.step()
        total_loss += loss.item() * batch[next(iter(batch))].size(0)
        n += batch[next(iter(batch))].size(0)
    if scheduler is not None:
        scheduler.step()
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, warmup, step_fn):
    model.eval()
    total_loss, n = 0.0, 0
    rows = []
    for batch in tqdm(loader, leave=False, desc="eval"):
        loss, pos, ret, dates, tickers = step_fn(model, batch, device, warmup)
        bs = pos.size(0)
        total_loss += loss.item() * bs
        n += bs
        # take last-step position for backtesting
        p_last = pos[:, -1].cpu().numpy() if pos.dim() == 2 else pos.cpu().numpy()
        r_last = ret[:, -1].cpu().numpy() if ret.dim() == 2 else ret.cpu().numpy()
        for i in range(bs):
            rows.append({"date": dates[i], "ticker": tickers[i],
                         "position": float(p_last[i]),
                         "target_return": float(r_last[i])})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["strategy_return"] = df["position"] * df["target_return"]
    daily = df.groupby("date")["strategy_return"].mean().sort_index()
    return {
        "loss": total_loss / max(n, 1),
        "pred_df": df,
        "daily_returns": daily,
        "sharpe": annualised_sharpe(daily),
        "max_drawdown": max_drawdown(daily),
    }
  
# ═══════════════════════════════════════════════════════════════════════════════
# Actual training 
# ═══════════════════════════════════════════════════════════════════════════════
def fit(model, train_loader, val_loader, device,
        step_fn,                         # _baseline_step or _xtrend_step
        tcfg = None,
        mcfg = None):
    if tcfg is None: tcfg = TRAIN
    if mcfg is None: mcfg = MODEL
    warmup = mcfg["warmup_steps"]

    optim = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=tcfg["epochs"])

    best_state = copy.deepcopy(model.state_dict())
    best_sharpe = -float("inf")
    wait = 0
    history = []

    for ep in range(1, tcfg["epochs"] + 1):
        tl = train_epoch(model, train_loader, optim, device, warmup, tcfg["max_grad_norm"], step_fn, scheduler)
        vs = eval_epoch(model, val_loader, device, warmup, step_fn)

        cur_lr = scheduler.get_last_lr()[0]
        row = {"epoch": ep, "train_loss": tl, "val_loss": vs["loss"],
               "val_sharpe": vs["sharpe"], "val_mdd": vs["max_drawdown"], "lr": cur_lr}
        history.append(row)
        print(f"Ep {ep:03d} | trn {tl:.4f} | val {vs['loss']:.4f} | "
              f"sharpe {vs['sharpe']:.4f} | mdd {vs['max_drawdown']:.4f} | lr {cur_lr:.2e}")

        if vs["sharpe"] > best_sharpe + 1e-4:
            best_sharpe = vs["sharpe"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= tcfg["patience"]:
                print(f"Early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

