from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yfinance as yf

from .config import (
    DATA, VOL_TARGET, VOL_LOOKBACK,
    RETURN_HORIZONS, MACD_PAIRS, EPS, DEFAULT_TICKERS,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Feature engineering
# ═══════════════════════════════════════════════════════════════════════════════
def _halflife(ts: int) -> float:
    return np.log(0.5) / np.log(1.0 - 1.0 / ts)

def _macd(close: pd.Series, S: int, L: int) -> pd.Series:
    """MACD feature normalised by rolling std (Eq. 4a-b)."""
    s_ewm = close.ewm(halflife=_halflife(S), adjust=False).mean()
    l_ewm = close.ewm(halflife=_halflife(L), adjust=False).mean()
    m = (s_ewm - l_ewm) / close.rolling(63, min_periods=63).std().clip(lower=EPS)
    return m / m.rolling(252, min_periods=252).std().clip(lower=EPS)


def build_panel(cfg=None) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """Download ETF prices and build the full feature panel."""
    if cfg is None:
        cfg = DATA

    tickers = list(cfg.get("tickers", DEFAULT_TICKERS))
    prices = yf.download(tickers, start=cfg["start"], end=cfg["end"], progress=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.sort_index().dropna(how="all")

    temp = []
    for ticker in prices.columns:
        close = prices[ticker].dropna()
        close = close[close > EPS].copy()
        df = pd.DataFrame({"date": close.index, "ticker": ticker, "close": close.values})

        ret1 = close.pct_change()
        vol  = ret1.ewm(span=VOL_LOOKBACK, min_periods=VOL_LOOKBACK, adjust=False).std()
        scaled = ret1 * VOL_TARGET / (vol.shift(1) * np.sqrt(252)).clip(lower=EPS)

        df["daily_vol"]      = vol.values
        df["target_return"]  = scaled.shift(-1).values          # next-day vol-scaled return

        for h in RETURN_HORIZONS:
            df[f"norm_ret_{h}"] = ((close / close.shift(h) - 1) / vol.clip(lower=EPS) / np.sqrt(h)).values
        for S, L in MACD_PAIRS:
            df[f"macd_{S}_{L}"] = _macd(close, S, L).values

        temp.append(df)

    panel = pd.concat(temp, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])

    # common start so every asset exists on every date
    common_start = panel.groupby("ticker")["date"].min().max()
    panel = panel[panel["date"] >= common_start].copy()

    feature_cols = [f"norm_ret_{h}" for h in RETURN_HORIZONS] + \
                   [f"macd_{S}_{L}" for S, L in MACD_PAIRS]
    panel = panel.dropna(subset=["target_return", "daily_vol"] + feature_cols).copy()

    tks = sorted(panel["ticker"].unique())
    tk2id = {t: i for i, t in enumerate(tks)}
    panel["asset_id"] = panel["ticker"].map(tk2id).astype(np.int64)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel, feature_cols, tk2id


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Time-ordered split
# ═══════════════════════════════════════════════════════════════════════════════
def time_split(panel: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    dates = sorted(panel["date"].unique())
    n = len(dates)
    t1, t2 = int(n * train_frac), int(n * (train_frac + val_frac))
    return (
        pd.DatetimeIndex(dates[:t1]),
        pd.DatetimeIndex(dates[t1:t2]),
        pd.DatetimeIndex(dates[t2:]),
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Baseline window dataset
# ═══════════════════════════════════════════════════════════════════════════════
class WindowDataset(Dataset):
    """Each sample: [lookback, F] features  →  scalar target_return at endpoint."""

    def __init__(self, panel: pd.DataFrame, feature_cols: Sequence[str],
                 dates: Sequence[pd.Timestamp], lookback: int = 126):
        self.fcols = list(feature_cols)
        self.lb = lookback
        dates_set = set(pd.to_datetime(dates))

        self.groups: Dict[str, dict] = {}
        self.samples: List[Tuple[str, int]] = []

        for tk, frame in panel.groupby("ticker", sort=False):
            frame = frame.sort_values("date").reset_index(drop=True)
            self.groups[tk] = {
                "x": torch.tensor(frame[self.fcols].values, dtype=torch.float32),
                "y": torch.tensor(frame["target_return"].values, dtype=torch.float32),
                "dates": pd.to_datetime(frame["date"]).tolist(),
                "asset_id": int(frame["asset_id"].iloc[0]),
            }
            for i in range(self.lb - 1, len(frame)):
                if frame["date"].iloc[i] in dates_set:
                    self.samples.append((tk, i))

        # sort by endpoint index and ticker
        self.samples.sort(key=lambda s: (self.groups[s[0]]["dates"][s[1]], s[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tk, end = self.samples[idx]
        frame = self.groups[tk]
        start = end - self.lb + 1
        return {
            "x":     frame["x"][start:end+1],                          # [T,F]
            "y":     frame["y"][start:end+1],                          # scalar
            "sid":   torch.tensor(frame["asset_id"], dtype=torch.long),
            "date":  frame["dates"][end].strftime("%Y-%m-%d"),
            "ticker": tk,
        }


def _window_collate(batch):
    return {
        "x":      torch.stack([b["x"]   for b in batch]),
        "y":      torch.stack([b["y"]   for b in batch]),
        "sid":    torch.stack([b["sid"]  for b in batch]),
        "date":   [b["date"]  for b in batch],
        "ticker": [b["ticker"] for b in batch],
    }
    
# Data Loaders
def build_baseline_loaders(panel, feature_cols, train_d, val_d, test_d, cfg = None):
    if cfg is None:
        cfg = DATA

    sets = {
        "train": WindowDataset(panel, feature_cols, train_d, lookback=cfg["lookback"]),
        "val":   WindowDataset(panel, feature_cols, val_d,   lookback=cfg["lookback"]),
        "test":  WindowDataset(panel, feature_cols, test_d,  lookback=cfg["lookback"]),
    }

    loaders = {}
    loaders = {
        "train": DataLoader(
            sets["train"],
            batch_size=cfg["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=_window_collate,
        ),
        "val": DataLoader(
            sets["val"],
            batch_size=cfg["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=_window_collate,
        ),
        "test": DataLoader(
            sets["test"],
            batch_size=cfg["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=_window_collate,
        ),
    }

    return sets, loaders
