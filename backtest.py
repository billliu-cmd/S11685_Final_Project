"""
Backtest utilities: metrics, equity curves, comparison plots.
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .train import annualised_sharpe, max_drawdown


def annualised_return(daily: pd.Series) -> float:
    x = daily.dropna().values.astype(np.float64)
    if len(x) == 0:
        return np.nan
    return float(np.prod(1 + x) ** (252 / len(x)) - 1)


def annualised_vol(daily: pd.Series) -> float:
    x = daily.dropna().values.astype(np.float64)
    return float(np.std(x, ddof=0) * np.sqrt(252)) if len(x) else np.nan


def calmar(daily: pd.Series, eps=1e-9) -> float:
    return annualised_return(daily) / max(abs(max_drawdown(daily)), eps)


def summary(daily: pd.Series) -> Dict[str, float]:
    return {
        "ann_return":  annualised_return(daily),
        "ann_vol":     annualised_vol(daily),
        "sharpe":      annualised_sharpe(daily),
        "max_dd":      max_drawdown(daily),
        "calmar":      calmar(daily),
        "hit_rate":    float((daily.dropna() > 0).mean()) if len(daily) else np.nan,
        "days":        len(daily.dropna()),
    }


def turnover(pred_df: pd.DataFrame) -> pd.Series:
    """Average daily absolute position change across tickers."""
    df = pred_df.sort_values(["ticker", "date"]).copy()
    df["prev_pos"] = df.groupby("ticker")["position"].shift(1).fillna(0)
    df["to"] = (df["position"] - df["prev_pos"]).abs()
    return df.groupby("date")["to"].mean().sort_index()


def run_backtest(pred_df: pd.DataFrame, cost_bps: float = 0.0,
                 label: str = "") -> Dict:
    """Full backtest from a prediction dataframe."""
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["strategy_return"] = df["position"] * df["target_return"]

    daily = df.groupby("date")["strategy_return"].mean().sort_index()
    to = turnover(df)
    cost = cost_bps / 1e4
    daily_net = daily - to * cost

    return {
        "label": label,
        "gross": summary(daily),
        "net":   summary(daily_net),
        "daily_gross": daily,
        "daily_net":   daily_net,
        "turnover":    to,
        "equity_gross": (1 + daily).cumprod(),
        "equity_net":   (1 + daily_net).cumprod(),
    }


# ── plotting ──────────────────────────────────────────────────────────────────
def compare_equity(results: list[Dict], title="Equity Curves"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for r in results:
        axes[0].plot(r["equity_gross"], label=r["label"])
        dd = r["equity_gross"] / r["equity_gross"].cummax() - 1
        axes[1].fill_between(dd.index, dd, alpha=0.3, label=r["label"])
    axes[0].set_title(title); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].set_title("Drawdown"); axes[1].legend(); axes[1].grid(alpha=.3)
    fig.tight_layout()
    return fig


def print_comparison(results: list[Dict]):
    rows = []
    for r in results:
        row = {"model": r["label"]}
        row.update({f"gross_{k}": v for k, v in r["gross"].items()})
        row.update({f"net_{k}": v for k, v in r["net"].items()})
        row["avg_turnover"] = float(r["turnover"].mean())
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    print(df.to_string(float_format="%.4f"))
    return df
