#!/usr/bin/env python3
"""
End-to-end runner:  data → baseline DMN → X-Trend (Sharpe) → backtest comparison.

Usage:
    python -m pipeline.run                     # defaults
    python -m pipeline.run --epochs 30 --seed 0
"""

from __future__ import annotations
import argparse, torch

from .config import DataConfig, ModelConfig, TrainConfig
from .data   import build_panel, time_split, build_baseline_loaders, build_xtrend_loaders
from .Baseline import BaselineDMN
from .xtrend   import XTrend
from .train    import fit_baseline, fit_xtrend, evaluate_baseline, evaluate_xtrend
from .backtest import run_backtest, compare_equity, print_comparison


def main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",  type=int, default=50)
    p.add_argument("--patience",type=int, default=10)
    p.add_argument("--batch",   type=int, default=64)
    p.add_argument("--hidden",  type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--cost_bps",type=float, default=0.0)
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--skip_xtrend",   action="store_true")
    a = p.parse_args(args)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    dcfg = DataConfig(batch_size=a.batch, seed=a.seed)
    mcfg = ModelConfig(hidden_dim=a.hidden, dropout=a.dropout)
    tcfg = TrainConfig(lr=a.lr, epochs=a.epochs, patience=a.patience)

    # ── data ──────────────────────────────────────────────────────────────
    print("Building panel …")
    panel, fcols, tk2id = build_panel(dcfg)
    train_d, val_d, test_d = time_split(panel, dcfg.train_frac, dcfg.val_frac)
    n_assets = len(tk2id)
    n_feat   = len(fcols)
    print(f"  {n_assets} assets | {n_feat} features | "
          f"train {len(train_d)} | val {len(val_d)} | test {len(test_d)} days")

    results = []

    # ── baseline DMN ──────────────────────────────────────────────────────
    if not a.skip_baseline:
        print("\n══ Baseline DMN ══")
        _, bl_loaders = build_baseline_loaders(panel, fcols, train_d, val_d, test_d, dcfg)
        bl_model = BaselineDMN(n_feat, n_assets, mcfg).to(device)
        bl_model, bl_hist = fit_baseline(bl_model, bl_loaders["train"], bl_loaders["val"],
                                         device, tcfg, mcfg)
        bl_test = evaluate_baseline(bl_model, bl_loaders["test"], device, mcfg)
        print(f"  Test sharpe={bl_test['sharpe']:.4f}  mdd={bl_test['max_drawdown']:.4f}")
        results.append(run_backtest(bl_test["pred_df"], a.cost_bps, "Baseline DMN"))

    # ── X-Trend (Sharpe) ─────────────────────────────────────────────────
    if not a.skip_xtrend:
        print("\n══ X-Trend (Sharpe) ══")
        _, xt_loaders = build_xtrend_loaders(panel, fcols, train_d, val_d, test_d, dcfg)
        xt_model = XTrend(n_feat, n_assets, mcfg).to(device)
        xt_model, xt_hist = fit_xtrend(xt_model, xt_loaders["train"], xt_loaders["val"],
                                        device, tcfg, mcfg)
        xt_test = evaluate_xtrend(xt_model, xt_loaders["test"], device, mcfg)
        print(f"  Test sharpe={xt_test['sharpe']:.4f}  mdd={xt_test['max_drawdown']:.4f}")
        results.append(run_backtest(xt_test["pred_df"], a.cost_bps, "X-Trend (Sharpe)"))

    # ── comparison ────────────────────────────────────────────────────────
    if len(results) >= 1:
        print("\n══ Backtest Summary ══")
        print_comparison(results)
    if len(results) >= 2:
        compare_equity(results, "Baseline DMN vs X-Trend (Sharpe)")
        import matplotlib.pyplot as plt
        plt.savefig("equity_comparison.png", dpi=150, bbox_inches="tight")
        print("Saved equity_comparison.png")


if __name__ == "__main__":
    main()
