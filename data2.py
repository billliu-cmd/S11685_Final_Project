from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yfinance as yf

from .config import (
    DATA, VOL_TARGET, VOL_LOOKBACK,
    RETURN_HORIZONS, MACD_PAIRS, EPS, DEFAULT_TICKERS, CPD
)
from .cpd import get_cached_regimes


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

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  X-Trend episode dataset
# ═══════════════════════════════════════════════════════════════════════════════
class EpisodeDataset(Dataset):
    """
    Each sample:
      target  – [l_t, F] features + [l_t] returns
      context – [C, l_c, F] features + [C, l_c] returns  (regime-aligned via CPD)

    Train:
      - uses a single regime set built from train history only
      - samples contexts freely from that train context pool

    Val/Test:
      - uses a causal regime cache keyed by date
      - each target date only sees contexts available strictly before that date
    """

    def __init__(self, panel, feature_cols, target_dates, ctx_pool_dates,
                 regimes=None, regime_cache=None,
                 target_len=126, ctx_len=21, num_ctx=10,
                 mode="train", seed=42):
        assert mode in ("train", "val", "test")
        if mode == "train" and regimes is None:
            raise ValueError("EpisodeDataset(train) requires `regimes`.")
        if mode in ("val", "test") and regime_cache is None:
            raise ValueError(f"EpisodeDataset({mode}) requires `regime_cache`.")

        self.mode, self.seed = mode, seed
        self.fcols = list(feature_cols)
        self.tl, self.cl, self.nc = target_len, ctx_len, num_ctx

        tgt_set = set(pd.to_datetime(target_dates))
        ctx_set = set(pd.to_datetime(ctx_pool_dates))

        self.groups = {}
        self.targets = []
        self.ctx_pool = []
        self.ctx_pool_by_date = {}
        self.original_num_targets = 0
        self.dropped_targets = 0

        for tk, g in panel.groupby("ticker", sort=False):
            g = g.sort_values("date").reset_index(drop=True)
            rec = {
                "x": torch.tensor(g[self.fcols].values, dtype=torch.float32),
                "y": torch.tensor(g["target_return"].values, dtype=torch.float32),
                "dates": pd.to_datetime(g["date"]).tolist(),
                "asset_id": int(g["asset_id"].iloc[0]),
            }
            self.groups[tk] = rec

            for i in range(self.tl - 1, len(g)):
                if g["date"].iloc[i] in tgt_set:
                    self.targets.append((tk, i))

        self.targets.sort(key=lambda s: (self.groups[s[0]]["dates"][s[1]], s[0]))
        self.original_num_targets = len(self.targets)

        if self.mode == "train":
            self.ctx_pool = self._build_ctx_pool(regimes, ctx_set)
            if len(self.ctx_pool) == 0:
                raise ValueError("Training context pool is empty. Check CPD segmentation and train dates.")
        else:
            unique_target_dates = sorted({self.groups[tk]["dates"][end] for tk, end in self.targets})

            for tgt_date in unique_target_dates:
                tgt_ts = pd.Timestamp(tgt_date)
                regimes_t = get_cached_regimes(regime_cache, tgt_ts)
                self.ctx_pool_by_date[tgt_ts] = self._build_ctx_pool(
                    regimes_t, ctx_set, tgt_date=tgt_ts
                )

            kept_targets = []
            for tk, end in self.targets:
                tgt_ts = pd.Timestamp(self.groups[tk]["dates"][end])
                if len(self.ctx_pool_by_date.get(tgt_ts, [])) > 0:
                    kept_targets.append((tk, end))

            self.targets = kept_targets
            self.dropped_targets = self.original_num_targets - len(self.targets)

    def __len__(self):
        return len(self.targets)

    def _build_ctx_pool(self, regimes, ctx_set, tgt_date=None):
        pool = []
        tgt_ts = pd.Timestamp(tgt_date) if tgt_date is not None else None

        for tk, intervals in regimes.items():
            if tk not in self.groups:
                continue
            rec = self.groups[tk]

            for rs, re in intervals:
                end_date = pd.Timestamp(rec["dates"][re])
                if end_date not in ctx_set:
                    continue
                if tgt_ts is not None and end_date >= tgt_ts:
                    continue
                pool.append((tk, rs, re, end_date))

        pool.sort(key=lambda s: (s[3], s[0]))
        return pool

    def _pick_ctx_entries(self, tgt_date, idx):
        if self.mode == "train":
            pool = self.ctx_pool
            if len(pool) == 0:
                raise IndexError("Training context pool is empty.")
            picks = np.random.choice(len(pool), self.nc, replace=(len(pool) < self.nc))
        else:
            pool = self.ctx_pool_by_date[pd.Timestamp(tgt_date)]
            if len(pool) == 0:
                raise IndexError(f"No causal contexts available for target date {tgt_date}.")
            rng = np.random.default_rng(self.seed + idx)
            picks = rng.choice(len(pool), self.nc, replace=(len(pool) < self.nc))

        picks = np.atleast_1d(picks).tolist()
        return [pool[p] for p in picks]

    def _slice_ctx(self, tk, rs, re):
        """Extract one context window, pad/truncate to l_c."""
        cg = self.groups[tk]
        regime_len = re - rs + 1
        if regime_len >= self.cl:
            cx = cg["x"][re - self.cl + 1:re + 1]
            cy = cg["y"][re - self.cl + 1:re + 1]
        else:
            pad = self.cl - regime_len
            cx = torch.cat([torch.zeros(pad, cg["x"].size(1)), cg["x"][rs:re + 1]])
            cy = torch.cat([torch.zeros(pad), cg["y"][rs:re + 1]])
        return cx, cy, cg["asset_id"]

    def __getitem__(self, idx):
        tk, end = self.targets[idx]
        g = self.groups[tk]
        s = end - self.tl + 1
        tgt_date = g["dates"][end]

        entries = self._pick_ctx_entries(tgt_date, idx)
        cx, cy, cid = [], [], []
        for ctk, rs, re, _ in entries:
            x, y, aid = self._slice_ctx(ctk, rs, re)
            cx.append(x)
            cy.append(y)
            cid.append(aid)

        return {
            "target_x":  g["x"][s:end + 1],
            "target_y":  g["y"][s:end + 1],
            "target_id": torch.tensor(g["asset_id"], dtype=torch.long),
            "ctx_x":     torch.stack(cx),
            "ctx_y":     torch.stack(cy),
            "ctx_id":    torch.tensor(cid, dtype=torch.long),
            "date":      tgt_date.strftime("%Y-%m-%d"),
            "ticker":    tk,
        }

def _episode_collate(batch):
    return {
        "target_x":  torch.stack([b["target_x"]  for b in batch]),
        "target_y":  torch.stack([b["target_y"]  for b in batch]),
        "target_id": torch.stack([b["target_id"] for b in batch]),
        "ctx_x":     torch.stack([b["ctx_x"]     for b in batch]),
        "ctx_y":     torch.stack([b["ctx_y"]     for b in batch]),
        "ctx_id":    torch.stack([b["ctx_id"]    for b in batch]),
        "date":      [b["date"]   for b in batch],
        "ticker":    [b["ticker"] for b in batch],
    }

def build_episode_loaders(panel, feature_cols, train_d, val_d, test_d,
                          train_regimes, cfg=None, regime_caches=None):
    if cfg is None:
        cfg = DATA
    if regime_caches is None:
        regime_caches = {}

    all_train_val = train_d.append(val_d) if hasattr(train_d, "append") \
                    else train_d.union(val_d)

    sets = {
        "train": EpisodeDataset(
            panel, feature_cols,
            target_dates=train_d, ctx_pool_dates=train_d,
            regimes=train_regimes,
            target_len=cfg["lookback"], ctx_len=cfg["context_len"],
            num_ctx=cfg["num_context"], mode="train", seed=cfg["seed"],
        ),
        "val": EpisodeDataset(
            panel, feature_cols,
            target_dates=val_d, ctx_pool_dates=all_train_val,
            regime_cache=regime_caches.get("val"),
            target_len=cfg["lookback"], ctx_len=cfg["context_len"],
            num_ctx=cfg["num_context"], mode="val", seed=cfg["seed"],
        ),
        "test": EpisodeDataset(
            panel, feature_cols,
            target_dates=test_d, ctx_pool_dates=all_train_val.union(test_d),
            regime_cache=regime_caches.get("test"),
            target_len=cfg["lookback"], ctx_len=cfg["context_len"],
            num_ctx=cfg["num_context"], mode="test", seed=cfg["seed"],
        ),
    }

    for split, ds in sets.items():
        print(
            f"{split}: kept {len(ds):,}/{ds.original_num_targets:,} targets "
            f"({ds.dropped_targets:,} dropped for no causal contexts)"
        )
        if split != "train":
            avg_pool = np.mean([len(v) for v in ds.ctx_pool_by_date.values()]) if ds.ctx_pool_by_date else 0.0
            print(f"  avg causal context pool per target date: {avg_pool:.1f}")

    loaders = {
        "train": DataLoader(
            sets["train"],
            batch_size=cfg["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=_episode_collate,
        ),
        "val": DataLoader(
            sets["val"],
            batch_size=cfg["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=_episode_collate,
        ),
        "test": DataLoader(
            sets["test"],
            batch_size=cfg["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=_episode_collate,
        ),
    }

    return sets, loaders


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SJM-driven EpisodeDataset  (new section — only ADDED to this file,
#     nothing above is modified).
#
# Goals, relative to EpisodeDataset:
#   (1) Similarity retrieval — instead of uniform-random sampling over the
#       causally-filtered CPD pool, rank candidates by cosine similarity
#       between the target's 6-dim SJM summary-statistic vector and each
#       candidate's 6-dim vector at its regime end date.
#   (2) Static cross-attention on SJM soft-probs — the target-date posterior
#       p_tgt ∈ Δ^K is the "query", each candidate's posterior p_ctx ∈ Δ^K is
#       the "key", and with the default identity affinity A = I we use the
#       plain dot product  p_tgt · p_ctx  as a probability-compatibility
#       score. This is a frozen / non-learned cross-attention — everything
#       fits inside the Dataset, no autograd boundary crossed.
#
# Final candidate score:   score = alpha * cos(feat_tgt, feat_ctx)
#                                + beta  * (p_tgt · p_ctx)
# Picks: top-C by `score` (deterministic, no random sampling — per user spec).
#
# Causality: we keep EpisodeDataset's CPD machinery verbatim, so segment
# endpoints + per-target pool filtering (end_date < tgt_date) still gate every
# candidate. The SJM probs/features themselves are also sourced causally:
#   - train mode: one SJM fit on panel rows up to max(train date universe),
#     the analogue of the single `train_regimes` snapshot.
#   - val/test mode: a snapshot schedule identical to jump_model.build_jm_cache
#     (every `recompute_every` days, always including the last date). Each
#     target is matched to the *latest snapshot ≤ its own date*, and every
#     candidate's probs for that target are read from the same snapshot. That
#     guarantees target and contexts agree on the same causal fit and never
#     look past tgt_date.
#
# Memory strategy (user asked for the most efficient cache):
#   - We precompute the 6-dim SJM feature per (ticker, date) ONCE over the
#     full panel (z-score is applied globally per-ticker by _build_features).
#     Cosine is scale-invariant on the sample axis, so a global z-score is
#     fine.
#   - For soft-probs we do NOT keep every snapshot in RAM. We iterate
#     snapshots in order; for each snapshot we extract only the rows we
#     actually need (rows for targets mapped to this snap + rows for every
#     context end-date in those targets' pools) into flat per-target /
#     per-pool arrays, then drop the full fit. Peak memory is
#     O(#targets × K + #pool_rows × K).
#   - Lookups inside __getitem__ reduce to a slice + two numpy dot products
#     + one argpartition. No dict access, no Python loop over candidates.
# ═══════════════════════════════════════════════════════════════════════════════

# We need fit_panel_jm, fit_panel_jm_until, JM_DEFAULTS, and the private
# _build_features helper to keep z-scoring identical to the SJM internals.
from .jump_model import (
    JM_DEFAULTS,
    fit_panel_jm,
    fit_panel_jm_until,
    _build_features as _jm_build_features,
)

try:
    from tqdm.auto import tqdm as _sjm_tqdm
except ImportError:
    # tqdm is listed in requirements.txt, but fall back gracefully if missing.
    def _sjm_tqdm(x, **_k):
        return x


def _sjm_prob_map_from_fit(jm_out):
    """
    Collapse `fit_panel_jm(...)` output into the minimal lookup form this
    module needs: {ticker: (dates_np[T], soft_probs[T, K])}. Everything else
    returned by SJM (mus, covs, loss_history, ...) is discarded so the caller
    can throw the fit away immediately after extraction — this is what keeps
    peak memory at ~one snapshot.
    """
    prob_map = {}
    for tk, res in jm_out.items():
        prob_map[tk] = (
            np.asarray(res["dates"], dtype="datetime64[ns]"),
            np.asarray(res["soft_probs"], dtype=np.float64),
        )
    return prob_map


def _sjm_row_at(series, query_date):
    """
    Given a (dates_np, values) tuple sorted by date and an arbitrary query
    timestamp, return `values[i]` where i is the largest index with
    dates[i] <= query_date. Returns None if no such index exists. This is
    how we align an SJM snapshot to a specific (ticker, date) row.
    """
    dates_np, values = series
    q = np.datetime64(pd.Timestamp(query_date))
    i = int(np.searchsorted(dates_np, q, side="right") - 1)
    if i < 0:
        return None
    return values[i]


class SJMEpisodeDataset(EpisodeDataset):
    """
    Episode dataset where each target's `C` contexts are the top-C candidates
    (from the same causally-filtered CPD pool EpisodeDataset already builds)
    under a fixed score combining:

        score_i = alpha * cos(feat_tgt,      feat_ctx_i)
                + beta  * (p_tgt @ A @ p_ctx_i)          # A = I by default

    where `feat` is the 6-dim SJM summary vector and `p` is the K-dim SJM
    soft-posterior. SJM fits are done inside `__init__`; no external SJM cache
    input is required (though `regimes` / `regime_cache` for the CPD pool are
    still required, matching EpisodeDataset's contract).

    Parameters
    ----------
    panel, feature_cols, target_dates, ctx_pool_dates,
    regimes, regime_cache, target_len, ctx_len, num_ctx, mode, seed :
        Same meaning as in EpisodeDataset. Forwarded unchanged to super().

    K, lam, cov_mode, feature_window :
        SJM hyper-parameters (defaults from JM_DEFAULTS).

    recompute_every :
        Snapshot cadence for val/test causal fits (defaults from JM_DEFAULTS).

    alpha, beta :
        Scalar weights on the feature-cosine term and the SJM-prob term.

    verbose :
        If True, wrap the val/test snapshot-fitting loop in tqdm.
    """

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------
    def __init__(
        self,
        panel,
        feature_cols,
        target_dates,
        ctx_pool_dates,
        *,
        regimes=None,
        regime_cache=None,
        target_len=126,
        ctx_len=21,
        num_ctx=10,
        mode="train",
        seed=42,
        # ---- SJM-specific kwargs (all default from JM_DEFAULTS) ----------
        K=None,
        lam=None,
        cov_mode=None,
        feature_window=None,
        recompute_every=None,
        alpha=1.0,
        beta=1.0,
        verbose=True,
    ):
        # Build the base EpisodeDataset state first — this populates
        # self.groups / self.targets / self.ctx_pool / self.ctx_pool_by_date
        # with exactly the same CPD-causal filtering as before.
        super().__init__(
            panel=panel,
            feature_cols=feature_cols,
            target_dates=target_dates,
            ctx_pool_dates=ctx_pool_dates,
            regimes=regimes,
            regime_cache=regime_cache,
            target_len=target_len,
            ctx_len=ctx_len,
            num_ctx=num_ctx,
            mode=mode,
            seed=seed,
        )

        # Resolve SJM hyper-params (do not mutate JM_DEFAULTS).
        K              = JM_DEFAULTS["K"]               if K               is None else K
        lam            = JM_DEFAULTS["lam"]             if lam             is None else lam
        cov_mode       = JM_DEFAULTS["cov_mode"]        if cov_mode        is None else cov_mode
        feature_window = JM_DEFAULTS["feature_window"]  if feature_window  is None else feature_window
        recompute_every = JM_DEFAULTS["recompute_every"] if recompute_every is None else recompute_every

        self._K = int(K)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._fit_kwargs = dict(K=K, lam=lam, cov_mode=cov_mode,
                                feature_window=feature_window)
        self._feature_window = int(feature_window)

        # ------------------------------------------------------------------
        # Step 1 — precompute per-ticker SJM summary features over full panel.
        # These are the same 6-dim rows SJM builds internally (we reuse
        # jump_model._build_features), so scale/units match the probs below.
        # Memory is trivial (~T_total * 6 * 8 bytes).
        # ------------------------------------------------------------------
        self._feats_by_tk: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for tk, g in panel.groupby("ticker", sort=False):
            g = g.sort_values("date").reset_index(drop=True)
            close = pd.Series(
                g["close"].values.astype(np.float64),
                index=pd.to_datetime(g["date"]),
            )
            Y, dates, _, _ = _jm_build_features(close, window=feature_window)
            self._feats_by_tk[tk] = (
                np.asarray(dates, dtype="datetime64[ns]"),
                Y.astype(np.float64),
            )

        # ------------------------------------------------------------------
        # Step 2 — pre-extract per-target 6-dim feature vectors.
        # ------------------------------------------------------------------
        self._tgt_feat = np.zeros((len(self.targets), 6), dtype=np.float64)
        for t_idx, (tk, end) in enumerate(self.targets):
            self._tgt_feat[t_idx] = self._lookup_feat(
                tk, self.groups[tk]["dates"][end]
            )

        # ------------------------------------------------------------------
        # Step 3 — pre-extract per-candidate 6-dim feature vectors. Shape
        # depends on mode: one shared (N, 6) for train; a dict of (N_d, 6)
        # keyed by target date for val/test.
        # ------------------------------------------------------------------
        if mode == "train":
            self._pool_feat = np.zeros((len(self.ctx_pool), 6), dtype=np.float64)
            for p_idx, (tk, _rs, _re, end_date) in enumerate(self.ctx_pool):
                self._pool_feat[p_idx] = self._lookup_feat(tk, end_date)
        else:
            self._pool_feat_by_date: Dict[pd.Timestamp, np.ndarray] = {}
            for tgt_ts, pool in self.ctx_pool_by_date.items():
                pf = np.zeros((len(pool), 6), dtype=np.float64)
                for p_idx, (tk, _rs, _re, end_date) in enumerate(pool):
                    pf[p_idx] = self._lookup_feat(tk, end_date)
                self._pool_feat_by_date[tgt_ts] = pf

        # ------------------------------------------------------------------
        # Step 4 — SJM probability extraction (the expensive bit).
        # Default everything to uniform 1/K so that when a fit is missing or a
        # row falls outside the fit's date coverage, the prob term degenerates
        # to a constant (no effect on top-C ranking).
        # ------------------------------------------------------------------
        self._tgt_prob = np.full((len(self.targets), K), 1.0 / K, dtype=np.float64)

        if mode == "train":
            # One SJM fit on panel truncated to max(train target dates ∪
            # train ctx_pool dates). That is exactly the train history — no
            # val/test leakage. Mirrors how build_episode_loaders passes a
            # single `train_regimes` for CPD.
            train_max = max(
                pd.to_datetime(
                    pd.DatetimeIndex(target_dates)
                    .union(pd.DatetimeIndex(ctx_pool_dates))
                )
            )
            panel_cut = panel[pd.to_datetime(panel["date"]) <= train_max].copy()
            if verbose:
                print(f"[SJMEpisodeDataset:train] fitting SJM on panel <= {train_max.date()}")
            prob_map = _sjm_prob_map_from_fit(
                fit_panel_jm(panel_cut, **self._fit_kwargs)
            )

            # Targets.
            for t_idx, (tk, end) in enumerate(self.targets):
                if tk not in prob_map:
                    continue
                p = _sjm_row_at(prob_map[tk], self.groups[tk]["dates"][end])
                if p is not None:
                    self._tgt_prob[t_idx] = p

            # Pool.
            self._pool_prob = np.full((len(self.ctx_pool), K), 1.0 / K,
                                      dtype=np.float64)
            for p_idx, (tk, _rs, _re, end_date) in enumerate(self.ctx_pool):
                if tk not in prob_map:
                    continue
                p = _sjm_row_at(prob_map[tk], end_date)
                if p is not None:
                    self._pool_prob[p_idx] = p

            # Release the fit immediately.
            del prob_map

        else:
            # --------------------------------------------------------------
            # Val/test: schedule causal snapshots and fit them one at a time,
            # extracting only the rows we need (targets routed to this snap +
            # pool candidates of those targets), then dropping the fit.
            # --------------------------------------------------------------
            self._pool_prob_by_date: Dict[pd.Timestamp, np.ndarray] = {
                k: np.full((len(v), K), 1.0 / K, dtype=np.float64)
                for k, v in self.ctx_pool_by_date.items()
            }

            # Snapshot schedule over target_dates ∪ ctx_pool_dates (same
            # universe the CPD cache would see).
            all_dates = pd.DatetimeIndex(
                sorted(
                    pd.to_datetime(target_dates)
                    .union(pd.to_datetime(ctx_pool_dates))
                )
            )
            if len(all_dates) == 0:
                return
            snap_list = list(all_dates[::recompute_every])
            if snap_list[-1] != all_dates[-1]:
                snap_list.append(all_dates[-1])
            snap_list = [pd.Timestamp(s) for s in snap_list]
            snap_np = np.array(
                [np.datetime64(s) for s in snap_list],
                dtype="datetime64[ns]",
            )

            # Target -> snap-index map (largest snap <= target date).
            tgt_ts_arr = np.array(
                [np.datetime64(pd.Timestamp(self.groups[tk]["dates"][end]))
                 for (tk, end) in self.targets],
                dtype="datetime64[ns]",
            )
            tgt_snap_idx = np.searchsorted(snap_np, tgt_ts_arr, side="right") - 1

            snap_to_tgts: Dict[int, List[int]] = {}
            for t_idx, s_idx in enumerate(tgt_snap_idx):
                s_idx = int(s_idx)
                if s_idx < 0:
                    continue
                snap_to_tgts.setdefault(s_idx, []).append(t_idx)

            # Pool-date -> snap-index map. A pool is keyed by its target date,
            # so it inherits the same snap as its target — this is exactly
            # what keeps target/context SJM views consistent and causal.
            pool_snap_idx = {
                tgt_ts: int(np.searchsorted(
                    snap_np,
                    np.datetime64(pd.Timestamp(tgt_ts)),
                    side="right") - 1)
                for tgt_ts in self.ctx_pool_by_date.keys()
            }
            snap_to_pools: Dict[int, List[pd.Timestamp]] = {}
            for tgt_ts, s_idx in pool_snap_idx.items():
                if s_idx < 0:
                    continue
                snap_to_pools.setdefault(s_idx, []).append(tgt_ts)

            # Iterate snapshots (skip ones with nothing routed to them).
            active_snaps = sorted(
                set(snap_to_tgts.keys()) | set(snap_to_pools.keys())
            )
            iterator = _sjm_tqdm(
                active_snaps,
                desc=f"SJM {mode} snapshots",
                disable=not verbose,
            )
            for s_idx in iterator:
                snap_ts = snap_list[s_idx]
                # Causal fit — uses only rows with date <= snap_ts.
                prob_map = _sjm_prob_map_from_fit(
                    fit_panel_jm_until(panel, snap_ts, **self._fit_kwargs)
                )

                # Targets routed to this snapshot.
                for t_idx in snap_to_tgts.get(s_idx, []):
                    tk, end = self.targets[t_idx]
                    if tk not in prob_map:
                        continue
                    p = _sjm_row_at(prob_map[tk],
                                    self.groups[tk]["dates"][end])
                    if p is not None:
                        self._tgt_prob[t_idx] = p

                # Pools routed to this snapshot.
                for tgt_ts in snap_to_pools.get(s_idx, []):
                    pool = self.ctx_pool_by_date[tgt_ts]
                    buf = self._pool_prob_by_date[tgt_ts]
                    for p_idx, (tk, _rs, _re, end_date) in enumerate(pool):
                        if tk not in prob_map:
                            continue
                        p = _sjm_row_at(prob_map[tk], end_date)
                        if p is not None:
                            buf[p_idx] = p

                # Drop the fit. Peak memory stays ~ one snapshot's worth.
                del prob_map

    # ---------------------------------------------------------------------
    # Lookups
    # ---------------------------------------------------------------------
    def _lookup_feat(self, tk, query_date):
        """Return the 6-dim SJM feature row for (tk, query_date); zeros if
        the ticker has no feature series or query_date precedes its first
        feature-valid day (the first `feature_window - 1` days are NaN and
        were dropped by _build_features)."""
        if tk not in self._feats_by_tk:
            return np.zeros(6, dtype=np.float64)
        row = _sjm_row_at(self._feats_by_tk[tk], query_date)
        if row is None:
            return np.zeros(6, dtype=np.float64)
        return row

    # ---------------------------------------------------------------------
    # Override: deterministic top-C by (alpha*cos + beta*p_tgt·p_ctx)
    # ---------------------------------------------------------------------
    def _pick_ctx_entries(self, tgt_date, idx):
        # Resolve which pool + prefetched feat/prob matrices apply here.
        if self.mode == "train":
            pool = self.ctx_pool
            pool_feat = self._pool_feat
            pool_prob = self._pool_prob
        else:
            tgt_ts = pd.Timestamp(tgt_date)
            pool = self.ctx_pool_by_date[tgt_ts]
            pool_feat = self._pool_feat_by_date[tgt_ts]
            pool_prob = self._pool_prob_by_date[tgt_ts]

        N = len(pool)
        if N == 0:
            raise IndexError(
                f"SJMEpisodeDataset: empty causal context pool at {tgt_date}"
            )

        tgt_feat = self._tgt_feat[idx]
        tgt_prob = self._tgt_prob[idx]

        # Similarity term A — cosine over the 6-dim SJM summary features.
        # A zero target vector (fallback) produces a zero cosine, which still
        # yields a valid argpartition below.
        t_norm = float(np.linalg.norm(tgt_feat)) + 1e-12
        p_norm = np.linalg.norm(pool_feat, axis=1) + 1e-12
        cos_sim = (pool_feat @ tgt_feat) / (p_norm * t_norm)

        # Probability term D — identity-affinity cross-attention p_tgt · p_ctx.
        # With soft-probs this lives in [0, 1]; it peaks when target and
        # context put mass on the same regime.
        prob_sim = pool_prob @ tgt_prob

        score = self.alpha * cos_sim + self.beta * prob_sim

        # Deterministic top-C pick. argpartition is O(N); the subsequent sort
        # is O(C log C) and only over the C winners.
        nc = self.nc
        if N >= nc:
            top = np.argpartition(-score, nc - 1)[:nc]
            picks = top[np.argsort(-score[top])].tolist()
        else:
            # Pool smaller than num_ctx: return all, then pad with the best
            # candidate repeated (same "duplicate" fallback EpisodeDataset
            # uses via np.random.choice(..., replace=True)).
            order = np.argsort(-score).tolist()
            picks = order + [order[0]] * (nc - N)

        return [pool[p] for p in picks]


def build_sjm_episode_loaders(
    panel, feature_cols, train_d, val_d, test_d,
    train_regimes, cfg=None, regime_caches=None,
    sjm_kwargs=None,
):
    """
    Parallel to build_episode_loaders but constructs SJMEpisodeDataset on each
    split. `sjm_kwargs` is passed through to the dataset (K, lam, cov_mode,
    feature_window, recompute_every, alpha, beta, verbose).
    """
    if cfg is None:
        cfg = DATA
    if regime_caches is None:
        regime_caches = {}
    if sjm_kwargs is None:
        sjm_kwargs = {}

    all_train_val = train_d.append(val_d) if hasattr(train_d, "append") \
                    else train_d.union(val_d)

    sets = {
        "train": SJMEpisodeDataset(
            panel, feature_cols,
            target_dates=train_d, ctx_pool_dates=train_d,
            regimes=train_regimes,
            target_len=cfg["lookback"], ctx_len=cfg["context_len"],
            num_ctx=cfg["num_context"], mode="train", seed=cfg["seed"],
            **sjm_kwargs,
        ),
        "val": SJMEpisodeDataset(
            panel, feature_cols,
            target_dates=val_d, ctx_pool_dates=all_train_val,
            regime_cache=regime_caches.get("val"),
            target_len=cfg["lookback"], ctx_len=cfg["context_len"],
            num_ctx=cfg["num_context"], mode="val", seed=cfg["seed"],
            **sjm_kwargs,
        ),
        "test": SJMEpisodeDataset(
            panel, feature_cols,
            target_dates=test_d, ctx_pool_dates=all_train_val.union(test_d),
            regime_cache=regime_caches.get("test"),
            target_len=cfg["lookback"], ctx_len=cfg["context_len"],
            num_ctx=cfg["num_context"], mode="test", seed=cfg["seed"],
            **sjm_kwargs,
        ),
    }

    for split, ds in sets.items():
        print(
            f"{split}: kept {len(ds):,}/{ds.original_num_targets:,} targets "
            f"({ds.dropped_targets:,} dropped for no causal contexts)"
        )
        if split != "train":
            avg_pool = (np.mean([len(v) for v in ds.ctx_pool_by_date.values()])
                        if ds.ctx_pool_by_date else 0.0)
            print(f"  avg causal context pool per target date: {avg_pool:.1f}")

    loaders = {
        "train": DataLoader(
            sets["train"], batch_size=cfg["batch_size"], shuffle=True,
            drop_last=False, num_workers=4, pin_memory=True,
            collate_fn=_episode_collate,
        ),
        "val": DataLoader(
            sets["val"], batch_size=cfg["batch_size"], shuffle=False,
            drop_last=False, num_workers=0, pin_memory=True,
            collate_fn=_episode_collate,
        ),
        "test": DataLoader(
            sets["test"], batch_size=cfg["batch_size"], shuffle=False,
            drop_last=False, num_workers=0, pin_memory=True,
            collate_fn=_episode_collate,
        ),
    }
    return sets, loaders
