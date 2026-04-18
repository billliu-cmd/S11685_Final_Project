"""
Statistical Jump Model (SJM) for asset-level regime classification.

Implements the SJM of Bemporad et al. (2018) "Fitting jump models" and
Nystrup, Lindstrom & Madsen (2020) "Learning hidden Markov models with
persistent states by penalising jumps", extended to a QDA-style setting
where each regime carries its own full covariance Sigma_k. In that mode
the per-state assignment cost is the Gaussian negative-log-likelihood

        d_k(Y_t) = (Y_t - mu_k)^T Sigma_k^{-1} (Y_t - mu_k) + log|Sigma_k|,

and the model is mathematically equivalent to a Viterbi-decoded Gaussian
HMM whose learned transition matrix has been replaced by a fixed jump
penalty lambda on off-diagonal moves.

Three covariance modes are supported via `cov_mode`:
  - "shared"             : single Sigma pooled across regimes
                           (classic SJM, squared Mahalanobis distance)
  - "diagonal_per_regime": per-regime diag(sigma_k^2) - robust middle ground
  - "full_per_regime"    : QDA-style full per-regime Sigma_k (primary mode)

Public API mirrors cpd.py for drop-in use with X-Trend's causal
context-cache machinery. Pure numpy/scipy/pandas, no sklearn / hmmlearn.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# =============================================================================
# Module-level defaults (until a `JM` dict is added to config.py).
# =============================================================================
JM_DEFAULTS = {
    "K":               3,
    "lam":             25.0,
    "cov_mode":        "full_per_regime",
    "n_init":          5,
    "max_iter":        50,
    "tol":             1e-4,
    "ridge":           1e-3,
    "feature_window":  21,
    "recompute_every": 21,
    "min_cluster":     5,
    "seed":            42,
}


# =============================================================================
# Feature engineering
# =============================================================================
def _build_features(close: pd.Series, window: int = 21):
    """
    Build regime-discriminating features from a single asset's close series.

    Returns
    -------
    Y    : np.ndarray (T_eff, F) float64, z-scored, NaN rows dropped
    dates: pd.DatetimeIndex aligned with Y rows
    cols : list[str] feature names
    ret_mean_idx : int index of the "ret_mean" column (used to canonicalise
                   regime labels so 0=most bearish, K-1=most bullish)

    Features (F = 6):
      ret_mean     : rolling mean of log-returns
      ret_std      : rolling std of log-returns
      downside_dev : rolling std of min(r_t, 0)
      drawdown     : close / rolling_max(close) - 1
      ret_skew     : rolling skew of log-returns
      abs_ret_ewm  : EWM of |r_t| (vol-clustering proxy, halflife = window/2)

    All features are computed causally: Y_t uses only data up to time t.
    Z-scoring uses the in-sample mean/std of the returned rows, so when this
    helper is called from `fit_panel_jm_until` with a truncated price series
    the standardisation itself is also causal.
    """
    close = pd.Series(close).astype(np.float64).sort_index()
    r = np.log(close).diff()
    downside = r.where(r < 0.0, 0.0)

    feats = pd.DataFrame(index=close.index)
    feats["ret_mean"]     = r.rolling(window, min_periods=window).mean()
    feats["ret_std"]      = r.rolling(window, min_periods=window).std()
    feats["downside_dev"] = downside.rolling(window, min_periods=window).std()
    roll_max = close.rolling(window, min_periods=window).max()
    feats["drawdown"]     = close / roll_max - 1.0
    feats["ret_skew"]     = r.rolling(window, min_periods=window).skew()
    hl = max(window / 2.0, 1.0)
    feats["abs_ret_ewm"]  = r.abs().ewm(halflife=hl, adjust=False).mean()

    feats = feats.dropna()
    Y = feats.values.astype(np.float64)

    if Y.shape[0] == 0:
        return Y, feats.index, list(feats.columns), 0

    mu = Y.mean(axis=0)
    sd = Y.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Y = (Y - mu) / sd

    return Y, feats.index, list(feats.columns), 0


# =============================================================================
# Cholesky primitives (mirror cpd._log_marglik style)
# =============================================================================
def _chol_factor(S, ridge=0.0):
    """Cholesky with ridge-doubling fallback on LinAlgError."""
    F = S.shape[0]
    try:
        return np.linalg.cholesky(S + ridge * np.eye(F))
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(S + (2.0 * ridge + 1e-3) * np.eye(F))


def _chol_inv(L):
    """Given L = chol(A), return A^{-1} via two triangular solves."""
    F = L.shape[0]
    Ident = np.eye(F)
    Y = np.linalg.solve(L, Ident)
    return np.linalg.solve(L.T, Y)


def _log_det_chol(L):
    return 2.0 * float(np.sum(np.log(np.diag(L))))


# =============================================================================
# k-means++ seeding
# =============================================================================
def _kmeans_plus_plus_init(Y, K, rng):
    """D^2-weighted seeding for random restarts."""
    T = Y.shape[0]
    mus = np.empty((K, Y.shape[1]))
    idx0 = int(rng.integers(T))
    mus[0] = Y[idx0]
    d2 = np.sum((Y - mus[0]) ** 2, axis=1)
    for k in range(1, K):
        total = d2.sum()
        if total <= 0.0 or not np.isfinite(total):
            idx = int(rng.integers(T))
        else:
            probs = d2 / total
            idx = int(rng.choice(T, p=probs))
        mus[k] = Y[idx]
        new_d2 = np.sum((Y - mus[k]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)
    return mus


# =============================================================================
# Viterbi assignment - agnostic to cov_mode
# =============================================================================
def _per_state_costs(Y, mus, precisions, log_dets):
    """cost[t, k] = (Y_t - mu_k)^T P_k (Y_t - mu_k) + log|Sigma_k|."""
    T = Y.shape[0]
    K = mus.shape[0]
    cost = np.empty((T, K))
    for k in range(K):
        diff = Y - mus[k]
        cost[:, k] = np.einsum("ti,ij,tj->t", diff, precisions[k], diff) + log_dets[k]
    return cost


def _assign_viterbi(Y, mus, precisions, log_dets, lam):
    """
    DP over the (T, K) trellis using the best/second-best trick so each
    forward step is O(K) rather than O(K^2). Returns hard int labels.
    """
    cost = _per_state_costs(Y, mus, precisions, log_dets)
    T, K = cost.shape

    V  = np.empty((T, K))
    bp = np.full((T, K), -1, dtype=np.int64)
    V[0] = cost[0]

    ks = np.arange(K)
    for t in range(1, T):
        prev = V[t - 1]
        best_idx = int(np.argmin(prev))
        best_val = prev[best_idx]
        prev_masked = prev.copy()
        prev_masked[best_idx] = np.inf
        second_idx = int(np.argmin(prev_masked))
        second_val = prev_masked[second_idx]

        switch_from = np.where(ks == best_idx, second_idx, best_idx)
        switch_val  = np.where(ks == best_idx, second_val, best_val) + lam
        stay_val    = prev

        choose_switch = switch_val < stay_val
        path_cost = np.where(choose_switch, switch_val, stay_val)
        path_src  = np.where(choose_switch, switch_from, ks)

        V[t] = path_cost + cost[t]
        bp[t] = path_src

    labels = np.empty(T, dtype=np.int64)
    labels[-1] = int(np.argmin(V[-1]))
    for t in range(T - 2, -1, -1):
        labels[t] = bp[t + 1, labels[t + 1]]
    return labels


# =============================================================================
# M-step helpers
# =============================================================================
def _update_means(Y, labels, K, min_cluster, rng, cost_per_obs=None):
    """Cluster means; reseed under-populated clusters from worst-fit point."""
    T, F = Y.shape
    mus = np.zeros((K, F))
    used = set()
    for k in range(K):
        mask = labels == k
        n_k = int(mask.sum())
        if n_k >= min_cluster:
            mus[k] = Y[mask].mean(axis=0)
        else:
            pick = None
            if cost_per_obs is not None:
                order = np.argsort(-cost_per_obs)
                for idx in order:
                    if int(idx) not in used:
                        pick = int(idx)
                        break
            if pick is None:
                pick = int(rng.integers(T))
            mus[k] = Y[pick]
            used.add(pick)
    return mus


def _pooled_scatter(Y, labels, mus, K, ridge):
    """Classic SJM: one pooled Sigma broadcast across regimes."""
    T, F = Y.shape
    S = np.zeros((F, F))
    for k in range(K):
        mask = labels == k
        if not mask.any():
            continue
        d = Y[mask] - mus[k]
        S += d.T @ d
    S = S / max(T - K, 1) + ridge * np.eye(F)
    L = _chol_factor(S, ridge=0.0)
    log_det = _log_det_chol(L)
    P = _chol_inv(L)

    covs       = np.broadcast_to(S, (K, F, F)).copy()
    precisions = np.broadcast_to(P, (K, F, F)).copy()
    log_dets   = np.full(K, log_det)
    return covs, precisions, log_dets


def _diag_per_regime(Y, labels, mus, K, ridge):
    """LDA/QDA middle ground: diag(sigma_k^2 + ridge) per regime."""
    T, F = Y.shape
    covs       = np.zeros((K, F, F))
    precisions = np.zeros((K, F, F))
    log_dets   = np.zeros(K)
    for k in range(K):
        mask = labels == k
        n_k = int(mask.sum())
        if n_k > 1:
            d = Y[mask] - mus[k]
            var = (d * d).sum(axis=0) / max(n_k - 1, 1)
        else:
            var = np.ones(F)
        var = var + ridge
        covs[k]       = np.diag(var)
        precisions[k] = np.diag(1.0 / var)
        log_dets[k]   = float(np.log(var).sum())
    return covs, precisions, log_dets


def _full_per_regime(Y, labels, mus, K, ridge):
    """QDA-style: full per-regime Sigma_k + ridge*I, with Cholesky fallback."""
    T, F = Y.shape
    covs       = np.zeros((K, F, F))
    precisions = np.zeros((K, F, F))
    log_dets   = np.zeros(K)
    I = np.eye(F)
    for k in range(K):
        mask = labels == k
        n_k = int(mask.sum())
        if n_k > 1:
            d = Y[mask] - mus[k]
            S = d.T @ d / max(n_k - 1, 1)
        else:
            S = np.eye(F)
        S = S + ridge * I
        L = _chol_factor(S, ridge=0.0)
        covs[k]       = S
        precisions[k] = _chol_inv(L)
        log_dets[k]   = _log_det_chol(L)
    return covs, precisions, log_dets


# =============================================================================
# Loss and soft probs
# =============================================================================
def _total_loss(Y, labels, mus, precisions, log_dets, lam):
    """Sigma_t d_{s_t}(Y_t) + lambda * jumps. Monotone under CD."""
    diffs = Y - mus[labels]
    quad  = np.einsum("ti,tij,tj->t", diffs, precisions[labels], diffs)
    state_loss = float((quad + log_dets[labels]).sum())
    jumps = int((labels[1:] != labels[:-1]).sum())
    return state_loss + lam * jumps


def _soft_probs(Y, mus, precisions, log_dets):
    """
    Row-wise softmax over -d_k(Y_t). LOCAL posterior under the fitted
    Gaussian emissions; deliberately does NOT run forward-backward over
    a constructed transition matrix (see module docstring). Consumers who
    want lambda-smoothed hard membership use `labels`.
    """
    cost = _per_state_costs(Y, mus, precisions, log_dets)
    neg_d = -cost
    m = neg_d.max(axis=1, keepdims=True)
    e = np.exp(neg_d - m)
    return e / e.sum(axis=1, keepdims=True)


# =============================================================================
# Label canonicalisation
# =============================================================================
def _sort_states_by_return(result, return_feature_idx=0):
    """
    Permute regime labels so regime 0 = lowest mean of feature
    `return_feature_idx` (most bearish), regime K-1 = highest (most bullish).
    Keeps soft_probs columns comparable across causal refits.
    """
    mus = result["mus"]
    K = mus.shape[0]
    order = np.argsort(mus[:, return_feature_idx])
    inv = np.empty(K, dtype=np.int64)
    inv[order] = np.arange(K)

    result["labels"]     = inv[result["labels"]]
    result["soft_probs"] = result["soft_probs"][:, order]
    result["mus"]        = result["mus"][order]
    result["covs"]       = result["covs"][order]
    result["precisions"] = result["precisions"][order]
    result["log_dets"]   = result["log_dets"][order]
    return result


# =============================================================================
# Main fit: coordinate descent with random restarts
# =============================================================================
def _fit_jm(Y, K=None, lam=None, cov_mode=None,
            n_init=None, max_iter=None, tol=None,
            ridge=None, min_cluster=None, seed=None):
    """
    Fit one Statistical Jump Model on feature matrix Y.

    Parameters
    ----------
    Y         : (T, F) float64 feature matrix (should be standardised)
    K         : number of regimes
    lam       : jump penalty (units of per-state assignment cost)
    cov_mode  : "shared" | "diagonal_per_regime" | "full_per_regime"
    n_init    : random k-means++ restarts
    max_iter  : max coordinate-descent sweeps per restart
    tol       : early-stop on |delta loss| / max(|loss|, 1) < tol
    ridge     : Tikhonov added to every Sigma during M-step
    min_cluster : clusters smaller than this get reseeded
    seed      : base RNG seed; restart i uses seed + i

    Returns a dict with: labels, soft_probs, mus, covs, precisions,
    log_dets, loss, loss_history, cov_mode.

    Note on lambda scale: under "shared" with z-scored features
    lambda ~ 5-25 is typical; under "full_per_regime" per-state log|Sigma_k|
    terms shift the additive baseline and lambda may need re-tuning.
    """
    K           = JM_DEFAULTS["K"]            if K           is None else K
    lam         = JM_DEFAULTS["lam"]          if lam         is None else lam
    cov_mode    = JM_DEFAULTS["cov_mode"]     if cov_mode    is None else cov_mode
    n_init      = JM_DEFAULTS["n_init"]       if n_init      is None else n_init
    max_iter    = JM_DEFAULTS["max_iter"]     if max_iter    is None else max_iter
    tol         = JM_DEFAULTS["tol"]          if tol         is None else tol
    ridge       = JM_DEFAULTS["ridge"]        if ridge       is None else ridge
    min_cluster = JM_DEFAULTS["min_cluster"]  if min_cluster is None else min_cluster
    seed        = JM_DEFAULTS["seed"]         if seed        is None else seed

    if cov_mode not in ("shared", "diagonal_per_regime", "full_per_regime"):
        raise ValueError(f"unknown cov_mode={cov_mode!r}")

    T, F = Y.shape
    Ident = np.eye(F)
    best = None

    for init_idx in range(n_init):
        rng = np.random.default_rng(seed + init_idx)

        mus = _kmeans_plus_plus_init(Y, K, rng)
        covs       = np.broadcast_to(Ident, (K, F, F)).copy()
        precisions = np.broadcast_to(Ident, (K, F, F)).copy()
        log_dets   = np.zeros(K)

        prev_loss = np.inf
        loss_hist = []
        labels = np.zeros(T, dtype=np.int64)
        loss = np.inf

        for _ in range(max_iter):
            labels = _assign_viterbi(Y, mus, precisions, log_dets, lam)

            diffs = Y - mus[labels]
            cost_per_obs = (
                np.einsum("ti,tij,tj->t", diffs, precisions[labels], diffs)
                + log_dets[labels]
            )

            mus = _update_means(Y, labels, K, min_cluster, rng, cost_per_obs)

            if cov_mode == "shared":
                covs, precisions, log_dets = _pooled_scatter(Y, labels, mus, K, ridge)
            elif cov_mode == "diagonal_per_regime":
                covs, precisions, log_dets = _diag_per_regime(Y, labels, mus, K, ridge)
            else:
                covs, precisions, log_dets = _full_per_regime(Y, labels, mus, K, ridge)

            loss = _total_loss(Y, labels, mus, precisions, log_dets, lam)
            loss_hist.append(loss)
            if abs(prev_loss - loss) / max(abs(loss), 1.0) < tol:
                break
            prev_loss = loss

        if best is None or loss < best["loss"]:
            best = {
                "labels":       labels.copy(),
                "mus":          mus.copy(),
                "covs":         covs.copy(),
                "precisions":   precisions.copy(),
                "log_dets":     log_dets.copy(),
                "loss":         loss,
                "loss_history": loss_hist,
                "cov_mode":     cov_mode,
            }

    best["soft_probs"] = _soft_probs(Y, best["mus"], best["precisions"], best["log_dets"])
    return best


# =============================================================================
# Panel-level wrappers (mirror cpd.py)
# =============================================================================
def fit_panel_jm(panel, K=None, lam=None, cov_mode=None,
                 feature_window=None, **fit_kwargs):
    """
    Fit SJM independently on every ticker in `panel`.

    Returns
    -------
    {ticker: {
        "dates":        pd.DatetimeIndex aligned with the rows below,
        "labels":       np.ndarray (T_eff,),
        "soft_probs":   np.ndarray (T_eff, K),
        "mus":          (K, F), "covs": (K, F, F),
        "precisions":   (K, F, F), "log_dets": (K,),
        "loss":         float,   "loss_history": list[float],
        "cov_mode":     str,
        "feature_cols": list[str],
    }}
    """
    K              = JM_DEFAULTS["K"]              if K              is None else K
    lam            = JM_DEFAULTS["lam"]            if lam            is None else lam
    cov_mode       = JM_DEFAULTS["cov_mode"]       if cov_mode       is None else cov_mode
    feature_window = JM_DEFAULTS["feature_window"] if feature_window is None else feature_window

    out = {}
    min_rows = max(K * 5, 20)
    for tk, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        close = pd.Series(
            g["close"].values.astype(np.float64),
            index=pd.to_datetime(g["date"]),
        )
        Y, dates, cols, ret_idx = _build_features(close, window=feature_window)
        if len(Y) < min_rows:
            continue
        result = _fit_jm(Y, K=K, lam=lam, cov_mode=cov_mode, **fit_kwargs)
        result = _sort_states_by_return(result, return_feature_idx=ret_idx)
        result["dates"] = dates
        result["feature_cols"] = cols
        out[tk] = result
    return out


def fit_panel_jm_until(panel, end_date, **kwargs):
    """Causal refit using only rows with date <= end_date (analogue of
    cpd.segment_panel_until)."""
    end_date = pd.Timestamp(end_date)
    panel_cut = panel[pd.to_datetime(panel["date"]) <= end_date].copy()
    return fit_panel_jm(panel_cut, **kwargs)


def build_jm_cache(panel, dates, recompute_every=None, **kwargs):
    """
    Build a causal SJM cache keyed by snapshot date. Identical schedule
    to cpd.build_regime_cache: take dates[::recompute_every] and
    force-include the final date.
    """
    if recompute_every is None:
        recompute_every = JM_DEFAULTS["recompute_every"]

    dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    if len(dates) == 0:
        return {}

    update_dates = list(dates[::recompute_every])
    if update_dates[-1] != dates[-1]:
        update_dates.append(dates[-1])

    cache = {}
    for dt in update_dates:
        cache[pd.Timestamp(dt)] = fit_panel_jm_until(panel, dt, **kwargs)
    return cache


def get_cached_regime_probs(jm_cache, target_date):
    """
    Return latest SJM snapshot at or before `target_date` as
    {ticker: soft_probs[rows_with_date <= target_date, K]}.
    Mirrors cpd.get_cached_regimes.
    """
    target_date = pd.Timestamp(target_date)
    if not jm_cache:
        return {}

    keys = sorted(jm_cache)
    key_ns = np.array([k.value for k in keys], dtype=np.int64)
    idx = int(np.searchsorted(key_ns, target_date.value, side="right") - 1)
    if idx < 0:
        return {}

    snapshot = jm_cache[keys[idx]]
    out = {}
    for tk, result in snapshot.items():
        dates = result["dates"]
        probs = result["soft_probs"]
        mask = dates <= target_date
        out[tk] = probs[mask.values] if hasattr(mask, "values") else probs[mask]
    return out
