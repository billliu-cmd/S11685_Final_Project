"""
GP-based Change-Point Detection (Revised): 
Segments each asset's price series into stationary regimes
that serve as the context pool for episodic training.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
import pandas as pd

from .config import CPD


# ═══════════════════════════════════════════════════════════════════════════════
# GP primitives
# ═══════════════════════════════════════════════════════════════════════════════
def _matern32(t, sig2, ell):
    """Matérn 3/2 covariance matrix."""
    d = np.abs(t[:, None] - t[None, :])
    r = np.sqrt(3.0) * d / max(ell, 1e-6)
    return sig2 * (1.0 + r) * np.exp(-r)


def _log_marglik(y, K, noise=1e-4):
    """Log marginal likelihood via Cholesky."""
    n = len(y)
    Ky = K + noise * np.eye(n)
    try:
        L = np.linalg.cholesky(Ky)
    except np.linalg.LinAlgError:
        Ky += 1e-3 * np.eye(n)
        L = np.linalg.cholesky(Ky)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return float(-0.5 * y @ alpha
                 - np.sum(np.log(np.diag(L)))
                 - 0.5 * n * np.log(2.0 * np.pi))


def _fit_gp(t, y):
    """Fit Matérn 3/2 GP, return (log_marglik, sig2, ell)."""
    def neg(params):
        sig2, ell = np.exp(params)
        return -_log_marglik(y, _matern32(t, sig2, ell))
    res = minimize(neg, x0=[0.0, 1.0], method="L-BFGS-B")
    sig2, ell = np.exp(res.x)
    return -res.fun, sig2, ell


# ═══════════════════════════════════════════════════════════════════════════════
# Revised Algorithm 1 from Paper
# ═══════════════════════════════════════════════════════════════════════════════
def segment_series(prices, lbw=None, nu=None, l_min=None, l_max=None):
    """Backward CPD scan on one price series. Returns [(start, end), ...]."""
    lbw   = lbw   or CPD["lbw"]
    nu    = nu    or CPD["nu"]
    l_min = l_min or CPD["l_min"]
    l_max = l_max or CPD["l_max"]

    T = len(prices)
    if T < lbw:
        return [(0, T - 1)] if T >= l_min else []

    t = T - 1
    t1 = T - 1
    regimes = []

    while t >= 0:
        w0 = max(t - lbw + 1, 0)
        w = prices[w0:t + 1].astype(np.float64)
        if len(w) < 5:
            t -= 1
            continue
        w = (w - w.mean()) / (w.std() + 1e-8)
        ti = np.arange(len(w), dtype=np.float64)

        # standard GP
        L_M, sig2, ell = _fit_gp(ti, w)

        # change-point GP: grid search over split points, reuse hyperparams
        best_L_C, best_cp = -np.inf, len(w) // 2
        for sp in range(3, len(w) - 2):
            L1 = _log_marglik(w[:sp], _matern32(ti[:sp], sig2, ell))
            L2 = _log_marglik(w[sp:], _matern32(ti[sp:] - ti[sp], sig2, ell))
            if L1 + L2 > best_L_C:
                best_L_C = L1 + L2
                best_cp = sp

        # severity: sigmoid(L_C - L_M) for numerical stability
        severity = 1.0 / (1.0 + np.exp(L_M - best_L_C))

        if severity >= nu:
            cp_abs = w0 + best_cp
            if t1 - cp_abs >= l_min:
                regimes.append((cp_abs, t1))
            t = cp_abs - 1
            t1 = t
        else:
            t -= 1
            if t1 - t > l_max:
                t = t1 - l_max
            if t1 - t == l_max:
                regimes.append((t, t1))
                t1 = t

    if t1 - max(t, 0) >= l_min:
        regimes.append((max(t, 0), t1))

    regimes.sort()
    return regimes


def segment_panel(panel):
    """Run CPD on every ticker. Returns {ticker: [(start, end), ...]}."""
    out = {}
    for tk, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        out[tk] = segment_series(g["close"].values)
    return out
