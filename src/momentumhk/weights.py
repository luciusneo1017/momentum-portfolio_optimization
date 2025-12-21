# src/momentumhk/weights.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import cvxpy as cp
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf


# --------- helpers ---------

def _slice_until(df: pd.DataFrame, asof: pd.Timestamp, window: int | None) -> pd.DataFrame:
    """Return df up to asof (inclusive), optionally taking the last `window` rows."""
    df_ = df.loc[:pd.Timestamp(asof)]
    if window is not None:
        df_ = df_.tail(int(window))
    return df_


def _simple_shrinkage_cov(X: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """
    Shrink sample cov toward its diagonal: (1-α)Σ + α*diag(Σ).
    X: returns (T×N), columns=tickers
    """
    Sig = X.cov()
    D = pd.DataFrame(np.diag(np.diag(Sig.values)), index=Sig.index, columns=Sig.columns)
    return (1 - alpha) * Sig + alpha * D


def _ledoit_wolf_cov(X: pd.DataFrame) -> pd.DataFrame:
    """Ledoit–Wolf covariance (sklearn)."""
    lw = LedoitWolf().fit(X.dropna())
    Sig = pd.DataFrame(lw.covariance_, index=X.columns, columns=X.columns)
    return Sig


def _normalize_long_only(w: pd.Series) -> pd.Series:
    w = w.clip(lower=0.0).fillna(0.0)
    s = w.sum()
    return w / s if s > 0 else w


def _normalize_net1(w: pd.Series) -> pd.Series:
    """Normalize to sum(w)=1 (can be negative weights)."""
    s = w.sum()
    return w / s if s != 0 else w


# --------- 1) Equal weight ---------

def equal_weight(top: List[str]) -> pd.Series:
    n = len(top)
    if n == 0:
        return pd.Series(dtype=float)
    w = pd.Series(1.0 / n, index=top)
    return w


# --------- 2) Inverse-vol ---------

def inverse_vol_weights(rets: pd.DataFrame, asof: pd.Timestamp, top: List[str],
                        vol_window: int = 126) -> pd.Series:
    R = _slice_until(rets[top], asof, vol_window)
    sig = R.std().replace(0, np.nan)
    w = 1.0 / sig
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _normalize_long_only(w)


# --------- 3) PCA tilt ---------

def pca_tilt_weights(rets: pd.DataFrame, asof: pd.Timestamp, top: List[str],
                     pca_window: int = 252, penalty: float = 1.0) -> pd.Series:
    """
    Down-weight stocks with large |PC1 loading|.
    weight_i ∝ 1 / (1 + penalty * |loading_i|)
    """
    X = _slice_until(rets[top], asof, pca_window).dropna(how="any")
    if X.shape[0] < 10:  # too little data
        return equal_weight(top)

    p = PCA(n_components=1).fit(X)
    load = pd.Series(np.abs(p.components_[0]), index=top)
    scale = 1.0 / (1.0 + penalty * load)
    return _normalize_long_only(scale)


# --------- 4/5) Min-variance (long-only), with/without shrinkage ---------

def mvo_long_only(rets: pd.DataFrame, asof: pd.Timestamp, top: List[str],
                         cov_window: int = 126, max_w: Optional[float] = None,
                         shrink: Optional[str] = None, shrink_alpha: float = 0.1) -> pd.Series:
    """
    Minimize w' Σ w  s.t. sum w = 1, 0 <= w <= max_w
    shrink: None | 'diag' (simple diagonal shrink) | 'lw' (Ledoit-Wolf)
    """
    if cp is None:
        raise ImportError("cvxpy is required for MVO weights.")

    X = _slice_until(rets[top], asof, cov_window).dropna(how="any")
    if X.shape[0] < len(top):
        # fallback when data is scarce
        return inverse_vol_weights(rets, asof, top, cov_window)

    if shrink is None:
        Sig = X.cov()
    elif shrink == "diag":
        Sig = _simple_shrinkage_cov(X, alpha=shrink_alpha)
    elif shrink == "lw":
        Sig = _ledoit_wolf_cov(X)
    else:
        raise ValueError("Unknown shrink option.")

    n = len(top)
    w = cp.Variable(n)
    Sigma = Sig.values

    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    if max_w is not None:
        constraints.append(w <= max_w)

    obj = cp.Minimize(cp.quad_form(w, Sigma))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    wv = np.array(w.value).ravel()
    wv = np.where(wv < 0, 0, wv)
    wv = wv / wv.sum() if wv.sum() > 0 else wv
    return pd.Series(wv, index=top)


# --------- 6/7) Unconstrained min-variance (long–short), with/without shrink ---------

def mvo_long_short(rets: pd.DataFrame, asof: pd.Timestamp, top: List[str],
                          cov_window: int = 126, shrink: Optional[str] = None,
                          shrink_alpha: float = 0.1, gross_cap: Optional[float] = None) -> pd.Series:
    """
    Minimize w' Σ w  s.t. sum w = 1  (weights can be +/-)
    Optional gross exposure cap: ||w||_1 <= G
    """
    if cp is None:
        raise ImportError("cvxpy is required for MVO weights.")

    X = _slice_until(rets[top], asof, cov_window).dropna(how="any")
    if X.shape[0] < len(top):
        # fallback: small ridge on diag to avoid singularity
        Sig = X.cov()
        Sig = Sig + 1e-6 * np.eye(len(Sig))
    else:
        if shrink is None:
            Sig = X.cov()
        elif shrink == "diag":
            Sig = _simple_shrinkage_cov(X, alpha=shrink_alpha)
        elif shrink == "lw":
            Sig = _ledoit_wolf_cov(X)
        else:
            raise ValueError("Unknown shrink option.")

    n = len(top)
    w = cp.Variable(n)
    Sigma = Sig.values

    constraints = [cp.sum(w) == 1]
    if gross_cap is not None:
        constraints.append(cp.norm1(w) <= float(gross_cap))

    obj = cp.Minimize(cp.quad_form(w, Sigma))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    wv = np.array(w.value).ravel()
    # normalize to net=1 in case solver drifted
    wv = wv / wv.sum() if wv.sum() != 0 else wv #cannot div by 0
    return pd.Series(wv, index=top)


# --------- orchestrator ---------

def make_weights_all(*,                     # keyword only function, all inputs must be passed  by name
    prices: pd.DataFrame,
    rets: pd.DataFrame,
    asof: pd.Timestamp,
    top: List[str],
    vol_window: int = 126,
    pca_window: int = 252,
    pca_penalty: float = 1.0,
    cov_window: int = 126,
    max_w: float = 0.10,
    gross_cap: Optional[float] = 2.0,         # for long–short
    shrink_alpha: float = 0.10                # for 'diag' shrink
) -> Dict[str, pd.Series]:
    """
    Compute all weight schemes on the same basket/date.
    Returns dict of pd.Series (index=top tickers).
    """
    # Clean list to available tickers at asof
    top = [t for t in top if t in prices.columns]
    if len(top) == 0:
        return {k: pd.Series(dtype=float) for k in
                ["equal", "inv_vol", "pca", "mvo", "mvo_shr_diag", "mvo_ls", "mvo_ls_shr_diag"]}

    weights = {
        "equal": equal_weight(top),
        "inv_vol": inverse_vol_weights(rets, asof, top, vol_window=vol_window),
        "pca": pca_tilt_weights(rets, asof, top, pca_window=pca_window, penalty=pca_penalty),
        "mvo": mvo_long_only(rets, asof, top, cov_window=cov_window, max_w=max_w, shrink=None),
        "mvo_shr_diag": mvo_long_only(rets, asof, top, cov_window=cov_window, max_w=max_w,
                                             shrink="diag", shrink_alpha=shrink_alpha),
        "mvo_ls": mvo_long_short(rets, asof, top, cov_window=cov_window, shrink=None,
                                        gross_cap=gross_cap),
        "mvo_ls_shr_diag": mvo_long_short(rets, asof, top, cov_window=cov_window,
                                                 shrink="diag", shrink_alpha=shrink_alpha,
                                                 gross_cap=gross_cap),
    }
    return weights
