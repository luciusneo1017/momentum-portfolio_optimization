# src/momentumhk/backtest.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .signals import select_topN
from .weights import make_weights_all


# -------------------------- helpers --------------------------

def _period_end_trading_dates(idx: pd.DatetimeIndex, freq: str) -> List[pd.Timestamp]:
    """
    Return last trading day of each calendar period (M, Q, A...) present in idx.
    """
    # Calendar period ends
    period_end = idx.to_period(freq).to_timestamp(freq)
    # Map each period-end to the last trading date <= that timestamp
    pos = np.searchsorted(idx.values, period_end.values, side="right") - 1
    pos = np.clip(pos, 0, len(idx) - 1)
    dates = pd.DatetimeIndex(idx.values[pos]).unique().sort_values()
    return list(dates)

def month_end_rebalance_dates(idx: pd.DatetimeIndex, freq: str = "Q") -> List[pd.Timestamp]:
    """
    Convenience wrapper: 'M' (monthly), 'Q' (quarterly), 'A' (annual) etc.
    Returns last trading date per period.
    """
    return _period_end_trading_dates(idx, freq=freq)

def open_to_open_returns(open_px: pd.DataFrame) -> pd.DataFrame:
    """
    Open->Open simple returns.
    """
    return open_px.pct_change()

def perf_stats(returns: pd.Series, freq: int = 252) -> dict:
    """
    Basic performance summary on a daily return series.
    """
    r = returns.dropna()
    if r.empty:
        return dict(CAGR=np.nan, Vol=np.nan, Sharpe=np.nan, MaxDD=np.nan, AvgTurnover=np.nan)

    cum = (1.0 + r).cumprod()
    days = len(r)
    CAGR = (cum.iloc[-1] ** (freq / max(days, 1))) - 1.0
    Vol = r.std() * np.sqrt(freq)
    Sharpe = (r.mean() * freq) / Vol if Vol > 0 else np.nan
    roll_max = cum.cummax()
    dd = (cum / roll_max) - 1.0
    MaxDD = dd.min()
    return dict(CAGR=CAGR, Vol=Vol, Sharpe=Sharpe, MaxDD=MaxDD)

# -------------------------- core backtest --------------------------

def run_backtest(
    *,                               # parameters are keywords only
    prices: pd.DataFrame,            # adj close (dates x tickers)
    open_px: pd.DataFrame,           # open prices for execution
    rets_close: Optional[pd.DataFrame] = None,  # if None, computed from prices
    adv_df: Optional[pd.DataFrame] = None,      # optional liquidity filter
    N: int = 40,
    signal: str = "12-1",            # "12-1" | "6-12" | "3-6-12"
    signal_weights: Optional[Tuple[float, ...]] = None,
    min_price: Optional[float] = 3.0,
    min_adv: Optional[float] = 1e6,
    buffer_rank: Optional[int] = 60,
    tc_bps: float = 5.0,             # turnover cost in bps per unit L1 turnover
    rebalance_freq: str = "Q",       # 'M', 'Q', 'A', etc.
    schemes: Optional[List[str]] = None,  # subset of schemes to compute (keys of make_weights_all)
    # kwargs passed through to make_weights_all (windows, caps, shrinkage, etc.)
    vol_window: int = 126,
    pca_window: int = 252,
    pca_penalty: float = 1.0,
    cov_window: int = 126,
    max_w: float = 0.10,
    gross_cap: Optional[float] = 2.0,
    shrink_alpha: float = 0.10,
) -> dict:
    """
    Event-lite, vectorized backtest:
      - Select Top-N at each period-end trading day (no look-ahead).
      - Compute multiple weight schemes at the rebalance close.
      - Apply weights from the *next open*, charge turnover costs, aggregate performance.

    Returns
    -------
    dict with:
      - 'weights': dict[name -> DataFrame] weight time series (dates x tickers)
      - 'turnover': dict[name -> Series] daily portfolio turnover
      - 'pnl': dict[name -> Series] daily net returns (after TC)
      - 'stats': dict[name -> metrics dict]
      - 'rebalance_dates': List[pd.Timestamp]
    """
    # Align/validate inputs
    prices = prices.sort_index()
    open_px = open_px.reindex(prices.index).sort_index()
    if rets_close is None:
        rets_close = prices.pct_change()
    else:
        rets_close = rets_close.reindex(prices.index).sort_index()
    if adv_df is not None:
        adv_df = adv_df.reindex(prices.index).sort_index()

    # Rebalance schedule
    idx = prices.index
    rebals = month_end_rebalance_dates(idx, freq=rebalance_freq)

    # Which schemes?
    # make_weights_all returns keys:
    default_names = ["equal", "inv_vol", "pca", "mvo", "mvo_shr_diag", "mvo_ls", "mvo_ls_shr_diag"]
    use_schemes = schemes or default_names

    # Storage
    W = {k: pd.DataFrame(0.0, index=idx, columns=prices.columns, dtype=float) for k in use_schemes}
    daily_turnover = {k: pd.Series(0.0, index=idx, dtype=float) for k in use_schemes}

    # Turnover buffer memory
    current_holdings: List[str] = []

    # Selection + weights at each rebalance date
    for dt in rebals:
        # 1) Select basket
        top = select_topN(
            prices=prices, asof=dt, N=N,
            min_price=min_price,
            adv=adv_df, min_adv=min_adv,
            keep_current=current_holdings, buffer_rank=buffer_rank,
            signal=signal, signal_weights=signal_weights
        )
        if not top:
            current_holdings = []
            continue

        # 2) Compute weights per scheme (as-of dt close)
        all_w = make_weights_all(
            prices=prices,
            rets=rets_close,
            asof=dt,
            top=top,
            vol_window=vol_window,
            pca_window=pca_window,
            pca_penalty=pca_penalty,
            cov_window=cov_window,
            max_w=max_w,
            gross_cap=gross_cap,
            shrink_alpha=shrink_alpha,
        )

        # 3) Store at dt
        for name in use_schemes:
            w = all_w.get(name)
            if w is None or w.empty:
                continue
            W[name].loc[dt, w.index] = w.values

        current_holdings = top

    # 4) Execution: shift weights by +1 bar (trade next open)
    W = {k: df.replace([np.inf, -np.inf], 0.0).fillna(0.0).ffill().shift(1).fillna(0.0) for k, df in W.items()}

    # 5) PnL & turnover costs
    oo = open_to_open_returns(open_px).reindex(idx)
    tc = float(tc_bps) / 1e4  # bps -> fraction

    pnl: Dict[str, pd.Series] = {}
    stats: Dict[str, dict] = {}

    for name, wts in W.items():
        # Gross returns
        gross = (wts * oo).sum(axis=1).fillna(0.0)

        # Turnover: daily L1 change in weights
        turn = (wts - wts.shift(1).fillna(0.0)).abs().sum(axis=1)
        daily_turnover[name] = turn

        # Net after TC
        net = gross - tc * turn
        pnl[name] = net
        s = perf_stats(net)
        # Attach average daily turnover for reference
        s["AvgTurnover"] = turn.mean()
        stats[name] = s

    return {
        "weights": W,
        "turnover": daily_turnover,
        "pnl": pnl,
        "stats": stats,
        "rebalance_dates": rebals,
    }


# -------------------------- (optional) CLI/demo --------------------------

if __name__ == "__main__":
    # Minimal smoke test (expects you to load your own data)
    # Example:
    #   prices = pd.read_parquet("cleaned_data/prices.parquet")      # adj close
    #   open_px = pd.read_parquet("cleaned_data/open.parquet")        # open
    #   adv_df  = pd.read_parquet("cleaned_data/adv.parquet")         # optional
    #   results = run_backtest(prices=prices, open_px=open_px, adv_df=adv_df,
    #                          N=40, signal="3-6-12", signal_weights=(0.2,0.3,0.5),
    #                          rebalance_freq="Q", tc_bps=5.0)
    #   print(pd.DataFrame(results["stats"]).T)
    print("Run me from your notebook or pipeline. See __main__ for example usage.")
