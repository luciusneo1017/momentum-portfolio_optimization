# src/momentumhk/signals.py
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Iterable, Optional, Tuple, List, Callable


# ----------------------------- #
# Rebalance date helpers
# ----------------------------- #

def quarter_end_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Return the trading days that are quarter-ends based on `dates`.
    Equivalent to resample('Q').last() on a calendar, but using trading days.
    """
    # Group by (year, quarter), take last date in each group
    q = pd.Series(dates, index=dates)
    grp = q.groupby([dates.year, dates.quarter]).last()
    return pd.DatetimeIndex(grp.values)


def quarterly_rebalance_dates(
    prices: pd.DataFrame,
    warmup_days: int = 252 + 21,  # ~13 months warmup for 12-1 momentum
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DatetimeIndex:
    """
    Compute quarterly rebalance dates (last trading day of each quarter),
    optionally clipped to [start, end], and with an initial warmup period.
    """
    idx = prices.index
    qend = quarter_end_dates(idx)

    if start is not None:
        qend = qend[qend >= pd.Timestamp(start)]
    if end is not None:
        qend = qend[qend <= pd.Timestamp(end)]

    # Apply warmup: drop any rebalance dates earlier than warmup_days into the series
    first_valid = idx[min(len(idx) - 1, warmup_days)]
    qend = qend[qend >= first_valid]
    return qend


# ----------------------------- #
# Momentum (12-1) computation
# ----------------------------- #

def _month_shift_index(idx: pd.DatetimeIndex, months: int) -> pd.DatetimeIndex:
    """
    For each date t in idx, take the CALENDAR month-end that is `months` back,
    then snap to the last available trading day <= that target month-end.
    """
    # Month-end of t's month, then back `months` month-ends
    target_me = (idx + pd.offsets.MonthEnd(0)) - pd.offsets.MonthEnd(months)
    # For each target date, find last idx <= target (binary search)
    pos = np.searchsorted(idx.values, target_me.values, side="right") - 1
    pos = np.clip(pos, 0, len(idx) - 1)
    return pd.DatetimeIndex(idx.values[pos])


def momentum_12m_minus_1(prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    idx = prices.index
    asof = pd.Timestamp(asof)

    # locate/align asof to trading calendar
    try:
        t_pos = idx.get_loc(asof)
    except KeyError:
        t_pos = np.searchsorted(idx.values, asof.to_datetime64(), side="right") - 1
        if t_pos < 0:
            raise ValueError("`asof` is before the first available trading day.")
    if t_pos < 252:
        raise ValueError("Not enough history for 12-1 momentum (~13 months).")

    t_minus_1_idx  = _month_shift_index(idx, months=1)
    t_minus_12_idx = _month_shift_index(idx, months=12)

    t1  = t_minus_1_idx[t_pos]
    t12 = t_minus_12_idx[t_pos]

    p1  = prices.loc[t1]
    p12 = prices.loc[t12]
    return (p1 / p12) - 1.0




# Selection (Top-N) with filters

def select_topN(
    prices: pd.DataFrame,
    asof: pd.Timestamp,
    N: int = 40,
    *,
    # Filters
    min_price: Optional[float] = None,
    adv: Optional[pd.DataFrame] = None,
    min_adv: Optional[float] = None,
    require_history_days: int = 252 + 21,
    # Turnover buffer
    keep_current: Optional[Iterable[str]] = None,
    buffer_rank: Optional[int] = None,
    # Signal selection
    signal: str = "12-1",                          # {"12-1","6-12","3-6-12"}
    signal_weights: Optional[Tuple[float, ...]] = None,
) -> List[str]:
    """
    Select Top-N tickers by a momentum-like cross-sectional score at `asof`,
    with optional price/ADV filters and a 'keep' buffer to reduce turnover.

    Parameters
    ----------
    prices : DataFrame
        Adjusted close prices (index = trading dates, columns = tickers).
    asof : Timestamp
        Rebalance date (trading day).
    N : int
        Number of names to return.
    min_price : float, optional
        Drop tickers whose *asof* price is below this.
    adv : DataFrame, optional
        Average daily (dollar) volume aligned to `prices.index`.
    min_adv : float, optional
        Drop tickers with ADV below this threshold (units must match `adv`).
    require_history_days : int
        Require at least this many trading days of history up to `asof`.
    keep_current : iterable[str], optional
        Current holdings to consider keeping if still ranked within `buffer_rank`.
    buffer_rank : int, optional
        If given with `keep_current`, keep those names with rank <= buffer_rank
        before filling the rest up to N in rank order.
    signal : str
        Which built-in signal to use: "12-1", "6-12", "3-6-12".
    signal_weights : tuple, optional
        Weights for blended signals. Defaults: (0.4,0.6) for "6-12"; (0.2,0.3,0.5) for "3-6-12".
    

    Returns
    -------
    list[str]
        Selected tickers (length <= N).
    """
    asof = pd.Timestamp(asof)

    # 1) Use only data up to asof (no look-ahead)
    px = prices.loc[:asof]

    # 2) History requirement per column
    enough_hist = px.count() >= int(require_history_days)
    candidates = prices.columns[enough_hist.values]
    if len(candidates) == 0:
        return []

    # 3) Compute cross-sectional score on candidates


    sig = signal.lower()
    if sig == "12-1":
        score = momentum_12m_minus_1(px[candidates], px.index[-1])

    elif sig == "6-12":
        w = signal_weights or (0.4, 0.6)
        score = blended_rank_6_12(px[candidates], px.index[-1], weights=w)

    elif sig == "3-6-12":
        w = signal_weights or (0.2, 0.3, 0.5)
        score = blended_rank_3_6_12(px[candidates], px.index[-1], weights=w)


    # 4) As-of price filter
    if min_price is not None:
        last_px = px.iloc[-1].reindex(score.index)
        ok_price = last_px[last_px >= float(min_price)].index
        score = score.loc[score.index.intersection(ok_price)]

    # 5) As-of ADV filter
    if adv is not None and min_adv is not None and len(score) > 0:
        adv_asof = adv.loc[:asof].iloc[-1].reindex(score.index).dropna()
        ok_adv = adv_asof[adv_asof >= float(min_adv)].index
        score = score.loc[score.index.intersection(ok_adv)]

    if len(score) == 0:
        return []

    # 6) Rank high→low (best rank=1) and sort
    ranks = score.rank(ascending=False, method="first").sort_values()

    # 7) Keep-buffer to reduce turnover
    if keep_current and buffer_rank:
        keep_current = [t for t in keep_current if t in ranks.index]
        keepable = [t for t in keep_current if ranks[t] <= buffer_rank]
        kept_sorted = sorted(keepable, key=lambda t: ranks[t])  # preserve rank order
        remainder = [t for t in ranks.index if t not in kept_sorted]
        ordered = kept_sorted + remainder
    else:
        ordered = list(ranks.index)

    return ordered[: min(N, len(ordered))]




def blended_rank_6_12(
    prices: pd.DataFrame,
    asof: pd.Timestamp,
    weights: tuple[float, float] = (0.4, 0.6)
) -> pd.Series:
    idx = prices.index
    asof = pd.Timestamp(asof)

    try:
        t_pos = idx.get_loc(asof)
    except KeyError:
        t_pos = np.searchsorted(idx.values, asof.to_datetime64(), side="right") - 1
        if t_pos < 0:
            raise ValueError("`asof` precedes first trading day.")
    if t_pos < 252:
        raise ValueError("Not enough history for 12-1 momentum (~13 months).")

    t1  = _month_shift_index(idx, 1)[t_pos]
    t6  = _month_shift_index(idx, 6)[t_pos]
    t12 = _month_shift_index(idx,12)[t_pos]

    p1  = prices.loc[t1]
    r6  = p1 / prices.loc[t6]  - 1.0   # 6-1
    r12 = p1 / prices.loc[t12] - 1.0   # 12-1

    w6, w12 = weights
    return (
        r6.rank(ascending=False)  * w6 +
        r12.rank(ascending=False) * w12
    )


def blended_rank_3_6_12(
    prices: pd.DataFrame,
    asof: pd.Timestamp,
    weights: tuple[float, float, float] = (0.2, 0.3, 0.5)
) -> pd.Series:
    w3, w6, w12 = weights
    idx = prices.index
    asof = pd.Timestamp(asof)

    try:
        t_pos = idx.get_loc(asof)
    except KeyError:
        t_pos = np.searchsorted(idx.values, asof.to_datetime64(), side="right") - 1
        if t_pos < 0:
            raise ValueError("`asof` precedes first trading day.")
    if t_pos < 252:
        raise ValueError("Not enough history for 12-1 momentum (~13 months).")

    t1  = _month_shift_index(idx, 1)[t_pos]
    t3  = _month_shift_index(idx, 3)[t_pos]
    t6  = _month_shift_index(idx, 6)[t_pos]
    t12 = _month_shift_index(idx,12)[t_pos]

    p1 = prices.loc[t1]
    r3  = p1 / prices.loc[t3]  - 1.0
    r6  = p1 / prices.loc[t6]  - 1.0
    r12 = p1 / prices.loc[t12] - 1.0

    return (
        r3.rank(ascending=False)  * w3 +
        r6.rank(ascending=False)  * w6 +
        r12.rank(ascending=False) * w12
    )
