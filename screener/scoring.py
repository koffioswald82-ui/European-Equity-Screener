import numpy as np
import pandas as pd
from scipy import stats

WEIGHTS: dict[str, float] = {
    "value":    0.30,   # P/E + EV/EBIT (inverted — lower is better)
    "quality":  0.30,   # ROE + net margin
    "momentum": 0.25,   # returns over 3M / 6M / 12M
    "revision": 0.15,   # earnings & revenue growth as proxy for analyst revisions
}


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std < 1e-9:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def compute_value_score(df: pd.DataFrame) -> pd.Series:
    pe  = _zscore(1.0 / df["pe_ratio"].clip(lower=0.1))
    ev  = _zscore(1.0 / df["ev_ebit"].clip(lower=0.1))
    # fill if only one factor is available
    return pe.fillna(0).add(ev.fillna(0)) / 2


def compute_quality_score(df: pd.DataFrame) -> pd.Series:
    roe    = _zscore(df["roe"].fillna(0))
    margin = _zscore(df["net_margin"].fillna(0))
    roce   = _zscore(df["roce"].fillna(0))
    return (roe + margin + roce) / 3


def compute_momentum_score(prices: pd.DataFrame) -> pd.Series:
    """
    Prices: DataFrame with dates as index, tickers as columns.
    Returns a Series indexed by ticker.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    r3m  = _zscore(log_returns.iloc[-63:].sum())
    r6m  = _zscore(log_returns.iloc[-126:].sum())
    r12m = _zscore(log_returns.iloc[-252:].sum())
    return 0.4 * r3m + 0.3 * r6m + 0.3 * r12m


def compute_revision_score(df: pd.DataFrame) -> pd.Series:
    """Use earnings_growth + revenue_growth as proxy for analyst revisions."""
    eg = _zscore(df["earnings_growth"].fillna(0))
    rg = _zscore(df["revenue_growth"].fillna(0))
    return (eg + rg) / 2


def composite_score(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    sentiment: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Returns fundamentals DataFrame enriched with factor scores and a
    composite_score column in [0, 100] (percentile rank).

    fundamentals : one row per ticker, must include columns used by factor fns.
    prices       : OHLCV Close DataFrame (dates × tickers).
    sentiment    : optional DataFrame with columns [ticker, sentiment_score].
    """
    df = fundamentals.copy().set_index("ticker")

    # Align prices to tickers present in fundamentals
    common = [t for t in df.index if t in prices.columns]
    prices_aligned = prices[common] if common else prices

    # Factor scores
    df["value_score"]    = compute_value_score(df)
    df["quality_score"]  = compute_quality_score(df)
    df["revision_score"] = compute_revision_score(df)

    mom = compute_momentum_score(prices_aligned)
    mom_aligned = mom.reindex(df.index)
    prices_available = mom_aligned.notna().any()
    if prices_available:
        df["momentum_score"] = mom_aligned.fillna(0)
    # If no price data, skip momentum_score column entirely

    if sentiment is not None and not sentiment.empty:
        sent = sentiment.set_index("ticker")["sentiment_score"].reindex(df.index).fillna(0)
        df["sentiment_score"] = sent

    # Composite — only include factors with actual data, then renormalise weights
    active = {k: v for k, v in WEIGHTS.items() if f"{k}_score" in df.columns}
    total_w = sum(active.values()) or 1.0
    composite = sum(
        df[f"{k}_score"].fillna(0) * (v / total_w)
        for k, v in active.items()
    )

    # Percentile rank → [0, 100]
    df["composite_score"] = (
        stats.rankdata(composite.fillna(0)) / len(composite) * 100
    ).round(1)

    return (
        df
        .reset_index()
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )
