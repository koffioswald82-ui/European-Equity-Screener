import numpy as np
import pandas as pd


def sector_min_weight(
    tickers: list[str],
    fundamentals: pd.DataFrame,
    sector_col: str = "sector",
    min_weight: float = 0.05,
) -> list[dict]:
    """
    Build SLSQP inequality constraints ensuring each sector has >= min_weight
    of total portfolio weight.
    """
    sector_map = (
        fundamentals.set_index("ticker")[sector_col]
        .reindex(tickers)
        .fillna("Unknown")
    )
    constraints = []
    for sector in sector_map.unique():
        idx = np.array([i for i, t in enumerate(tickers) if sector_map[t] == sector])
        if len(idx) == 0:
            continue
        constraints.append({
            "type": "ineq",
            "fun": lambda w, ix=idx, mw=min_weight: w[ix].sum() - mw,
        })
    return constraints


def tracking_error_constraint(
    benchmark_weights: np.ndarray,
    cov: np.ndarray,
    max_te: float = 0.05,
) -> dict:
    """
    Inequality constraint: annualised tracking error vs benchmark <= max_te.
    benchmark_weights must be aligned with the optimisation universe.
    """
    def _te(w):
        diff = w - benchmark_weights
        te   = float(np.sqrt(diff @ cov @ diff))
        return max_te - te   # >= 0  means feasible

    return {"type": "ineq", "fun": _te}


def max_single_weight(w_max: float = 0.10, n: int | None = None) -> list[tuple]:
    """Return scipy bounds list capping each weight at w_max (1–10% range)."""
    if n is None:
        return []
    return [(0.01, w_max)] * n
