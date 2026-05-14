import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.optimize import minimize


@dataclass
class PortfolioResult:
    tickers: list[str]
    weights: np.ndarray
    expected_ret: float
    volatility: float
    sharpe: float
    weights_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        if self.weights_df.empty and self.tickers:
            self.weights_df = pd.DataFrame(
                {"ticker": self.tickers, "weight": self.weights}
            ).sort_values("weight", ascending=False).reset_index(drop=True)


def _portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.03):
    ret  = float(w @ mu)
    vol  = float(np.sqrt(w @ cov @ w))
    shr  = (ret - rf) / max(vol, 1e-9)
    return ret, vol, shr


def max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    tickers: list[str],
    rf: float = 0.03,
    w_min: float = 0.01,
    w_max: float = 0.10,
    extra_constraints: list | None = None,
) -> PortfolioResult:
    """Maximise the Sharpe ratio subject to weight bounds."""
    n = len(mu)
    x0 = np.ones(n) / n
    bounds = [(w_min, w_max)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if extra_constraints:
        constraints.extend(extra_constraints)

    res = minimize(
        lambda w: -_portfolio_stats(w, mu, cov, rf)[2],
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    ret, vol, shr = _portfolio_stats(res.x, mu, cov, rf)
    return PortfolioResult(tickers, res.x, ret, vol, shr)


def min_variance(
    mu: np.ndarray,
    cov: np.ndarray,
    tickers: list[str],
    w_min: float = 0.01,
    w_max: float = 0.10,
    extra_constraints: list | None = None,
) -> PortfolioResult:
    """Minimise portfolio variance (minimum-variance portfolio)."""
    n = len(mu)
    x0 = np.ones(n) / n
    bounds = [(w_min, w_max)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if extra_constraints:
        constraints.extend(extra_constraints)

    res = minimize(
        lambda w: float(w @ cov @ w),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )
    ret, vol, shr = _portfolio_stats(res.x, mu, cov)
    return PortfolioResult(tickers, res.x, ret, vol, shr)


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    tickers: list[str],
    n_points: int = 50,
    rf: float = 0.03,
    w_min: float = 0.01,
    w_max: float = 0.10,
) -> list[PortfolioResult]:
    """Trace the efficient frontier by scanning target return levels."""
    target_rets = np.linspace(mu.min(), mu.max(), n_points)
    results = []
    for target in target_rets:
        extra = [{"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t}]
        try:
            p = max_sharpe(mu, cov, tickers, rf=rf, w_min=w_min, w_max=w_max, extra_constraints=extra)
            results.append(p)
        except Exception:
            pass
    return results


def compute_cov_matrix(prices: pd.DataFrame, annualise: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute annualised mean returns and covariance matrix from a prices DataFrame.
    Returns (mu, cov) — both numpy arrays.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    factor  = 252 if annualise else 1
    mu  = log_ret.mean().values * factor
    cov = log_ret.cov().values * factor
    return mu, cov
