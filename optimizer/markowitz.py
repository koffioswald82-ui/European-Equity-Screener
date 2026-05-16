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
    model: str = "Markowitz"

    def __post_init__(self):
        if self.weights_df.empty and self.tickers:
            self.weights_df = pd.DataFrame(
                {"ticker": self.tickers, "weight": self.weights}
            ).sort_values("weight", ascending=False).reset_index(drop=True)


def _portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.03):
    ret = float(w @ mu)
    vol = float(np.sqrt(max(float(w @ cov @ w), 0.0)))
    shr = (ret - rf) / max(vol, 1e-9)
    return ret, vol, shr


# ── Covariance estimators ──────────────────────────────────────────────────────

def compute_cov_matrix(prices: pd.DataFrame, annualise: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Historical log-return covariance (standard Markowitz)."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    factor = 252 if annualise else 1
    mu = log_ret.mean().values * factor
    cov = log_ret.cov().values * factor
    return mu, cov


def ledoit_wolf_cov(prices: pd.DataFrame, annualise: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Ledoit-Wolf analytical shrinkage covariance.
    Shrinks sample covariance toward scaled identity — dramatically reduces
    estimation error for small-sample / large-universe problems.
    """
    from sklearn.covariance import LedoitWolf
    log_ret = np.log(prices / prices.shift(1)).dropna()
    factor = 252 if annualise else 1
    mu = log_ret.mean().values * factor
    lw = LedoitWolf().fit(log_ret.values)
    return mu, lw.covariance_ * factor


# ── Black-Litterman ───────────────────────────────────────────────────────────

def black_litterman(
    cov: np.ndarray,
    tickers: list[str],
    market_weights: np.ndarray | None = None,
    delta: float = 2.5,
    tau: float = 0.05,
    views: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Black-Litterman posterior (μ_BL, Σ_BL).

    Parameters
    ----------
    market_weights : market-cap proxy weights; defaults to equal weights.
    delta          : risk-aversion coefficient (CAPM: ~2.5).
    tau            : prior uncertainty scalar (0.025–0.1).
    views          : list of absolute-view dicts:
                     {"assets": ["MC.PA"], "weights": [1.0], "return": 0.12}
                     or relative: {"assets": ["MC.PA","OR.PA"], "weights": [1,-1], "return": 0.05}

    Returns
    -------
    (mu_bl, cov_bl) — posterior expected returns and posterior covariance.
    """
    n = len(tickers)
    if market_weights is None:
        market_weights = np.ones(n) / n

    # Reverse-optimisation: implied equilibrium excess returns
    pi = delta * cov @ market_weights

    if not views:
        return pi, cov

    ticker_idx = {t: i for i, t in enumerate(tickers)}
    P_rows, q_vals = [], []
    for v in views:
        row = np.zeros(n)
        for asset, w in zip(v["assets"], v["weights"]):
            if asset in ticker_idx:
                row[ticker_idx[asset]] = w
        if np.any(row != 0):
            P_rows.append(row)
            q_vals.append(float(v["return"]))

    if not P_rows:
        return pi, cov

    P = np.array(P_rows)
    q = np.array(q_vals)

    # Diagonal view-uncertainty matrix (He & Litterman proportional form)
    omega = np.diag(np.diag(tau * P @ cov @ P.T))

    tau_cov = tau * cov
    try:
        inv_tau_cov = np.linalg.inv(tau_cov)
        inv_omega = np.linalg.inv(omega)
        M = np.linalg.inv(inv_tau_cov + P.T @ inv_omega @ P)
        mu_bl = M @ (inv_tau_cov @ pi + P.T @ inv_omega @ q)
        cov_bl = cov + M
    except np.linalg.LinAlgError:
        return pi, cov

    return mu_bl, cov_bl


# ── Risk Parity (Equal Risk Contribution) ─────────────────────────────────────

def risk_parity(
    cov: np.ndarray,
    tickers: list[str],
    rf: float = 0.03,
    mu: np.ndarray | None = None,
    w_min: float = 0.01,
    w_max: float = 0.30,
) -> PortfolioResult:
    """
    Equal Risk Contribution portfolio — each asset contributes the same
    fraction of total portfolio variance.  No expected-return assumption needed.
    """
    n = len(tickers)

    def _rc(w: np.ndarray) -> np.ndarray:
        sigma = float(np.sqrt(max(float(w @ cov @ w), 1e-18)))
        return (w * (cov @ w)) / sigma

    def _objective(w: np.ndarray) -> float:
        rc = _rc(w)
        target = np.sum(rc) / n
        return float(np.sum((rc - target) ** 2))

    x0 = np.ones(n) / n
    bounds = [(w_min, w_max)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    res = minimize(
        _objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    _mu = mu if mu is not None else np.zeros(n)
    ret, vol, shr = _portfolio_stats(res.x, _mu, cov, rf)
    result = PortfolioResult(tickers, res.x, ret, vol, shr, model="Risk Parity")
    return result


# ── Standard Markowitz optimisers ─────────────────────────────────────────────

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
    result = PortfolioResult(tickers, res.x, ret, vol, shr)
    result.model = "Max Sharpe"
    return result


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
    result = PortfolioResult(tickers, res.x, ret, vol, shr)
    result.model = "Min Variance"
    return result


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
            p = max_sharpe(mu, cov, tickers, rf=rf, w_min=w_min, w_max=w_max,
                           extra_constraints=extra)
            results.append(p)
        except Exception:
            pass
    return results
