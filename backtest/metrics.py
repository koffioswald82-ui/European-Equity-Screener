import numpy as np
import pandas as pd
from backtest.engine import BacktestResult


def sharpe_ratio(returns: pd.Series, rf: float = 0.03, periods: int = 252) -> float:
    excess = returns - rf / periods
    if excess.std() < 1e-9:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, rf: float = 0.03, periods: int = 252) -> float:
    excess = returns - rf / periods
    downside = excess[excess < 0].std()
    if downside < 1e-9:
        return 0.0
    return float(excess.mean() / downside * np.sqrt(periods))


def max_drawdown(nav: pd.Series) -> float:
    """Return the maximum drawdown as a positive fraction (e.g. 0.25 = 25%)."""
    peak = nav.cummax()
    dd = (nav - peak) / peak
    return float(abs(dd.min()))


def calmar_ratio(returns: pd.Series, nav: pd.Series, periods: int = 252) -> float:
    ann_ret = float((1 + returns.mean()) ** periods - 1)
    mdd = max_drawdown(nav)
    return ann_ret / mdd if mdd > 1e-9 else 0.0


def cagr(nav: pd.Series, periods: int = 252) -> float:
    n = len(nav)
    if n < 2:
        return 0.0
    return float((nav.iloc[-1] / nav.iloc[0]) ** (periods / n) - 1)


def alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: float = 0.03,
    periods: int = 252,
) -> tuple[float, float]:
    """
    OLS regression of excess strategy returns on excess benchmark returns.
    Returns (alpha_annualised, beta).
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return 0.0, 1.0
    strat  = aligned.iloc[:, 0] - rf / periods
    bench  = aligned.iloc[:, 1] - rf / periods
    bench_var = max(float(np.var(bench)), 1e-9)
    beta   = float(np.cov(strat, bench)[0, 1] / bench_var)
    alpha  = float((strat.mean() - beta * bench.mean()) * periods)
    return alpha, beta


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods: int = 252,
) -> float:
    diff = strategy_returns - benchmark_returns
    if diff.std() < 1e-9:
        return 0.0
    return float(diff.mean() / diff.std() * np.sqrt(periods))


def var_95(returns: pd.Series) -> float:
    """Historical 95% 1-day VaR (positive number)."""
    return float(abs(np.percentile(returns.dropna(), 5)))


def compute_all(result: BacktestResult, rf: float = 0.03) -> dict:
    """Compute the full set of performance metrics from a BacktestResult."""
    sr = result.portfolio_returns
    br = result.benchmark_returns
    pn = result.portfolio_nav
    a, b = alpha_beta(sr, br, rf)
    return {
        "CAGR":              f"{cagr(pn):.2%}",
        "Sharpe":            f"{sharpe_ratio(sr, rf):.2f}",
        "Sortino":           f"{sortino_ratio(sr, rf):.2f}",
        "Calmar":            f"{calmar_ratio(sr, pn):.2f}",
        "Max Drawdown":      f"{max_drawdown(pn):.2%}",
        "Alpha (ann.)":      f"{a:.2%}",
        "Beta":              f"{b:.2f}",
        "Info Ratio":        f"{information_ratio(sr, br):.2f}",
        "VaR 95% (1d)":      f"{var_95(sr):.2%}",
        "Benchmark CAGR":    f"{cagr(result.benchmark_nav):.2%}",
    }
