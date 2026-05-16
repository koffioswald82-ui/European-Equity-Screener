import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestResult:
    portfolio_returns: pd.Series      # daily returns of the strategy
    benchmark_returns: pd.Series      # daily returns of benchmark
    portfolio_nav: pd.Series          # cumulative NAV starting at 100
    benchmark_nav: pd.Series
    rebalance_dates: pd.DatetimeIndex


def run_backtest(
    prices: pd.DataFrame,
    weights_fn,                        # callable(prices_window) -> dict {ticker: weight}
    benchmark_ticker: str = "EXW1.DE", # iShares Euro Stoxx 50 ETF
    rebalance_freq: str = "ME",        # month-end
    lookback_days: int = 252,
    start_date: str | None = None,
    end_date: str | None = None,
) -> BacktestResult:
    """
    Walk-forward backtest.

    At each rebalance date the strategy calls *weights_fn* with the trailing
    *lookback_days* of prices, then holds those weights until the next date.

    prices     : Close prices DataFrame (dates × tickers, benchmark included).
    weights_fn : function(prices_window: DataFrame) -> dict[str, float].
    """
    # Trim date range
    if start_date:
        prices = prices[prices.index >= start_date]
    if end_date:
        prices = prices[prices.index <= end_date]

    # Separate benchmark
    bench_col = benchmark_ticker if benchmark_ticker in prices.columns else None
    asset_prices = prices.drop(columns=[bench_col], errors="ignore")

    rebalance_dates = pd.date_range(
        start=asset_prices.index[lookback_days],
        end=asset_prices.index[-1],
        freq=rebalance_freq,
    )
    if rebalance_dates.empty:
        rebalance_dates = pd.DatetimeIndex([asset_prices.index[lookback_days]])

    daily_ret = asset_prices.pct_change().fillna(0)

    strategy_daily: list[tuple[pd.Timestamp, float]] = []
    current_weights: dict[str, float] = {}

    rebal_iter = iter(rebalance_dates)
    next_rebal = next(rebal_iter, None)

    for date in asset_prices.index[lookback_days:]:
        # Rebalance?
        if next_rebal is not None and date >= next_rebal:
            window = asset_prices.loc[:date].iloc[-lookback_days:]
            try:
                current_weights = weights_fn(window)
            except Exception:
                pass
            next_rebal = next(rebal_iter, None)

        if current_weights:
            day_ret = sum(
                current_weights.get(t, 0.0) * daily_ret.loc[date, t]
                for t in current_weights
                if t in daily_ret.columns
            )
        else:
            day_ret = 0.0

        strategy_daily.append((date, day_ret))

    strat_series = pd.Series(
        {d: r for d, r in strategy_daily}, name="strategy"
    )

    # Benchmark daily returns
    if bench_col:
        bench_series = prices[bench_col].pct_change().reindex(strat_series.index).fillna(0)
    else:
        bench_series = pd.Series(0.0, index=strat_series.index, name="benchmark")

    bench_series.name = "benchmark"
    strat_nav = (1 + strat_series).cumprod() * 100
    bench_nav = (1 + bench_series).cumprod() * 100

    return BacktestResult(
        portfolio_returns=strat_series,
        benchmark_returns=bench_series,
        portfolio_nav=strat_nav,
        benchmark_nav=bench_nav,
        rebalance_dates=rebalance_dates,
    )
