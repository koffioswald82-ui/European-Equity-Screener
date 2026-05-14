import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from screener.data import fetch_ohlcv
from optimizer.markowitz import compute_cov_matrix, max_sharpe
from backtest.engine import run_backtest
from backtest.metrics import compute_all

st.set_page_config(page_title="Backtest", page_icon="📈", layout="wide")
st.title("📈 Walk-forward Backtest vs Euro Stoxx 50")

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Backtest settings")

    default_tickers = (
        "MC.PA,OR.PA,AIR.PA,ASML.AS,SAP.DE,SIE.DE,ALV.DE,HEIA.AS,"
        "TTE.PA,BNP.PA,SAN.PA,BAYN.DE,ENEL.MI,ENI.MI,IBE.MC,BBVA.MC"
    )
    ticker_input = st.text_area("Tickers (comma-separated)", value=default_tickers, height=120)

    benchmark = st.text_input(
        "Benchmark ticker (yfinance)", value="EXW1.DE",
        help="EXW1.DE = iShares Euro Stoxx 50 ETF"
    )
    period      = st.selectbox("History", ["2y", "3y", "5y"], index=1)
    rebal_freq  = st.selectbox("Rebalance frequency", ["ME", "QE", "YE"], index=0,
                               format_func=lambda x: {"ME": "Monthly", "QE": "Quarterly", "YE": "Yearly"}[x])
    rf          = st.number_input("Risk-free rate (%)", value=3.0) / 100
    w_max       = st.slider("Max weight per stock (%)", 5, 30, 10) / 100
    lookback    = st.slider("Lookback window (days)", 63, 504, 252)
    run         = st.button("Run backtest", type="primary")

if not run:
    st.info("Configure settings in the sidebar and click **Run backtest**.")
    st.stop()

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
all_tickers = list(set(tickers + [benchmark]))

with st.spinner("Downloading prices..."):
    prices = fetch_ohlcv(all_tickers, period=period)

prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.6))
asset_tickers = [t for t in tickers if t in prices.columns]

if len(asset_tickers) < 2:
    st.error("Not enough valid tickers for backtest.")
    st.stop()


def weights_fn(window: pd.DataFrame):
    valid = window.dropna(axis=1, thresh=int(len(window) * 0.8))
    if len(valid.columns) < 2:
        return {}
    mu, cov = compute_cov_matrix(valid)
    result  = max_sharpe(mu, cov, list(valid.columns), rf=rf, w_min=0.01, w_max=w_max)
    return dict(zip(result.tickers, result.weights))


with st.spinner("Running walk-forward backtest..."):
    bt = run_backtest(
        prices=prices,
        weights_fn=weights_fn,
        benchmark_ticker=benchmark,
        rebalance_freq=rebal_freq,
        lookback_days=lookback,
    )

# ── metrics table ──────────────────────────────────────────────────────────────
st.subheader("Performance metrics")
metrics = compute_all(bt, rf=rf)
cols = st.columns(5)
items = list(metrics.items())
for i, (k, v) in enumerate(items):
    cols[i % 5].metric(k, v)

# ── NAV chart ──────────────────────────────────────────────────────────────────
st.subheader("Cumulative performance (base 100)")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=bt.portfolio_nav.index, y=bt.portfolio_nav,
    name="Strategy", line=dict(color="#1B2A4A", width=2),
))
fig.add_trace(go.Scatter(
    x=bt.benchmark_nav.index, y=bt.benchmark_nav,
    name=f"Benchmark ({benchmark})", line=dict(color="#B8962E", width=1.5, dash="dash"),
))
for rd in bt.rebalance_dates:
    fig.add_vline(x=rd, line_width=0.5, line_dash="dot", line_color="grey")
fig.update_layout(
    yaxis_title="NAV (base 100)",
    legend=dict(x=0.02, y=0.98),
    height=450,
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── drawdown chart ─────────────────────────────────────────────────────────────
st.subheader("Drawdown")
dd_strat = (bt.portfolio_nav / bt.portfolio_nav.cummax() - 1) * 100
dd_bench = (bt.benchmark_nav / bt.benchmark_nav.cummax() - 1) * 100

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=dd_strat.index, y=dd_strat,
    fill="tozeroy", name="Strategy drawdown",
    line=dict(color="#1B2A4A"), fillcolor="rgba(27,42,74,0.2)",
))
fig2.add_trace(go.Scatter(
    x=dd_bench.index, y=dd_bench,
    fill="tozeroy", name="Benchmark drawdown",
    line=dict(color="#B8962E", dash="dash"), fillcolor="rgba(184,150,46,0.15)",
))
fig2.update_layout(yaxis_title="Drawdown (%)", height=280, hovermode="x unified")
st.plotly_chart(fig2, use_container_width=True)

# ── rolling Sharpe ──────────────────────────────────────────────────────────────
st.subheader("Rolling 252-day Sharpe ratio")
roll_mean = bt.portfolio_returns.rolling(252).mean()
roll_std  = bt.portfolio_returns.rolling(252).std().clip(1e-9)
roll_sharpe = (roll_mean - rf / 252) / roll_std * np.sqrt(252)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=roll_sharpe.index, y=roll_sharpe,
    name="Rolling Sharpe", line=dict(color="#185FA5"),
))
fig3.add_hline(y=0, line_dash="dash", line_color="grey")
fig3.update_layout(yaxis_title="Sharpe", height=260)
st.plotly_chart(fig3, use_container_width=True)

# ── export ─────────────────────────────────────────────────────────────────────
export_df = pd.DataFrame({
    "strategy_return":   bt.portfolio_returns,
    "benchmark_return":  bt.benchmark_returns,
    "strategy_nav":      bt.portfolio_nav,
    "benchmark_nav":     bt.benchmark_nav,
})
csv = export_df.to_csv().encode()
st.download_button("⬇️ Export backtest data CSV", csv, "backtest_results.csv", "text/csv")
