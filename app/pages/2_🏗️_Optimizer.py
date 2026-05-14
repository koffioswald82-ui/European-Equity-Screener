import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from screener.data import fetch_ohlcv
from optimizer.markowitz import compute_cov_matrix, max_sharpe, min_variance, efficient_frontier

st.set_page_config(page_title="Optimizer", page_icon="🏗️", layout="wide")
st.title("🏗️ Portfolio Optimizer — Markowitz")

with st.sidebar:
    st.header("Universe & constraints")
    default_tickers = (
        "MC.PA,OR.PA,AIR.PA,ASML.AS,SAP.DE,SIE.DE,ALV.DE,HEIA.AS,"
        "TTE.PA,BNP.PA,SAN.PA,BAYN.DE,ENEL.MI,ENI.MI,IBE.MC,BBVA.MC"
    )
    ticker_input = st.text_area("Tickers (comma-separated)", value=default_tickers, height=120)
    period     = st.selectbox("Price history", ["1y", "2y", "3y", "5y"], index=2)
    rf         = st.number_input("Risk-free rate (%)", value=3.0, step=0.1) / 100
    w_max      = st.slider("Max weight per stock (%)", 5, 30, 10) / 100
    w_min      = st.slider("Min weight per stock (%)", 1, 5, 1) / 100
    opt_target = st.radio("Optimisation target", ["Max Sharpe", "Min Variance"])
    n_frontier = st.slider("Frontier points", 20, 100, 50)
    run        = st.button("Optimise", type="primary")

if not run:
    st.info("Enter tickers in the sidebar and click **Optimise**.")
    st.stop()

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

with st.spinner("Downloading prices..."):
    prices = fetch_ohlcv(tickers, period=period)

prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
valid  = list(prices.columns)

if len(valid) < 2:
    st.error("Not enough valid tickers.")
    st.stop()

mu, cov = compute_cov_matrix(prices[valid])

with st.spinner("Optimising..."):
    if opt_target == "Max Sharpe":
        result = max_sharpe(mu, cov, valid, rf=rf, w_min=w_min, w_max=w_max)
    else:
        result = min_variance(mu, cov, valid, w_min=w_min, w_max=w_max)
    frontier = efficient_frontier(mu, cov, valid, n_points=n_frontier, rf=rf,
                                  w_min=w_min, w_max=w_max)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected Return (ann.)", f"{result.expected_ret:.2%}")
c2.metric("Volatility (ann.)",      f"{result.volatility:.2%}")
c3.metric("Sharpe Ratio",           f"{result.sharpe:.2f}")
c4.metric("Stocks selected",        str(len([w for w in result.weights if w > 0.005])))

st.subheader("Efficient Frontier")
f_vols   = [p.volatility for p in frontier]
f_rets   = [p.expected_ret for p in frontier]
f_sharpe = [p.sharpe for p in frontier]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=f_vols, y=f_rets, mode="markers",
    marker=dict(color=f_sharpe, colorscale="Viridis", size=6,
                colorbar=dict(title="Sharpe")),
    name="Frontier",
    text=[f"Sharpe: {s:.2f}" for s in f_sharpe],
    hovertemplate="%{text}<br>Ret: %{y:.2%}<br>Vol: %{x:.2%}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=[result.volatility], y=[result.expected_ret],
    mode="markers", marker=dict(color="red", size=14, symbol="star"),
    name=opt_target,
))
fig.update_layout(xaxis_title="Annualised Volatility", yaxis_title="Annualised Return",
                  height=450, legend=dict(x=0.02, y=0.98))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Portfolio weights")
wdf = result.weights_df[result.weights_df["weight"] > 0.005].copy()
wdf["weight_pct"] = wdf["weight"] * 100
col1, col2 = st.columns(2)
with col1:
    fig2 = px.pie(wdf, names="ticker", values="weight_pct",
                  title="Allocation (%)", hole=0.4)
    fig2.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    fig3 = px.bar(wdf, x="ticker", y="weight_pct", title="Weight per ticker (%)",
                  color="weight_pct", color_continuous_scale="Blues")
    st.plotly_chart(fig3, use_container_width=True)
st.dataframe(wdf.rename(columns={"weight_pct": "Weight (%)"}), use_container_width=True)
csv = wdf.to_csv(index=False).encode()
st.download_button("⬇️ Export weights CSV", csv, "portfolio_weights.csv", "text/csv")
