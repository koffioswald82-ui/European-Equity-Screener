# ruff: noqa: E402
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from screener.data import fetch_ohlcv
from optimizer.markowitz import (
    compute_cov_matrix, ledoit_wolf_cov,
    black_litterman,
    max_sharpe, min_variance, risk_parity,
    efficient_frontier, PortfolioResult,
)

st.set_page_config(page_title="Optimizer", page_icon="🏗️", layout="wide")
st.title("🏗️ Portfolio Optimizer — Multi-Model")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Universe & constraints")
    default_tickers = (
        "MC.PA,OR.PA,AIR.PA,ASML.AS,SAP.DE,SIE.DE,ALV.DE,HEIA.AS,"
        "TTE.PA,BNP.PA,SAN.PA,BAYN.DE,ENEL.MI,ENI.MI,IBE.MC,BBVA.MC"
    )
    ticker_input = st.text_area("Tickers (comma-separated)", value=default_tickers, height=120)
    period    = st.selectbox("Price history", ["1y", "2y", "3y", "5y"], index=2)
    rf        = st.number_input("Risk-free rate (%)", value=3.0, step=0.1) / 100
    w_max     = st.slider("Max weight per stock (%)", 5, 40, 10) / 100
    w_min     = st.slider("Min weight per stock (%)", 1, 5, 1) / 100
    n_frontier = st.slider("Frontier points", 20, 100, 40)
    st.markdown("---")
    st.subheader("Covariance estimator")
    use_lw = st.checkbox(
        "Ledoit-Wolf shrinkage",
        value=True,
        help="Stabilises the covariance matrix — strongly recommended for < 100 observations per asset",
    )
    st.markdown("---")
    st.subheader("Black-Litterman views")
    st.caption("Optional absolute views on individual assets. Leave empty for equilibrium-only.")
    n_views = st.number_input("Number of views", 0, 10, 0, step=1)
    views: list[dict] = []
    for i in range(int(n_views)):
        cols = st.columns([3, 2])
        asset = cols[0].text_input(f"Asset {i+1}", key=f"bl_asset_{i}", placeholder="MC.PA")
        ret_v = cols[1].number_input(f"Return % {i+1}", key=f"bl_ret_{i}", value=10.0, step=0.5)
        if asset.strip():
            views.append({"assets": [asset.strip().upper()], "weights": [1.0], "return": ret_v / 100})
    st.markdown("---")
    run = st.button("Optimise all models", type="primary")

if not run:
    st.info(
        "Configure the universe in the sidebar and click **Optimise all models**.\n\n"
        "**Models available:**\n"
        "- **Max Sharpe** — classical Markowitz, maximises risk-adjusted return\n"
        "- **Min Variance** — lowest-risk portfolio on the efficient frontier\n"
        "- **Black-Litterman** — blends market equilibrium with your views\n"
        "- **Risk Parity** — each stock contributes equally to portfolio risk"
    )
    st.stop()

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

with st.spinner("Downloading prices..."):
    prices = fetch_ohlcv(tickers, period=period)

prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
valid  = list(prices.columns)

if len(valid) < 2:
    st.error("Not enough valid tickers (need at least 2 with sufficient history).")
    st.stop()

# ── Covariance ────────────────────────────────────────────────────────────────
with st.spinner("Estimating covariance..."):
    if use_lw:
        mu, cov = ledoit_wolf_cov(prices[valid])
        cov_label = "Ledoit-Wolf shrinkage"
    else:
        mu, cov = compute_cov_matrix(prices[valid])
        cov_label = "Historical sample"

# ── Run all models ─────────────────────────────────────────────────────────────
with st.spinner("Running optimisations..."):
    res_ms  = max_sharpe(mu, cov, valid, rf=rf, w_min=w_min, w_max=w_max)
    res_mv  = min_variance(mu, cov, valid, w_min=w_min, w_max=w_max)

    mu_bl, cov_bl = black_litterman(cov, valid, views=views if views else None)
    res_bl = max_sharpe(mu_bl, cov_bl, valid, rf=rf, w_min=w_min, w_max=w_max)
    res_bl.model = "Black-Litterman"

    rp_wmax = min(w_max, 0.30)
    res_rp  = risk_parity(cov, valid, rf=rf, mu=mu, w_min=w_min, w_max=rp_wmax)

    frontier = efficient_frontier(mu, cov, valid, n_points=n_frontier, rf=rf,
                                  w_min=w_min, w_max=w_max)

st.caption(f"Covariance: **{cov_label}** · {len(valid)} tickers · {period} history")

# ── Model comparison table ────────────────────────────────────────────────────
st.subheader("Model comparison")
comparison = pd.DataFrame([
    {
        "Model":            r.model,
        "Expected Return":  f"{r.expected_ret:.2%}",
        "Volatility":       f"{r.volatility:.2%}",
        "Sharpe":           f"{r.sharpe:.2f}",
        "Max Weight":       f"{r.weights.max():.1%}",
        "Effective stocks": int((r.weights > 0.005).sum()),
    }
    for r in [res_ms, res_mv, res_bl, res_rp]
])
st.dataframe(comparison.set_index("Model"), use_container_width=True)

# ── Efficient frontier (shared) ───────────────────────────────────────────────
if frontier:
    st.subheader("Efficient Frontier (Markowitz)")
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
    for r, color, sym in [
        (res_ms, "red",    "star"),
        (res_mv, "blue",   "diamond"),
        (res_bl, "orange", "triangle-up"),
        (res_rp, "green",  "square"),
    ]:
        fig.add_trace(go.Scatter(
            x=[r.volatility], y=[r.expected_ret], mode="markers",
            marker=dict(color=color, size=14, symbol=sym),
            name=r.model,
            hovertemplate=f"{r.model}<br>Ret: {r.expected_ret:.2%}<br>Vol: {r.volatility:.2%}<br>Sharpe: {r.sharpe:.2f}<extra></extra>",
        ))
    fig.update_layout(xaxis_title="Annualised Volatility", yaxis_title="Annualised Return",
                      height=450, legend=dict(x=0.02, y=0.98))
    st.plotly_chart(fig, use_container_width=True)

# ── Per-model tabs ────────────────────────────────────────────────────────────
st.subheader("Portfolio weights by model")
tab_ms, tab_mv, tab_bl, tab_rp = st.tabs([
    "📈 Max Sharpe", "🛡️ Min Variance", "🔭 Black-Litterman", "⚖️ Risk Parity"
])


def _render_weights(result: PortfolioResult):
    wdf = result.weights_df[result.weights_df["weight"] > 0.005].copy()
    wdf["weight_pct"] = wdf["weight"] * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Return", f"{result.expected_ret:.2%}")
    c2.metric("Volatility",      f"{result.volatility:.2%}")
    c3.metric("Sharpe Ratio",    f"{result.sharpe:.2f}")
    c4.metric("Stocks",          str(len(wdf)))

    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(wdf, names="ticker", values="weight_pct",
                         title="Allocation (%)", hole=0.4)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        fig_bar = px.bar(wdf, x="ticker", y="weight_pct",
                         title="Weight per ticker (%)",
                         color="weight_pct", color_continuous_scale="Blues")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(wdf.rename(columns={"weight_pct": "Weight (%)"}), use_container_width=True)
    csv = wdf.to_csv(index=False).encode()
    st.download_button(
        f"⬇️ Export {result.model} weights CSV",
        csv,
        f"weights_{result.model.lower().replace(' ', '_')}.csv",
        "text/csv",
    )


with tab_ms:
    st.markdown(
        "**Markowitz Max Sharpe** — maximises risk-adjusted return. "
        "Concentrates in assets with best return/risk ratio."
    )
    _render_weights(res_ms)

with tab_mv:
    st.markdown(
        "**Markowitz Min Variance** — lowest-risk point on the efficient frontier. "
        "Useful for capital preservation mandates."
    )
    _render_weights(res_mv)

with tab_bl:
    if views:
        st.markdown(
            f"**Black-Litterman** — equilibrium prior (δ=2.5) blended with "
            f"**{len(views)} analyst view(s)**. Posterior μ replaces historical mean."
        )
    else:
        st.markdown(
            "**Black-Litterman (no views)** — pure market equilibrium (reverse-optimised CAPM). "
            "Add views in the sidebar to override equilibrium returns for specific assets."
        )
    _render_weights(res_bl)

with tab_rp:
    st.markdown(
        "**Risk Parity** — each asset contributes equally to total portfolio variance. "
        "Does not rely on expected-return estimates — robust to forecasting errors."
    )
    # Show risk contributions
    w_rp = res_rp.weights
    sigma = float(np.sqrt(max(float(w_rp @ cov @ w_rp), 1e-18)))
    rc = (w_rp * (cov @ w_rp)) / sigma
    rc_df = pd.DataFrame({"ticker": valid, "weight": w_rp, "risk_contrib": rc})
    rc_df["rc_pct"] = rc_df["risk_contrib"] / rc_df["risk_contrib"].sum() * 100
    rc_df = rc_df[rc_df["weight"] > 0.005].sort_values("rc_pct", ascending=False)

    _render_weights(res_rp)

    st.markdown("**Risk contribution per asset**")
    fig_rc = px.bar(rc_df, x="ticker", y="rc_pct",
                    title="Risk Contribution (% of total portfolio risk)",
                    color="rc_pct", color_continuous_scale="Reds")
    fig_rc.add_hline(y=100 / len(rc_df), line_dash="dash",
                     annotation_text="Equal target", line_color="grey")
    st.plotly_chart(fig_rc, use_container_width=True)

st.markdown("---")
st.caption("Réalisé par **Oswald Jaures KOFFI**")
