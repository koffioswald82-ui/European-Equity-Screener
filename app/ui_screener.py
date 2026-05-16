import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import plotly.express as px

from screener.universe import get_universe, EU_COUNTRIES
from screener.data import fetch_fundamentals, fetch_ohlcv
from screener.scoring import composite_score, WEIGHTS

st.set_page_config(page_title="Screener", page_icon="📊", layout="wide")
st.title("📊 Multi-factor Equity Screener")

# ── sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    countries = st.multiselect(
        "Countries",
        EU_COUNTRIES,
        default=["France", "Germany", "Netherlands", "Italy", "Spain"],
    )

    min_cap = st.number_input("Min market cap (€M)", value=500, step=100) * 1e6

    sectors = st.multiselect(
        "Sectors (GICS)",
        ["Technology", "Financials", "Industrials", "Consumer Discretionary",
         "Consumer Staples", "Healthcare", "Energy", "Materials",
         "Communication Services", "Real Estate", "Utilities"],
        default=[],
    )

    st.markdown("---")
    st.subheader("Factor weights")
    w_value    = st.slider("Value",    0, 100, int(WEIGHTS["value"]    * 100), 5) / 100
    w_quality  = st.slider("Quality",  0, 100, int(WEIGHTS["quality"]  * 100), 5) / 100
    w_momentum = st.slider("Momentum", 0, 100, int(WEIGHTS["momentum"] * 100), 5) / 100
    w_revision = st.slider("Revision", 0, 100, int(WEIGHTS["revision"] * 100), 5) / 100

    include_sentiment = st.checkbox("Include FinBERT sentiment (slow)", value=False)
    top_n = st.slider("Show top N", 10, 100, 30)
    run   = st.button("Run screener", type="primary")

# ── main panel ─────────────────────────────────────────────────────────────────
if not run:
    st.info("Configure filters in the sidebar and click **Run screener**.")
    st.stop()

with st.spinner("Loading universe..."):
    tickers = get_universe(min_market_cap=min_cap, countries=countries or None,
                           sectors=sectors or None)

st.caption(f"Universe: **{len(tickers)}** tickers")

with st.spinner("Fetching fundamentals (cached after first run)..."):
    fundamentals = fetch_fundamentals(tickers)

if fundamentals.empty:
    st.error("No fundamental data available. Check your internet connection.")
    st.stop()

with st.spinner("Downloading price history..."):
    valid_tickers = fundamentals["ticker"].dropna().tolist()
    prices = fetch_ohlcv(valid_tickers, period="2y")

sentiment_df = None
if include_sentiment:
    from screener.sentiment import batch_sentiment
    with st.spinner("Running FinBERT sentiment analysis..."):
        sentiment_df = batch_sentiment(valid_tickers, progress=False)

custom_weights = {
    "value": w_value, "quality": w_quality,
    "momentum": w_momentum, "revision": w_revision,
}
import screener.scoring as sc_mod
sc_mod.WEIGHTS.update(custom_weights)

with st.spinner("Computing scores..."):
    scored = composite_score(fundamentals, prices, sentiment_df)

top = scored.head(top_n).copy()

# ── score heatmap ──────────────────────────────────────────────────────────────
st.subheader("Top picks — composite score heatmap")
factor_cols = [c for c in ["value_score", "quality_score", "momentum_score",
                            "revision_score", "sentiment_score"] if c in top.columns]

if factor_cols:
    heatmap_data = top.set_index("ticker")[factor_cols + ["composite_score"]]
    fig = px.imshow(
        heatmap_data.T,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Factor scores heatmap (z-score based)",
        labels={"x": "Ticker", "y": "Factor", "color": "Score"},
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ── data table ─────────────────────────────────────────────────────────────────
st.subheader("Ranked stocks")
display_cols = ["ticker", "name", "sector", "composite_score"] + factor_cols + [
    "pe_ratio", "roe", "net_margin", "market_cap"
]
display_cols = [c for c in display_cols if c in top.columns]

st.dataframe(
    top[display_cols].style.background_gradient(subset=["composite_score"], cmap="RdYlGn"),
    use_container_width=True,
    height=500,
)

# ── export ─────────────────────────────────────────────────────────────────────
csv = scored.to_csv(index=False).encode()
st.download_button("⬇️ Export full screener CSV", csv, "screener_results.csv", "text/csv")
