# ruff: noqa: E402
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
import plotly.express as px
from datetime import datetime, timezone

from screener.universe import get_universe, EU_COUNTRIES
from screener.data import fetch_fundamentals, fetch_ohlcv, CACHE_DIR
from screener.scoring import composite_score, WEIGHTS

st.set_page_config(page_title="Screener", page_icon="📊", layout="wide")
st.title("📊 Multi-factor Equity Screener")

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    countries = st.multiselect(
        "Countries", EU_COUNTRIES,
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
    price_period = st.selectbox("Price history", ["1y", "2y", "3y"], index=1)
    st.markdown("---")
    st.subheader("Factor weights")
    w_value    = st.slider("Value",    0, 100, int(WEIGHTS["value"]    * 100), 5) / 100
    w_quality  = st.slider("Quality",  0, 100, int(WEIGHTS["quality"]  * 100), 5) / 100
    w_momentum = st.slider("Momentum", 0, 100, int(WEIGHTS["momentum"] * 100), 5) / 100
    w_revision = st.slider("Revision", 0, 100, int(WEIGHTS["revision"] * 100), 5) / 100
    include_sentiment = st.checkbox("Include FinBERT sentiment (slow)", value=False)
    top_n = st.slider("Show top N", 10, 100, 30)
    run = st.button("▶ Run screener", type="primary")
    st.markdown("---")
    if st.button("🗑️ Clear cache"):
        from screener.data import clear_cache
        clear_cache()
        st.success("Cache cleared — next run re-downloads all data.")

# ── info banner ────────────────────────────────────────────────────────────────
run_date = datetime.now(timezone.utc).strftime("%d %B %Y at %H:%M UTC")

if not run:
    st.info("Configure filters in the sidebar and click **▶ Run screener**.")
    st.stop()

# ── run ────────────────────────────────────────────────────────────────────────
with st.spinner("Loading universe..."):
    tickers = get_universe(min_market_cap=min_cap, countries=countries or None,
                           sectors=sectors or None)

with st.spinner(f"Fetching fundamentals for {len(tickers)} tickers (cached after first run)..."):
    fundamentals = fetch_fundamentals(tickers)

if fundamentals.empty:
    st.error("No fundamental data available. Check your internet connection.")
    st.stop()

# Indiquer si les données viennent du cache ou sont fraîches
cached_files = list(CACHE_DIR.glob("*.parquet")) if CACHE_DIR.exists() else []
cache_dates  = [datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                for f in cached_files if f.stat().st_size > 0]
if cache_dates:
    oldest = min(cache_dates).strftime("%d %b %Y %H:%M UTC")
    newest = max(cache_dates).strftime("%d %b %Y %H:%M UTC")
    cache_info = f"📦 Fundamentals cache — oldest: **{oldest}** · newest: **{newest}**"
else:
    cache_info = "📡 Fundamentals downloaded fresh (no cache yet)"

with st.spinner(f"Downloading {price_period} price history..."):
    valid_tickers = fundamentals["ticker"].dropna().tolist()
    prices = fetch_ohlcv(valid_tickers, period=price_period)

# Récupérer la plage de dates réelle des prix
if not prices.empty:
    price_start = prices.index[0].strftime("%d %b %Y")
    price_end   = prices.index[-1].strftime("%d %b %Y")
    price_info  = f"📈 Prices from **{price_start}** to **{price_end}** ({price_period} window)"
else:
    price_info = "⚠️ Price data unavailable"

sentiment_df = None
sentiment_info = ""
if include_sentiment:
    from screener.sentiment import batch_sentiment
    with st.spinner("Running sentiment analysis (HF API → VADER fallback)..."):
        sentiment_df = batch_sentiment(valid_tickers, progress=False)
    if sentiment_df is not None and not sentiment_df.empty and "method" in sentiment_df.columns:
        methods = sentiment_df["method"].value_counts().to_dict()
        method_str = " · ".join(f"{k}: {v}" for k, v in methods.items())
        sentiment_info = f"🧠 Sentiment methods used — {method_str}"

import screener.scoring as sc_mod
sc_mod.WEIGHTS.update({"value": w_value, "quality": w_quality,
                        "momentum": w_momentum, "revision": w_revision})

with st.spinner("Computing composite scores..."):
    scored = composite_score(fundamentals, prices, sentiment_df)

top = scored.head(top_n).copy()

# ── info temporelle ────────────────────────────────────────────────────────────
st.markdown(f"""
> 🕐 **Screener run:** {run_date}
> {price_info}
> {cache_info}
> {"  \n> " + sentiment_info if sentiment_info else ""}
""")

# Compter les actions avec données valides vs vides
valid_rows   = fundamentals["pe_ratio"].notna().sum()
missing_rows = len(fundamentals) - valid_rows
if missing_rows > 0:
    st.warning(
        f"⚠️ {missing_rows}/{len(fundamentals)} tickers returned no fundamental data "
        f"(Yahoo Finance rate limit or unknown ticker). "
        f"Clear the cache and retry, or reduce the universe."
    )

# ── heatmap ────────────────────────────────────────────────────────────────────
st.subheader("Top picks — composite score heatmap")
factor_cols = [c for c in ["value_score", "quality_score", "momentum_score",
                            "revision_score", "sentiment_score"] if c in top.columns]

if factor_cols:
    heatmap_data = top.set_index("ticker")[factor_cols + ["composite_score"]]
    fig = px.imshow(
        heatmap_data.T, color_continuous_scale="RdYlGn", aspect="auto",
        title=f"Factor scores — top {top_n} stocks (run: {run_date})",
        labels={"x": "Ticker", "y": "Factor", "color": "Score"},
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ── tableau ────────────────────────────────────────────────────────────────────
st.subheader(f"Ranked stocks · data as of {run_date}")
display_cols = ["ticker", "name", "sector", "composite_score"] + factor_cols + [
    "pe_ratio", "roe", "net_margin", "market_cap"]
display_cols = [c for c in display_cols if c in top.columns]
st.dataframe(
    top[display_cols].style.background_gradient(subset=["composite_score"], cmap="RdYlGn"),
    use_container_width=True, height=520,
)

# ── export ─────────────────────────────────────────────────────────────────────
export = scored.copy()
export.insert(0, "screener_date", run_date)
csv = export.to_csv(index=False).encode()
st.download_button(
    f"⬇️ Export CSV (run: {datetime.now(timezone.utc).strftime('%Y-%m-%d')})",
    csv, f"screener_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv", "text/csv"
)

st.markdown("---")
st.caption("Réalisé par **Oswald Jaures KOFFI**")
