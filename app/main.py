import streamlit as st

st.set_page_config(
    page_title="European Equity Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("European Equity Screener")
st.sidebar.markdown(
    "Multi-factor screener, Markowitz optimiser and walk-forward backtest "
    "for European large-cap equities."
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pages**\n"
    "- Screener\n"
    "- Optimizer\n"
    "- Backtest"
)
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Oswald Jaures KOFFI**")

st.title("European Equity Screener")
st.markdown("**Developed by Oswald Jaures KOFFI**")
st.markdown("---")
st.markdown(
    """
    Use the sidebar to navigate between the three tools:

    | Page | Description |
    |---|---|
    | **Screener** | Filter and rank ~600 European large-caps on Value, Quality, Momentum and Sentiment |
    | **Optimizer** | Build an optimised portfolio — Markowitz, Black-Litterman, Risk Parity |
    | **Backtest** | Walk-forward simulation vs Euro Stoxx 50 with full tearsheet |

    Data is sourced from [yfinance](https://github.com/ranaroussi/yfinance)
    and [financedatabase](https://github.com/JerBouma/FinanceDatabase).
    Sentiment powered by [FinBERT](https://huggingface.co/ProsusAI/finbert).
    """
)

st.markdown("---")
st.caption("Developed by **Oswald Jaures KOFFI**")
