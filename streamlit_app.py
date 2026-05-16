import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(
    page_title="European Equity Screener",
    page_icon="🇪🇺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🇪🇺 European Equity Screener")
st.sidebar.markdown(
    "Multi-factor screener, Markowitz optimiser and walk-forward backtest "
    "for European large-cap equities."
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Pages**\n"
    "- 📊 Screener\n"
    "- 🏗️ Optimizer\n"
    "- 📈 Backtest"
)
st.sidebar.markdown("---")
st.sidebar.markdown("*Réalisé par **Oswald Jaures KOFFI***")

st.title("🇪🇺 European Equity Screener")
st.markdown(
    """
    Bienvenue ! Utilisez la sidebar pour naviguer entre les trois outils :

    | Page | Description |
    |---|---|
    | **📊 Screener** | Filtrer et classer ~600 actions européennes sur Value · Quality · Momentum · Sentiment |
    | **🏗️ Optimizer** | Construire un portefeuille optimal sur la frontière efficiente de Markowitz |
    | **📈 Backtest** | Simulation walk-forward vs Euro Stoxx 50 avec tearsheet complet |

    > Données : [yfinance](https://github.com/ranaroussi/yfinance) · [stooq](https://stooq.com) · [financedatabase](https://github.com/JerBouma/FinanceDatabase)
    > Sentiment NLP : [FinBERT](https://huggingface.co/ProsusAI/finbert)
    """
)
st.markdown("---")
st.caption("Réalisé par **Oswald Jaures KOFFI**")
