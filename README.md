# 🇪🇺 European Equity Screener

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B)
![License: MIT](https://img.shields.io/badge/License-MIT-0F6E56)

End-to-end Python pipeline to screen, score and optimise portfolios of European large-cap equities. Combines multi-factor fundamental analysis, FinBERT sentiment NLP and Markowitz optimisation in an interactive Streamlit dashboard.

## Features

- Universe of ~600 European equities (France, Germany, Italy, Netherlands, Spain, Belgium…)
- Composite scoring: **Value · Quality · Momentum · Analyst Revisions**
- Sentiment NLP via **FinBERT** on latest 10 news per stock
- **Markowitz optimisation** — Max Sharpe, Min Variance, full efficient frontier
- **Walk-forward backtest** vs Euro Stoxx 50 — Sharpe, drawdown, alpha/beta, information ratio
- Local Parquet cache to avoid re-fetching fundamentals
- Export screener results and backtest data as CSV

## Installation

```bash
git clone https://github.com/ojklabs/equity-screener
cd equity-screener
pip install -r requirements.txt
streamlit run app/main.py
```

## Docker

```bash
docker build -t equity-screener .
docker run -p 8501:8501 equity-screener
# open http://localhost:8501
```

## Project structure

```
equity-screener/
├── screener/
│   ├── data.py          # yfinance + Parquet cache
│   ├── universe.py      # Euro Stoxx 600 universe (financedatabase)
│   ├── scoring.py       # Multi-factor composite score
│   └── sentiment.py     # FinBERT news sentiment
├── optimizer/
│   ├── markowitz.py     # Efficient frontier + max Sharpe
│   └── constraints.py   # Sector, tracking-error constraints
├── backtest/
│   ├── engine.py        # Walk-forward backtest engine
│   └── metrics.py       # Sharpe, drawdown, alpha, beta…
├── app/
│   ├── main.py          # Streamlit entry point
│   ├── ui_screener.py   # Screener page
│   ├── ui_optimizer.py  # Portfolio construction page
│   └── ui_backtest.py   # Backtest page
├── tests/
│   ├── test_data.py
│   └── test_scoring.py
├── .github/workflows/ci.yml
├── requirements.txt
└── Dockerfile
```

## Tech stack

| Library | Role |
|---|---|
| yfinance | OHLCV + fundamentals |
| financedatabase | European equity universe |
| FinBERT (HuggingFace) | News sentiment NLP |
| scipy.optimize | Markowitz optimisation |
| quantstats | Backtest tearsheet metrics |
| Streamlit | UI + deployment |

## License

MIT
