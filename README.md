---
title: European Equity Screener
emoji: 📊
colorFrom: indigo
colorTo: yellow
sdk: docker
pinned: true
license: mit
short_description: Multi-factor screener & portfolio optimizer
---

# European Equity Screener

**Developed by Oswald Jaures KOFFI**

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B)
![License: MIT](https://img.shields.io/badge/License-MIT-0F6E56)

End-to-end Python platform for screening, scoring and optimising portfolios of European large-cap equities. Combines multi-factor fundamental analysis, NLP sentiment (FinBERT) and professional-grade portfolio construction models in an interactive Streamlit interface.

## Features

- Universe of ~600 European equities (France, Germany, Italy, Netherlands, Spain...)
- Composite scoring: **Value · Quality · Momentum · Analyst Revisions**
- NLP sentiment via **FinBERT** (HuggingFace) with VADER fallback
- **Multi-model optimizer** — Markowitz Max Sharpe, Min Variance, Black-Litterman, Risk Parity
- **Ledoit-Wolf shrinkage** covariance estimator for robust portfolio construction
- **Walk-forward backtest** vs Euro Stoxx 50 — Sharpe, drawdown, alpha/beta, IR
- Parquet cache for fast subsequent runs
- CSV export of results

## Pages

| Page | Description |
|---|---|
| Screener | Filter and rank stocks on 4 factors + sentiment |
| Optimizer | Build portfolios with Markowitz, Black-Litterman and Risk Parity models |
| Backtest | Walk-forward simulation vs Euro Stoxx 50 with full tearsheet |

## Technical stack

| Library | Role |
|---|---|
| yfinance + financedatabase | OHLCV data + European universe |
| scikit-learn | Ledoit-Wolf covariance shrinkage |
| scipy.optimize | SLSQP optimisation (Sharpe, ERC, BL) |
| FinBERT (HuggingFace) | NLP sentiment analysis |
| quantstats | Backtest metrics |
| Streamlit | Interface + deployment |

## Local installation

```bash
git clone https://github.com/koffioswald82-ui/European-Equity-Screener
cd European-Equity-Screener
pip install -r requirements.txt
streamlit run app/main.py
```

## Docker

```bash
docker build -t equity-screener .
docker run -p 7860:7860 equity-screener
```
