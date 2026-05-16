---
title: European Equity Screener
emoji: 📊
colorFrom: indigo
colorTo: yellow
sdk: docker
pinned: true
license: mit
short_description: Multi-factor screener & Markowitz optimizer
---

# 🇪🇺 European Equity Screener

**Réalisé par Oswald Jaures KOFFI**

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B)
![License: MIT](https://img.shields.io/badge/License-MIT-0F6E56)

Pipeline Python end-to-end pour **screener, scorer et optimiser** des portefeuilles d'actions européennes large cap. Combine analyse fondamentale multi-facteurs, sentiment NLP (FinBERT) et optimisation Markowitz dans une interface Streamlit interactive.

## Fonctionnalités

- Univers de ~600 actions européennes (France, Allemagne, Italie, Pays-Bas, Espagne…)
- Scoring composite : **Value · Quality · Momentum · Révisions analystes**
- Sentiment NLP via **FinBERT** (HuggingFace) + fallback VADER
- **Optimisation Markowitz** — Max Sharpe, Min Variance, frontière efficiente
- **Backtest walk-forward** vs Euro Stoxx 50 — Sharpe, drawdown, alpha/beta, IR
- Cache Parquet local pour accélérer les relances
- Export CSV des résultats

## Pages

| Page | Description |
|---|---|
| 📊 Screener | Filtrer et classer les actions sur 4 facteurs + sentiment |
| 🏗️ Optimizer | Construire un portefeuille optimal sur la frontière efficiente |
| 📈 Backtest | Simulation walk-forward vs Euro Stoxx 50 avec tearsheet complet |

## Stack technique

| Librairie | Rôle |
|---|---|
| yfinance + stooq | Données OHLCV + fondamentaux |
| financedatabase | Univers investissable européen |
| FinBERT (HuggingFace) | Analyse de sentiment NLP |
| scipy.optimize | Optimisation Markowitz (SLSQP) |
| quantstats | Métriques backtest |
| Streamlit | Interface + déploiement |

## Installation locale

```bash
git clone https://github.com/koffioswald82-ui/European-Equity-Screener
cd European-Equity-Screener
pip install -r requirements.txt
streamlit run app/main.py
```

## Docker

```bash
docker build -t equity-screener .
docker run -p 8501:8501 equity-screener
```
