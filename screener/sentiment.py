"""
Sentiment analysis — cascade :
  1. HuggingFace Inference API (FinBERT HTTP, sans torch)
  2. FinBERT local (transformers+torch, si disponible)
  3. VADER (vaderSentiment, pur Python, toujours disponible)
"""
import os
import time
import pandas as pd
import yfinance as yf

HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_TOKEN   = os.environ.get("HF_TOKEN", "")


# ── extraction des textes (robuste aux changements de format yfinance) ─────────

def _get_news_texts(ticker: str) -> list[str]:
    """Retourne la liste des textes (titre + résumé) des dernières news."""
    try:
        # Utilise la session partagée si disponible
        try:
            from screener.data import _SESSION
            t = yf.Ticker(ticker, session=_SESSION)
        except Exception:
            t = yf.Ticker(ticker)

        news = t.news or []
        texts = []
        for n in news[:10]:
            # yfinance >= 0.2.x : titre dans "title" ou "headline"
            # résumé dans "summary", "description", ou absent
            title   = (n.get("title") or n.get("headline") or "").strip()
            summary = (n.get("summary") or n.get("description") or
                       n.get("content") or "").strip()
            text = f"{title}. {summary}" if summary else title
            if text:
                texts.append(text[:512])   # tronque à 512 chars
        return texts
    except Exception:
        return []


# ── niveau 1 : HF Inference API ───────────────────────────────────────────────

def _hf_api_score(texts: list[str]) -> list[float] | None:
    try:
        import requests
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        resp = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": [t[:400] for t in texts]},
            timeout=25,
        )
        # Modèle en cours de chargement → on attend et on réessaie
        if resp.status_code == 503:
            time.sleep(20)
            resp = requests.post(HF_API_URL, headers=headers,
                                 json={"inputs": [t[:400] for t in texts]},
                                 timeout=25)
        if resp.status_code != 200:
            return None

        data = resp.json()
        # Cas : erreur renvoyée en JSON
        if isinstance(data, dict) and "error" in data:
            return None
        if not isinstance(data, list) or not data:
            return None

        scores = []
        for item in data:
            # item peut être une liste ou un dict selon le nb de textes envoyés
            if isinstance(item, list):
                d = {x["label"]: x["score"] for x in item}
            elif isinstance(item, dict) and "label" in item:
                d = {item["label"]: item["score"]}
            else:
                scores.append(0.0)
                continue
            scores.append(d.get("positive", 0.0) - d.get("negative", 0.0))
        return scores if scores else None
    except Exception:
        return None


# ── niveau 2 : FinBERT local ──────────────────────────────────────────────────

_local_pipe = None

def _local_finbert_score(texts: list[str]) -> list[float] | None:
    global _local_pipe
    try:
        if _local_pipe is None:
            from transformers import pipeline as hf_pipeline
            _local_pipe = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                return_all_scores=True,
                truncation=True,
                max_length=512,
            )
        results = _local_pipe(texts)
        return [
            {x["label"]: x["score"] for x in r}.get("positive", 0.0)
            - {x["label"]: x["score"] for x in r}.get("negative", 0.0)
            for r in results
        ]
    except Exception:
        return None


# ── niveau 3 : VADER ──────────────────────────────────────────────────────────

_vader = None

def _vader_score(texts: list[str]) -> list[float]:
    global _vader
    try:
        if _vader is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        return [_vader.polarity_scores(t)["compound"] for t in texts]
    except ImportError:
        return [0.0] * len(texts)


# ── fonction publique ─────────────────────────────────────────────────────────

def get_sentiment_score(ticker: str) -> dict:
    """Retourne score net bullish [-1, +1] pour un titre."""
    neutral = {"ticker": ticker, "sentiment_score": 0.0,
               "label": "neutral", "n_articles": 0, "method": "no_news"}

    texts = _get_news_texts(ticker)
    if not texts:
        return neutral

    net_scores = _hf_api_score(texts)
    method = "finbert-api"
    if net_scores is None:
        net_scores = _local_finbert_score(texts)
        method = "finbert-local"
    if net_scores is None:
        net_scores = _vader_score(texts)
        method = "vader"

    avg = sum(net_scores) / len(net_scores) if net_scores else 0.0
    label = "bullish" if avg > 0.1 else "bearish" if avg < -0.1 else "neutral"
    return {
        "ticker":          ticker,
        "sentiment_score": round(avg, 4),
        "label":           label,
        "n_articles":      len(texts),
        "method":          method,
    }


def batch_sentiment(tickers: list[str], progress: bool = False) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers):
        if progress:
            print(f"  Sentiment {i+1}/{len(tickers)}: {t}", end="\r")
        rows.append(get_sentiment_score(t))
    if progress:
        print()
    return pd.DataFrame(rows)
