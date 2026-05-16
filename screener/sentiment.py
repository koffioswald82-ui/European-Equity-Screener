"""
Sentiment analysis pipeline — trois niveaux de fallback :
  1. HuggingFace Inference API (FinBERT via HTTP, pas besoin de torch)
  2. FinBERT local via transformers+torch (si installé)
  3. VADER  (vaderSentiment, pur Python, léger, fallback garanti)
"""
import os
import time
import pandas as pd
import yfinance as yf

# ── niveau 1 : HF Inference API ───────────────────────────────────────────────
HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_TOKEN   = os.environ.get("HF_TOKEN", "")   # optionnel — améliore la limite de taux


def _hf_api_score(texts: list[str]) -> list[float] | None:
    """
    Envoie les textes à l'API Inference HuggingFace.
    Retourne une liste de scores nets [pos - neg] ou None si l'API échoue.
    """
    try:
        import requests
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        # L'API accepte max ~500 tokens par texte ; on tronque
        truncated = [t[:400] for t in texts]
        resp = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": truncated},
            timeout=20,
        )
        if resp.status_code == 503:
            # Modèle en cours de chargement côté HF — on attend et on réessaie
            time.sleep(15)
            resp = requests.post(HF_API_URL, headers=headers,
                                 json={"inputs": truncated}, timeout=20)
        if resp.status_code != 200:
            return None

        data = resp.json()
        # Réponse : [[{"label":"positive","score":0.8},…], […], …]
        if not isinstance(data, list) or not data:
            return None

        scores = []
        for item in data:
            if isinstance(item, list):
                d = {x["label"]: x["score"] for x in item}
            elif isinstance(item, dict) and "label" in item:
                # réponse à un seul texte (liste aplatie)
                d = {item["label"]: item["score"]}
            else:
                scores.append(0.0)
                continue
            scores.append(d.get("positive", 0.0) - d.get("negative", 0.0))
        return scores if scores else None

    except Exception:
        return None


# ── niveau 2 : FinBERT local (torch) ─────────────────────────────────────────
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
        scores = []
        for r in results:
            d = {x["label"]: x["score"] for x in r}
            scores.append(d.get("positive", 0.0) - d.get("negative", 0.0))
        return scores
    except Exception:
        return None


# ── niveau 3 : VADER (fallback garanti) ──────────────────────────────────────
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
    """
    Retourne un score net bullish [-1, +1] pour un titre.
    Essaie HF API → FinBERT local → VADER dans cet ordre.
    """
    neutral = {"ticker": ticker, "sentiment_score": 0.0,
               "label": "neutral", "n_articles": 0, "method": "none"}
    try:
        news = yf.Ticker(ticker).news[:10]
    except Exception:
        return neutral

    texts = [
        n.get("title", "") + ". " + n.get("summary", "")
        for n in news if n.get("title")
    ]
    if not texts:
        return neutral

    # Cascade
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
    """Score sentiment pour une liste de tickers. Retourne un DataFrame."""
    rows = []
    for i, t in enumerate(tickers):
        if progress:
            print(f"  Sentiment {i+1}/{len(tickers)}: {t}", end="\r")
        rows.append(get_sentiment_score(t))
    if progress:
        print()
    return pd.DataFrame(rows)
