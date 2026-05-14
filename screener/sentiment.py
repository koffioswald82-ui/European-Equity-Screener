import pandas as pd

try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False

import yfinance as yf

_pipe = None  # lazy-loaded once

FINBERT_MODEL = "ProsusAI/finbert"


def _get_pipe():
    global _pipe
    if _pipe is None:
        if not _TRANSFORMERS_OK:
            raise ImportError(
                "transformers is required for sentiment analysis. "
                "pip install transformers torch"
            )
        _pipe = hf_pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )
    return _pipe


def get_sentiment_score(ticker: str) -> dict:
    """
    Fetch latest news for *ticker* via yfinance and score with FinBERT.
    Returns a dict with keys: ticker, sentiment_score (float in [-1, +1]),
    label (bullish/neutral/bearish), n_articles.
    """
    neutral = {"ticker": ticker, "sentiment_score": 0.0, "label": "neutral", "n_articles": 0}
    try:
        news = yf.Ticker(ticker).news[:10]
    except Exception:
        return neutral

    texts = [
        n.get("title", "") + ". " + n.get("summary", "")
        for n in news
        if n.get("title")
    ]
    if not texts:
        return neutral

    try:
        pipe = _get_pipe()
        results = pipe(texts)
    except Exception:
        return neutral

    net_scores = []
    for r in results:
        scores = {x["label"]: x["score"] for x in r}
        net = scores.get("positive", 0.0) - scores.get("negative", 0.0)
        net_scores.append(net)

    avg = sum(net_scores) / len(net_scores)
    label = "bullish" if avg > 0.1 else "bearish" if avg < -0.1 else "neutral"
    return {
        "ticker": ticker,
        "sentiment_score": round(avg, 4),
        "label": label,
        "n_articles": len(texts),
    }


def batch_sentiment(tickers: list[str], progress: bool = False) -> pd.DataFrame:
    """Run sentiment for a list of tickers. Returns DataFrame with one row per ticker."""
    rows = []
    for i, t in enumerate(tickers):
        if progress:
            print(f"  Sentiment {i+1}/{len(tickers)}: {t}", end="\r")
        rows.append(get_sentiment_score(t))
    if progress:
        print()
    return pd.DataFrame(rows)
