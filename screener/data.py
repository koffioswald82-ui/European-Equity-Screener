import os
import time
import requests
import yfinance as yf
import pandas as pd
from pathlib import Path

# Cache absolu, indépendant du répertoire de travail courant
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.environ.get("SCREENER_CACHE",
                                str(_PROJECT_ROOT / "data" / "cache")))

# Session HTTP avec User-Agent humain pour éviter le blocage Yahoo Finance
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
})


def fetch_ohlcv(tickers: list[str], period: str = "3y") -> pd.DataFrame:
    """Télécharge les prix de clôture ajustés pour une liste de tickers."""
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(
            tickers, period=period, auto_adjust=True,
            progress=False, session=_SESSION,
        )
    except Exception:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if len(tickers) > 1 else raw.rename(columns={"Close": tickers[0]})

    return prices.dropna(how="all")


def _safe_get(info: dict, *keys, default=None):
    """Essaie plusieurs clés dans l'ordre et retourne la première trouvée."""
    for k in keys:
        v = info.get(k)
        if v is not None:
            return v
    return default


def _fetch_info(ticker: str) -> dict:
    """
    Tente de récupérer le dict info de yfinance avec la session personnalisée.
    Essaie plusieurs méthodes si la première échoue.
    """
    info = {}
    try:
        t = yf.Ticker(ticker, session=_SESSION)
        info = t.info or {}
    except Exception:
        pass

    # Si info est vide ou trop partiel, essai sans session
    if len(info) < 5:
        try:
            t2 = yf.Ticker(ticker)
            info2 = t2.info or {}
            if len(info2) > len(info):
                info = info2
        except Exception:
            pass

    # Enrichir avec fast_info (market cap, beta) si possible
    try:
        fi = yf.Ticker(ticker, session=_SESSION).fast_info
        if not info.get("marketCap"):
            info["marketCap"] = getattr(fi, "market_cap", None)
        if not info.get("beta"):
            info["beta"] = getattr(fi, "three_month_average_volume", None)
    except Exception:
        pass

    return info


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Récupère les données fondamentales pour chaque ticker.
    - Utilise le cache Parquet local si disponible
    - Rate limiting : pause toutes les 5 requêtes pour éviter le blocage Yahoo
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for i, ticker in enumerate(tickers):
        cache_path = CACHE_DIR / f"{ticker.replace('/', '_')}.parquet"

        # Lecture cache
        if cache_path.exists():
            try:
                records.append(pd.read_parquet(cache_path))
                continue
            except Exception:
                cache_path.unlink(missing_ok=True)

        # Pause anti-rate-limit toutes les 5 requêtes
        if i > 0 and i % 5 == 0:
            time.sleep(1.5)

        info = _fetch_info(ticker)

        row = {
            "ticker":          ticker,
            "name":            _safe_get(info, "longName", "shortName", default=ticker),
            "sector":          _safe_get(info, "sector", default="Unknown"),
            "country":         _safe_get(info, "country", default="Unknown"),
            "market_cap":      _safe_get(info, "marketCap"),
            "pe_ratio":        _safe_get(info, "trailingPE", "forwardPE"),
            "ev_ebit":         _safe_get(info, "enterpriseToEbitda"),
            "roe":             _safe_get(info, "returnOnEquity"),
            "roce":            _safe_get(info, "returnOnAssets"),
            "debt_to_ebitda":  _safe_get(info, "debtToEquity"),
            "net_margin":      _safe_get(info, "profitMargins"),
            "revenue_growth":  _safe_get(info, "revenueGrowth"),
            "earnings_growth": _safe_get(info, "earningsGrowth"),
            "dividend_yield":  _safe_get(info, "dividendYield"),
            "beta":            _safe_get(info, "beta"),
            "_info_keys":      len(info),   # diagnostic : nb de champs reçus
        }

        df_row = pd.DataFrame([row])
        try:
            df_row.drop(columns=["_info_keys"]).to_parquet(cache_path)
        except Exception:
            pass
        records.append(df_row)

    if not records:
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)
    # Supprimer la colonne de diagnostic si elle existe
    if "_info_keys" in result.columns:
        result = result.drop(columns=["_info_keys"])
    return result


def clear_cache(ticker: str | None = None) -> None:
    """Supprime le cache Parquet. ticker=None pour tout effacer."""
    if ticker:
        p = CACHE_DIR / f"{ticker.replace('/', '_')}.parquet"
        p.unlink(missing_ok=True)
    else:
        for p in CACHE_DIR.glob("*.parquet"):
            p.unlink()
