import yfinance as yf
import pandas as pd
from pathlib import Path

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlcv(tickers: list[str], period: str = "3y") -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers."""
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
    return prices.dropna(how="all")


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """Fetch P/E, EV/EBIT, ROE, debt and margins for each ticker. Results are cached as Parquet."""
    records = []
    for ticker in tickers:
        cache_path = CACHE_DIR / f"{ticker.replace('/', '_')}.parquet"
        if cache_path.exists():
            records.append(pd.read_parquet(cache_path))
            continue
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            info = {}
        row = {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "country": info.get("country", "Unknown"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "ev_ebit": info.get("enterpriseToEbitda"),
            "roe": info.get("returnOnEquity"),
            "roce": info.get("returnOnAssets"),
            "debt_to_ebitda": info.get("debtToEquity"),
            "net_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
        }
        df_row = pd.DataFrame([row])
        try:
            df_row.to_parquet(cache_path)
        except Exception:
            pass
        records.append(df_row)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def clear_cache(ticker: str | None = None) -> None:
    """Remove cached Parquet files. Pass a ticker to clear one, or None to clear all."""
    if ticker:
        p = CACHE_DIR / f"{ticker.replace('/', '_')}.parquet"
        p.unlink(missing_ok=True)
    else:
        for p in CACHE_DIR.glob("*.parquet"):
            p.unlink()
