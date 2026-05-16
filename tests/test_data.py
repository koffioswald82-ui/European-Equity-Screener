import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _make_ticker_mock(info: dict):
    m = MagicMock()
    m.info = info
    m.fast_info = MagicMock()
    m.fast_info.market_cap = info.get("marketCap")
    return m


@pytest.fixture()
def sample_info():
    return {
        "longName": "LVMH Moët Hennessy",
        "sector": "Consumer Discretionary",
        "country": "France",
        "marketCap": 350_000_000_000,
        "trailingPE": 20.5,
        "enterpriseToEbitda": 12.3,
        "returnOnEquity": 0.21,
        "returnOnAssets": 0.10,
        "debtToEquity": 45.0,
        "profitMargins": 0.17,
        "revenueGrowth": 0.08,
        "earningsGrowth": 0.12,
        "dividendYield": 0.018,
        "beta": 0.9,
    }


def test_fetch_fundamentals_returns_dataframe(sample_info, tmp_path):
    import screener.data as data_mod
    original_cache = data_mod.CACHE_DIR
    data_mod.CACHE_DIR = tmp_path / "cache"

    try:
        with patch("yfinance.Ticker", return_value=_make_ticker_mock(sample_info)):
            df = data_mod.fetch_fundamentals(["MC.PA"])
    finally:
        data_mod.CACHE_DIR = original_cache

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "MC.PA"
    assert df.iloc[0]["pe_ratio"] == pytest.approx(20.5)


def test_fetch_fundamentals_uses_cache(sample_info, tmp_path):
    import screener.data as data_mod
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    cached = pd.DataFrame([{"ticker": "MC.PA", "pe_ratio": 99.0}])
    cached.to_parquet(cache_dir / "MC.PA.parquet")

    original_cache = data_mod.CACHE_DIR
    data_mod.CACHE_DIR = cache_dir

    try:
        with patch("yfinance.Ticker") as mock_yf:
            df = data_mod.fetch_fundamentals(["MC.PA"])
            mock_yf.assert_not_called()
    finally:
        data_mod.CACHE_DIR = original_cache

    assert df.iloc[0]["pe_ratio"] == pytest.approx(99.0)


def test_fetch_fundamentals_handles_missing_fields(tmp_path):
    import screener.data as data_mod
    original_cache = data_mod.CACHE_DIR
    data_mod.CACHE_DIR = tmp_path / "cache"

    try:
        with patch("yfinance.Ticker", return_value=_make_ticker_mock({})):
            df = data_mod.fetch_fundamentals(["FAKE.PA"])
    finally:
        data_mod.CACHE_DIR = original_cache

    assert "pe_ratio" in df.columns
    assert pd.isna(df.iloc[0]["pe_ratio"])
