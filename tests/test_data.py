import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _make_ticker_mock(info: dict):
    m = MagicMock()
    m.info = info
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


def test_fetch_fundamentals_returns_dataframe(sample_info, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "cache").mkdir(parents=True)

    with patch("yfinance.Ticker", return_value=_make_ticker_mock(sample_info)):
        from screener.data import fetch_fundamentals
        df = fetch_fundamentals(["MC.PA"])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "MC.PA"
    assert df.iloc[0]["pe_ratio"] == pytest.approx(20.5)


def test_fetch_fundamentals_uses_cache(sample_info, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "data" / "cache"
    cache_dir.mkdir(parents=True)

    cached = pd.DataFrame([{"ticker": "MC.PA", "pe_ratio": 99.0}])
    cached.to_parquet(cache_dir / "MC.PA.parquet")

    with patch("yfinance.Ticker") as mock_yf:
        from screener.data import fetch_fundamentals, CACHE_DIR
        import screener.data as data_mod
        data_mod.CACHE_DIR = cache_dir
        df = fetch_fundamentals(["MC.PA"])
        mock_yf.assert_not_called()

    assert df.iloc[0]["pe_ratio"] == pytest.approx(99.0)


def test_fetch_fundamentals_handles_missing_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "cache").mkdir(parents=True)

    with patch("yfinance.Ticker", return_value=_make_ticker_mock({})):
        from screener.data import fetch_fundamentals
        df = fetch_fundamentals(["FAKE.PA"])

    assert "pe_ratio" in df.columns
    assert pd.isna(df.iloc[0]["pe_ratio"])
