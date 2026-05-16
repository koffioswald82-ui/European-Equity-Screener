import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from screener.scoring import (
    _zscore,
    compute_value_score,
    compute_quality_score,
    compute_momentum_score,
    composite_score,
)


@pytest.fixture()
def sample_fundamentals():
    return pd.DataFrame({
        "ticker":         ["A", "B", "C", "D"],
        "pe_ratio":       [15.0, 25.0, 10.0, 30.0],
        "ev_ebit":        [10.0, 18.0,  8.0, 22.0],
        "roe":            [0.20, 0.10,  0.30,  0.05],
        "roce":           [0.15, 0.08,  0.25,  0.04],
        "net_margin":     [0.15, 0.08,  0.22,  0.03],
        "earnings_growth":[0.10, 0.05,  0.20, -0.02],
        "revenue_growth": [0.08, 0.04,  0.15, -0.01],
    })


@pytest.fixture()
def sample_prices():
    np.random.seed(42)
    idx = pd.date_range("2022-01-01", periods=300, freq="B")
    data = {t: 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, 300))
            for t in ["A", "B", "C", "D"]}
    return pd.DataFrame(data, index=idx)


def test_zscore_mean_zero_std_one():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = _zscore(s)
    assert abs(z.mean()) < 1e-10
    assert abs(z.std() - 1.0) < 1e-10


def test_zscore_constant_series():
    s = pd.Series([5.0, 5.0, 5.0])
    z = _zscore(s)
    assert (z == 0.0).all()


def test_value_score_cheaper_stock_higher(sample_fundamentals):
    df = sample_fundamentals.set_index("ticker")
    scores = compute_value_score(df)
    assert scores["C"] > scores["D"]


def test_quality_score_better_roe_higher(sample_fundamentals):
    df = sample_fundamentals.set_index("ticker")
    scores = compute_quality_score(df)
    assert scores["C"] > scores["D"]


def test_momentum_score_returns_series(sample_prices):
    scores = compute_momentum_score(sample_prices)
    assert isinstance(scores, pd.Series)
    assert set(scores.index) == {"A", "B", "C", "D"}


def test_composite_score_sorted_descending(sample_fundamentals, sample_prices):
    result = composite_score(sample_fundamentals, sample_prices)
    assert list(result["composite_score"]) == sorted(result["composite_score"], reverse=True)


def test_composite_score_range(sample_fundamentals, sample_prices):
    result = composite_score(sample_fundamentals, sample_prices)
    assert result["composite_score"].min() >= 0
    assert result["composite_score"].max() <= 100


def test_composite_score_with_sentiment(sample_fundamentals, sample_prices):
    sentiment = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "sentiment_score": [0.5, -0.3, 0.8, -0.1],
    })
    result = composite_score(sample_fundamentals, sample_prices, sentiment)
    assert "composite_score" in result.columns
    assert len(result) == 4
