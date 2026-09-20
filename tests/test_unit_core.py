"""
Unit tests for core functions without external network reliance.
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from core import (
    _add_jitter,
    fit_egarch_model,
    fit_model_comparison,
    load_stock_data,
    load_gdelt_sentiment,
    load_trends_data,
    news_sentiment_pipeline,
)


def test_add_jitter_constant_series():
    """Verify jitter adds non-zero variance to a constant series."""
    constant = pd.Series([5.0] * 50)
    assert constant.std() == 0.0 or np.isnan(constant.std())
    jittered = _add_jitter(constant, seed=42)
    assert len(jittered) == len(constant)
    assert jittered.std() > 0.0
    # Values should be very close to original
    np.testing.assert_allclose(jittered.values, constant.values, atol=1e-3)


def test_add_jitter_deterministic_seed():
    """Verify seed produces reproducible jitter."""
    s = pd.Series([1.0] * 30)
    j1 = _add_jitter(s, seed=123)
    j2 = _add_jitter(s, seed=123)
    pd.testing.assert_series_equal(j1, j2)


def test_load_stock_data_single_level_columns():
    """Verify stock data parsing and log returns calculation for standard columns."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 110.0]
    mock_df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 1 for p in prices],
            "Low": [p - 1 for p in prices],
            "Close": prices,
            "Volume": [1000] * 10,
        },
        index=dates,
    )

    with patch("yfinance.download", return_value=mock_df):
        result = load_stock_data("^NSEI", "2023-01-01", "2023-01-10")
        assert not result.empty
        assert "returns" in result.columns
        assert len(result) == 9  # first row dropped due to NaN returns
        expected_first_return = np.log(102.0 / 100.0)
        assert np.isclose(result["returns"].iloc[0], expected_first_return)


def test_load_stock_data_multiindex_columns():
    """Verify stock data parsing for MultiIndex columns as returned by newer yfinance."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    prices = [100.0, 102.0, 101.0, 105.0, 103.0]
    tuples = [
        ("Close", "^NSEI"),
        ("High", "^NSEI"),
        ("Low", "^NSEI"),
        ("Open", "^NSEI"),
        ("Volume", "^NSEI"),
    ]
    columns = pd.MultiIndex.from_tuples(tuples, names=["Price", "Ticker"])
    mock_df = pd.DataFrame(
        [
            [100.0, 101.0, 99.0, 100.0, 1000],
            [102.0, 103.0, 101.0, 102.0, 1100],
            [101.0, 102.0, 100.0, 101.0, 1200],
            [105.0, 106.0, 104.0, 105.0, 1300],
            [103.0, 104.0, 102.0, 103.0, 1400],
        ],
        index=dates,
        columns=columns,
    )

    with patch("yfinance.download", return_value=mock_df):
        result = load_stock_data("^NSEI", "2023-01-01", "2023-01-05")
        assert not result.empty
        assert "returns" in result.columns
        assert len(result) == 4


def test_load_stock_data_empty_raises():
    """Verify empty yfinance response raises ValueError."""
    with patch("yfinance.download", return_value=pd.DataFrame()):
        with pytest.raises(ValueError, match="No data for"):
            load_stock_data("BAD_TICKER", "2023-01-01", "2023-01-10")


def test_fit_egarch_model_min_obs_guard():
    """Verify fitting raises ValueError when observations < 100."""
    returns = np.random.normal(0, 0.01, 50)
    sentiment = np.random.normal(0, 1, 50)
    with pytest.raises(ValueError, match="need ≥100"):
        fit_egarch_model(returns, sentiment, p=1, q=1)


def test_fit_egarch_model_non_finite_returns_guard():
    """Verify non-finite returns raise ValueError."""
    returns = np.random.normal(0, 0.01, 150)
    returns[10] = np.nan
    sentiment = np.random.normal(0, 1, 150)
    with pytest.raises(ValueError, match="Returns contain NaN/inf"):
        fit_egarch_model(returns, sentiment, p=1, q=1)


def test_fit_egarch_model_sanitises_sentiment_nan():
    """Verify sentiment with NaNs is sanitised without raising an error."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 150)
    sentiment = rng.normal(0, 1, 150)
    sentiment[5] = np.nan
    sentiment[20] = np.inf

    res = fit_egarch_model(returns, sentiment, p=1, q=1)
    assert res is not None
    assert hasattr(res, "params")
    assert len(res.conditional_volatility) == 150


def test_fit_model_comparison_structure():
    """Verify fit_model_comparison returns comparison table with GARCH, GJR-GARCH, EGARCH."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 250)
    cmp_df = fit_model_comparison(returns)

    assert isinstance(cmp_df, pd.DataFrame)
    assert not cmp_df.empty
    expected_cols = {"Model", "Log-Likelihood", "AIC", "BIC", "Best AIC", "Best BIC"}
    assert expected_cols.issubset(set(cmp_df.columns))
    assert set(cmp_df["Model"]).issubset({"GARCH(1,1)", "GJR-GARCH(1,1)", "EGARCH(1,1)"})
    assert "✅" in cmp_df["Best AIC"].values
    assert "✅" in cmp_df["Best BIC"].values


def test_fit_egarch_model_estimates_asymmetry_and_sentiment():
    """Verify EGARCH(1,1) estimates asymmetric leverage (gamma) and exogenous sentiment (x0)."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.015, 300)
    sentiment = rng.normal(0, 1.0, 300)

    res = fit_egarch_model(returns, sentiment, p=1, q=1, o=1)
    assert res is not None
    assert hasattr(res, "params")
    param_names = res.params.index.tolist()
    assert "gamma[1]" in param_names
    assert "alpha[1]" in param_names
    assert "beta[1]" in param_names
    assert "x0" in param_names


def test_fit_egarch_model_supports_none_sentiment():
    """Verify fit_egarch_model works when sentiment is None (pure EGARCH)."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.015, 200)

    res = fit_egarch_model(returns, None, p=1, q=1, o=1)
    assert res is not None
    assert hasattr(res, "params")
    param_names = res.params.index.tolist()
    assert "gamma[1]" in param_names


def test_load_stock_data_missing_close_raises():
    """Verify load_stock_data raises ValueError if Close column is missing."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df_no_close = pd.DataFrame({"Open": [100.0] * 5, "Volume": [1000] * 5}, index=dates)
    if hasattr(load_stock_data, "clear"):
        load_stock_data.clear()
    with patch("yfinance.download", return_value=df_no_close):
        with pytest.raises(ValueError, match="does not contain 'Close' price"):
            load_stock_data("NO_CLOSE_TICKER", "2023-01-01", "2023-01-05")


