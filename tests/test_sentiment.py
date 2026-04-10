"""
Tests for sentiment loading functions.
Network calls are expected; failures are handled gracefully.
"""
import pandas as pd

from core import load_gdelt_sentiment, load_trends_data, news_sentiment_pipeline


def test_gdelt_sentiment_fast_mode():
    """GDELT may return None if the API is unavailable — that is acceptable."""
    df, err = load_gdelt_sentiment("2023-01-01", "2023-01-31", fast_mode=True)
    if df is not None:
        assert isinstance(df, pd.DataFrame)
        assert "sentiment" in df.columns
        assert len(df) > 0
    else:
        assert isinstance(err, str) and len(err) > 0


def test_load_trends_data_returns_valid_types():
    """Trends may fail due to rate-limiting — check return types only."""
    result = load_trends_data("2023-01-01", "2023-06-01")
    assert isinstance(result, tuple) and len(result) == 3
    df, keywords, err = result
    if df is not None:
        assert isinstance(df, pd.DataFrame)
        assert isinstance(keywords, list)
        assert len(keywords) > 0
    else:
        assert isinstance(err, str)


def test_news_sentiment_pipeline_returns_series():
    """News pipeline should always return a Series (may be demo data)."""
    series, err, source = news_sentiment_pipeline(api_key=None)
    assert isinstance(series, pd.Series)
    assert isinstance(source, str)
    assert len(series) > 0
