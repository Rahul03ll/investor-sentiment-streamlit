"""
Tests for sentiment loading functions.
Network calls are handled gracefully, and mock tests ensure pipeline integrity.
"""
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

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


def test_news_sentiment_pipeline_network_handling():
    """News pipeline should return (Series, None, source) or (None, err, 'None')."""
    series, err, source = news_sentiment_pipeline(api_key=None)
    assert isinstance(source, str)
    if series is not None:
        assert isinstance(series, pd.Series)
        assert len(series) > 0
    else:
        assert isinstance(err, str)


def test_news_sentiment_pipeline_with_mocked_rss():
    """Verify sentiment analysis calculation when RSS returns valid headlines."""
    mock_headlines = [
        "Nifty surges to all-time high amid strong corporate earnings",
        "Sensex rallies 500 points led by banking and IT stocks",
        "Markets trade higher as foreign investors boost inflows",
        "Stock market gains momentum on positive global cues",
        "Nifty breaks key resistance level with high trading volumes",
        "Bulls dominate Dalal Street as inflation eases",
        "Sensex climbs as economic growth data beats expectations",
        "Market sentiment turns bullish with robust retail participation",
        "Broader markets outperform benchmarks in solid rally",
        "Nifty crosses record milestone with broad-based buying",
        "Sensex surges further as auto and metal stocks rally",
        "Indian equities continue uptrend for fourth straight session",
    ]

    with patch("core._newsapi_texts", return_value=([], "No API key.")), \
         patch("core._yfinance_texts", return_value=([], "No Yahoo news.")), \
         patch("core._rss_texts", return_value=(mock_headlines, None)):
        series, err, source = news_sentiment_pipeline()
        assert err is None
        assert source == "RSS Feed"
        assert isinstance(series, pd.Series)
        assert len(series) > 0
        assert series.name == "sentiment"
        # Sentiment for bullish headlines should be positive
        assert series.mean() > 0


def test_news_sentiment_pipeline_all_sources_fail():
    """Verify graceful failure when all three news sources fail."""
    with patch("core._newsapi_texts", return_value=([], "Mock NewsAPI error")), \
         patch("core._yfinance_texts", return_value=([], "Mock YFinance error")), \
         patch("core._rss_texts", return_value=([], "Mock RSS error")):
        series, err, source = news_sentiment_pipeline()
        assert series is None
        assert source == "None"
        assert "Mock RSS error" in err
