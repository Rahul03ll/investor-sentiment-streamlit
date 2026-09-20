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


def test_gdelt_sentiment_with_mocked_api():
    """Verify GDELT parsing, tone filtering, and resampling with mocked HTTP responses."""
    mock_articles = [
        {"tone": f"{-i:.1f},1,2,3", "seendate": f"202301{i:02d}T120000Z"}
        for i in range(1, 15)
    ]
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"articles": mock_articles}

    if hasattr(load_gdelt_sentiment, "clear"):
        load_gdelt_sentiment.clear()

    with patch("core.requests.get", return_value=mock_resp):
        df, err = load_gdelt_sentiment("2023-01-01", "2023-01-31", fast_mode=True)
        assert err is None
        assert df is not None
        assert "sentiment" in df.columns
        assert len(df) > 5


def test_load_trends_data_with_mocked_pytrends():
    """Verify Google Trends payload creation and column filtering with mocked pytrends."""
    dates = pd.date_range("2023-01-01", periods=10, freq="W")
    mock_trends = pd.DataFrame(
        {
            "stock market crash": [10, 15, 20, 18, 12, 14, 16, 25, 30, 22],
            "Nifty crash": [5, 8, 12, 10, 6, 7, 9, 15, 18, 11],
            "Sensex fall": [8, 12, 16, 14, 9, 10, 11, 20, 24, 15],
            "isPartial": [False] * 10,
        },
        index=dates,
    )
    if hasattr(load_trends_data, "clear"):
        load_trends_data.clear()

    with patch("core.TrendReq") as mock_trendreq_cls:
        mock_instance = MagicMock()
        mock_instance.interest_over_time.return_value = mock_trends
        mock_trendreq_cls.return_value = mock_instance

        df, keywords, err = load_trends_data("2023-01-01", "2023-03-01")
        assert err is None
        assert df is not None
        assert len(keywords) == 3
        assert "isPartial" not in df.columns


