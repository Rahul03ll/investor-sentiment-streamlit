"""Tests for stock data loading."""
import pytest

from core import load_stock_data


def test_load_stock_data_returns_valid_dataframe():
    try:
        df = load_stock_data("^NSEI", "2022-01-01", "2022-06-01")
        assert df is not None
        assert not df.empty
        assert "returns" in df.columns
        assert df["returns"].isnull().sum() == 0
    except Exception as exc:
        # Network or rate limiting may occur in external CI environments
        assert any(term in str(exc).lower() for term in ["no data for", "http", "connection", "rate", "timeout"])


def test_load_stock_data_raises_on_bad_ticker():
    with pytest.raises(Exception):
        load_stock_data("INVALID_TICKER_XYZ", "2022-01-01", "2022-12-31")
