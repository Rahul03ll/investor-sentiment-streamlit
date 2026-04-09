from core import load_gdelt_sentiment


def test_gdelt_sentiment_fast_mode():
    df, err = load_gdelt_sentiment("2023-01-01", "2023-01-10", fast_mode=True)
    if df is not None:
        assert "sentiment" in df.columns
        assert len(df) > 0
    else:
        assert isinstance(err, str)
