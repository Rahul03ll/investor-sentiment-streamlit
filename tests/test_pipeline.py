"""
Smoke test for the full analysis pipeline.
Uses live network data when available, with resilient fallback to test pipeline integrity.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from core import fit_egarch_model, load_gdelt_sentiment, load_stock_data, load_trends_data


def test_full_pipeline_smoke():
    try:
        data = load_stock_data("^NSEI", "2022-01-01", "2022-06-01")
    except Exception:
        # Fallback to deterministic synthetic stock data if network is throttled
        dates = pd.date_range("2022-01-01", "2022-06-01", freq="B")
        rng = np.random.default_rng(42)
        close = 15000 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, len(dates))))
        data = pd.DataFrame({"Close": close}, index=dates)
        data["returns"] = np.log(data["Close"] / data["Close"].shift(1))
        data = data.dropna(subset=["returns"])

    assert data is not None and not data.empty
    assert "returns" in data.columns

    # Try GDELT first; fall back to Trends; fall back to generated sentiment
    gdelt_df = None
    try:
        gdelt_df, _ = load_gdelt_sentiment("2022-01-01", "2022-06-01", fast_mode=True)
    except Exception:
        gdelt_df = None

    if gdelt_df is not None and not gdelt_df.empty:
        data = data.merge(gdelt_df, left_index=True, right_index=True, how="left")
        data["sentiment"] = data["sentiment"].ffill().bfill()
        data["sentiment_index"] = data["sentiment"]
    else:
        trends = None
        keywords = []
        try:
            trends, keywords, _ = load_trends_data("2022-01-01", "2022-06-01")
        except Exception:
            trends, keywords = None, []

        if trends is not None and keywords:
            data = data.merge(trends, left_index=True, right_index=True, how="left")
            data[keywords] = data[keywords].ffill().bfill()
            scaled = StandardScaler().fit_transform(data[keywords].fillna(0))
            data["sentiment_index"] = PCA(n_components=1).fit_transform(scaled)
        else:
            rng = np.random.default_rng(42)
            data["sentiment_index"] = rng.normal(0, 0.5, len(data))

    data = data.dropna(subset=["returns", "sentiment_index"])
    assert len(data) > 0, "No data left after cleaning."

    result = fit_egarch_model(
        data["returns"].values,
        data["sentiment_index"].values,
        1,
        1,
    )
    assert result is not None
    assert hasattr(result, "params")
    assert len(result.conditional_volatility) > 0
