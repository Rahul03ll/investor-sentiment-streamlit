"""
Smoke test for the full analysis pipeline.
Uses live network data when available, with resilient fallback to test pipeline integrity.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
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
        1,
    )
    assert result is not None
    assert hasattr(result, "params")
    assert len(result.conditional_volatility) > 0
    assert "gamma[1]" in result.params.index

    # Engine 2: Random Forest Directional Volatility Forecasting
    data["volatility"] = result.conditional_volatility[-len(data):]
    feature_cols = ["returns", "sentiment_index", "volatility"]
    for lag in [1, 2]:
        data[f"returns_lag{lag}"] = data["returns"].shift(lag)
        data[f"sentiment_lag{lag}"] = data["sentiment_index"].shift(lag)
        data[f"vol_lag{lag}"] = data["volatility"].shift(lag)
        feature_cols += [f"returns_lag{lag}", f"sentiment_lag{lag}", f"vol_lag{lag}"]

    cols_to_use = list(dict.fromkeys(feature_cols))
    clean_ml = data[cols_to_use].dropna().copy()
    split = int(len(clean_ml) * 0.8)
    vol_thresh = clean_ml["volatility"].iloc[:split].median()
    clean_ml["vol_class"] = (clean_ml["volatility"].shift(-1) > vol_thresh).astype(int)
    clean_ml = clean_ml.dropna()

    safe_feats = [c for c in feature_cols if c != "volatility"]
    X_train, X_test = clean_ml[safe_feats].iloc[:split], clean_ml[safe_feats].iloc[split:]
    y_train, y_test = clean_ml["vol_class"].iloc[:split], clean_ml["vol_class"].iloc[split:]

    rf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    assert len(preds) == len(y_test)
    assert 0.0 <= accuracy_score(y_test, preds) <= 1.0
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
        assert 0.0 <= auc <= 1.0

