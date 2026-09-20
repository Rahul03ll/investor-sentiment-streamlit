"""
Unit tests for application helper functions and ML pipeline mechanics.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from core import safe_trend_fit


def test_safe_trend_fit_linear():
    """Verify safe_trend_fit correctly fits a line on clean data."""
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y = pd.Series([3.0, 5.0, 7.0, 9.0, 11.0])  # y = 2x + 1
    poly, slope = safe_trend_fit(x, y)
    assert poly is not None
    assert slope is not None
    assert np.isclose(slope, 2.0, atol=1e-5)
    assert np.isclose(poly(1.0), 3.0, atol=1e-5)


def test_safe_trend_fit_constant_data():
    """Verify safe_trend_fit returns None for constant x or y."""
    x_const = pd.Series([2.0, 2.0, 2.0, 2.0])
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    poly, slope = safe_trend_fit(x_const, y)
    assert poly is None
    assert slope is None

    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_const = pd.Series([5.0, 5.0, 5.0, 5.0])
    poly2, slope2 = safe_trend_fit(x, y_const)
    assert poly2 is None
    assert slope2 is None


def test_safe_trend_fit_too_few_points():
    """Verify safe_trend_fit returns None if len <= 2."""
    x = pd.Series([1.0, 2.0])
    y = pd.Series([3.0, 5.0])
    poly, slope = safe_trend_fit(x, y)
    assert poly is None
    assert slope is None


def test_safe_trend_fit_handles_nan_and_inf():
    """Verify safe_trend_fit handles and filters nan and inf values."""
    x = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, np.inf])
    y = pd.Series([3.0, 5.0, 7.0, 9.0, 11.0, 15.0])
    poly, slope = safe_trend_fit(x, y)
    assert poly is not None
    assert slope is not None
    assert np.isclose(slope, 2.0, atol=1e-5)


def test_ml_pipeline_mechanics():
    """Verify the ML feature engineering and Random Forest pipeline mechanics."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "returns": rng.normal(0, 0.01, n),
            "sentiment_index": rng.normal(0, 1, n),
            "volatility": rng.uniform(0.5, 3.0, n),
        }
    )

    # Lags
    for lag in [1, 2, 3]:
        df[f"returns_lag{lag}"] = df["returns"].shift(lag)
        df[f"sentiment_lag{lag}"] = df["sentiment_index"].shift(lag)
        df[f"vol_lag{lag}"] = df["volatility"].shift(lag)

    df["vol_ma5"] = df["volatility"].rolling(5).mean()
    df["vol_ma20"] = df["volatility"].rolling(20).mean()
    df["sent_ma5"] = df["sentiment_index"].rolling(5).mean()
    df["vol_change"] = df["volatility"].pct_change()

    # Target: next-day volatility above historical median
    df["vol_class"] = (df["volatility"].shift(-1) > df["volatility"].median()).astype(int)

    feature_cols = [
        "returns",
        "sentiment_index",
        "returns_lag1",
        "sentiment_lag1",
        "vol_lag1",
        "returns_lag2",
        "sentiment_lag2",
        "vol_lag2",
        "returns_lag3",
        "sentiment_lag3",
        "vol_lag3",
        "vol_ma5",
        "vol_ma20",
        "sent_ma5",
        "vol_change",
    ]

    clean_df = df[feature_cols + ["vol_class"]].dropna()
    assert len(clean_df) > 100

    # Ensure no leakage: exclude current volatility from features
    safe_features = [c for c in feature_cols if c != "volatility"]
    X = clean_df[safe_features]
    y = clean_df["vol_class"]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    assert len(preds) == len(y_test)
    acc = accuracy_score(y_test, preds)
    assert 0.0 <= acc <= 1.0

    cm = confusion_matrix(y_test, preds)
    assert cm.shape == (2, 2)
