import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def test_ml_training():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "returns": rng.normal(0, 0.01, 220),
            "sentiment_index": rng.normal(0, 1, 220),
            "volatility": rng.uniform(0.1, 2.0, 220),
        }
    )
    df["target"] = (df["volatility"].shift(-1) > df["volatility"].median()).astype(int)
    df = df.dropna()

    X = df[["returns", "sentiment_index", "volatility"]]
    y = df["target"]

    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
