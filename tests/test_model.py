import numpy as np

from core import fit_egarch_model


def test_egarch_model():
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 300)
    sentiment = rng.normal(0, 1, 300)

    result = fit_egarch_model(returns, sentiment, 1, 1)
    assert result is not None
    assert hasattr(result, "params")
