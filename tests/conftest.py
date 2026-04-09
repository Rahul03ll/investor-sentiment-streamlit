import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "returns": rng.normal(0, 0.01, 100),
            "sentiment_index": rng.normal(0, 1, 100),
        }
    )
