import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from core import fit_egarch_model, load_gdelt_sentiment, load_stock_data, load_trends_data


def test_full_pipeline_smoke():
    data = load_stock_data("^NSEI", "2022-01-01", "2022-06-01")
    assert data is not None and not data.empty

    gdelt_df, _ = load_gdelt_sentiment("2022-01-01", "2022-06-01", fast_mode=True)
    if gdelt_df is not None and not gdelt_df.empty:
        data = data.merge(gdelt_df, left_index=True, right_index=True, how="left")
        data["sentiment"] = data["sentiment"].ffill().bfill()
        data["sentiment_index"] = data["sentiment"]
    else:
        trends, keywords, _ = load_trends_data("2022-01-01", "2022-06-01")
        if trends is not None and keywords:
            data = data.merge(trends, left_index=True, right_index=True, how="left")
            data[keywords] = data[keywords].ffill().bfill()
            scaled = StandardScaler().fit_transform(data[keywords])
            data["sentiment_index"] = PCA(n_components=1).fit_transform(scaled)
        else:
            data["sentiment_index"] = np.random.default_rng(42).normal(0, 0.5, len(data))

    result = fit_egarch_model(
        data["returns"].values,
        data["sentiment_index"].values,
        1,
        1,
    )
    assert result is not None
