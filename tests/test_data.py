from core import load_stock_data


def test_load_stock_data():
    df = load_stock_data("^NSEI", "2022-01-01", "2022-12-31")
    assert df is not None
    assert not df.empty
    assert "returns" in df.columns
    assert df["returns"].isnull().sum() == 0
