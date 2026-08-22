"""共用 fixture。

`engine` 是 repo 根目錄底下的正常 package，pytest 從根目錄跑就 import 得到，
不需要任何 sys.path 操作（rootdir 會被加進 sys.path）。
"""
import numpy as np
import pandas as pd
import pytest

from engine.paths import DATA_DIR, PROJECT_ROOT  # noqa: F401  （測試可直接取用）


@pytest.fixture
def sample_price_df():
    """最小價格 DataFrame，供各測試使用。"""
    dates = pd.date_range("2024-01-02", periods=60, freq="B")
    np.random.seed(42)
    close = 100 * (1 + np.random.randn(60).cumsum() * 0.01)
    return pd.DataFrame({
        "date": dates,
        "stock_id": "2330",
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.randint(1000, 10000, 60).astype(float),
        "amount": close * np.random.randint(1000, 10000, 60),
    })
