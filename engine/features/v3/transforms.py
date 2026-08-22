"""v3 特徵的兩個基礎轉換。

兩個都嚴格只用到 t 為止的資訊：
* 方向 A `self_zscore`：rolling 視窗以當列結尾（pandas rolling 預設 closed="right"），
  不含任何未來列。
* 方向 B `cross_section_rank`：只用「同一個交易日」的橫斷面，不跨日。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SELF_WINDOW = 250
SELF_MIN_PERIODS = 60
STD_EPSILON = 1e-9
# z-score 截尾。少數股票在長期停牌後復牌會產生 |z| > 50 的值，
# 留著會讓「跨年可比」這件事在極端列上失效。
Z_CLIP = 8.0


def self_zscore(
    values: pd.Series,
    group: pd.Series,
    window: int = SELF_WINDOW,
    min_periods: int = SELF_MIN_PERIODS,
) -> pd.Series:
    """方向 A：該值相對「這檔股票自己過去 window 日」的 z-score。

    呼叫端必須保證 values 在每個 group 內是日期遞增的，否則 rolling 會用到未來。
    """
    grouped = values.groupby(group, sort=False)
    mean = grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    std = grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).std())
    z = (values - mean) / std.where(std > STD_EPSILON)
    return z.clip(-Z_CLIP, Z_CLIP)


def cross_section_rank(values: pd.Series, date: pd.Series) -> pd.Series:
    """方向 B：當日全市場橫斷面百分位（0~1）。NaN 保持 NaN。"""
    return values.groupby(date, sort=False).rank(pct=True)


def assert_within_group_ascending(dates: pd.Series, group: pd.Series) -> None:
    """守門：確認每個 stock_id 內部日期遞增，否則 rolling 會取到未來資料。"""
    diffs = dates.groupby(group, sort=False).diff()
    bad = int((diffs < pd.Timedelta(0)).sum())
    if bad:
        raise ValueError(f"{bad} 列的日期在同一個 stock_id 內不是遞增的，rolling 會洩漏未來資料")


def summarise_column(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"nan_pct": 1.0, "median": np.nan, "p10": np.nan, "p90": np.nan}
    p10, p50, p90 = np.percentile(finite, [10, 50, 90])
    return {
        "nan_pct": float(1.0 - finite.size / values.size),
        "median": float(p50),
        "p10": float(p10),
        "p90": float(p90),
    }
