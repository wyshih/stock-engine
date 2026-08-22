"""型態的全市場歷史條件統計：「過去出現這個型態 N 次，之後怎麼走」。

這是這頁跟網紅唯一的差別 —— 話術後面掛得出樣本數與勝率。

方法（刻意跟回測對齊）：
- 進場價 = 訊號日**隔一天**的收盤（訊號要收盤後才知道，當天買不到）。
- 出場價 = 進場後第 5 / 20 個交易日的收盤，與 ground truth `label_up20`
  的 20 個交易日視窗一致。
- 對照組是**全市場所有日子**的同一組報酬。沒有基準的勝率沒有意義：
  多頭年份隨便買 20 天勝率都有五成以上。

已知的限制（畫面上一定要一起講）：
- `data/price.parquet` 不含已下市股票，統計本身帶生存偏差，數字偏樂觀。
- 這是全市場統計，不是這一檔的統計，個股的產業與籌碼結構沒有納入。
- 樣本數低於 `MIN_SAMPLES` 一律不給結論。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st


from engine.app.frontend.pattern_base import Pattern
from engine.paths import DATA_DIR

# 特徵欄位可能落在這幾個檔案，實際歸屬由 parquet schema 決定，不寫死對照表。
# 順序有意義：同名欄位以先出現的檔案為準（`price_features` 與 `talib_features`
# 都有 rsi 家族，前者才是本專案自己算的那組）。
FEATURE_FILES = ("swing_features", "trendline_features", "price_features",
                 "talib_features", "chip_features", "relative_features")
HOLD_DAYS = (5, 20)
MIN_SAMPLES = 30
# 勝率差距小於這個值視為沒有差別
EDGE_TOLERANCE = 0.01


@st.cache_data(ttl=3600, show_spinner=False)
def _column_index() -> dict[str, str]:
    """欄位名 → 所在的 parquet 檔名。只讀 schema，不讀資料。"""
    index: dict[str, str] = {}
    for name in FEATURE_FILES:
        path = DATA_DIR / f"{name}.parquet"
        if not path.exists():
            continue
        for column in pq.read_schema(path).names:
            index.setdefault(column, name)
    return index


@st.cache_data(ttl=3600, show_spinner=False)
def forward_returns() -> pd.DataFrame:
    """全市場每日的未來報酬（進場 = 隔日收盤）。"""
    price = pd.read_parquet(DATA_DIR / "price.parquet",
                            columns=["date", "stock_id", "close"])
    price["date"] = pd.to_datetime(price["date"])
    price = price.sort_values(["stock_id", "date"])
    entry = price.groupby("stock_id")["close"].shift(-1)
    out = price[["date", "stock_id"]].copy()
    for days in HOLD_DAYS:
        exit_price = price.groupby("stock_id")["close"].shift(-(days + 1))
        out[f"ret{days}"] = (exit_price - entry) / entry
    return out


def _summarise(frame: pd.DataFrame) -> dict:
    out: dict = {"n": int(len(frame))}
    for days in HOLD_DAYS:
        series = frame[f"ret{days}"].dropna()
        out[f"n{days}"] = int(len(series))
        out[f"win{days}"] = float((series > 0).mean()) if len(series) else None
        out[f"median{days}"] = float(series.median()) if len(series) else None
        out[f"mean{days}"] = float(series.mean()) if len(series) else None
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def baseline() -> dict:
    """對照組：全市場所有日子隨便買的結果。"""
    return _summarise(forward_returns())


@st.cache_data(ttl=3600, show_spinner=False)
def _load_columns(columns: tuple[str, ...]) -> pd.DataFrame:
    """把散在不同 parquet 的欄位併成一張 (date, stock_id) + 指定欄位的表。"""
    index = _column_index()
    by_file: dict[str, list[str]] = {}
    for column in columns:
        source = index.get(column)
        if source is None:
            continue
        by_file.setdefault(source, []).append(column)

    merged: pd.DataFrame | None = None
    for name, cols in by_file.items():
        part = pd.read_parquet(DATA_DIR / f"{name}.parquet",
                               columns=["date", "stock_id", *cols])
        part["date"] = pd.to_datetime(part["date"])
        merged = part if merged is None else merged.merge(
            part, on=["date", "stock_id"], how="inner")
    return merged if merged is not None else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _stats_for(key: str, columns: tuple[str, ...]) -> dict | None:
    """快取的實際計算。以 key 當快取鍵：predicate 是 lambda，不可雜湊。"""
    from engine.app.frontend.patterns import PATTERNS_BY_KEY

    pattern = PATTERNS_BY_KEY[key]
    features = _load_columns(columns)
    if features.empty:
        return None
    hit = features[pattern.predicate(features).fillna(False)]
    if hit.empty:
        return {"n": 0}
    joined = hit[["date", "stock_id"]].merge(
        forward_returns(), on=["date", "stock_id"], how="inner")
    return _summarise(joined)


def pattern_stats(pattern: Pattern) -> dict | None:
    """某個型態的歷史統計；欄位缺失回 None。"""
    return _stats_for(pattern.key, pattern.columns)


@st.cache_data(ttl=3600, show_spinner=False)
def joint_stats(keys: tuple[str, ...]) -> dict | None:
    """**同時**符合這幾個條件的歷史日子，之後怎麼走。

    綜合結論一定要走這裡，不能把各條型態的勝率平均起來 —— 今天成立的十條說法
    彼此高度重疊（均線多頭排列與創新高幾乎同時發生），平均等於把同一件事算了
    很多次，得到的數字沒有任何意義。
    """
    from engine.app.frontend.patterns import PATTERNS_BY_KEY

    patterns = [PATTERNS_BY_KEY[k] for k in keys if k in PATTERNS_BY_KEY]
    if not patterns:
        return None
    columns = tuple(dict.fromkeys(c for p in patterns for c in p.columns))
    features = _load_columns(columns)
    if features.empty:
        return None

    mask = pd.Series(True, index=features.index)
    for pattern in patterns:
        mask &= pattern.predicate(features).fillna(False)
    hit = features[mask]
    if hit.empty:
        return {"n": 0, "n5": 0, "n20": 0}
    joined = hit[["date", "stock_id"]].merge(
        forward_returns(), on=["date", "stock_id"], how="inner")
    return _summarise(joined)


@st.cache_data(ttl=3600, show_spinner=False)
def crossing_history(stock_id: str, price: float, downward: bool) -> dict:
    """這檔股票歷史上「穿越某個價位」之後的走勢。

    事件定義是**穿越當天**：前一天收盤在價位的另一側，當天收盤穿過去。用「收盤在
    價位下方的所有日子」當樣本是錯的 —— 那會把一路陰跌的整段期間重複計入，樣本
    彼此高度重疊，勝率會被同一段行情灌爆。
    """
    price_all = pd.read_parquet(DATA_DIR / "price.parquet",
                                columns=["date", "stock_id", "close"])
    one = price_all[price_all["stock_id"] == stock_id].copy()
    if one.empty:
        return {"n": 0}
    one["date"] = pd.to_datetime(one["date"])
    one = one.sort_values("date")
    previous = one["close"].shift(1)
    if downward:
        event = (previous > price) & (one["close"] <= price)
    else:
        event = (previous < price) & (one["close"] >= price)

    hit = one[event.fillna(False)]
    if hit.empty:
        return {"n": 0}
    joined = hit[["date", "stock_id"]].merge(
        forward_returns(), on=["date", "stock_id"], how="inner")
    return _summarise(joined)


def is_conclusive(stats: dict | None, days: int = 20) -> bool:
    return bool(stats) and (stats.get(f"n{days}") or 0) >= MIN_SAMPLES


def verdict(stats: dict, base: dict, days: int = 20) -> str:
    """跟對照組比出來的一句話。差距小於 1 個百分點視為沒有差別。"""
    edge = (stats.get(f"win{days}") or 0) - (base.get(f"win{days}") or 0)
    if abs(edge) < EDGE_TOLERANCE:
        return "與隨機進場沒有明顯差別"
    return f"勝率{'高' if edge > 0 else '低'}於隨機進場 {abs(edge):.1%}"
