"""型態規則的共用型別與欄位存取工具。

抽成獨立檔的原因：規則分成三組（趨勢線/指標、TA-Lib K 棒型態、籌碼與相對強弱），
三組都要用 `Pattern`，而 `patterns.py` 又要把三組組裝成總表 —— 型別留在 `patterns.py`
會造成循環 import。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

BULLISH, BEARISH, NEUTRAL = "bullish", "bearish", "neutral"

TONE_ICON = {BULLISH: "🔺", BEARISH: "🔻", NEUTRAL: "▪️"}
TONE_NAME = {BULLISH: "偏多", BEARISH: "偏空", NEUTRAL: "中性"}


@dataclass(frozen=True)
class Pattern:
    """一條型態規則。

    `predicate` 是唯一的判定來源：畫面上的敘述與旁邊的歷史統計都呼叫它，
    一個吃單列（這檔股票這一天），一個吃全市場所有列。
    """

    key: str
    name: str
    tone: str
    columns: tuple[str, ...]
    predicate: Callable[[pd.DataFrame], pd.Series]
    text: str
    group: str = "技術型態"
    # 敘述裡要填的具體數字（例：收斂還剩幾天），拿不到就只顯示 text
    detail: Optional[Callable[[pd.Series], Optional[str]]] = field(default=None)


def flag(df: pd.DataFrame, col: str) -> pd.Series:
    """可為空的 Int8 旗標 → 純布林。NA 視為 False（算不出來就不該報這個型態）。"""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).gt(0.5)


def num(df: pd.DataFrame, col: str) -> pd.Series:
    """數值欄。缺欄位回全 NaN，讓後續比較自然變成 False。"""
    if col not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def fmt_int(row: pd.Series, col: str, template: str) -> Optional[str]:
    value = row.get(col)
    if value is None or pd.isna(value):
        return None
    return template.format(int(round(float(value))))


def fmt_float(row: pd.Series, col: str, template: str) -> Optional[str]:
    value = row.get(col)
    if value is None or pd.isna(value):
        return None
    return template.format(float(value))
