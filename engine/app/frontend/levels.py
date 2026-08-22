"""把既有特徵欄位反推成「絕對價位」的支撐／壓力線。

為什麼是反推而不是重算：`data_v2/swing_features.parquet` 與
`trendline_features.parquet` 已經算好轉折點與趨勢線，而且做了前視偏誤防護
（樞紐點要 5 根 K 之後才確認，見 `features_v2/swing_core.py`）。前端自己再算一次
一定會跟模型看到的不一樣，畫出來的線與敘述就對不上訓練資料。

反推用到的欄位語意（來源：`features_v2/build_swing.py` 與 `build_trendline.py`
的檔頭說明）::

    high1_dist   = close / high[樞紐當日] - 1      → 價位 = close / (1 + dist)
    low1_dist    = close / low[樞紐當日]  - 1      → 同上
    vh1_dist     = 大量高點，基準價同樣是當日 high
    *_days       = log1p(距今交易日數)             → 天數 = expm1(值)
    support_dist = close / 迴歸線預測價 - 1        → 線價 = close / (1 + dist)
    support_slope= 每根 K 的斜率 / 預測價          → 絕對斜率 = slope * 線價
    tl_*         = §8 的包絡線版本，dist/slope 定義同上
    close_ma20_ratio = close / ma20                → 均線 = close / ratio

⚠️ `*_dist` 的分母是**基準價不是現價**，所以不能寫成 `close * (1 + dist)`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

# 距離現價超過這個比例的線不顯示：離現價 35% 以外的「壓力」對當下的進出場沒有
# 意義（台積電那種長期大漲的股票會把三年前的大量高點一路帶進來），而且會把圖的
# y 軸拉爛，K 棒全被壓成一條線。
MAX_DISTANCE_FRAC = 0.35
# 均線要畫的窗口。240 日線在多數個股上跟 120 日線幾乎重疊，不畫。
MA_WINDOWS = (20, 60, 120)
# 併成同一「區」的相對價差
CLUSTER_TOLERANCE = 0.02


@dataclass(frozen=True)
class Level:
    """一條水平價位線。"""

    price: float
    label: str          # 顯示名稱，例：「前波高點 ①」
    source: str         # 來源分類：pivot / volume_pivot / ma
    days_ago: Optional[int] = None
    strength: Optional[float] = None   # 大量高低點的量能倍數（60 日均量的幾倍）


@dataclass(frozen=True)
class TrendLine:
    """一條斜的趨勢線，用當日線價與絕對斜率表達。"""

    price_now: float
    slope_per_bar: float   # 每根 K 的價格變化量（元）
    label: str
    r2: Optional[float] = None
    touches: Optional[float] = None

    def price_at(self, bars_from_now: int) -> float:
        """往前（負）或往後（正）第 n 根 K 的線價。"""
        return self.price_now + self.slope_per_bar * bars_from_now


def _finite(value) -> Optional[float]:
    """把 NaN / None / 無限大一律收斂成 None，呼叫端只要判斷 is None。"""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _price_from_dist(close: float, dist) -> Optional[float]:
    """dist = close / 基準價 − 1，反推基準價。"""
    d = _finite(dist)
    if d is None or d <= -1.0:
        return None
    price = close / (1.0 + d)
    return price if price > 0 else None


def _days_from_log(value) -> Optional[int]:
    v = _finite(value)
    return None if v is None else int(round(math.expm1(v)))


def _pivot_levels(row: pd.Series, close: float, prefix: str, name: str,
                  source: str) -> list[Level]:
    """`high1~3` / `low1~3` / `vh1~3` / `vl1~3` 這四組共用的展開邏輯。"""
    out: list[Level] = []
    for rank in (1, 2, 3):
        price = _price_from_dist(close, row.get(f"{prefix}{rank}_dist"))
        if price is None:
            continue
        out.append(Level(
            price=price,
            label=f"{name} {'①②③'[rank - 1]}",
            source=source,
            days_ago=_days_from_log(row.get(f"{prefix}{rank}_days")),
            strength=_finite(row.get(f"{prefix}{rank}_strength")),
        ))
    return out


def _ma_levels(row: pd.Series, close: float) -> list[Level]:
    out: list[Level] = []
    for window in MA_WINDOWS:
        ratio = _finite(row.get(f"close_ma{window}_ratio"))
        if ratio is None or ratio <= 0:
            continue
        out.append(Level(price=close / ratio, label=f"{window} 日均線", source="ma"))
    return out


def extract_levels(row: pd.Series, close: float) -> list[Level]:
    """一列（單股單日）合併特徵 → 所有水平價位線，依價格排序。

    `row` 需含 swing / price 兩組特徵欄位；缺欄位只會少幾條線，不會拋錯。
    """
    levels = (
        _pivot_levels(row, close, "high", "前波高點", "pivot")
        + _pivot_levels(row, close, "low", "前波低點", "pivot")
        + _pivot_levels(row, close, "vh", "大量高點", "volume_pivot")
        + _pivot_levels(row, close, "vl", "大量低點", "volume_pivot")
        + _ma_levels(row, close)
    )
    kept = [lv for lv in levels
            if abs(lv.price - close) / close <= MAX_DISTANCE_FRAC]
    return sorted(kept, key=lambda lv: lv.price)


def _trendline(row: pd.Series, close: float, dist_col: str, slope_col: str,
               label: str, r2_col: str, touch_col: str) -> Optional[TrendLine]:
    price = _price_from_dist(close, row.get(dist_col))
    slope_frac = _finite(row.get(slope_col))
    if price is None or slope_frac is None:
        return None
    return TrendLine(
        price_now=price,
        slope_per_bar=slope_frac * price,   # slope 欄位存的是 斜率/預測價
        label=label,
        r2=_finite(row.get(r2_col)),
        touches=_finite(row.get(touch_col)),
    )


def extract_trendlines(row: pd.Series, close: float) -> list[TrendLine]:
    """§8 的包絡線（教科書定義：壓力線連高點、支撐線連低點，且所有樞紐點同側）。

    §7 的 `support_*`/`resist_*` 是迴歸中線，一半樞紐點會在線上方，畫給人看會誤導，
    這裡只取 §8 那組（見 `build_trendline.py` 檔頭第 2 點）。
    """
    lines = [
        _trendline(row, close, "tl_resist_dist", "tl_resist_slope", "壓力線",
                   "tl_resist_r2", "tl_resist_touches"),
        _trendline(row, close, "tl_support_dist", "tl_support_slope", "支撐線",
                   "tl_support_r2", "tl_support_touches"),
    ]
    return [ln for ln in lines if ln is not None]


def classify(levels: list[Level], close: float) -> tuple[list[Level], list[Level]]:
    """依現價把價位線分成（壓力, 支撐）。

    刻意用現價判定而不是看名稱：前波高點一旦被站上就變成支撐，
    照名稱分類會在突破後給出完全相反的解讀。
    """
    resistance = [lv for lv in levels if lv.price > close]
    support = [lv for lv in levels if lv.price <= close]
    return resistance, support


def nearest(levels: list[Level], close: float, above: bool) -> Optional[Level]:
    side = [lv for lv in levels if (lv.price > close if above else lv.price <= close)]
    if not side:
        return None
    return min(side, key=lambda lv: abs(lv.price - close))


def cluster(levels: list[Level],
            tolerance: float = CLUSTER_TOLERANCE) -> list[list[Level]]:
    """把價位相近的線併成「區」—— 網紅講的是壓力**區**，不是單一價位。

    tolerance 是相對價差（0.02 = 2% 以內視為同一區）。輸入需已依價格排序。
    """
    groups: list[list[Level]] = []
    for lv in levels:
        if groups and abs(lv.price - groups[-1][-1].price) / groups[-1][-1].price <= tolerance:
            groups[-1].append(lv)
        else:
            groups.append([lv])
    return groups
