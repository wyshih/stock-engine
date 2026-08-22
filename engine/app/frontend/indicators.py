"""看盤圖用的疊圖計算：均線、布林、VWAP、費波那契、樞紐點、量價分布、缺口。

為什麼不直接用 `data_v2/price_features.parquet`：那裡的欄位是餵模型的**比值**
（`dif_ratio = dif / close`、`bb_width_pct`…），畫圖需要的是絕對價位序列。能反推
的（MACD 家族、KD、RSI）在頁面那邊直接反推，反推不回來的（布林上下軌的實際定義、
均線本身）在這裡用標準公式算。

⚠️ 這裡算的東西**只用於顯示**，不進模型、不進回測。模型看到的永遠是
`features_v2/` 產出的那份。兩邊若有細微差異（例如布林用母體或樣本標準差），
差異只影響畫面上的線，不影響任何決策數字。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MA_WINDOWS = (5, 10, 20, 60, 120, 240)
VOLUME_MA_WINDOWS = (5, 20)
BOLLINGER_WINDOW, BOLLINGER_K = 20, 2.0
FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)
VOLUME_PROFILE_BINS = 48
MIN_GAP_FRAC = 0.01     # 缺口至少要有 1%，否則滿圖都是雜訊


def moving_averages(ohlc: pd.DataFrame,
                    windows: tuple[int, ...] = MA_WINDOWS) -> dict[int, pd.Series]:
    """只回傳資料長度撐得住的均線 —— 60 根 K 畫 240 日線只會得到一整條 NaN。"""
    close = ohlc["close"]
    return {w: close.rolling(w, min_periods=w).mean()
            for w in windows if len(close) >= w}


def volume_averages(ohlc: pd.DataFrame,
                    windows: tuple[int, ...] = VOLUME_MA_WINDOWS) -> dict[int, pd.Series]:
    volume = ohlc["volume"]
    return {w: volume.rolling(w, min_periods=w).mean()
            for w in windows if len(volume) >= w}


def bollinger(ohlc: pd.DataFrame, window: int = BOLLINGER_WINDOW,
              k: float = BOLLINGER_K) -> dict[str, pd.Series]:
    close = ohlc["close"]
    if len(close) < window:
        return {}
    middle = close.rolling(window, min_periods=window).mean()
    deviation = close.rolling(window, min_periods=window).std(ddof=0) * k
    return {"中軌": middle, "上軌": middle + deviation, "下軌": middle - deviation}


def anchored_vwap(ohlc: pd.DataFrame) -> pd.Series:
    """從圖表起點錨定的 VWAP：這段期間的成交量加權平均成本。

    日線沒有盤中明細，用典型價 (H+L+C)/3 當每根 K 的代表價 —— 這是 VWAP 在日線圖
    上的標準近似。
    """
    typical = (ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3
    volume = ohlc["volume"].fillna(0)
    return (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)


def fibonacci_levels(ohlc: pd.DataFrame) -> dict[str, float]:
    """以視窗內的最高／最低點做回撤。上升段從低點往上量，下降段反之。"""
    high, low = float(ohlc["high"].max()), float(ohlc["low"].min())
    if not np.isfinite(high) or not np.isfinite(low) or high <= low:
        return {}
    high_first = ohlc["high"].idxmax() < ohlc["low"].idxmin()
    span = high - low
    levels = {"起點": low if high_first else high,
              "終點": high if high_first else low}
    for ratio in FIB_RATIOS:
        levels[f"{ratio:.1%}"] = (low + span * ratio if high_first
                                  else high - span * ratio)
    return levels


def pivot_points(ohlc: pd.DataFrame) -> dict[str, float]:
    """經典樞紐點：用最後一根 K 的高低收推算隔日的支撐壓力。"""
    last = ohlc.iloc[-1]
    high, low, close = float(last["high"]), float(last["low"]), float(last["close"])
    pivot = (high + low + close) / 3
    span = high - low
    return {
        "P": pivot,
        "R1": 2 * pivot - low, "S1": 2 * pivot - high,
        "R2": pivot + span, "S2": pivot - span,
    }


def volume_profile(ohlc: pd.DataFrame,
                   bins: int = VOLUME_PROFILE_BINS) -> pd.DataFrame:
    """量價分布：每個價格區間累積了多少成交量 —— 也就是套牢區在哪。

    把每根 K 的量整筆記在它的典型價上（不在 high~low 之間攤平）。日線沒有盤中
    明細，攤平只是換一種假設，不會比較準，卻會讓分布糊掉。
    """
    typical = (ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3
    low, high = float(typical.min()), float(typical.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return pd.DataFrame(columns=["price", "volume"])
    edges = np.linspace(low, high, bins + 1)
    which = np.clip(np.digitize(typical.to_numpy(), edges) - 1, 0, bins - 1)
    totals = np.zeros(bins)
    np.add.at(totals, which, ohlc["volume"].fillna(0).to_numpy())
    return pd.DataFrame({"price": (edges[:-1] + edges[1:]) / 2, "volume": totals})


def gaps(ohlc: pd.DataFrame, min_frac: float = MIN_GAP_FRAC) -> list[dict]:
    """跳空缺口。只回傳**還沒被回補**的 —— 補掉的缺口在盤面上已經沒有意義。"""
    high, low, close = ohlc["high"], ohlc["low"], ohlc["close"]
    previous_high, previous_low = high.shift(1), low.shift(1)
    up = (low > previous_high * (1 + min_frac)).fillna(False).to_numpy()
    down = (high < previous_low * (1 - min_frac)).fillna(False).to_numpy()

    later_low = close.iloc[::-1].cummin().iloc[::-1]
    later_high = close.iloc[::-1].cummax().iloc[::-1]
    found: list[dict] = []
    for position in np.flatnonzero(up):
        bottom, top = float(previous_high.iloc[position]), float(low.iloc[position])
        if float(later_low.iloc[position]) > bottom:      # 之後沒有跌回來
            found.append({"date": ohlc["date"].iloc[position], "low": bottom,
                          "high": top, "direction": "up"})
    for position in np.flatnonzero(down):
        bottom, top = float(high.iloc[position]), float(previous_low.iloc[position])
        if float(later_high.iloc[position]) < top:        # 之後沒有補上去
            found.append({"date": ohlc["date"].iloc[position], "low": bottom,
                          "high": top, "direction": "down"})
    return sorted(found, key=lambda g: g["date"])
