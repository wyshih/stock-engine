"""
前波高低點、大量高低點、趨勢線特徵（PLAN.md 5.10 ~ 5.12）。
依賴：price.parquet
輸出：data/swing_features.parquet
用法：python build_swing_features.py [--full]
"""
import logging
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 專案路徑一律走 code/paths.py（唯一來源），不要各檔自行推導
from engine.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402


WARMUP = 260   # 計算趨勢線需要足夠歷史


# ── 工具函數 ──────────────────────────────────────────────────────────────────

def _local_highs(close: np.ndarray, window: int = 5) -> np.ndarray:
    """回傳局部高點索引（5-bar 最大值）。"""
    n = len(close)
    is_high = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        seg = close[i - window: i + window + 1]
        if close[i] == seg.max() and close[i] > close[i - 1]:
            is_high[i] = True
    return np.where(is_high)[0]


def _local_lows(close: np.ndarray, window: int = 5) -> np.ndarray:
    """回傳局部低點索引（5-bar 最小值）。"""
    n = len(close)
    is_low = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        seg = close[i - window: i + window + 1]
        if close[i] == seg.min() and close[i] < close[i - 1]:
            is_low[i] = True
    return np.where(is_low)[0]


def _vol_highs(close: np.ndarray, vol: np.ndarray,
               vol_ma60: np.ndarray, window: int = 2) -> np.ndarray:
    """大量（量比 > 2）的局部高點索引。"""
    vol_ratio = vol / np.where(vol_ma60 > 0, vol_ma60, np.nan)
    high_idxs = _local_highs(close, window)
    return np.array([i for i in high_idxs
                     if not np.isnan(vol_ratio[i]) and vol_ratio[i] >= 2.0])


def _vol_lows(close: np.ndarray, vol: np.ndarray,
              vol_ma60: np.ndarray, window: int = 2) -> np.ndarray:
    """大量的局部低點索引。"""
    vol_ratio = vol / np.where(vol_ma60 > 0, vol_ma60, np.nan)
    low_idxs = _local_lows(close, window)
    return np.array([i for i in low_idxs
                     if not np.isnan(vol_ratio[i]) and vol_ratio[i] >= 2.0])


def _trendline(idxs: np.ndarray, prices: np.ndarray,
               current_i: int, lookback: int = 60, lag: int = 0) -> dict:
    """
    用最近 lookback 內的局部點做線性回歸趨勢線。
    回傳 dist（與趨勢線距離%）、slope（斜率/close）、r2、touches（觸及次數）。

    lag：idxs 裡的點需要未來幾天才能被確認為高/低點，只取已確認的點。
    """
    # 只取 lookback 內、且已被確認（idx + lag <= current_i）的點
    recent = idxs[(idxs >= current_i - lookback) & (idxs <= current_i - lag)]
    if len(recent) < 2:
        return {"dist": np.nan, "slope": np.nan, "r2": np.nan, "touches": 0, "days": 0}

    x = recent.astype(float)
    y = prices[recent]
    slope, intercept, r, _, _ = scipy_stats.linregress(x, y)
    predicted = slope * current_i + intercept
    close_now = prices[current_i]
    # 2026-08-01 修正：原本除以現價，股價極低時分母趨近零（實測 support_dist
    # 最低到 -1.1e4）。改除以趨勢線預測值 ＝「現價偏離趨勢線幾 %」，
    # 這也是「距離支撐/壓力多遠」的標準定義。
    # 2026-08-01 二次修正：上一版改除以 predicted，但線性外推到低價區時
    # predicted 會趨近零甚至為負，反而讓 support_dist 從 4,804 惡化到 75,285。
    # 加上「預測值至少為現價的 10%」的下限，並限制偏離幅度。
    _ok = np.isfinite(predicted) and predicted >= close_now * 0.1
    dist = (close_now - predicted) / predicted if _ok else np.nan
    if _ok and abs(dist) > 5.0:
        dist = np.nan
    slope_norm = slope / predicted if _ok else np.nan
    days_span = int(current_i - recent[0])

    # 觸及次數：所有 recent 索引處的真實價格與趨勢線偏差 < 1%
    touch_count = 0
    for idx in recent:
        pred_i = slope * idx + intercept
        if prices[idx] != 0 and abs(prices[idx] - pred_i) / prices[idx] < 0.01:
            touch_count += 1

    return {
        "dist":    float(dist),
        "slope":   float(slope_norm),
        "r2":      float(r ** 2),
        "touches": int(touch_count),
        "days":    days_span,
    }


# ── 逐支股票計算 ──────────────────────────────────────────────────────────────

def _compute_stock(grp: pd.DataFrame) -> pd.DataFrame:
    grp = grp.sort_values("date").reset_index(drop=True)
    n = len(grp)

    cl  = grp["close"].values.astype(float)
    hi  = grp["high"].values.astype(float)
    lo  = grp["low"].values.astype(float)
    vol = grp["volume"].values.astype(float)
    vol_ma60 = pd.Series(vol).rolling(60, min_periods=10).mean().values

    # ── 前波高低點（5.10）────────────────────────────────────────────────
    high_idxs = _local_highs(cl, window=5)
    low_idxs  = _local_lows(cl, window=5)

    # 大量高低點（5.11）
    vhigh_idxs = _vol_highs(cl, vol, vol_ma60, window=2)
    vlow_idxs  = _vol_lows(cl,  vol, vol_ma60, window=2)

    # 初始化輸出矩陣
    nan = np.full(n, np.nan, dtype="float32")
    out = {
        # 5.10 前波高低點（前 3 個）
        "high1_dist": nan.copy(), "high1_days": nan.copy(),
        "high2_dist": nan.copy(), "high2_days": nan.copy(),
        "high3_dist": nan.copy(), "high3_days": nan.copy(),
        "low1_dist":  nan.copy(), "low1_days":  nan.copy(),
        "low2_dist":  nan.copy(), "low2_days":  nan.copy(),
        "low3_dist":  nan.copy(), "low3_days":  nan.copy(),
        # 5.11 大量高低點（前 3 個）
        "vh1_dist": nan.copy(), "vh1_days": nan.copy(), "vh1_strength": nan.copy(),
        "vh2_dist": nan.copy(), "vh2_days": nan.copy(),
        "vh3_dist": nan.copy(), "vh3_days": nan.copy(),
        "vl1_dist": nan.copy(), "vl1_days": nan.copy(), "vl1_strength": nan.copy(),
        "vl2_dist": nan.copy(), "vl2_days": nan.copy(),
        "vl3_dist": nan.copy(), "vl3_days": nan.copy(),
        "is_above_vh1": np.zeros(n, "int8"),
        "is_below_vl1": np.zeros(n, "int8"),
        # 5.12 趨勢線
        "support_dist":    nan.copy(), "support_slope": nan.copy(),
        "support_r2":      nan.copy(), "support_touches": nan.copy(),
        "support_days":    nan.copy(),
        "resist_dist":     nan.copy(), "resist_slope":  nan.copy(),
        "resist_r2":       nan.copy(), "resist_touches": nan.copy(),
        "resist_days":     nan.copy(),
        "is_triangle":     np.zeros(n, "int8"),
        "slope_align":     np.zeros(n, "int8"),
    }

    def _prev_k(arr, i, k=3, lag=0):
        """arr 中已被確認（idx + lag <= i）的最後 k 個。

        lag 必須等於判定該 idx 是否為波段高/低點所需的未來窗口大小，
        否則會用到「這天當下還不知道」的未來資料（look-ahead）。
        """
        candidates = arr[arr <= i - lag]
        return candidates[-k:] if len(candidates) >= 1 else np.array([])

    for i in range(WARMUP, n):
        c = cl[i]
        if c <= 0:
            continue

        # ── 前波高低點 ───────────────────────────────────────────────────
        # high_idxs/low_idxs 需要未來 5 天才能確認，lag=5 避免用到未來資料
        # 2026-08-01 修正（兩處）：
        #   1. 原本除以「當前股價 c」，股價從高點崩跌時分母趨近零，實測
        #      high1_dist 最低到 -13,378（上界卻只有 0.99，嚴重不對稱）。
        #      改除以基準價 ＝ 自該高低點以來的報酬率，下界自然有界於 -1。
        #   2. 基準價原本取轉折日的「收盤價」，但既然叫前波「高點」，壓力位在
        #      技術分析上看的是最高價，故改用 hi[idx]；低點對應改用 lo[idx]。
        ph = _prev_k(high_idxs, i, 3, lag=5)
        for k, idx in enumerate(reversed(ph)):
            _base = hi[idx]
            dist = (c - _base) / _base if _base > 0 else np.nan
            days = float(np.log1p(i - idx))
            out[f"high{k+1}_dist"][i] = dist
            out[f"high{k+1}_days"][i] = days

        pl = _prev_k(low_idxs, i, 3, lag=5)
        for k, idx in enumerate(reversed(pl)):
            _base = lo[idx]
            dist = (c - _base) / _base if _base > 0 else np.nan
            days = float(np.log1p(i - idx))
            out[f"low{k+1}_dist"][i] = dist
            out[f"low{k+1}_days"][i] = days

        # ── 大量高低點 ───────────────────────────────────────────────────
        # vhigh_idxs/vlow_idxs 需要未來 2 天才能確認，lag=2
        pvh = _prev_k(vhigh_idxs, i, 3, lag=2)
        for k, idx in enumerate(reversed(pvh)):
            # 2026-08-01 修正：與 high/low_dist 同一個問題 —— 原本除以現價，
            # 股價崩跌時分母趨近零（實測 vh1_dist 最低到 -2.8e4）。改除以基準價。
            _b = hi[idx]
            dist = (c - _b) / _b if _b > 0 else np.nan
            days = float(np.log1p(i - idx))
            str_ = vol[idx] / vol_ma60[idx] if vol_ma60[idx] > 0 else np.nan
            out[f"vh{k+1}_dist"][i] = dist
            out[f"vh{k+1}_days"][i] = days
            if k == 0:
                out["vh1_strength"][i] = str_
                out["is_above_vh1"][i] = int(c > cl[idx])

        pvl = _prev_k(vlow_idxs, i, 3, lag=2)
        for k, idx in enumerate(reversed(pvl)):
            # 2026-08-01 修正：與 high/low_dist 同一個問題 —— 原本除以現價，
            # 股價崩跌時分母趨近零（實測 vl1_dist 最低到 -2.8e4）。改除以基準價。
            _b = lo[idx]
            dist = (c - _b) / _b if _b > 0 else np.nan
            days = float(np.log1p(i - idx))
            str_ = vol[idx] / vol_ma60[idx] if vol_ma60[idx] > 0 else np.nan
            out[f"vl{k+1}_dist"][i] = dist
            out[f"vl{k+1}_days"][i] = days
            if k == 0:
                out["vl1_strength"][i] = str_
                out["is_below_vl1"][i] = int(c < cl[idx])

        # ── 趨勢線（支撐 / 壓力）────────────────────────────────────────
        sup = _trendline(low_idxs, cl, i, lookback=120, lag=5)
        res = _trendline(high_idxs, cl, i, lookback=120, lag=5)

        out["support_dist"][i]    = sup["dist"]
        out["support_slope"][i]   = sup["slope"]
        out["support_r2"][i]      = sup["r2"]
        out["support_touches"][i] = sup["touches"]
        out["support_days"][i]    = float(np.log1p(sup["days"]))

        out["resist_dist"][i]     = res["dist"]
        out["resist_slope"][i]    = res["slope"]
        out["resist_r2"][i]       = res["r2"]
        out["resist_touches"][i]  = res["touches"]
        out["resist_days"][i]     = float(np.log1p(res["days"]))

        # 三角形（支撐上升 + 壓力下降）
        if not np.isnan(sup["slope"]) and not np.isnan(res["slope"]):
            out["is_triangle"][i] = int(sup["slope"] > 0 and res["slope"] < 0)
            out["slope_align"][i] = int(
                (sup["slope"] > 0 and res["slope"] > 0) or
                (sup["slope"] < 0 and res["slope"] < 0)
            )

    result_df = pd.DataFrame(out, index=grp.index)
    result_df.insert(0, "date",     grp["date"])
    result_df.insert(1, "stock_id", grp["stock_id"])
    return result_df


# ── I/O ───────────────────────────────────────────────────────────────────────

def _read(name: str) -> pd.DataFrame:
    p = DATA_DIR / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _upsert(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read("swing_features")
    combined = pd.concat([existing, df], ignore_index=True) if not existing.empty else df
    combined = (combined
                .drop_duplicates(subset=["date", "stock_id"], keep="last")
                .sort_values(["date", "stock_id"])
                .reset_index(drop=True))
    combined.to_parquet(DATA_DIR / "swing_features.parquet", index=False, engine="pyarrow")
    logger.info(f"swing_features.parquet：{len(combined)} 筆 × {len(combined.columns)} 欄")


# ── entry point ───────────────────────────────────────────────────────────────

def run(full: bool = False) -> pd.DataFrame:
    price = _read("price")
    if price.empty:
        raise RuntimeError("price.parquet 不存在")

    price["date"] = pd.to_datetime(price["date"])
    price = price.sort_values(["stock_id", "date"]).reset_index(drop=True)

    existing = _read("swing_features")
    if full or existing.empty:
        logger.info("全量計算 swing/trendline 特徵...")
        target = price
        new_dates = None
    else:
        existing["date"] = pd.to_datetime(existing["date"])
        done = set(existing["date"].dt.normalize().unique())
        new_dates = set(price["date"].dt.normalize().unique()) - done
        if not new_dates:
            logger.info("swing_features.parquet 已是最新")
            return pd.DataFrame()
        min_new = min(new_dates)
        warmup_start = min_new - pd.Timedelta(days=500)
        target = price[price["date"] >= warmup_start]
        logger.info(f"增量計算 {len(new_dates)} 天")

    stocks = target["stock_id"].unique()
    logger.info(f"共 {len(stocks)} 支股票（每支需 {WARMUP} 天 warmup，較慢）...")

    results = []
    for i, sid in enumerate(stocks):
        grp = target[target["stock_id"] == sid]
        try:
            feat = _compute_stock(grp)
            if new_dates is not None:
                feat = feat[feat["date"].dt.normalize().isin(new_dates)]
            results.append(feat)
        except Exception as e:
            logger.warning(f"{sid}: {e}")

        if (i + 1) % 100 == 0:
            logger.info(f"  {i+1}/{len(stocks)}")

    if not results:
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)
    _upsert(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run(full=args.full)
