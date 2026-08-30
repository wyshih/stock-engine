"""
趨勢線特徵 v2（2026-07-29 新增）：修正舊 `support_*`/`resist_*` 的定義問題，
並補上舊版完全沒有的「突破事件」。

## 舊版（build_swing_features.py）的五個定義問題

使用者 2026-07-29 問「這些線的定義跟網路上講的一樣嗎」，逐項核對後發現：

1. **用收盤價，不是高低價**。教科書壓力線連 `high`、支撐線連 `low`。
   舊版全部用 `close` → 盤中戳破壓力但收盤拉回，在舊定義下等於沒發生過。
2. **OLS 回歸線穿過所有點的中間，不是邊界線** ← 最嚴重。
   真正的壓力線是邊界（價格不該穿過），回歸線則大約一半高點在上、一半在下。
   後果：`resist_dist > 0` **不等於突破壓力**，只是「高於近期高點的回歸中線」，
   在盤整期間有一半時間都成立。
3. **兩個點就能成線**（`len(recent) < 2` 才回傳 NaN），而兩點必定 `r2 = 1.0`。
   教科書要求至少 3 個觸點：兩點可以連出任何一條線，第三點才是驗證。
4. **`touches` 循環論證**：數的是「拿來擬合這條線的那些點」離線多近。擬合本來
   就會讓線靠近它們，所以 touches 高只代表擬合得好，不代表被市場獨立測試過。
5. **`is_triangle` 只檢查 `sup.slope > 0 and res.slope < 0`**。標準對稱三角形還要求
   兩線真的收斂到頂點、頂點在合理時間內、且成交量遞減。

## 本模組的修正

1. 壓力線用 `high`、支撐線用 `low`
2. OLS 擬合後**把截距平移到邊界**：壓力線抬到所有樞紐點之上（`+max(residual)`），
   支撐線壓到所有點之下（`+min(residual)`）→ 線變成真正的上/下包絡
3. **至少 3 個已確認樞紐點**才輸出，否則 NaN
4. `touches` 改數**擬合窗口內、但不是樞紐點本身**的獨立回測次數
5. 三角形要求兩線收斂**且頂點落在未來 5~60 根 K 內**，另檢查量能是否遞減

## 舊版做對、本模組必須維持的地方

**`lag=5` 的前視防護**：局部高低點需要未來 5 根 K 才能確認是不是極值，
所以只能取 `idx <= current_i - lag` 的樞紐點。舊版這點寫得正確，
ML 稽核也確認過無洩漏（doc/AUDIT_20260728.md §1.1）。**本模組沿用同一紀律。**

## 舊版完全沒有的：突破「事件」

舊版只有 `*_dist`（狀態量：現在離線多遠），沒有「**昨天在線下、今天站上去**」
這個事件本身。本模組補上 break 事件、突破量能倍數、以及（僅用過去資料判斷的）
假突破計數。

⚠️ 本專案加特徵的成功率是 **0/5**（KMeans 價量形態、Meta 加原始特徵、盤整放量
突破濾網、真訓練 BUY5_2STAGE、自我正規化特徵族）。使用者知情後仍要求進行，
這是第 6 次嘗試。驗收一律用對照組重訓 + `benchmark.py` 的 alpha 中位數。

輸出：`data/trendline_features.parquet`
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 專案路徑一律走 code/paths.py（唯一來源），不要各檔自行推導
from engine.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402


PIVOT_WINDOW = 5      # 局部極值的左右各看幾根
CONFIRM_LAG = 5       # 樞紐點需要幾根未來 K 才算確認（= PIVOT_WINDOW，前視防護）
LOOKBACK = 120        # 趨勢線取多久內的樞紐點
MIN_PIVOTS = 3        # 至少幾個樞紐點才算有效線（修正舊版的 2）
TOUCH_TOL = 0.01      # 觸及判定：距離線 1% 以內
APEX_MIN, APEX_MAX = 5, 60   # 三角形頂點必須落在未來這個區間內
FALSE_BREAK_WINDOW = 20      # 往回看幾天內的假突破次數

INDEX_IDS = {"TWII"}


def _pivot_highs(high: np.ndarray, window: int = PIVOT_WINDOW) -> np.ndarray:
    """局部高點索引：以 high 為準（修正舊版用 close）。"""
    n = len(high)
    out = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        seg = high[i - window: i + window + 1]
        if high[i] == seg.max():
            out[i] = True
    return np.where(out)[0]


def _pivot_lows(low: np.ndarray, window: int = PIVOT_WINDOW) -> np.ndarray:
    """局部低點索引：以 low 為準（修正舊版用 close）。"""
    n = len(low)
    out = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        seg = low[i - window: i + window + 1]
        if low[i] == seg.min():
            out[i] = True
    return np.where(out)[0]


def _boundary_line(idxs: np.ndarray, pivot_px: np.ndarray, bar_px: np.ndarray,
                   current_i: int, side: str,
                   lookback: int = LOOKBACK, lag: int = CONFIRM_LAG) -> dict | None:
    """
    擬合「邊界」趨勢線（修正舊版的中線問題）。

    做法：先對已確認的樞紐點做 OLS，再把截距平移，讓線落在所有樞紐點的
    上緣（壓力）或下緣（支撐）——這才是交易員畫的那條線。

    `pivot_px` 是樞紐點取值用的序列（壓力用 high、支撐用 low），
    `bar_px` 是計算獨立觸及次數用的逐日序列（同上）。

    前視防護：只取 `idx <= current_i - lag` 的樞紐點（需要 lag 根未來 K 才能確認）。
    回傳 None 表示樞紐點不足（< MIN_PIVOTS），呼叫端須填 NaN。
    """
    recent = idxs[(idxs >= current_i - lookback) & (idxs <= current_i - lag)]
    if len(recent) < MIN_PIVOTS:
        return None

    x = recent.astype(float)
    y = pivot_px[recent]
    slope, intercept, r, _, _ = scipy_stats.linregress(x, y)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None

    # 平移到邊界：壓力線抬到所有樞紐之上、支撐線壓到所有樞紐之下
    resid = y - (slope * x + intercept)
    intercept += resid.max() if side == "resist" else resid.min()

    line_now = slope * current_i + intercept
    close_now = bar_px[current_i]
    if close_now == 0 or not np.isfinite(close_now):
        return None

    # 獨立觸及次數（修正舊版的循環論證）：數的是擬合窗口內、
    # **不是樞紐點本身**的那些 bar，有幾根曾回來測試這條線
    lo = max(0, current_i - lookback)
    js = np.arange(lo, current_i + 1)
    line_j = slope * js + intercept
    with np.errstate(divide="ignore", invalid="ignore"):
        near = np.abs(bar_px[lo:current_i + 1] - line_j) / np.abs(line_j) < TOUCH_TOL
    near &= np.isfinite(line_j) & (line_j != 0)
    near[np.isin(js, recent)] = False   # 排除樞紐點本身（修正舊版的循環論證）
    touches = int(near.sum())

    return {
        # 2026-08-01 修正：原本除以現價，股價極低時分母趨近零（實測
        # tl_support_dist 最低到 -1.1e4）。改除以趨勢線價位，語意為
        # 「現價偏離趨勢線幾 %」。
        # 趨勢線外推到低價區時 line_now 可能趨近零甚至為負，需下限保護；
        # 偏離超過 5 倍代表該趨勢線已無參考價值。
        "dist":     _bounded_dist(close_now, line_now),
        "slope":    float(slope / close_now),
        "r2":       float(r ** 2),
        "touches":  int(touches),
        "n_pivots": int(len(recent)),
        "days":     int(current_i - recent[0]),
        "line":     float(line_now),
        "_slope_raw":     float(slope),
        "_intercept_raw": float(intercept),
    }


# 兩線交會點超過這個根數視為無意義（約兩年交易日），避免近平行時數值爆炸

def _bounded_dist(close_now: float, line_now: float) -> float:
    """現價相對趨勢線的偏離幅度，分母過小或偏離過大時視為無效。"""
    if not np.isfinite(line_now) or line_now < close_now * 0.1:
        return float("nan")
    d = (close_now - line_now) / line_now
    return d if abs(d) <= 5.0 else float("nan")


MAX_APEX_BARS = 500


def _apex_bars(sup: dict, res: dict, current_i: int) -> float:
    """
    兩線交會點**距今**幾根 K（負值代表已經交會過）。無解（兩線平行）回傳 NaN。

    ⚠️ 解出來的 x 是「絕對 bar 索引」，必須減去 current_i 才是距今根數
    （2026-07-29 初版漏減，導致 apex 中位數高達 657、只有 1.6% 落在 5~60 的
    合理區間，`tl_is_triangle` 因此幾乎恆為 0）。
    """
    ds = sup["_slope_raw"] - res["_slope_raw"]
    if abs(ds) < 1e-12:
        return np.nan
    x_cross = (res["_intercept_raw"] - sup["_intercept_raw"]) / ds
    apex = float(x_cross - current_i)
    # 2026-08-01 修正：兩線接近平行時 ds 雖大於 1e-12 但仍極小，apex 會爆到
    # 2.4e8 根 K 棒（實測）。超出 ±MAX_APEX_BARS 的交會點在交易上無意義，設 NaN。
    if not np.isfinite(apex) or abs(apex) > MAX_APEX_BARS:
        return np.nan
    return apex


def _build_one(grp: pd.DataFrame,
               seed: tuple[pd.Timestamp, int] | None = None) -> pd.DataFrame:
    """`seed`：(錨點日期, 該日距上次突破幾根)，用來接續 `tl_days_since_break`。

    這個特徵的回看**沒有上限**（可能是 500 根前的突破），再大的暖身視窗都不保證
    看得到。而且暖身視窗前段的趨勢線本身還沒成形（LOOKBACK=120 根），那段偵測到
    的突破與全量算的不一致 —— 所以錨點取「新日期的前一根」，那一天的值是既有檔案
    裡算好的、可信的，從它往後續算即可。存的是 log1p(gap)，expm1 還原成根數。
    """
    n = len(grp)
    high = grp["high"].to_numpy(float)
    low = grp["low"].to_numpy(float)
    close = grp["close"].to_numpy(float)
    vol = grp["volume"].to_numpy(float)

    vol_ma20 = pd.Series(vol).rolling(20, min_periods=5).mean().to_numpy()

    hi_idx = _pivot_highs(high)
    lo_idx = _pivot_lows(low)

    nan = lambda: np.full(n, np.nan, dtype="float32")  # noqa: E731
    out = {
        "tl_resist_dist": nan(), "tl_resist_slope": nan(), "tl_resist_r2": nan(),
        "tl_resist_touches": nan(), "tl_resist_pivots": nan(),
        "tl_support_dist": nan(), "tl_support_slope": nan(), "tl_support_r2": nan(),
        "tl_support_touches": nan(), "tl_support_pivots": nan(),
        "tl_channel_pos": nan(), "tl_channel_width": nan(),
        "tl_apex_bars": nan(),
        "tl_resist_break": np.zeros(n, "int8"),
        "tl_support_break": np.zeros(n, "int8"),
        "tl_break_vol": nan(),
        "tl_days_since_break": nan(),
        "tl_false_breaks_20d": np.zeros(n, "int8"),
        "tl_is_triangle": np.zeros(n, "int8"),
        "tl_triangle_vol_dry": np.zeros(n, "int8"),
    }

    prev_above_resist = None
    prev_below_support = None
    last_break_i = None
    break_history: list[tuple[int, float]] = []   # (突破的 bar, 突破時的壓力線價)

    start = max(LOOKBACK // 4, PIVOT_WINDOW + CONFIRM_LAG)
    for i in range(start, n):
        res = _boundary_line(hi_idx, high, close, i, "resist")
        sup = _boundary_line(lo_idx, low, close, i, "support")

        if res is not None:
            out["tl_resist_dist"][i] = res["dist"]
            out["tl_resist_slope"][i] = res["slope"]
            out["tl_resist_r2"][i] = res["r2"]
            out["tl_resist_touches"][i] = res["touches"]
            out["tl_resist_pivots"][i] = res["n_pivots"]
        if sup is not None:
            out["tl_support_dist"][i] = sup["dist"]
            out["tl_support_slope"][i] = sup["slope"]
            out["tl_support_r2"][i] = sup["r2"]
            out["tl_support_touches"][i] = sup["touches"]
            out["tl_support_pivots"][i] = sup["n_pivots"]

        # ── 通道位置與寬度（價格在上下軌之間的相對位置）───────────────────
        if res is not None and sup is not None and close[i] != 0:
            width = res["line"] - sup["line"]
            # 2026-08-01 修正：width > 0 不夠，通道極窄時 channel_pos 會爆到 7.3e6。
            # 要求通道寬度至少為股價的 0.1%，否則視為無效通道。
            if width > max(close[i] * 0.001, 1e-8):
                _pos = (close[i] - sup["line"]) / width
                # 2026-08-01 追加：0~1 表示在通道內，超出代表已脫離通道；
                # 但距離超過 5 個通道寬時該通道已無參考價值，設 NaN。
                out["tl_channel_pos"][i] = _pos if abs(_pos) <= 5 else np.nan
                _w = width / close[i]
                out["tl_channel_width"][i] = _w if _w <= 5.0 else np.nan

            apex = _apex_bars(sup, res, i)
            out["tl_apex_bars"][i] = apex
            # 三角形：兩線收斂（支撐上升、壓力下降）且頂點落在合理未來區間
            converging = sup["_slope_raw"] > 0 and res["_slope_raw"] < 0
            if converging and np.isfinite(apex) and APEX_MIN <= apex <= APEX_MAX:
                out["tl_is_triangle"][i] = 1
                # 量能遞減確認（標準三角形的必要條件，舊版沒有）
                if i >= 40 and np.isfinite(vol_ma20[i]) and np.isfinite(vol_ma20[i - 20]):
                    out["tl_triangle_vol_dry"][i] = int(vol_ma20[i] < vol_ma20[i - 20])

        # ── 突破「事件」（舊版完全沒有）──────────────────────────────────
        if res is not None:
            # ⚠️ 一定要 bool()：close[i] 是 np.float64，比較結果是 np.bool_，
            # 而 `np.bool_(False) is False` 會回傳 False（不同物件），
            # 導致下面的 `is False` 永遠不成立、突破事件永遠不觸發
            # （2026-07-29 初版踩到，實測 9 次真實轉換卻記錄到 0 次）。
            above = bool(close[i] > res["line"])
            if prev_above_resist is False and above:
                out["tl_resist_break"][i] = 1
                last_break_i = i
                break_history.append((i, res["line"]))
                if np.isfinite(vol_ma20[i]) and vol_ma20[i] > 0:
                    out["tl_break_vol"][i] = vol[i] / vol_ma20[i]
            prev_above_resist = above
        if sup is not None:
            below = bool(close[i] < sup["line"])   # 同上，必須轉成 Python bool
            if prev_below_support is False and below:
                out["tl_support_break"][i] = 1
            prev_below_support = below

        if last_break_i is not None:
            out["tl_days_since_break"][i] = float(np.log1p(i - last_break_i))

        # 假突破：過去 FALSE_BREAK_WINDOW 根內突破過壓力線、但現在又跌回線下。
        # 只用已發生的資料判斷，無前視。
        out["tl_false_breaks_20d"][i] = sum(
            1 for (bi, bline) in break_history
            if i - bi <= FALSE_BREAK_WINDOW and 0 < bi < i and close[i] < bline
        )

    if seed is not None:
        _reseed_days_since_break(out, grp["date"].to_numpy(), *seed)

    res_df = pd.DataFrame(out)
    res_df.insert(0, "stock_id", grp["stock_id"].values)
    res_df.insert(0, "date", grp["date"].values)
    return res_df


def _reseed_days_since_break(out: dict, dates: np.ndarray,
                             anchor_date: pd.Timestamp, gap: int) -> None:
    """從錨點往後重算 `tl_days_since_break`，蓋掉暖身區推得的不可靠值。"""
    pos = int(np.searchsorted(dates, np.datetime64(anchor_date)))
    if pos >= len(dates) or dates[pos] != np.datetime64(anchor_date):
        return                      # 這檔在錨點那天沒有資料，維持原樣
    # ⚠️ 只有「壓力線突破」會重置計數，支撐跌破不算 —— 這與主迴圈裡
    # `last_break_i = i` 的位置一致（它在 resist 分支內，不在 support 分支）。
    # 初版兩個都算，1583 在 2026-08-27 的支撐跌破就被誤判成重置。
    last = pos - int(gap)
    for i in range(pos, len(dates)):
        if i > pos and out["tl_resist_break"][i]:
            last = i
        out["tl_days_since_break"][i] = np.float32(np.log1p(i - last))


# 增量時要往回讀幾根 K：趨勢線取 LOOKBACK(120) 根內的樞紐點，樞紐點本身還要
# PIVOT_WINDOW + CONFIRM_LAG 根確認，另有 FALSE_BREAK_WINDOW(20) 與 rolling(20)。
# 取 200 根，比最長相依多出一截餘裕。
WARMUP_BARS = 200


def _existing() -> pd.DataFrame:
    path = DATA_DIR / "trendline_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    out = pd.read_parquet(path)
    out["date"] = pd.to_datetime(out["date"])
    return out


def _one(args: tuple) -> pd.DataFrame | None:
    """給行程池用的頂層函式（lambda / closure 不能被 pickle）。"""
    sid, grp, seed_gap = args
    if len(grp) < LOOKBACK // 2:
        return None
    return _build_one(grp.reset_index(drop=True), seed_gap)


def _seed_gaps(existing: pd.DataFrame,
               first_new: pd.Timestamp) -> dict[str, tuple[pd.Timestamp, int]]:
    """每檔在「第一個新日期之前」最後一根的 (日期, 距上次突破根數)。

    錨點刻意取新日期的前一根而不是暖身視窗起點 —— 那一天的值是既有檔案算好的，
    暖身區前段推得的值則不可信（趨勢線還沒成形）。
    """
    if existing.empty or "tl_days_since_break" not in existing.columns:
        return {}
    before = existing[existing["date"] < first_new]
    if before.empty:
        return {}
    last = (before.sort_values("date").groupby("stock_id")
            .agg(d=("date", "last"), v=("tl_days_since_break", "last")))
    last = last[last["v"].notna()]
    return {sid: (r.d, int(round(float(np.expm1(r.v))))) for sid, r in last.iterrows()}


def build(full: bool = False, jobs: int = 0) -> pd.DataFrame:
    """算趨勢線特徵。

    預設是增量的：只重算「有新資料的那幾天」，每檔往回多讀 WARMUP_BARS 根 K 當
    暖身，算完只留新日期。2026-08-29 之前這支沒有增量路徑，每次 `make update`
    都把 2,069 檔 × 全部歷史重算一遍，佔掉整個更新流程四分之一的時間。

    每檔股票彼此獨立，所以用行程池平行（jobs=0 表示自動取 CPU 數的一半，
    留餘裕給其他程式）。
    """
    logger.info("讀取 price.parquet…")
    px = pd.read_parquet(DATA_DIR / "price.parquet",
                         columns=["date", "stock_id", "high", "low", "close", "volume"])
    px["date"] = pd.to_datetime(px["date"])
    px = px[~px["stock_id"].isin(INDEX_IDS)]
    px = px.sort_values(["stock_id", "date"]).reset_index(drop=True)

    new_dates: set | None = None
    if not full:
        existing = _existing()
        if not existing.empty:
            done = set(existing["date"].unique())
            new_dates = set(px["date"].unique()) - done
            if not new_dates:
                logger.info("trendline_features.parquet 已是最新")
                return pd.DataFrame()
            calendar = sorted(px["date"].unique())
            pos = calendar.index(min(new_dates))
            window_start = calendar[max(0, pos - WARMUP_BARS)]
            px = px[px["date"] >= window_start]
            logger.info(f"增量補算 {len(new_dates)} 個交易日，"
                        f"暖身自 {pd.Timestamp(window_start).date()} 起"
                        f"（{WARMUP_BARS} 根 K）")
    if new_dates is None:
        logger.info("全量重算")

    seeds: dict[str, int] = {}
    if new_dates is not None:
        seeds = _seed_gaps(existing, pd.Timestamp(min(new_dates)))
        logger.info(f"  接續 {len(seeds)} 檔的 tl_days_since_break 狀態")
    groups = [(sid, grp, seeds.get(sid)) for sid, grp in px.groupby("stock_id", sort=False)]
    n_jobs = jobs or max(1, (os.cpu_count() or 2) // 2)
    logger.info(f"  {len(groups)} 檔，{n_jobs} 個行程")

    if n_jobs == 1:
        frames = [f for f in map(_one, groups) if f is not None]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            frames = [f for f in pool.map(_one, groups, chunksize=16) if f is not None]

    out = pd.concat(frames, ignore_index=True)
    if new_dates is not None:
        out = out[out["date"].isin(new_dates)]
    n_feat = len([c for c in out.columns if c not in ("date", "stock_id")])
    logger.info(f"完成：{len(out)} 列 × {n_feat} 個特徵")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="不管既有檔案，整段重算")
    parser.add_argument("--jobs", type=int, default=0, help="行程數，0＝自動")
    args = parser.parse_args()

    out = build(full=args.full, jobs=args.jobs)
    if out.empty:
        return
    path = DATA_DIR / "trendline_features.parquet"
    if args.full:
        out.to_parquet(path, index=False)
    else:
        existing = _existing()
        combined = pd.concat([existing, out], ignore_index=True) if not existing.empty else out
        combined = (combined.drop_duplicates(subset=["date", "stock_id"], keep="last")
                            .sort_values(["stock_id", "date"]).reset_index(drop=True))
        combined.to_parquet(path, index=False)
    logger.info(f"已寫入 {path}")

    body = out.drop(columns=["date", "stock_id"])
    logger.info("缺失率：\n" + body.isna().mean().sort_values(ascending=False).round(3).to_string())
    logger.info("突破事件發生率：\n" + body[["tl_resist_break", "tl_support_break",
                                             "tl_is_triangle"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
