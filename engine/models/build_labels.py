"""
產生 label_up20 ground truth（拆委員會後的唯一正式 label）。
輸出：data/labels.parquet

label_up20：未來 20 個交易日中，收盤價較前一日上漲的天數 >= 10 天記 1，否則
記 0（持平不算上漲，不要求累積報酬達標）。base rate ≈ 0.376。獨立實驗驗證過
（見 ~/stock_committee_experiments/updays/METHOD.md）：RF 在五個測試期的
test AUC 穩定在 0.60~0.64。

2026-08-05 精簡（Part A）：拆掉舊「13 子模型 + Meta 委員會」架構後，
label_T123/buy5-20系列/early_rally/local/trend/meta/persist7/persist10/
M1-M3/C1-C3/F1 等探索性或已棄用的 label 產生器（含 `_market_labels`/
`_chip_labels`/`_revenue_labels`）已一併移除，不再有任何模型使用。
5 個防洩漏原子工具（`_shift_forward`/`_triple_barrier`/`_triple_barrier_persist`/
`_local_extrema_label`/`_trend_scanning_label`）目前雖未被 label_up20 用到，
但刻意完整保留、一字未改——下一階段「label 重新設計」會用到，刪掉會造成
實質損失。

用法：python build_labels.py [--full]
"""
import logging
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 專案路徑一律走 code/paths.py（唯一來源），不要各檔自行推導
from engine.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402


# 增量模式每次要一併重算的尾端交易日數（2026-07-30 新增）。
# 需 >= 標籤的前瞻天數：label_up20 是 20 天，取 40 天留充分緩衝
# （2026-08-05：精簡成單一 label 後，前瞻天數不再是 F1 的 21 天，維持 40
# 不變，緩衝只增不減）。
RECOMPUTE_TAIL_DAYS = 40

def _read(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    p = DATA_DIR / f"{name}.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p, columns=columns) if columns else pd.read_parquet(p)


def _upsert(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read("labels")
    if not existing.empty:
        # 2026-07-26 修正（資料審核發現）：existing 裡可能殘留舊版程式碼已經不再
        # 產生的孤兒欄位（例如 label_buy5_tb_p5 等 6 個從沒被目前程式寫過的欄位），
        # concat 時若照樣保留，會在下次 --full 全部變成 100% NaN 的死欄位。
        # df 每次都代表目前程式碼實際會產出的完整欄位集合，drop 掉不在其中的舊欄位。
        orphan_cols = [c for c in existing.columns if c not in df.columns]
        if orphan_cols:
            logger.warning(f"清除孤兒欄位（目前程式碼已不再產生）：{orphan_cols}")
            existing = existing.drop(columns=orphan_cols)
    combined = pd.concat([existing, df], ignore_index=True) if not existing.empty else df
    combined = (combined
                .drop_duplicates(subset=["date", "stock_id"], keep="last")
                .sort_values(["date", "stock_id"])
                .reset_index(drop=True))
    combined.to_parquet(DATA_DIR / "labels.parquet", index=False, engine="pyarrow")
    logger.info(f"labels.parquet 寫入：{len(combined)} 筆")


# ── 工具 ───────────────────────────────────────────────────────────────────────

def _shift_forward(df: pd.DataFrame, col: str, n: int) -> pd.Series:
    """每支股票獨立 forward shift（未來 n 期）。"""
    return df.groupby("stock_id")[col].shift(-n)


def _triple_barrier(price: pd.DataFrame, entry: pd.Series, ma10: pd.Series,
                     horizon: int, profit: float) -> pd.Series:
    """
    Triple-barrier 標籤（de Prado 式）：從進場後第 1 天（t+2）開始逐日檢查收盤價，
    先觸及停利門檻（收盤 >= entry*(1+profit)）記 1；停損用「連續兩天收盤都跌破
    MA10」才觸發記 0（避免單日正常拉回被誤判成停損，只跌破一天會重置警戒）；
    horizon 天內都沒觸及（vertical barrier）記 0。停利不需連續確認（想要的結果
    不用防呆），停損需要連續確認（會提早出場、容易誤殺真訊號，需要防呆）。

    2026-07-30 修正資料尾端污染（doc/AUDIT_20260728.md §A-1）：
    原本回傳 int8，資料尾端沒有未來收盤價時 `_shift_forward` 是 NaN，而
    `NaN >= profit` 在 pandas 回傳 **False 而不是 NaN**，導致最後 horizon 天
    被靜默標成「沒中」而不是「還不知道」，下游 dropna() 完全攔不到。
    這個 bug 2026-07-26 已為 persist 系列修過，卻在本函式（含 label_meta）復發，
    直接原因是缺少回歸測試——現已補在 tests/test_models/test_build_labels.py。
    改用 pandas 可空整數 Int8 + valid mask：只有「已在 horizon 內判定勝負」或
    「未來資料完整（走到 vertical barrier）」的列才輸出 0/1，其餘標 NA。
    """
    resolved = pd.Series(False, index=price.index)
    below_prev = pd.Series(False, index=price.index)   # 前一天是否已跌破 MA10（警戒中）
    label = pd.Series(0, index=price.index, dtype="int8")
    price_ma10 = price.assign(_ma10=ma10)
    for k in range(2, horizon + 1):
        ck  = _shift_forward(price, "close", k)
        mak = _shift_forward(price_ma10, "_ma10", k)
        ret_k = (ck - entry) / entry
        below_today = ck < mak
        hit_profit = (ret_k >= profit) & (~resolved)
        hit_stop   = below_today & below_prev & (~resolved) & (~hit_profit)
        label = label.where(~hit_profit, 1)
        resolved = resolved | hit_profit | hit_stop
        below_prev = below_today

    # 未在 horizon 內判定、且未來收盤價不齊 → 「還不知道」，必須是 NA 不是 0
    has_full_horizon = _shift_forward(price, "close", horizon).notna()
    return label.astype("Int8").where(resolved | has_full_horizon, pd.NA)


def _triple_barrier_persist(price: pd.DataFrame, entry: pd.Series, ma_stop: pd.Series,
                             horizon: int, profit: float) -> pd.Series:
    """
    Triple-barrier + 持續性確認（2026-07-26 新增，label_meta 30天版專用）：
    跟 `_triple_barrier` 一樣的停利/停損/vertical barrier 邏輯，但停利額外要求
    「觸價當下累積上漲天數 > 已過天數的一半」，避免單日跳空就達標、其餘天數都在
    盤整或下跌的假訊號（跟 label_buy10_persist7 的「持續漲」精神一致）。
    """
    resolved = pd.Series(False, index=price.index)
    below_prev = pd.Series(False, index=price.index)
    label = pd.Series(0, index=price.index, dtype="int8")
    price_ma = price.assign(_ma=ma_stop)
    prev_close = entry
    up_days = pd.Series(0, index=price.index, dtype="int16")
    for k in range(2, horizon + 1):
        ck  = _shift_forward(price, "close", k)
        mak = _shift_forward(price_ma, "_ma", k)
        ret_k = (ck - entry) / entry
        is_up = (ck > prev_close).astype("int16")
        up_days = up_days + is_up
        below_today = ck < mak
        hit_profit = (ret_k >= profit) & (~resolved) & (up_days > k / 2)
        hit_stop   = below_today & below_prev & (~resolved) & (~hit_profit)
        label = label.where(~hit_profit, 1)
        resolved = resolved | hit_profit | hit_stop
        below_prev = below_today
        prev_close = ck

    # 尾端未定列必須是 NA 不是 0，理由同 `_triple_barrier`（2026-07-30 修正）
    has_full_horizon = _shift_forward(price, "close", horizon).notna()
    return label.astype("Int8").where(resolved | has_full_horizon, pd.NA)


def _local_extrema_label(price: pd.DataFrame, ret_5: pd.Series,
                          window: int = 5, profit: float = 0.07) -> pd.Series:
    """
    局部低點標籤（N-Period Min-Max / NPMM）：t 當天收盤是前後 window 天內的
    最低點，才算「轉折點」；再交集買進後 window 天漲幅達門檻，避免把「隨便一天
    買到、後來剛好漲很多」跟「買在真正的局部底部」混為一談。
    """
    rolling_min = price.groupby("stock_id")["close"].transform(
        lambda s: s.rolling(window * 2 + 1, center=True, min_periods=window * 2 + 1).min()
    )
    is_local_min = price["close"] <= rolling_min
    return (is_local_min & (ret_5 >= profit)).astype("int8")


def _trend_scanning_label(closes: list[pd.Series]) -> pd.Series:
    """
    Trend-scanning 標籤（de Prado 式，簡化版）：closes 為 t+1 起連續 n 天的收盤價
    （x=0..n-1）。對每個長度 L=3..n 的子窗口做 y~x 線性回歸，取 |t-stat| 最大的
    窗口，label = 該窗口迴歸斜率的正負號。不預設門檻/固定天數，讓資料自己選出
    最顯著的趨勢窗口。
    """
    n = len(closes)
    best_abs_t = None
    best_slope = None
    for L in range(3, n + 1):
        xs = np.arange(L, dtype="float64")
        Sx, Sxx = xs.sum(), (xs ** 2).sum()
        denom = L * Sxx - Sx ** 2

        y_stack = pd.concat(closes[:L], axis=1)
        y_stack.columns = range(L)
        Sy  = y_stack.sum(axis=1)
        Sxy = y_stack.mul(xs, axis=1).sum(axis=1)

        slope = (L * Sxy - Sx * Sy) / denom
        intercept = (Sy - slope * Sx) / L
        pred = pd.concat([intercept + slope * x for x in xs], axis=1)
        pred.columns = range(L)
        sse = ((y_stack - pred) ** 2).sum(axis=1)

        se_slope = np.sqrt((sse / (L - 2)) / (Sxx - Sx ** 2 / L))
        abs_t = (slope / se_slope.replace(0, np.nan)).abs()

        if best_abs_t is None:
            best_abs_t, best_slope = abs_t, slope
        else:
            better = abs_t > best_abs_t
            best_abs_t = best_abs_t.where(~better, abs_t)
            best_slope = best_slope.where(~better, slope)

    return (best_slope > 0).astype("int8")


# ── Price labels ───────────────────────────────────────────────────────────────

def _price_labels(price: pd.DataFrame) -> pd.DataFrame:
    price = price.sort_values(["stock_id", "date"]).reset_index(drop=True)

    logger.info("計算 forward closes...")
    cN = {k: _shift_forward(price, "close", k) for k in range(1, 21)}

    # label_up20（2026-08-02 新增，UP20 子模型用，2026-08-05 起為唯一正式
    # label）：未來 20 天「上漲天數 >= 10」（過半即可，不要求累積報酬達標）。
    # 這是獨立實驗驗證過的 label，RF 在五個測試期的 test AUC 穩定在
    # 0.60~0.64（見 ~/stock_committee_experiments/updays/METHOD.md）。
    # 資料尾端沒有未來收盤價時，用 valid mask 明確標成 NaN（pandas 可空整數
    # Int8）而非 0，避免被下游當成負樣本吃進訓練/評估（同一批 tb/persist
    # 系列曾踩過的坑，見 doc/AUDIT_20260728.md §A-1 / §C-10 / §C-11）。
    closes_hold20 = [price["close"]] + [cN[k] for k in range(1, 21)]
    up_days20 = None
    for k in range(1, 21):
        is_up = (closes_hold20[k] > closes_hold20[k - 1]).astype("int8")
        up_days20 = is_up if up_days20 is None else up_days20 + is_up
    valid_20 = pd.concat(closes_hold20[1:], axis=1).notna().all(axis=1)
    label_up20 = (up_days20 >= 10).astype("Int8").where(valid_20)

    out = pd.DataFrame({
        "date":       price["date"],
        "stock_id":   price["stock_id"],
        "label_up20": label_up20,
    })
    return out


# ── entry point ───────────────────────────────────────────────────────────────

def run(full: bool = False) -> pd.DataFrame:
    price = _read("price")
    if price.empty:
        raise RuntimeError("price.parquet 不存在")
    price["date"] = pd.to_datetime(price["date"])

    # 增量：找出未完成的日期
    existing = _read("labels")
    if full or existing.empty:
        target_dates = None
    else:
        existing["date"] = pd.to_datetime(existing["date"])
        done = set(existing["date"].dt.normalize().unique())
        all_dates = set(price["date"].dt.normalize().unique())
        new = all_dates - done

        # 2026-07-30 修正（doc/AUDIT_20260728.md §C-10 的「未爆引信」）：
        # 每天新增的那一天，當下必然缺少未來 horizon 天的收盤價，標籤只能是
        # 「未定」；若之後永不重算，整份 labels.parquet 會被逐日蠶食成假陰性
        # （`_upsert` 用 keep="last" 保留舊列，舊的未定值會一直留著）。
        # 因此每次增量都要把最近 RECOMPUTE_TAIL_DAYS 個交易日一併重算——
        # 那些日子現在已經有足夠的未來資料可以定案了。
        recent = sorted(done)[-RECOMPUTE_TAIL_DAYS:] if done else []
        target_dates = new | set(recent)
        if not new:
            logger.info("labels.parquet 已是最新（仍重算尾端以定案未定列）")

        price = price[price["date"].dt.normalize().isin(target_dates)]
        logger.info(f"增量補算 {len(new)} 天新資料 + 重算尾端 {len(recent)} 天")

    # ── Price labels ──────────────────────────────────────────────────────
    # 需要完整 price 才能算 forward return（即使只更新部分日期，也要帶上 warmup）
    full_price = _read("price") if target_dates else price
    full_price["date"] = pd.to_datetime(full_price["date"])

    price_lbl = _price_labels(full_price)
    if target_dates:
        price_lbl = price_lbl[price_lbl["date"].dt.normalize().isin(target_dates)]

    logger.info(f"labels：{len(price_lbl)} 筆 × {len(price_lbl.columns)} 欄")
    logger.info(f"正負比（label_up20）：{price_lbl['label_up20'].mean():.3f}")

    _upsert(price_lbl)
    return price_lbl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    df = run(full=args.full)
    if not df.empty:
        label_cols = [c for c in df.columns if c.startswith("label_")]
        print("\n標籤分布：")
        for c in label_cols:
            if c in df.columns:
                pos = df[c].sum()
                total = df[c].notna().sum()
                print(f"  {c:<15} pos={pos:>7,}  total={total:>8,}  rate={pos/total:.3f}" if total else f"  {c}: N/A")
