"""產生 v3 轉換所需的兩份稽核檔 —— 讓 v3 特徵集可以從 repo 完整重建。

背景（2026-08-22）：`spec.py` 需要 `feature_audit.csv` 與 `volproxy_old.csv`，但
repo 裡沒有任何程式產生它們 —— 當初是在 scratchpad 臨時算的，session 結束就沒了。
結果是 m1~m10 這十個模型**無法重現**：換掉資料源之後想重訓，卻連 v3 特徵集都建不
出來。這支把那一步固定下來。

⚠️ 這是**重建**，不是還原。原始的分類門檻沒有留下紀錄，下面的規則是依 `spec.py`
的說明與各分類的用途重新定義的，因此 v3 欄位集**不保證與 2026-08-16 那版逐欄相同**，
舊實驗的數字不能直接沿用比較。

分類（`category`，決定 `spec.plan_for()` 走哪條路）：
    market_level   大盤層級（`mkt_*`）—— 排名會抹掉「現在是什麼市況」，原樣保留
    binary         只取 0/1 的旗標 —— 本來就跨年可比
    sparse_count   稀疏計數（K 棒型態、觸線次數）—— 多數為 0，排名沒有意義
    self_relative  已經是「跟自己歷史比」的滾動百分位（`rel_pct_*`、`*_rank`）
    drift          逐年尺度漂移過大 —— 原值不可跨年比較，只留 _sz / _szx
    continuous     其餘尺度無關的連續值

`scale_ratio`：逐年 P10~P90 寬度的 max/min。等於 1 代表尺度穩定，越大代表這個特徵
的數值範圍逐年變動越劇烈（例如成交金額類）。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from engine.paths import DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)

KEY_COLS = ("date", "stock_id")
AUDIT_YEARS = (2020, 2022, 2024, 2026)
# 逐年 P10~P90 寬度的 max/min 超過這個倍數，視為尺度逐年漂移
DRIFT_SCALE_RATIO = 2.0
# 稀疏計數的判定：整數值、相異值不多、零佔比高
SPARSE_MAX_UNIQUE = 12
SPARSE_MIN_ZERO_RATE = 0.7
# Spearman 相關的抽樣列數。全量 330 萬列 × 380 欄的秩相關要跑很久，而相關係數
# 這種統計量在 20 萬列上已經穩定到小數點後兩位。
SAMPLE_ROWS = 200_000
SAMPLE_SEED = 20260822
REALIZED_VOL_WINDOW = 20


def _classify(name: str, values: pd.Series, scale_ratio: float) -> str:
    if name.startswith("mkt_"):
        return "market_level"

    clean = values.dropna()
    if clean.empty:
        return "continuous"

    unique = pd.unique(clean)
    if len(unique) <= 2 and set(np.round(unique, 6)).issubset({0.0, 1.0}):
        return "binary"

    is_integer = bool(np.allclose(clean, np.round(clean), atol=1e-9))
    zero_rate = float((clean == 0).mean())
    if is_integer and len(unique) <= SPARSE_MAX_UNIQUE and zero_rate >= SPARSE_MIN_ZERO_RATE:
        return "sparse_count"

    if name.startswith("rel_pct_") or name.endswith("_rank"):
        return "self_relative"

    if np.isfinite(scale_ratio) and scale_ratio > DRIFT_SCALE_RATIO:
        return "drift"
    return "continuous"


def _yearly_stats(frame: pd.DataFrame, column: str) -> dict:
    """逐年中位數與 P10~P90 寬度，以及寬度的 max/min。"""
    stats: dict = {}
    widths = []
    for year in AUDIT_YEARS:
        chunk = frame.loc[frame["year"] == year, column].dropna()
        if len(chunk) < 100:
            stats[f"median_{year}"] = np.nan
            stats[f"width_{year}"] = np.nan
            continue
        low, high = np.percentile(chunk, [10, 90])
        stats[f"median_{year}"] = float(np.median(chunk))
        stats[f"width_{year}"] = float(high - low)
        if high - low > 0:
            widths.append(high - low)
    stats["scale_ratio"] = (max(widths) / min(widths)) if len(widths) >= 2 else 1.0
    return stats


def build_audit(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for position, column in enumerate(columns, start=1):
        stats = _yearly_stats(features, column)
        stats["feature"] = column
        stats["category"] = _classify(column, features[column], stats["scale_ratio"])
        rows.append(stats)
        if position % 50 == 0:
            logger.info(f"  稽核 {position}/{len(columns)} 欄")
    return pd.DataFrame(rows).set_index("feature")


def build_volproxy(features: pd.DataFrame, columns: list[str],
                   price: pd.DataFrame) -> pd.DataFrame:
    """每個特徵與「當日成交金額」「20 日已實現波動」的 Spearman 相關。

    這兩個是「個股身分」的代理變數：跟它們高度相關的特徵，其絕對水位主要在說
    「這是一檔什麼樣的股票」，而不是「現在發生了什麼」。
    """
    price = price.sort_values(["stock_id", "date"]).copy()
    returns = price.groupby("stock_id")["close"].pct_change()
    price["rv20"] = (returns.groupby(price["stock_id"])
                     .rolling(REALIZED_VOL_WINDOW, min_periods=REALIZED_VOL_WINDOW)
                     .std().reset_index(level=0, drop=True))
    merged = features.merge(price[["date", "stock_id", "amount", "rv20"]],
                            on=["date", "stock_id"], how="inner")
    merged = merged.dropna(subset=["amount", "rv20"])
    if len(merged) > SAMPLE_ROWS:
        merged = merged.sample(SAMPLE_ROWS, random_state=SAMPLE_SEED)
    logger.info(f"  相關性抽樣 {len(merged):,} 列")

    amount_rank = merged["amount"].rank()
    vol_rank = merged["rv20"].rank()
    rows = []
    for column in columns:
        series = merged[column]
        if series.notna().sum() < 1000:
            rows.append({"feature": column, "sp_amount": 0.0, "sp_rv20": 0.0})
            continue
        ranked = series.rank()
        rows.append({
            "feature": column,
            "sp_amount": float(ranked.corr(amount_rank)),
            "sp_rv20": float(ranked.corr(vol_rank)),
        })
    return pd.DataFrame(rows).set_index("feature").fillna(0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DATA_DIR / "features.parquet")
    parser.add_argument("--price", type=Path, default=DATA_DIR / "price.parquet")
    parser.add_argument("--audit-out", type=Path, default=DATA_DIR / "feature_audit.csv")
    parser.add_argument("--volproxy-out", type=Path, default=DATA_DIR / "volproxy.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    features = pd.read_parquet(args.features)
    features["date"] = pd.to_datetime(features["date"])
    features["year"] = features["date"].dt.year
    columns = [c for c in features.columns
               if c not in KEY_COLS and c != "year"
               and pd.api.types.is_numeric_dtype(features[c])]
    logger.info(f"特徵 {len(columns)} 欄、{len(features):,} 列")

    audit = build_audit(features, columns)
    audit.to_csv(args.audit_out)
    logger.info(f"寫出 {args.audit_out}")
    logger.info(f"分類分布：{audit['category'].value_counts().to_dict()}")

    price = pd.read_parquet(args.price, columns=["date", "stock_id", "close", "amount"])
    price["date"] = pd.to_datetime(price["date"])
    volproxy = build_volproxy(features, columns, price)
    volproxy.to_csv(args.volproxy_out)
    logger.info(f"寫出 {args.volproxy_out}")
    identity = ((volproxy["sp_rv20"].abs() > 0.30)
                | (volproxy["sp_amount"].abs() > 0.30)).sum()
    logger.info(f"身分編碼型特徵（|Spearman| > 0.30）：{identity} 欄")


if __name__ == "__main__":
    main()
