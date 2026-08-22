"""`label_nobear`：把「當下處於空頭排列」的樣本整列排除後的 label_up20。

背景（2026-08-22）：這個 label 是 m6~m10 五個模型的 ground truth，但 repo 裡沒有
任何程式產生它 —— `data/labels_nobear.parquet` 是 2026-08-16 在 scratchpad 臨時算的。
換掉資料源要重訓時才發現重建不出來。這支把它固定下來。

定義（來源：doc/BACKTEST_LOG.md #26）：
    label_nobear = label_up20，但「目前空頭排列」的列**整列排除**（不標 0）。

「整列排除」與「標 0」的差別是這個 label 的全部重點：空頭排列的股票之後照樣可能
上漲，把它們標成 0 等於教模型「空頭排列 → 不會漲」這個未必成立的規則；整列排除
則是不對這些情境表態，讓模型只在非空頭的情境裡學。

空頭排列取 `bear_3ma_1d`（短中長三條均線由下而上排列，見 features_v2/price_trend.py）。

驗收基準（2026-08-16 那版的全買基準列數）：up20 693,909 → nobear 503,643，
保留約 72.6%。列數比例明顯偏離時要先查是不是 `bear_3ma_1d` 的定義變了。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


from engine.paths import DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)

SOURCE_LABEL = "label_up20"
OUTPUT_LABEL = "label_nobear"
BEAR_FLAG = "bear_3ma_1d"


def build(labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merged = labels.merge(features[["date", "stock_id", BEAR_FLAG]],
                          on=["date", "stock_id"], how="inner")
    # NA 代表算不出來（暖機期均線還沒成形），那也不該拿來當訓練樣本
    is_bear = pd.to_numeric(merged[BEAR_FLAG], errors="coerce").fillna(1.0).gt(0.5)
    kept = merged.loc[~is_bear, ["date", "stock_id", SOURCE_LABEL]].copy()
    kept = kept.rename(columns={SOURCE_LABEL: OUTPUT_LABEL})
    logger.info(f"原始 {len(merged):,} 列 → 排除空頭排列 {int(is_bear.sum()):,} 列 "
                f"→ 保留 {len(kept):,} 列（{len(kept) / len(merged):.1%}）")
    return kept.dropna(subset=[OUTPUT_LABEL]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DATA_DIR / "labels.parquet")
    parser.add_argument("--features", type=Path, default=DATA_DIR / "features.parquet")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "labels_nobear.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    labels = pd.read_parquet(args.labels, columns=["date", "stock_id", SOURCE_LABEL])
    features = pd.read_parquet(args.features, columns=["date", "stock_id", BEAR_FLAG])
    for frame in (labels, features):
        frame["date"] = pd.to_datetime(frame["date"])

    result = build(labels, features)
    result.to_parquet(args.out, index=False)
    logger.info(f"寫出 {args.out}：{len(result):,} 列，"
                f"正例率 {result[OUTPUT_LABEL].mean():.4f}")


if __name__ == "__main__":
    main()
