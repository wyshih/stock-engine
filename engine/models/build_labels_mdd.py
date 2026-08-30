"""`label_mdd10`：在 `label_up20` 之上再要求「持有期間不曾跌破 −10%」。

為什麼要這個標的：`label_up20` 只數未來 20 個交易日的上漲天數，**完全不看路徑**。
4739 在 2026-07-20 有 13/20 天上漲、第 20 天 +12.33%，順利達標 —— 但期間最低
收盤是 −13.95%，實際持有過程要先被套住兩週。模型因此學到「剛崩跌就買」是好的，
卻分不出「跌完直接彈」與「跌完還會再破底」。

規則（與 `build_labels.py` 同為**收盤制**，不混用盤中低點）：

    label_mdd10 = 1  ⟺  label_up20 == 1  且  未來 20 日最低收盤 / 當日收盤 - 1 > -0.10

原本標為 1、但期間跌破 −10% 的樣本改標成 **0**（不是整列排除）—— 那些是模型
應該學會避開的反例，丟掉的話等於沒教到。這點與 `label_nobear` 不同，後者是
整列排除，因為那些列的**問題在於不該進場**，而非答案是否定的。

驗收：正例率應從 39.6% 降到約 36.1%（少掉約 8.8% 的正例）。偏離太多要先查
`price.parquet` 是不是變了。

用法：
  python -m engine.models.build_labels_mdd
  python -m engine.models.build_labels_mdd --max-drawdown 0.07
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from engine.paths import DATA_DIR

logger = logging.getLogger(__name__)

SOURCE_LABEL = "label_up20"
OUTPUT_LABEL = "label_mdd10"
HOLD_BARS = 20
DEFAULT_MAX_DRAWDOWN = 0.10


def forward_min_close(price: pd.DataFrame, bars: int = HOLD_BARS) -> pd.Series:
    """未來 `bars` 個交易日的最低收盤 / 當日收盤 - 1。

    與 `build_labels.py` 的前瞻視窗定義一致：看的是 t+1 ~ t+bars，不含當日。
    """
    frame = price.sort_values(["stock_id", "date"])
    grouped = frame.groupby("stock_id")["close"]
    ahead = pd.concat([grouped.shift(-k) for k in range(1, bars + 1)], axis=1)
    return ahead.min(axis=1) / frame["close"] - 1


def build(labels: pd.DataFrame, price: pd.DataFrame,
          max_drawdown: float = DEFAULT_MAX_DRAWDOWN) -> pd.DataFrame:
    frame = price[["date", "stock_id", "close"]].sort_values(["stock_id", "date"])
    frame["mae"] = forward_min_close(frame)

    merged = labels.merge(frame[["date", "stock_id", "mae"]],
                          on=["date", "stock_id"], how="inner")
    merged = merged.dropna(subset=[SOURCE_LABEL, "mae"])

    hit = merged[SOURCE_LABEL].astype("int8").eq(1)
    survived = merged["mae"] > -abs(max_drawdown)
    merged[OUTPUT_LABEL] = (hit & survived).astype("int8")

    demoted = int((hit & ~survived).sum())
    logger.info(
        f"{len(merged):,} 列：原正例 {int(hit.sum()):,}"
        f"（{hit.mean():.1%}）→ 扣掉跌破 {max_drawdown:.0%} 的 {demoted:,} 筆"
        f"（{demoted / max(int(hit.sum()), 1):.1%}）"
        f"→ 新正例率 {merged[OUTPUT_LABEL].mean():.1%}")
    return merged[["date", "stock_id", OUTPUT_LABEL]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DATA_DIR / "labels.parquet")
    parser.add_argument("--price", type=Path, default=DATA_DIR / "price.parquet")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "labels_mdd10.parquet")
    parser.add_argument("--max-drawdown", type=float, default=DEFAULT_MAX_DRAWDOWN)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    labels = pd.read_parquet(args.labels, columns=["date", "stock_id", SOURCE_LABEL])
    labels["date"] = pd.to_datetime(labels["date"])
    price = pd.read_parquet(args.price, columns=["date", "stock_id", "close"])
    price["date"] = pd.to_datetime(price["date"])

    out = build(labels, price, args.max_drawdown)
    out.to_parquet(args.out, index=False)
    logger.info(f"{len(out):,} 列 → {args.out}")


if __name__ == "__main__":
    main()
