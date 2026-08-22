"""列出 data/ 底下各 parquet 的列數與最新日期。

供 `make status` 使用 —— 每日更新後一眼看出哪個資料源沒跟上。
與 `validate_data.py` 的差別：那支驗**單日**的正確性（OHLC 不變式、缺值率、
price/chip 列數比），這支看**跨檔案的新鮮度與連續性**。只讀不寫。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from engine.data_source.fetch_from import find_gaps  # noqa: E402
from engine.paths import DATA_DIR  # noqa: E402


def is_frozen(name: str) -> bool:
    """訓練期產物：訓練當下就固定，日期停在該切分的結尾是正常的。

    這些不該算進新鮮度檢查 —— 全部列出來還標「落後 1808 天」，
    100 多行會把真正該看的每日資料洗掉。`score_live_*` 例外，那是每日更新的。
    """
    if name == "shape_features.parquet":
        return True   # 2026-08-05 因資料洩漏停用，不再重算（見 build_features.py）
    return (name.startswith(("score_", "sigtrades_"))
            and not name.startswith("score_live_"))


def main() -> None:
    daily, frozen, gaps = [], [], []
    for path in sorted(DATA_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(path, columns=["date"])
        except Exception:
            continue  # 沒有 date 欄的檔案跳過
        dates = pd.to_datetime(df["date"])
        row = (path.name, len(df), dates.max().date())
        if is_frozen(path.name):
            frozen.append(row)
            continue
        daily.append(row)
        # 只看最新日期會漏掉「中間缺一段」—— 2026-08 就是這樣：所有檔案都顯示
        # 最新 8/13 一切正常，實際上 8/3~8/12 共 8 個交易日全缺，兩週後才發現
        found = find_gaps(pd.DatetimeIndex(sorted(dates.unique())))
        gaps += [(path.name, a, b) for a, b in found]

    if not daily:
        print("data/ 底下沒有帶 date 欄的 parquet")
        return

    newest = max(r[2] for r in daily)
    print(f"每日更新的資料（最新交易日 {newest}）")
    for name, n, last in daily:
        lag = (newest - last).days
        flag = "" if lag == 0 else f"   ← 落後 {lag} 天"
        print(f"  {name:36s} {n:>10,} 列   最新 {last}{flag}")

    if frozen:
        print(f"\n訓練期產物 {len(frozen)} 個檔案（日期固定在各切分結尾，不用更新）")

    if gaps:
        print(f"\n⚠️  近期有斷層（{len(gaps)} 處）—— 跑 `make update` 會自動補齊")
        for name, before, after in gaps:
            missing = (after - before).days - 1
            print(f"  {name:36s} {before.date()} → {after.date()}"
                  f"（中間缺 {missing} 個日曆日）")
    else:
        print("\n✅ 近期沒有斷層")


if __name__ == "__main__":
    main()
