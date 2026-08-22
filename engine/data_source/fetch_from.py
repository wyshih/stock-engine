"""印出「該從哪一天開始抓」，給 Makefile 的 data target 用。

為什麼需要：各 fetch_*.py 的 `--date` 是單日模式，`make update` 若只抓當天，
中間沒跑到的日子就會留洞（2026-08-14 實際發生過：7/31 之後直接跳到 8/13，
中間 8 個交易日全缺）。改成從資料裡的最後一天接著抓到今天，隔多久沒跑都能補齊。

輸出一行 YYYY-MM-DD。price.parquet 不存在時回退到 FALLBACK_START。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from engine.paths import DATA_DIR  # noqa: E402

# 沒有任何價格資料時的起點（專案資料從 2019 起，見 doc/PLAN.md）
FALLBACK_START = "2019-01-01"
# 從起點再往前抓幾天。盤中跑過一次的話那天可能是不完整的，
# 往前重抓一段覆蓋掉（各 fetcher 都是 upsert，重複抓無害）。
OVERLAP_DAYS = 3
# 相鄰兩個有資料的日期相隔超過幾天就視為斷層。正常週末是 3 天（五→一），
# 加上國定假日抓到 5 天；超過就是真的漏抓了。
GAP_DAYS = 5
# 只往回看這麼多天找斷層。不掃全歷史的原因：農曆年休市 9 天會被誤判成斷層，
# 每年都從年初重抓一次沒有意義。近三個月夠涵蓋「幾週沒跑」的情況。
LOOKBACK_DAYS = 90


def trading_dates(name: str = "price") -> pd.DatetimeIndex | None:
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["date"])
    if df.empty:
        return None
    return pd.DatetimeIndex(sorted(pd.to_datetime(df["date"]).unique()))


def find_gaps(dates: pd.DatetimeIndex,
              lookback_days: int = LOOKBACK_DAYS) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """近期資料裡的斷層，回傳 [(斷層前一天, 斷層後一天), ...]。

    `make status` 的健檢也用這支 —— 判定規則只寫這一份，兩邊才不會各自漂移。
    """
    if len(dates) < 2:
        return []
    recent = dates[dates >= dates[-1] - pd.Timedelta(days=lookback_days)]
    if len(recent) < 2:
        return []

    diffs = recent.to_series().diff().dt.days
    holes = recent[diffs > GAP_DAYS]
    return [(recent[recent < h][-1], h) for h in holes if len(recent[recent < h])]


def start_from(dates: pd.DatetimeIndex) -> pd.Timestamp:
    """該從哪天開始抓：近期第一個斷層的前緣，沒有斷層就從最後一天接著抓。"""
    gaps = find_gaps(dates)
    return gaps[0][0] if gaps else dates[-1]


def main() -> None:
    # --last：印出資料裡最新的交易日。給 validate_data 用 —— 早上開盤前跑
    # `make update` 時「今天」根本還沒有資料，拿今天去驗一定失敗，整條流程會斷
    want_last = "--last" in sys.argv

    dates = trading_dates()
    if dates is None:
        print(FALLBACK_START)
        return
    if want_last:
        print(dates[-1].date())
        return
    print((start_from(dates) - pd.Timedelta(days=OVERLAP_DAYS)).date())


if __name__ == "__main__":
    main()
