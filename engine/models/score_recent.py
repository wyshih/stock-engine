"""對最近這段新資料做即時推論，補上訓練時分數檔沒涵蓋的日期。

為什麼需要這一支：`finalists.py` 存下的分數檔（`bundle.score_path()`）在訓練
當下就固定了，只涵蓋 val_sel / test / test2 / test3 的日期區間。之後每天抓進來
的新資料沒有分數，前端的「個股歷史預測曲線」與「累積達標天數」就會停在測試期
的最後一天。今日推薦頁走的是 bundle 即時推論所以不受影響，但那兩處要靠這份。

輸出：data/score_live_<model_key>.parquet（10 個模型各一份），欄位 date /
stock_id / score，與訓練期分數檔同 schema，前端直接接起來用。

增量：已經算過的日期會跳過，所以每日更新只會算新的那一天（約 3 秒）。
首次執行要補 --days 天（60 天約 3 分鐘）。

已驗證：這裡算出來的分數與訓練期分數檔在重疊日期**逐列相同**（max|diff| = 0）。
註：訓練期分數檔只算 label 非空的股票，這裡算全市場，列數會多一些 —— 推論時
看不到未來的 label，全市場才是對的口徑。

用法：
  python code/models/score_recent.py            # 補最近 60 個交易日
  python code/models/score_recent.py --days 120
  python code/models/score_recent.py --rebuild  # 不管既有檔案，整段重算
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd


from engine.models import bundle as bundle_mod  # noqa: E402
from engine.paths import DATA_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DAYS = 60


def existing_dates(key: str) -> set[pd.Timestamp]:
    path = bundle_mod.live_score_path(key)
    if not path.exists():
        return set()
    return set(pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"]).unique())


def target_dates(feat: pd.DataFrame, days: int) -> list[pd.Timestamp]:
    """特徵檔裡最後 `days` 個交易日。"""
    all_dates = sorted(feat["date"].unique())
    return list(all_dates[-days:])


def score_one_date(feat: pd.DataFrame, date: pd.Timestamp,
                   keys: list[str]) -> dict[str, pd.DataFrame]:
    """一天 → {model_key: DataFrame[stock_id, score]}。"""
    return {key: bundle_mod.score_for_date(key, feat, date) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"往回補幾個交易日（預設 {DEFAULT_DAYS}）")
    parser.add_argument("--features", default=str(DATA_DIR / "features.parquet"))
    parser.add_argument("--rebuild", action="store_true",
                        help="忽略既有分數檔，整段重算")
    args = parser.parse_args()

    keys = bundle_mod.available_keys()
    if not keys:
        logger.error("models/ 底下沒有任何 bundle，請先執行 `make train`")
        sys.exit(1)

    feat = pd.read_parquet(args.features)
    feat["date"] = pd.to_datetime(feat["date"])
    dates = target_dates(feat, args.days)
    logger.info(f"模型 {len(keys)} 個，候選日期 {dates[0].date()} ~ {dates[-1].date()}"
                f"（{len(dates)} 個交易日）")

    done = {} if args.rebuild else {k: existing_dates(k) for k in keys}
    todo = [d for d in dates if any(d not in done.get(k, set()) for k in keys)]
    if not todo:
        logger.info("分數已是最新，不用重算")
        return
    logger.info(f"要算 {len(todo)} 天：{todo[0].date()} ~ {todo[-1].date()}")

    started = time.time()
    rows: dict[str, list[pd.DataFrame]] = {k: [] for k in keys}
    for i, date in enumerate(todo, 1):
        for key, frame in score_one_date(feat, date, keys).items():
            if frame.empty:
                continue
            rows[key].append(frame.assign(date=date)[["date", "stock_id", "score"]])
        if i % 10 == 0 or i == len(todo):
            logger.info(f"  {i}/{len(todo)} 天　{time.time() - started:.0f}s")

    for key in keys:
        if not rows[key]:
            continue
        fresh = pd.concat(rows[key], ignore_index=True)
        path = bundle_mod.live_score_path(key)
        if path.exists() and not args.rebuild:
            old = pd.read_parquet(path)
            old["date"] = pd.to_datetime(old["date"])
            # 新算的優先：同一天重算過就以新的為準
            fresh = pd.concat([old[~old["date"].isin(fresh["date"])], fresh],
                              ignore_index=True)
        fresh = fresh.sort_values(["stock_id", "date"]).reset_index(drop=True)
        fresh.to_parquet(path, index=False)
        logger.info(f"  {key:12s} {len(fresh):>9,} 列 → {path.name}")

    logger.info(f"完成，共 {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
