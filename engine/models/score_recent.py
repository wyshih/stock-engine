"""對最近這段新資料做即時推論，補上訓練時分數檔沒涵蓋的日期。

為什麼需要這一支：`finalists.py` 存下的分數檔（`bundle.score_path()`）在訓練
當下就固定了，只涵蓋 val_sel / test / test2 / test3 的日期區間。之後每天抓進來
的新資料沒有分數，前端的「個股歷史預測曲線」與「累積達標天數」就會停在測試期
的最後一天。今日推薦頁走的是 bundle 即時推論所以不受影響，但那兩處要靠這份。

輸出：data/score_live_<model_key>.parquet（每個模型各一份），欄位 date /
stock_id / score，與訓練期分數檔同 schema，前端直接接起來用。

增量：已經算過的日期會跳過，所以每日更新只會算新的那一天（約 3 秒）。
首次執行要補 --days 天（60 天約 3 分鐘）。

已驗證：這裡算出來的分數與訓練期分數檔在重疊日期**逐列相同**（max|diff| = 0）。
註：訓練期分數檔只算 label 非空的股票，這裡算全市場，列數會多一些 —— 推論時
看不到未來的 label，全市場才是對的口徑。

⚠️ 每個模型讀**自己訓練時那一份特徵檔**（bundle 裡的 `features_file`）。
2026-09-02 移除 v3 家族後現行兩個模型都是 features.parquet，但這層間接刻意留著
—— 餵錯的話缺欄會被訓練期中位數靜默補掉，分數不會報錯但整份是錯的。

用法：
  python -m engine.models.score_recent            # 補最近 60 個交易日
  python -m engine.models.score_recent --days 120
  python -m engine.models.score_recent --rebuild  # 不管既有檔案，整段重算
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


def group_by_features(keys: list[str]) -> dict[str, list[str]]:
    """{特徵檔名: [模型代號]}。

    2026-09-02 移除 v3 家族後只剩一群，但分組邏輯保留 —— 這支的坑就是出在
    「對所有模型都餵 `--features` 那一份」：當時 m3/m8 有 41% 的欄位被訓練期
    中位數填掉，寫進 score_live 的分數整份是錯的，而且不報錯。
    問 bundle 自己要哪一份才是對的做法，同一份特徵檔的模型併成一組、只載入一次。
    """
    groups: dict[str, list[str]] = {}
    for key in keys:
        groups.setdefault(bundle_mod.features_file_for_key(key), []).append(key)
    return groups


def score_dates(bundle: dict, feat: pd.DataFrame,
                dates: list[pd.Timestamp]) -> list[pd.DataFrame]:
    """一個模型 × 多個日期 → [DataFrame[date, stock_id, score]]。"""
    frames = []
    for date in dates:
        frame = bundle_mod.score_single(bundle, feat, date)
        if not frame.empty:
            frames.append(frame.assign(date=date)[["date", "stock_id", "score"]])
    return frames


def merge_with_existing(key: str, fresh: pd.DataFrame, rebuild: bool) -> pd.DataFrame:
    """新算的分數併回既有的 score_live 檔（同一天重算過就以新的為準）。"""
    path = bundle_mod.live_score_path(key)
    if rebuild or not path.exists():
        return fresh
    old = pd.read_parquet(path)
    old["date"] = pd.to_datetime(old["date"])
    return pd.concat([old[~old["date"].isin(fresh["date"])], fresh], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"往回補幾個交易日（預設 {DEFAULT_DAYS}）")
    parser.add_argument("--features", default=None,
                        help="覆寫特徵檔（除錯用）。預設依每個 bundle 自己記的那一份")
    parser.add_argument("--rebuild", action="store_true",
                        help="忽略既有分數檔，整段重算")
    args = parser.parse_args()

    keys = bundle_mod.available_keys()
    if not keys:
        logger.error("models/ 底下沒有任何 bundle，請先執行 `make train`")
        sys.exit(1)

    # --features 覆寫時所有模型共用同一份（除錯用；正常路徑一律依 bundle 分組）
    groups = ({Path(args.features).name: keys} if args.features
              else group_by_features(keys))
    logger.info(f"模型 {len(keys)} 個，特徵檔 {len(groups)} 份："
                + "、".join(f"{f}（{len(ks)} 個模型）" for f, ks in groups.items()))

    started = time.time()
    for features_file, group in groups.items():
        path = Path(args.features) if args.features else DATA_DIR / features_file
        logger.info(f"載入 {path.name} → {group}")
        feat = pd.read_parquet(path)
        feat["date"] = pd.to_datetime(feat["date"])
        dates = target_dates(feat, args.days)
        logger.info(f"  候選日期 {dates[0].date()} ~ {dates[-1].date()}"
                    f"（{len(dates)} 個交易日）")

        for key in group:
            done = set() if args.rebuild else existing_dates(key)
            todo = [d for d in dates if d not in done]
            if not todo:
                logger.info(f"  {key:16s} 已是最新，不用重算")
                continue
            logger.info(f"  {key:16s} 要算 {len(todo)} 天："
                        f"{todo[0].date()} ~ {todo[-1].date()}")
            frames = score_dates(bundle_mod.load_by_key(key), feat, todo)
            if not frames:
                continue
            fresh = merge_with_existing(key, pd.concat(frames, ignore_index=True),
                                        args.rebuild)
            fresh = fresh.sort_values(["stock_id", "date"]).reset_index(drop=True)
            out_path = bundle_mod.live_score_path(key)
            fresh.to_parquet(out_path, index=False)
            logger.info(f"  {key:16s} {len(fresh):>9,} 列 → {out_path.name}"
                        f"　{time.time() - started:.0f}s")
        del feat

    logger.info(f"完成，共 {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
