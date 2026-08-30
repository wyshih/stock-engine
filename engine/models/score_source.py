"""模型分數的**唯一**組裝入口 —— 回測與公開資料包都必須走這裡。

為什麼需要這支：訓練期分數檔（`score_{key}_{split}.parquet`）在訓練當下就固定
了，蓋不到期間最後幾天；`score_recent.py` 產的 live 分數補那一段。但兩份的
**股票宇宙不同**：

  * 訓練期分數檔只算 label 非空的股票；nobear 家族更是「去空頭排列」過濾後的
    子集（某些日子只剩 75 檔）。
  * live 分數是對全市場算的，**不套 nobear 過濾**。

2026-08-27 踩過的坑：匯出程式把 live 直接併進來，m6 在 2026-07-25~31 冒出
1,412 筆訊號，其中 1,342 筆是空頭排列的股票 —— 依定義根本不該被 m6 評分。
同時回測那條路徑沒跟著改，同一個網站兩頁對同一個模型給出差 5~14% 的訊號數。

所以這裡定兩條規矩：

1. **live 只補訓練期分數檔完全沒有的日期**，不補既有日期裡缺的股票。
   （既有日期缺的那些股票是被過濾掉的，不是漏算的。）
2. **nobear 家族的 live 分數要套上同一份去空頭過濾**，規則直接沿用
   `build_labels_nobear` 的 `BEAR_FLAG`，不在這裡另寫一份判斷。
"""

from __future__ import annotations

import logging

import pandas as pd

from engine.models.bundle import live_score_path, load_by_key, score_path
from engine.models.build_labels_nobear import BEAR_FLAG
from engine.paths import DATA_DIR

logger = logging.getLogger(__name__)

NOBEAR_LABEL = "label_nobear"


def _is_nobear(key: str) -> bool:
    """看 bundle 記的 label_name，不從模型代號猜。"""
    return load_by_key(key).get("label_name") == NOBEAR_LABEL


def _load_bear_flags() -> pd.DataFrame:
    flags = pd.read_parquet(DATA_DIR / "features.parquet",
                            columns=["date", "stock_id", BEAR_FLAG])
    flags["date"] = pd.to_datetime(flags["date"])
    return flags


def _drop_bear_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """套用與 `build_labels_nobear` 完全相同的去空頭過濾。"""
    flags = _load_bear_flags()
    merged = frame.merge(flags, on=["date", "stock_id"], how="left")
    is_bear = pd.to_numeric(merged[BEAR_FLAG], errors="coerce").fillna(1.0).gt(0.5)
    return merged.loc[~is_bear, frame.columns].reset_index(drop=True)


def combined_scores(key: str, splits: tuple[str, ...],
                    start: str | None = None,
                    end: str | None = None) -> pd.DataFrame:
    """該模型在指定期間的分數：訓練期分數檔 + live 補未涵蓋的日期。"""
    frames = []
    for split in splits:
        path = score_path(key, split)
        if path.exists():
            part = pd.read_parquet(path)
            part["date"] = pd.to_datetime(part["date"])
            frames.append(part[["date", "stock_id", "score"]])
    if not frames:
        raise SystemExit(
            f"找不到 {key} 的任何分數檔（{', '.join(splits)}）——請先 `make train`")
    trained = pd.concat(frames, ignore_index=True)

    live_path = live_score_path(key)
    if live_path.exists():
        live = pd.read_parquet(live_path)
        live["date"] = pd.to_datetime(live["date"])
        # 規矩 1：只補訓練期完全沒有的日期。
        live = live[~live["date"].isin(set(trained["date"]))]
        # 規矩 2：nobear 家族要套同一份過濾。
        if not live.empty and _is_nobear(key):
            before = len(live)
            live = _drop_bear_rows(live[["date", "stock_id", "score"]])
            logger.info(f"  {key}：live 去空頭過濾 {before:,} → {len(live):,} 列")
        if not live.empty:
            frames.append(live[["date", "stock_id", "score"]])

    out = pd.concat(frames, ignore_index=True)
    if start:
        out = out[out["date"] >= start]
    if end:
        out = out[out["date"] <= end]
    return (out.drop_duplicates(subset=["date", "stock_id"], keep="first")
               .sort_values(["date", "stock_id"]).reset_index(drop=True))
