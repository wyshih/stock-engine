"""用「替換的 ground truth」訓練一個 RF，並產出前端可直接載入的全套檔案。

為什麼需要這一支：`train_bundles.py` 與 `finalists.py` 都綁死 `train_single.LABEL`
（`label_up20`），沒辦法換 label。label 實驗當初是 agent 在 scratchpad 臨時寫的，
session 一結束就沒了，無法重跑也無法排進佇列。這支把它固定下來 —— 之後每個新
label 都是一行指令。
（`train_submodels.py` 是已停用的舊委員會系統，切分與用途都不同，不能重用。）

產出（走 `bundle.py` 的實驗模型命名慣例，檔案就位前端就會自動出現）：
    models/bundle_<key>.pkl
    data/score_<key>_<split>.parquet
    data/sigcurve_<key>_val_sel.csv      ← 由 run_queue.sh 接著用 threshold_curve 產生

⚠️ 刻意的設計：
- 切分預設 **Round 4**（可用 --round 覆寫）—— 不同 label 必須同切分才比得起來
- 超參數沿用該輪 sweep CSV 的 val_sel 最佳組，不重新搜尋（要比的是 label，不是調參）
- 特徵集走 `train_single.load_data()`（內部用 `submodel_config.feature_cols()`），
  不可直接用 features.parquet 全欄
- 前處理統計量只從訓練期算
- label 為 NA 的列一律剔除，不可當負例（未定的未來視窗補 0 是本專案修過的 bug）
- **不挑門檻、不下結論**：只訓練與存分數，門檻由人看曲線決定（專案規則）

用法：
  python code/models/train_label_variant.py \
      --label-file data/labels_ab.parquet --label-col label_A \
      --key labelA_r2 --desc "最大漲幅>1.5σ（Round 2）"
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


from engine.models.sweep_config import best_config  # noqa: E402
from engine.paths import DATA_DIR, MODEL_DIR  # noqa: E402
from engine.models.sweep_round1 import build_fitted  # noqa: E402
from engine.models.train_single import (  # noqa: E402
    eval_splits, evaluate, fit_stats, load_data, preprocess,
    split_frame, splits, use_round,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 預設 Round 4（train 2020-01~2023-11 / val 2024 / test 2025-02~2026-07）。
# 2026-08-15 使用者要求「全部模型都要用 round4」，原本是 Round 2。
DEFAULT_SPLIT_ROUND = 4
# 超參數來源預設為 None＝讀該模型自己的 sweep_{key}_rf.csv。
# 規定：每個模型各自調參，不得共用組態（見 sweep_config.py 的說明）。
# 傳 --config-round N 會改走舊式的共用檔 sweep_round{N}_rf.csv，只供讀取
# 既有歷史檔案，新模型不得使用。
DEFAULT_CONFIG_ROUND = None
CURVE_SPLIT = "val_sel"


def load_with_label(features_path: Path, label_file: Path, label_col: str,
                    drop_prefixes: tuple[str, ...] = (),
                    drop_cols: tuple[str, ...] = ()) -> dict:
    """讀特徵 + 外部 label，切分並前處理。與 finalists.prepare_with_index 同一套流程。

    `drop_prefixes`：把指定前綴的特徵整組剔除。用途是做「拿掉某一類資訊」的
    對照實驗，例如去掉 `mkt_` 觀察模型在沒有大盤資訊時還剩多少選股能力。
    """
    df, cols = load_data(features_path)
    if drop_prefixes:
        before = len(cols)
        cols = [c for c in cols if not c.startswith(drop_prefixes)]
        logger.info(f"剔除前綴 {drop_prefixes}：{before} → {len(cols)} 欄")
    if drop_cols:
        before = len(cols)
        cols = [c for c in cols if c not in set(drop_cols)]
        logger.info(f"剔除指定欄位 {len(drop_cols)} 個：{before} → {len(cols)} 欄")
    logger.info(f"特徵 {len(cols)} 欄，資料 {len(df):,} 列")

    df["date"] = pd.to_datetime(df["date"])
    # `load_data()` 已經把 data/labels.parquet 的欄位併進來了，若要的 label 就在
    # 裡面就別再 merge 一次 —— 重複 merge 會變成 label_up20_x / _y，後面取不到
    if label_col in df.columns:
        logger.info(f"label `{label_col}` 已在特徵表中，不重複合併")
    else:
        lab = pd.read_parquet(label_file, columns=["date", "stock_id", label_col])
        lab["date"] = pd.to_datetime(lab["date"])
        df = df.merge(lab, on=["date", "stock_id"], how="left")
    n_valid = int(df[label_col].notna().sum())
    logger.info(f"label `{label_col}`：有效 {n_valid:,} 列（{n_valid / len(df):.1%}），"
                f"基準率 {df[label_col].mean():.4f}")

    frames = {name: split_frame(df, name) for name in splits()}
    # label 為 NA 的列一律剔除：未定的未來視窗不能當負例
    frames = {k: v[v[label_col].notna()] for k, v in frames.items()}
    for name, frame in frames.items():
        logger.info(f"  {name:8s} {len(frame):>9,} 列　正例 {frame[label_col].mean():.3f}")

    train = frames["train"]
    others = {k: v for k, v in frames.items() if k != "train"}
    x_train, x_others = preprocess(train, others, cols)

    return {
        "cols": cols, "stats": fit_stats(train, cols),
        "x_train": x_train, "y_train": train[label_col].to_numpy(dtype=np.int8),
        "x": x_others,
        "y": {k: v[label_col].to_numpy(dtype=np.int8) for k, v in others.items()},
        "frames": others,
        "groups_train": train.groupby("date").size().to_numpy(),
        "groups_es": others["val_es"].groupby("date").size().to_numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-file", required=True,
                        help="含 date/stock_id/<label-col> 的 parquet")
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--key", required=True, help="模型代號，決定所有輸出檔名")
    parser.add_argument("--desc", required=True, help="前端下拉選單顯示的名稱")
    parser.add_argument("--features", default=str(DATA_DIR / "features.parquet"))
    parser.add_argument("--round", type=int, default=DEFAULT_SPLIT_ROUND,
                        help="用哪一套切分")
    parser.add_argument("--config-round", type=int, default=DEFAULT_CONFIG_ROUND,
                        help="舊式：整輪共用的 sweep CSV。不給就讀本模型自己那份")
    parser.add_argument("--drop-prefix", action="append", default=[],
                        help="剔除此前綴的所有特徵，可重複（例：--drop-prefix mkt_）")
    parser.add_argument("--drop-file", default=None,
                        help="文字檔，每行一個要剔除的欄名（例：data/drop_volatility.txt）")
    args = parser.parse_args()

    use_round(args.round)
    # 每個模型讀自己那份 sweep_{key}_rf.csv（規定：不得共用組態）。
    # --config-round 只在明確指定時才走舊式共用路徑，供讀取歷史檔案用。
    params = (best_config("rf", key=args.key) if args.config_round is None
              else best_config("rf", args.config_round))
    source = (f"本模型自己的 sweep_{args.key}_rf.csv" if args.config_round is None
              else f"⚠️ 共用的 Round {args.config_round} 組態")
    logger.info(f"[{args.key}] Round {args.round} 切分，超參數來自{source}：{params}")

    drop_cols = tuple(Path(args.drop_file).read_text().split()) if args.drop_file else ()
    data = load_with_label(Path(args.features), Path(args.label_file), args.label_col,
                           tuple(args.drop_prefix), drop_cols)

    started = time.time()
    predict, model = build_fitted("rf", dict(params), data)
    logger.info(f"[{args.key}] 訓練完成 {time.time() - started:.0f}s")

    rows, val_scores = [], None
    for split in eval_splits():
        score = predict(data["x"][split])
        if split == CURVE_SPLIT:
            val_scores = score
        out = data["frames"][split][["date", "stock_id"]].copy()
        out["score"] = score.astype("float32")
        out.to_parquet(DATA_DIR / f"score_{args.key}_{split}.parquet", index=False)
        m = evaluate(data["y"][split], score)
        rows.append({"split": split, "n": m["n"], "auc": round(m["auc"], 4),
                     "pr_lift": round(m["pr_lift"], 3)})
        logger.info(f"  {split:8s} n={m['n']:>8,} auc={m['auc']:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "family": "rf", "round": args.round, "cols": data["cols"],
        # 推論時要用哪一份特徵檔。存**檔名**而非絕對路徑，bundle 搬到別台機器
        # 或 repo 換位置時仍然對得到（推論端用 DATA_DIR 接起來）。
        # 沒有這一欄的話推論端只能猜，而 v3 那兩個模型猜錯就是整份名單全錯。
        "features_file": Path(args.features).name,
        "stats": data["stats"], "model": model, "arch": None, "params": params,
        "config_round": args.config_round,
        "score_min": float(val_scores.min()), "score_max": float(val_scores.max()),
        "val_split": CURVE_SPLIT, "n_train": len(data["y_train"]),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "label_name": args.label_col, "desc": args.desc,
        "dropped_prefixes": list(args.drop_prefix),
        "dropped_cols": list(drop_cols),
    }
    with open(MODEL_DIR / f"bundle_{args.key}.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n=== {args.key}（{args.desc}）===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nbundle → models/bundle_{args.key}.pkl")
    print("下一步：threshold_curve.py 產生門檻曲線，由人挑門檻（本程式不挑）")


if __name__ == "__main__":
    main()
