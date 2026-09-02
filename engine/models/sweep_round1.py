"""超參數搜尋（表格模型）→ data/sweep_round{N}_{model}.csv

依 doc/PLAN.md §5 的搜尋空間，對單一 ground truth `label_up20` 比較模型類型。
本檔負責吃表格資料的模型；MLP / LSTM 需要 torch 訓練迴圈，另外一支處理。

用法（Round 1，不加 --round 就是這個）：
  python sweep_round1.py --model rf          # 96 種，GridSampler 全掃
  python sweep_round1.py --model extratrees  # 同上，bootstrap=False
  python sweep_round1.py --model lightgbm    # TPESampler 80 trials
  python sweep_round1.py --model lambdarank  # 同上 × truncation

Round 2（大訓練集，只剩 RF）：
  python sweep_round1.py --model rf --round 2   # 27 種，GridSampler 全掃

Round 4（新切分，只剩 RF，往 Round 2 最佳解的邊界外延伸）：
  python sweep_round1.py --model rf --round 4   # 12 種，GridSampler 全掃

方法論要點（踩過坑才定下來的）：

- **Optuna 目標是 val_es AUC**，樹模型也用同一段。神經網路要用 val 做 early
  stopping，會偷看 val 幾百次；讓所有模型都只用 val_es 當目標，六種模型才站得到
  同一條線上。
- **跨模型排名看 val_sel + 兩個 test 年**，不看 val_es。實測 fold3 出現過
  val 排名與 test 幾乎顛倒。
- **輸出每個組態在四段的成績**，不只有最佳解 —— Round 1 的存活組態由人挑，
  不設自動規則。
- **PR-AUC 一律附 lift**（除以正例率）。PR-AUC 的隨機基線等於正例率，
  不正規化就跨設定比較會誤導。
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


from engine.models.train_single import (  # noqa: E402
    DATA_DIR,
    LABEL,
    add_round_arg,
    current_round,
    eval_splits,
    evaluate,
    load_data,
    preprocess,
    selected_columns,
    split_frame,
    splits,
    use_round,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
# TPE_TRIALS 保留為文件：lightgbm / lambdarank 當初用 TPE 跑 80 trials。
# Optuna 已移除，現行流程用不到（見 run_sweep 的說明）。
TPE_TRIALS = 80
EARLY_STOPPING_ROUNDS = 100

# ── 搜尋空間（doc/PLAN.md §5，逐項與使用者確認過）────────────────────────────
FOREST_SPACE = {
    "max_features": ["sqrt", 10, 30, 40],
    "min_samples_leaf": [50, 150, 200, 300],
    "max_depth": [10, 14, 20],
    "class_weight": [None, "balanced"],
}
GBM_SPACE = {
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [50, 150, 200, 300],
    "learning_rate": [0.01, 0.03, 0.1],
    "feature_fraction": [0.1, 0.3, 0.6],
    "lambda_l2": [1, 10, 100],
}
LAMBDARANK_TRUNCATION = [10, 30, 100]

# ── Round 2 搜尋空間（使用者逐項確認過）────────────────────────────────────
# Round 1 砍到只剩 RF / MLP / LSTM，且改在 Round 1 最佳解「附近」小範圍搜尋。
# max_features 改用絕對欄數（Round 1 的 "sqrt" 約 18 欄，最佳解落在 30~40）。
FOREST_SPACE_R2 = {
    "max_features": [20, 40, 60],
    "min_samples_leaf": [200, 300, 400],
    "max_depth": [20, 30, 50],
}
# 固定不掃，但仍寫進 CSV，讓 finalists.py 讀得回完整組態
FOREST_FIXED_PARAMS_R2 = {"class_weight": None}

# ── Round 4 搜尋空間（2026-08-15，使用者逐項指定）──────────────────────────
# 27 組 = 3 × 3 × 3。Round 2 最佳解的 max_features 與 min_samples_leaf 都卡在舊
# 範圍的邊界，所以往外延伸；max_depth 的最佳值 30 是內部點，這次在其兩側加點確認。
#
# ⚠️ **搜尋與正式訓練用同一個樹數（300），不分兩階段**（2026-08-23 使用者指定）。
# 舊做法是「第一階段固定 150 掃組態、第二階段取前幾名再試 300/450」，理由是樹數
# 主要影響變異不影響偏誤、150 棵的組態排名與 300 幾乎一致。問題有兩個：
#   1. 第二階段從來沒跑過 —— 正式模型一直是 150 棵，也就是**用篩選階段的暫定值
#      當成最終組態**，而那個值當初只是為了加速。
#   2. 「排名會轉移」是假設不是事實。搜尋與訓練用同一個樹數就不需要這個假設。
# 代價是搜尋慢一倍，換掉的是一個沒被驗證的前提。
FOREST_SPACE_R4 = {
    "max_features": [10, 15, 20],
    "min_samples_leaf": [200, 400, 500],
    "max_depth": [15, 30, 45],
}
FOREST_FIXED_PARAMS_R4 = {"class_weight": None}   # n_estimators 走 FOREST_FIXED 的 300

# n_jobs=8（2026-08-23 使用者指定，原本 6）：機器 10 核，實測 n_jobs=6 時 CPU
# 有 27% 閒置（約 2.7 核）。模型一個一個訓練、外層不並行，內層就該吃滿，留 2 核
# 給系統。**不要改成外層並行** —— 兩層相乘會超賣，而且四座森林同時佔記憶體
# （舊 repo 2026-08-14 實測 load 衝到 23，每件都變慢）。
# 記憶體不受影響：sklearn RF 用 threading backend，執行緒共用同一份 X 矩陣。
FOREST_FIXED = dict(n_estimators=300, n_jobs=8, random_state=RANDOM_STATE)
GBM_FIXED = dict(bagging_fraction=0.7, n_estimators=3000, random_state=RANDOM_STATE, n_jobs=4)


def _fit_forest(cls, params, x_train, y_train, *_):
    # params 覆寫 FOREST_FIXED。現行組態不再覆寫 n_estimators（搜尋與訓練都用
    # 300），但保留這個覆寫機制 —— 讀既有的歷史 sweep CSV 時，裡面那欄
    # n_estimators=150 仍需要能生效，否則重現不了當初的模型。
    model = cls(**{**FOREST_FIXED, **params}).fit(x_train, y_train)
    return (lambda x: model.predict_proba(x)[:, 1]), model


def _fit_lightgbm(params, x_train, y_train, x_es, y_es, *_):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(**GBM_FIXED, **params, verbose=-1)
    model.fit(
        x_train, y_train,
        eval_set=[(x_es, y_es)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return (lambda x: model.predict_proba(x)[:, 1]), model


def _fit_lambdarank(params, x_train, y_train, x_es, y_es, groups_train, groups_es):
    """LambdaRank：每個交易日一組（全市場），relevance 用二元 label_up20。

    輸出是分數不是機率，門檻走每日分位數 —— 但 AUC 對單調轉換不變，
    所以與其他模型的 AUC 直接可比。
    """
    import lightgbm as lgb

    params = dict(params)
    truncation = params.pop("lambdarank_truncation_level")
    model = lgb.LGBMRanker(
        objective="lambdarank",
        lambdarank_truncation_level=truncation,
        **GBM_FIXED, **params, verbose=-1,
    )
    model.fit(
        x_train, y_train, group=groups_train,
        eval_set=[(x_es, y_es)], eval_group=[groups_es], eval_at=[10],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model.predict, model


def build_fitted(model_name: str, params: dict, data: dict):
    """訓練並回傳 (predict 函式, 模型物件)。

    2026-08-12 新增：前端要能重複使用訓練好的模型，必須把模型物件本身存下來
    （`build_predictor()` 只回傳閉包，模型用完就被 GC）。訓練邏輯完全共用，
    這裡只是多把 model 傳出來。
    """
    # val_es 只有神經網路與 LambdaRank 的 early stopping 會用；RF / ExtraTrees
    # 收下就丟掉。SWEEP_EVAL_SPLITS 不含 val_es 時它根本不會被載入，所以用 .get()
    # —— 要復用 NN/LambdaRank 就得把 val_es 加回 SWEEP_EVAL_SPLITS。
    args = (data["x_train"], data["y_train"],
            data["x"].get("val_es"), data["y"].get("val_es"))
    if model_name == "rf":
        return _fit_forest(RandomForestClassifier, params, *args)
    if model_name == "extratrees":
        return _fit_forest(ExtraTreesClassifier, {**params, "bootstrap": False}, *args)
    if model_name == "lightgbm":
        return _fit_lightgbm(params, *args)
    if model_name == "lambdarank":
        return _fit_lambdarank(params, *args, data["groups_train"], data["groups_es"])
    raise ValueError(f"未知模型：{model_name}")


def build_predictor(model_name: str, params: dict, data: dict):
    """只要 predict 函式時用這個（既有呼叫端行為不變）。"""
    return build_fitted(model_name, params, data)[0]


# ── 調參只看驗證期，不碰測試期（2026-08-23 使用者指定）─────────────────────
# 兩個理由：
# 1. **方法論**：組態選擇一律看 val_sel。在搜尋階段連算帶記 test 分數，等於把
#    測試期攤在眼前 —— 就算程式沒拿它排序，人看了就很難不受影響。測試期只在
#    最終回測時看一次。
# 2. **成本**：每一組都要對 test（約 44 萬列）與 test2（約 25 萬列）各做一次
#    推論，前處理階段也要多載入這兩個切分的 X 矩陣。
# 3. **val_es 也拿掉了**（2026-08-23）：它的用途是神經網路的 early stopping，
#    RF 根本不用。組態選擇一律看 val_sel，留著 val_es 只是多載入 22 萬列、
#    每組多做一次推論。要復用 NN / LambdaRank 時再加回來。
SWEEP_EVAL_SPLITS = ("val_sel",)


def sweep_eval_splits() -> tuple[str, ...]:
    """調參階段要評估的切分：只取驗證期，且必須真的存在於當前 round。"""
    return tuple(s for s in eval_splits() if s in SWEEP_EVAL_SPLITS)


def prepare(features_path: Path, intersect_with: Path | None,
            drop_prefixes: tuple[str, ...] = (), drop_cols: tuple[str, ...] = (),
            label_file: Path | None = None, label_col: str = LABEL) -> dict:
    df, cols = load_data(features_path)
    if label_col != LABEL:
        # 自己合併 label，不重用 train_label_variant.load_with_label ——
        # 那支回傳的是「讀檔＋合併＋切分＋前處理」做完的整包 dict，跟這裡需要的
        # (df, cols) 不是同一層抽象（2026-08-23 踩過：解包失敗，而且要等到第一個
        # 非預設 label 的模型 m6 才炸，前面 12 組全正常，掩蓋了三小時）。
        # 它還會強制載入 val_es 與 test —— 正是我們刻意砍掉的東西。
        if label_col in df.columns:
            logger.info(f"label `{label_col}` 已在特徵表中，不重複合併")
        else:
            lab = pd.read_parquet(label_file, columns=["date", "stock_id", label_col])
            lab["date"] = pd.to_datetime(lab["date"])
            df = df.merge(lab, on=["date", "stock_id"], how="left")
        n_valid = int(df[label_col].notna().sum())
        logger.info(f"label `{label_col}`：有效 {n_valid:,} 列（{n_valid / len(df):.1%}），"
                    f"基準率 {df[label_col].mean():.4f}")
    if intersect_with:
        shared = selected_columns(intersect_with)
        cols = [c for c in cols if c in shared]
    # 剔除規則要跟最終訓練完全一致，否則搜出來的組態是為別的特徵集調的
    if drop_prefixes:
        cols = [c for c in cols if not c.startswith(drop_prefixes)]
    if drop_cols:
        cols = [c for c in cols if c not in set(drop_cols)]
    logger.info(f"特徵 {len(cols)} 欄")

    # 只載入 train 與驗證期 —— 測試期在調參階段完全不碰（見 SWEEP_EVAL_SPLITS）
    wanted = ("train",) + sweep_eval_splits()
    frames = {name: split_frame(df, name) for name in splits() if name in wanted}
    frames = {k: v[v[label_col].notna()] for k, v in frames.items()}
    train = frames["train"]
    others = {k: v for k, v in frames.items() if k != "train"}
    x_train, x_others = preprocess(train, others, cols)

    return {
        "cols": cols,
        "x_train": x_train,
        "y_train": train[label_col].to_numpy(dtype=np.int8),
        "x": x_others,
        "y": {k: v[label_col].to_numpy(dtype=np.int8) for k, v in others.items()},
        # LambdaRank 的 group：每個交易日全市場一組，要跟列的排序一致
        "groups_train": train.groupby("date").size().to_numpy(),
        # 同上：沒載入 val_es 就沒有 group（只有 LambdaRank 要）
        "groups_es": (others["val_es"].groupby("date").size().to_numpy()
                      if "val_es" in others else None),
    }


# ── 每個模型自己的搜尋空間（2026-08-23 使用者逐個確認）─────────────────────
# 規定：每個模型各自調參，不得共用組態。空間以 norf 舊 repo 的 base 家族最佳解
# 為中心往外給格子（max_features=20, min_samples_leaf=200, max_depth=15）——
# 那個最佳解是從封存 bundle 直接讀出來的。
#
# ⚠️ `depth=10` 在 norf 搜過的範圍之外（norf 的 depth 是 15/30/45），是刻意往下探。
#    若最佳解落在這個邊界上，代表真正的最佳值可能更小，要再往下搜一輪 ——
#    別直接當成收斂了。
#
# 空間相同不代表結果會相同：兩者格子一樣，但各自用自己的 label 搜，選出來的
# 組態很可能不同。這正是「不得共用組態」的意義。
MODEL_SPACES = {
    # 三個模型同特徵集（base，344 欄）、同空間，**只差標的** ——
    #   m1_base_up20  label_up20      未來 20 日上漲天數 >= 10
    #   m1_mdd10      label_mdd10     同上，再要求最低收盤不跌破 −10%
    #   m1_steady20   label_steady20  報酬 > max(1.5×自身波動, 5%) 且站上 20 日線 >= 10 天
    # 空間刻意相同：這樣三者的差異只能來自標的，不會混進調參的運氣。
    #
    # max_depth 從 10/15/20 砍成兩個端點（2026-08-23）：前一輪 7 組實測，
    # depth 10/15/20 的 val_sel 平均分別是 0.6064 / 0.6079 / 0.6067，**差 0.0015**，
    # 但 depth=20 比 depth=10 慢 63%（1,078s vs 662s）。留兩個端點是為了保住
    # 「淺 vs 深」的對照 —— 萬一在別的標的上 depth 真的有影響，看得出來。
    # min_samples_leaf 固定（2026-08-23）：前一輪 7 組實測，leaf 100 vs 200 的
    # val_sel 平均是 0.6060 vs 0.6074（差 0.0013，三個參數中最小），**耗時只差
    # 1.02 倍**。也就是說它既沒訊號、也不影響速度，留在格子裡純粹讓組數翻倍。
    # 固定值取 200：實測較佳，且與 norf 最佳解一致。
    "m1_base_up20": {"max_features": [15, 20], "max_depth": [10, 20]},
    "m1_mdd10":     {"max_features": [15, 20], "max_depth": [10, 20]},
    "m1_steady20":  {"max_features": [15, 20], "max_depth": [10, 20]},
}
# 搜尋與正式訓練同樹數，且不再覆寫 n_estimators（走 FOREST_FIXED 的 300）
# 固定但仍寫進 CSV 的參數。min_samples_leaf 在這裡（不在搜尋空間裡），
# 所以 CSV 仍會記錄實際用的值，日後回查得到。
MODEL_FIXED_PARAMS = {
    "m1_base_up20": {"class_weight": None, "min_samples_leaf": 200},
    "m1_mdd10":     {"class_weight": None, "min_samples_leaf": 200},
    # class_weight 仍是 None：基準率 10% 比另外兩個低得多，但改成
    # balanced 等於同時換了標的與權重，兩個變因混在一起就比不出東西。
    # 要試權重，等這一輪比完、單獨開一個模型試。
    "m1_steady20":  {"class_weight": None, "min_samples_leaf": 200},
}



def search_space(model_name: str, round_no: int, key: str | None = None) -> tuple[dict, dict]:
    """回傳 (要搜的空間, 固定但仍寫進 CSV 的參數)。

    給了 `key` 就用該模型自己的空間（現行的唯一正路）；沒給則走舊的
    「整輪共用一份」路徑，只保留給重現既有歷史結果用。
    """
    if key:
        if key not in MODEL_SPACES:
            raise ValueError(
                f"{key} 沒有定義搜尋空間。每個模型都必須有自己的一份 —— "
                f"請在 MODEL_SPACES 補上，不要沿用別的模型的。"
                f"目前有：{', '.join(MODEL_SPACES)}")
        return dict(MODEL_SPACES[key]), dict(MODEL_FIXED_PARAMS[key])
    if round_no == 2:
        if model_name != "rf":
            raise ValueError(f"Round 2 的表格模型只留 rf，不支援 {model_name}")
        return dict(FOREST_SPACE_R2), dict(FOREST_FIXED_PARAMS_R2)
    if round_no == 4:
        if model_name != "rf":
            raise ValueError(f"Round 4 的表格模型只留 rf，不支援 {model_name}")
        return dict(FOREST_SPACE_R4), dict(FOREST_FIXED_PARAMS_R4)

    space = dict(FOREST_SPACE) if model_name in ("rf", "extratrees") else dict(GBM_SPACE)
    if model_name == "lambdarank":
        space["lambdarank_truncation_level"] = LAMBDARANK_TRUNCATION
    return space, {}


def params_key(params: dict) -> tuple[str, ...]:
    """組態的比對鍵。轉字串是為了跨 CSV 來回不受 int64／float64 影響。"""
    return tuple(str(params[k]) for k in sorted(params))


def load_done(out_path: Path, space: dict) -> tuple[list[dict], dict[tuple, float]]:
    """讀回上次中斷前完成的組態，讓 sweep 接著跑。

    Optuna study 沒有持久化（沒設 storage），重跑會從頭列舉整個 grid，
    所以拿輸出的 CSV 當進度檔：已跑過的組態直接回傳當時的 val_es AUC，不重訓。
    """
    if not out_path.exists():
        return [], {}
    df = pd.read_csv(out_path)
    records = df.to_dict("records")
    # 進度值用 val_sel_auc —— 選組態一律看它（best_config 也是），
    # 舊版這裡用 val_es_auc 是為了配合 Optuna 的目標值，那個耦合已經拿掉。
    done = {params_key({k: row[k] for k in space}): row["val_sel_auc"] for row in records}
    logger.info(f"接續 {out_path.name}：已完成 {len(done)} 組")
    return records, done


def run_sweep(model_name: str, data: dict, out_path: Path, jobs: int = 1,
              key: str | None = None) -> pd.DataFrame:
    """把搜尋空間的所有組合跑一遍。

    2026-08-23：**拿掉 Optuna，改成純網格迴圈**（使用者指定）。

    原本用 `optuna.samplers.GridSampler`，但 GridSampler 的行為就是「把所有組合
    跑一遍」—— 跟兩層迴圈完全等價，卻帶來兩個代價：

    1. 多一個相依（`optuna`，而且它正是舊 repo requirements 漏列的三個之一）
    2. **一個會咬人的耦合**：Optuna 需要 objective 回傳一個純量目標，這份程式
       寫死回傳 `val_es_auc`。2026-08-23 把 val_es 從評估切分拿掉之後，第一組
       跑完要回填時就 `KeyError` 炸掉 —— 模型其實已經訓練完、AUC 也算好了，
       純粹是死在這個不必要的耦合上。

    ⚠️ 用 TPE 的模型家族（lightgbm / lambdarank）不走這裡 —— 它們需要真正的
    貝氏搜尋。要復用那些家族就得把 Optuna 加回來，見 git 歷史。

    `jobs > 1` 已不支援：模型一個一個訓練、內層吃 n_jobs=8，外層再並行會超賣。
    """
    if model_name not in ("rf", "extratrees"):
        raise NotImplementedError(
            f"{model_name} 原本走 Optuna 的 TPE 搜尋，2026-08-23 移除 Optuna 時一併停用。"
            "現行流程只用 rf。要復用請看 git 歷史把 TPE 路徑加回來。")
    if jobs != 1:
        raise ValueError(
            f"--jobs={jobs}：外層並行已停用。模型一個一個跑、內層 n_jobs 吃滿，"
            "兩層相乘會超賣（舊 repo 2026-08-14 實測 load 衝到 23，每件都變慢）。")

    space, fixed = search_space(model_name, current_round(), key)
    names = list(space)
    combos = [dict(zip(names, values)) for values in itertools.product(*(space[n] for n in names))]
    logger.info(f"{model_name}：{len(combos)} 個組態")

    records, done = load_done(out_path, space)

    for combo in combos:
        pkey = params_key(combo)
        if pkey in done:
            logger.info(f"  [{len(records)}/{len(combos)}] 跳過已完成組態 {combo}")
            continue

        started = time.time()
        predict = build_predictor(model_name, {**combo, **fixed}, data)

        row = {
            "model": model_name, **combo, **fixed,
            "seconds": round(time.time() - started, 1),
        }
        for split in sweep_eval_splits():
            metrics = evaluate(data["y"][split], predict(data["x"][split]))
            row[f"{split}_auc"] = round(metrics["auc"], 4)
            row[f"{split}_lift"] = round(metrics["pr_lift"], 3)
        records.append(row)
        done[pkey] = row["val_sel_auc"]

        pd.DataFrame(records).to_csv(out_path, index=False)  # 中途被中斷也留得住
        logger.info(
            f"  [{len(records)}/{len(combos)}] "
            + " | ".join(f"{s} {row[f'{s}_auc']:.4f}" for s in sweep_eval_splits())
            + f" | {row['seconds']:.0f}s"
        )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=["rf", "extratrees", "lightgbm", "lambdarank"]
    )
    parser.add_argument("--features", default=str(DATA_DIR / "features.parquet"))
    parser.add_argument("--intersect-with")
    parser.add_argument("--drop-prefix", action="append", default=[])
    parser.add_argument("--drop-file", default=None)
    parser.add_argument("--label-file", default=None,
                        help="label 檔（預設 labels.parquet 的 label_up20）")
    parser.add_argument("--label-col", default=LABEL,
                        help="label 欄名。⚠️ 每個模型必須用**自己的** label 調參，"
                             "不可以拿別的 label 搜出來的組態套過來")
    parser.add_argument("--key", default=None,
                        help="模型代號。給了就寫 sweep_{key}_{model}.csv —— "
                             "每個模型一份，不共用")
    parser.add_argument("--jobs", type=int, default=1,
                        help="同時訓練幾個 trial（內層 n_jobs 固定 4，jobs×4 勿超過核心數）")
    add_round_arg(parser)
    args = parser.parse_args()
    use_round(args.round)

    drop_cols = tuple(Path(args.drop_file).read_text().split()) if args.drop_file else ()
    data = prepare(
        Path(args.features), Path(args.intersect_with) if args.intersect_with else None,
        tuple(args.drop_prefix), drop_cols,
        Path(args.label_file) if args.label_file else None, args.label_col,
    )
    out_path = (DATA_DIR / f"sweep_{args.key}_{args.model}.csv" if args.key
                else DATA_DIR / f"sweep_round{args.round}_{args.model}.csv")
    results = run_sweep(args.model, data, out_path, jobs=args.jobs, key=args.key)

    print(f"\n=== {args.model}：依 val_sel AUC 排序前 10 ===")
    top = results.sort_values("val_sel_auc", ascending=False).head(10)
    print(top[[f"{s}_auc" for s in sweep_eval_splits()]].to_string())
    print(f"\n完整結果：{out_path}")


if __name__ == "__main__":
    main()
