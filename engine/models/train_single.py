"""單一模型訓練與評估（Round 1 / Round 2 切分）→ 分數檔 + AUC

委員會已拆除，架構是「單一 ground truth `label_up20` + 多種模型類型」。
這支先支援 RF，供 Part D 的 v1/v2 特徵比較使用；六模型 sweep 另外接 Optuna。

用法：
  python train_single.py --features data/features.parquet     --tag v1
  python train_single.py --features data_v2/features.parquet  --tag v2

輸出：
  data/score_{tag}_{split}.parquet   欄位 date / stock_id / score
  → 直接餵給 backtest.simulate(score_path=...)

刻意的設計：
- **兩版的特徵欄位集合必須完全相同**，否則比較的就不只是「實作差異」。
- shape 特徵一律排除：KMeans 群心用 2023-12-31 前的資料 fit，而測試期在
  2022/2023，那 14 欄是分布層級的洩漏來源（見 doc/PLAN.md §3）。
- 前處理統計量**只從訓練期算**（補值中位數、標準化的平均與標準差），
  否則測試期資訊會反向洩漏進訓練。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


from engine.models.submodel_config import feature_cols  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 專案路徑一律走 code/paths.py（唯一來源），不要各檔自行推導
from engine.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402


LABEL = "label_up20"
MODEL_ID = "UP20"
SHAPE_PREFIXES = ("price_shape_", "volume_shape_")
WINSORIZE_SIGMA = 10.0

# 切分（doc/PLAN.md §4）。兩個 round 都是 embargo 一個月，避免 label 視窗跨界。
# Round 1 是小訓練集的模型類型篩選；Round 2 換成大訓練集，並多一段 test3。
ROUND_SPLITS: dict[int, dict[str, tuple[str, str]]] = {
    1: {
        "train": ("2020-01-01", "2020-11-30"),
        "val_es": ("2021-01-01", "2021-08-31"),
        "val_sel": ("2021-09-01", "2021-11-30"),
        "test": ("2022-01-01", "2022-12-31"),
        "test2": ("2023-01-01", "2023-12-31"),
    },
    2: {
        "train": ("2020-01-01", "2022-11-30"),
        "val_es": ("2023-01-01", "2023-08-31"),
        "val_sel": ("2023-09-01", "2023-11-30"),
        "test": ("2024-01-01", "2024-12-31"),
        "test2": ("2025-01-01", "2025-12-31"),
        "test3": ("2026-01-01", "2026-07-31"),
    },
    # Round 3＝「Round 1 的訓練/驗證期 + Round 2 的測試期」。
    # 用途只有一個：讓 Round 1 模型（小訓練集）跟 Round 2 模型（大訓練集）在
    # **完全相同的 2024/2025/2026 測試資料**上對決，各用自己校準的門檻。
    # 前處理統計量仍只從 train（2020）算，測試期資料不會反向洩漏。
    3: {
        "train": ("2020-01-01", "2020-11-30"),
        "val_es": ("2021-01-01", "2021-08-31"),
        "val_sel": ("2021-09-01", "2021-11-30"),
        "test": ("2024-01-01", "2024-12-31"),
        "test2": ("2025-01-01", "2025-12-31"),
        "test3": ("2026-01-01", "2026-07-31"),
    },
    # Round 4（2026-08-14 新增）：訓練期拉到 2023 底、驗證期整整一年。
    # 相對 Round 2 的三個好處：
    #   1. 訓練資料 +38%（1.29M → 1.78M 列），且多的是較近期的 2023 年
    #   2. 挑門檻用的 val_sel 從 3 個月（117k 列）拉到 6 個月，曲線尾端不會太薄
    #   3. val 期的正例基準率（~35%）與測試期（36.4%/35.2%）一致；Round 2 的
    #      val_sel 是 37.9%，在那上面挑的門檻搬到測試期本來就會偏
    # 兩個交界各留一個月 embargo（2023-12、2025-01）：label_up20 要看未來 20 個
    # 交易日，交界緊貼的話訓練期末端的答案會落在驗證期裡，等於偷看。
    # Round 5＝Round 4 的切分，僅供 v3 系列調參時使用（sweep CSV 分開存）
    5: {
        "train": ("2020-01-01", "2023-11-30"),
        "val_es": ("2024-01-01", "2024-06-30"),
        "val_sel": ("2024-07-01", "2024-12-31"),
        "test": ("2025-02-01", "2025-12-31"),
        "test2": ("2026-01-01", "2026-07-31"),
    },
    6: {
        "train": ("2020-01-01", "2023-11-30"),
        "val_es": ("2024-01-01", "2024-06-30"),
        "val_sel": ("2024-07-01", "2024-12-31"),
        "test": ("2025-02-01", "2025-12-31"),
        "test2": ("2026-01-01", "2026-07-31"),
    },
    7: {
        "train": ("2020-01-01", "2023-11-30"),
        "val_es": ("2024-01-01", "2024-06-30"),
        "val_sel": ("2024-07-01", "2024-12-31"),
        "test": ("2025-02-01", "2025-12-31"),
        "test2": ("2026-01-01", "2026-07-31"),
    },
    4: {
        "train": ("2020-01-01", "2023-11-30"),
        "val_es": ("2024-01-01", "2024-06-30"),    # 調參目標
        "val_sel": ("2024-07-01", "2024-12-31"),   # 挑門檻
        "test": ("2025-02-01", "2025-12-31"),
        "test2": ("2026-01-01", "2026-07-31"),
    },
}

# 每個 round 要評估／寫進 sweep CSV 的切分（train 不算）
ROUND_EVAL_SPLITS: dict[int, tuple[str, ...]] = {
    1: ("val_es", "val_sel", "test", "test2"),
    2: ("val_es", "val_sel", "test", "test2", "test3"),
    3: ("val_es", "val_sel", "test", "test2", "test3"),
    4: ("val_es", "val_sel", "test", "test2"),
    5: ("val_es", "val_sel", "test", "test2"),
    6: ("val_es", "val_sel", "test", "test2"),
    7: ("val_es", "val_sel", "test", "test2"),
}

DEFAULT_ROUND = 1
_active_round = DEFAULT_ROUND


def use_round(round_no: int) -> None:
    """切換使用哪一套切分。必須在載入資料之前呼叫。

    刻意做成行程層級的狀態而不是到處傳參數：切分是「這次實驗是哪個 round」的
    全域事實，四支腳本共用；混用兩套切分算出來的數字不能互相比較。
    """
    global _active_round
    if round_no not in ROUND_SPLITS:
        raise ValueError(f"未知的 round：{round_no}（可用 {sorted(ROUND_SPLITS)}）")
    _active_round = round_no


def current_round() -> int:
    return _active_round


def splits() -> dict[str, tuple[str, str]]:
    """目前 round 的切分。一律用函式取，不要 from-import 常數（會綁到舊值）。"""
    return ROUND_SPLITS[_active_round]


def eval_splits() -> tuple[str, ...]:
    return ROUND_EVAL_SPLITS[_active_round]


def add_round_arg(parser: argparse.ArgumentParser) -> None:
    """四支腳本共用的 --round 參數；不給就是 Round 1（維持既有行為）。"""
    parser.add_argument(
        "--round", type=int, default=DEFAULT_ROUND, choices=sorted(ROUND_SPLITS),
        help="用哪一套切分（1=Round 1 小訓練集，2=Round 2 大訓練集 + test3）",
    )

# Part D 比較用的固定組態：兩版必須完全一致，這裡不做搜參
RF_CONFIG = dict(
    n_estimators=300,
    max_depth=14,
    min_samples_leaf=150,
    max_features="sqrt",
    class_weight=None,
    # n_jobs=6（2026-08-23 使用者指定）：模型一個一個訓練，不做外層並行，
    # 所以內層可以吃滿。原本是 4，註解寫「避免與外層並行相乘把記憶體榨乾」——
    # 那是舊委員會時代多模型同時跑的遺留。
    # 記憶體不是限制：sklearn 的 RandomForest 用 threading backend，n_jobs 之間
    # 共用同一份 X 矩陣不複製（v3 訓練集約 1.7M 列 × 518 欄 float32 ≈ 3.5 GB，
    # 機器 24 GB）。真的不夠時才調降。
    n_jobs=6,
    random_state=42,
)


def selected_columns(features_path: Path) -> set[str]:
    """該檔案經 feature_cols() 選出、且排除 shape 後的特徵欄。"""
    import pyarrow.parquet as pq

    names = [c for c in pq.read_schema(features_path).names if c not in ("date", "stock_id")]
    return {c for c in feature_cols(MODEL_ID, names) if not c.startswith(SHAPE_PREFIXES)}


def load_data(features_path: Path) -> tuple[pd.DataFrame, list[str]]:
    labels = pd.read_parquet(DATA_DIR / "labels.parquet", columns=["date", "stock_id", LABEL])
    features = pd.read_parquet(features_path)
    features["date"] = pd.to_datetime(features["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    df = features.merge(labels, on=["date", "stock_id"], how="inner")
    all_cols = [c for c in df.columns if c not in ("date", "stock_id", LABEL)]
    cols = [c for c in feature_cols(MODEL_ID, all_cols) if not c.startswith(SHAPE_PREFIXES)]
    return df, sorted(cols)


def split_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = splits()[name]
    return df[(df["date"] >= start) & (df["date"] <= end)]


def fit_stats(train: pd.DataFrame, cols: list[str]) -> dict:
    """從**訓練期**算出補值中位數與標準化的平均／標準差。

    2026-08-12 抽出來（原本是 `preprocess()` 內的區域變數，用完即丟）：推論時
    必須沿用訓練期算出來的這一份，重算等於讓推論期資料反向洩漏進標準化。
    存進 model bundle 的就是這個 dict。
    """
    median = train[cols].median()
    filled = train[cols].fillna(median)
    return {
        "median": median,
        "mean": filled.mean(),
        "std": filled.std().replace(0, 1.0),
    }


def apply_stats(frame: pd.DataFrame, cols: list[str], stats: dict) -> np.ndarray:
    """用既有統計量做補值 + 標準化 + ±10σ winsorize。**不重算統計量**。"""
    z = (frame[cols].fillna(stats["median"]) - stats["mean"]) / stats["std"]
    return z.clip(-WINSORIZE_SIGMA, WINSORIZE_SIGMA).to_numpy(dtype=np.float32)


def preprocess(train: pd.DataFrame, others: dict[str, pd.DataFrame], cols: list[str]):
    """訓練期統計量補值 + 標準化 + ±10σ winsorize。統計量只從訓練期算。

    行為與 2026-08-12 之前完全相同；統計量本身要拿來存 bundle 時改呼叫
    `fit_stats()` + `apply_stats()`。
    """
    stats = fit_stats(train, cols)
    return (apply_stats(train, cols, stats),
            {k: apply_stats(v, cols, stats) for k, v in others.items()})


def evaluate(y_true: np.ndarray, score: np.ndarray) -> dict:
    base_rate = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, score))
    return {
        "n": len(y_true),
        "base_rate": base_rate,
        "auc": float(roc_auc_score(y_true, score)),
        "pr_auc": pr_auc,
        # PR-AUC 的隨機基線等於正例率，跨 label 比較一定要先正規化成 lift
        "pr_lift": pr_auc / base_rate if base_rate > 0 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="features.parquet 路徑")
    parser.add_argument("--tag", required=True, help="輸出檔名標籤，例如 v1 / v2")
    parser.add_argument(
        "--intersect-with",
        help="另一份 features.parquet 的路徑；只用兩邊都有的欄位。"
             "Part D 的 v1/v2 比較必須加這個，否則比的就不只是實作差異",
    )
    add_round_arg(parser)
    args = parser.parse_args()
    use_round(args.round)

    df, cols = load_data(Path(args.features))
    if args.intersect_with:
        shared = selected_columns(Path(args.intersect_with))
        dropped = [c for c in cols if c not in shared]
        cols = [c for c in cols if c in shared]
        logger.info(f"取交集後剔除 {len(dropped)} 欄：{dropped}")
    logger.info(f"特徵 {len(cols)} 欄，資料 {len(df):,} 列")

    frames = {name: split_frame(df, name) for name in splits()}
    for name, frame in frames.items():
        logger.info(f"  {name:8s} {len(frame):>9,} 列（有 label {frame[LABEL].notna().sum():,}）")

    train = frames["train"][frames["train"][LABEL].notna()]
    others = {k: v for k, v in frames.items() if k != "train"}
    x_train, x_others = preprocess(train, others, cols)
    y_train = train[LABEL].to_numpy(dtype=np.int8)

    logger.info(f"訓練 RF：{RF_CONFIG}")
    model = RandomForestClassifier(**RF_CONFIG).fit(x_train, y_train)

    results = {}
    for name, frame in others.items():
        score = model.predict_proba(x_others[name])[:, 1]
        out = frame[["date", "stock_id"]].copy()
        out["score"] = score.astype("float32")
        out.to_parquet(DATA_DIR / f"score_{args.tag}_{name}.parquet", index=False)

        labelled = frame[LABEL].notna().to_numpy()
        if labelled.sum():
            results[name] = evaluate(
                frame.loc[labelled, LABEL].to_numpy(dtype=np.int8), score[labelled]
            )

    print(f"\n=== {args.tag}（{len(cols)} 欄特徵）===")
    print(f"{'split':10s} {'n':>10s} {'base':>7s} {'AUC':>8s} {'PR-AUC':>8s} {'lift':>7s}")
    for name, r in results.items():
        print(
            f"{name:10s} {r['n']:>10,} {r['base_rate']:>7.3f} "
            f"{r['auc']:>8.4f} {r['pr_auc']:>8.4f} {r['pr_lift']:>7.3f}"
        )


if __name__ == "__main__":
    main()
