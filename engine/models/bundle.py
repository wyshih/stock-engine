"""訓練好的模型 bundle：存檔格式、載入、以及對「某一天」做推論。

前端（streamlit）與 `predict.py` 共用這一支。可用的模型＝`models/` 底下所有
`bundle_*.pkl`，現行是 m1~m10（5 種特徵集 × 2 種 label，全部 Round 4 切分）。

2026-08-14：MLP / LSTM / 兩種集成全部移除，只保留 RF。集成必須同輪三個單模都在
才算得出來，單模一移除就不可能存在，因此一併拿掉。要復原得重寫 torch 推論路徑
並重訓（見 git 歷史）。

2026-08-22（搬進本 repo 時）：移除 `r1_rf` / `r2_rf` / `r4_rf` 這三個內建代號
與它們的檔名前綴對照表。那些 bundle 已封存、不再產生，留著只會讓
`score_path()` / `sigcurve_path()` 多一條永遠走不到的分支。現在一律掃描
`models/bundle_*.pkl`，檔名規則與原本的「非內建模型」分支**完全相同**，
既有的 `score_{key}_{split}.parquet` / `sigcurve_{key}_val_sel.csv` 照樣讀得到。

⚠️ 防洩漏核心：bundle 裡的 `stats`（median/mean/std）是**訓練期**算出來的那一份，
推論時一律沿用，不得重算（重算等於把推論期的分布資訊倒灌回標準化）。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from engine.paths import DATA_DIR, MODEL_DIR
from engine.models.train_single import apply_stats


def model_label(key: str) -> str:
    """下拉選單顯示的名稱。從 bundle 的 metadata 取，取不到就用代號。"""
    try:
        with open(key_bundle_path(key), "rb") as f:
            meta = pickle.load(f)
        desc = meta.get("desc") or meta.get("label_name") or "實驗"
        return f"🧪 {desc}"
    except Exception:
        return f"🧪 {key}"


def key_bundle_path(key: str) -> Path:
    """模型代號 → bundle 檔路徑。"""
    return MODEL_DIR / f"bundle_{key}.pkl"


def sigcurve_path(key: str) -> Path:
    """該模型在**驗證期（val_sel）**的門檻曲線 CSV。

    曲線是**無去重（訊號層級）**口徑 —— 與使用者挑門檻時看的那條線相同
    （doc/BACKTEST_LOG.md #25）。
    """
    return DATA_DIR / f"sigcurve_{key}_val_sel.csv"


def score_path(key: str, split: str) -> Path:
    """該模型某個切分的分數檔（回測用）。"""
    return DATA_DIR / f"score_{key}_{split}.parquet"


def live_score_path(key: str) -> Path:
    """訓練後新資料的即時推論分數（`score_recent.py` 產生，每日更新時附加）。

    `score_path()` 的分數檔在訓練時就固定了，只涵蓋 val/test 各切分的日期區間；
    最近這段新資料不在裡面。前端的個股歷史曲線與「累積達標天數」要接得上
    今天，就得靠這一份。
    """
    return DATA_DIR / f"score_live_{key}.parquet"


def available_keys() -> list[str]:
    """可以用的模型代號：`models/` 底下所有 `bundle_*.pkl`。

    掃描目錄而不是寫死清單 —— 模型只要把 bundle 丟進 models/ 就會出現在
    前端選單，不用每加一個就改一次程式（2026-08-14 使用者要求可自行開來比較）。
    """
    return [path.stem[len("bundle_"):] for path in sorted(MODEL_DIR.glob("bundle_*.pkl"))]


# ── 推論 ──────────────────────────────────────────────────────────────────────

def _tabular_matrix(bundle: dict, feat_day: pd.DataFrame):
    """單日特徵 → 標準化矩陣。缺欄一律補成 NaN 再走訓練期中位數補值。"""
    cols = bundle["cols"]
    frame = feat_day.reindex(columns=cols)
    return apply_stats(frame, cols, bundle["stats"])


def score_single(bundle: dict, feat: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """單一模型對某一天的全市場推論 → DataFrame[stock_id, score]。"""
    feat_day = feat[feat["date"] == date]
    if feat_day.empty:
        return pd.DataFrame(columns=["stock_id", "score"])

    x = _tabular_matrix(bundle, feat_day)
    score = bundle["model"].predict_proba(x)[:, 1]
    return pd.DataFrame({"stock_id": feat_day["stock_id"].to_numpy(),
                         "score": score.astype("float32")})


def load_by_key(key: str) -> dict:
    path = key_bundle_path(key)
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，請先執行 `make train`")
    with open(path, "rb") as f:
        return pickle.load(f)


def score_for_date(key: str, feat: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """模型代號 → 某一天的推論 → DataFrame[stock_id, score]。"""
    return score_single(load_by_key(key), feat, date)


# ── 驗證期門檻曲線（前端滑桿旁邊即時顯示用）──────────────────────────────────

# 使用者在無去重曲線上挑定的門檻（doc/BACKTEST_LOG.md #24 / #25）。
# 沒挑過的代號 → `default_threshold()` 退回「訊號率 1%」那一點。
# 2026-08-15 使用者從 Round 4 各模型的 val_sel 曲線挑定下列五組。
CHOSEN_THRESHOLDS = {
    # 2026-08-16 使用者從 val_sel 曲線挑定的 10 個模型門檻
    "m1_base_up20": 0.60,      "m2_nomkt_up20": 0.60,
    "m3_v3_up20": 0.60,        "m4_v3nomkt_up20": 0.60,
    "m5_v3nomv_up20": 0.60,    "m6_base_nobear": 0.60,
    "m7_nomkt_nobear": 0.60,   "m8_v3_nobear": 0.58,
    "m9_v3nomkt_nobear": 0.57, "m10_v3nomv_nobear": 0.585,
}
FALLBACK_SIGNAL_RATE = 0.01


def default_threshold(key: str, curve: pd.DataFrame) -> float:
    """滑桿的預設位置。"""
    if key in CHOSEN_THRESHOLDS:
        return CHOSEN_THRESHOLDS[key]
    if curve.empty:
        return 0.5
    target = max(1, int(len(curve) * FALLBACK_SIGNAL_RATE))
    return float(curve.iloc[(curve["n"] - target).abs().idxmin()]["threshold"])


def load_sigcurve(key: str) -> pd.DataFrame:
    path = sigcurve_path(key)
    if not path.exists():
        return pd.DataFrame(columns=["threshold", "n", "win_rate", "avg_return"])
    return pd.read_csv(path)


def sigcurve_stats_at(curve: pd.DataFrame, threshold: float) -> dict | None:
    """曲線上「門檻 >= threshold」的那一段：筆數、勝率、平均報酬。

    曲線每一列是「以該列 threshold 當門檻」的累積統計，所以直接取**最接近且
    不低於**指定門檻的那一列即可（沒有就代表門檻高過全部訊號）。
    """
    if curve.empty:
        return None
    above = curve[curve["threshold"] >= threshold]
    if above.empty:
        return None
    row = above.iloc[-1]
    return {"n": int(row["n"]), "win_rate": float(row["win_rate"]),
            "avg_return": float(row["avg_return"]),
            "threshold": float(row["threshold"])}
