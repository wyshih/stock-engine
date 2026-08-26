"""訓練好的模型 bundle：存檔格式、載入、以及對「某一天」做推論。

前端（streamlit）與 `predict.py` 共用這一支。可用的模型＝`models/` 底下所有
`bundle_*.pkl`，現行是 ①②③⑥⑧ 五個（base/nomkt/v3 三種特徵集 × 兩種 label，Round 4 切分）。

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

import logging
import pickle
from pathlib import Path

import pandas as pd

from engine.paths import DATA_DIR, MODEL_DIR
from engine.models.train_single import apply_stats

logger = logging.getLogger(__name__)

# ── 模型 ↔ 特徵檔 ──────────────────────────────────────────────────────────────
# 五個模型分成兩群、用**不同的特徵檔**訓練（見 engine/models/train_all.sh）：
#   base / nomkt（m1/m2/m6）  data/features.parquet     378 欄
#   v3（m3/m8）               data/features_v3.parquet  518 欄
# 推論時餵錯檔案不會報錯 —— `reindex` 會把缺的欄位變成 NaN，再被訓練期中位數
# 補上。實測 m3 餵 features.parquet 有 41% 的欄位被中位數取代，Top-20 只剩
# 3/20 正確。所以哪一份特徵檔由 bundle 自己說了算，見 `features_file()`。
BASE_FEATURES_FILE = "features.parquet"
V3_FEATURES_FILE = "features_v3.parquet"

# v3 特徵轉換產生的後綴（engine/features/v3/spec.py 的 SUFFIX_*）：
#   _sz  對自己過去 250 日的 z-score
#   _szx _sz 的當日橫斷面百分位
#   _xs  原值的當日橫斷面百分位
# 只有 v3 特徵集會有這些欄位，base/nomkt 一個都沒有。
V3_COL_SUFFIXES = ("_sz", "_szx", "_xs")


def model_label(key: str) -> str:
    """下拉選單顯示的名稱。從 bundle 的 metadata 取，取不到就用代號。

    舊版會在名稱前掛 🧪 表示「非內建的實驗模型」。內建／實驗的區分已經隨
    r1/r2/r4 那組死代號一起移除 —— 現在 ①②③⑥⑧ 就是正式模型，全部掛 🧪 反而
    誤導。真正的實驗模型（自己丟進 models/ 的）靠 desc 自己說明。
    """
    try:
        with open(key_bundle_path(key), "rb") as f:
            meta = pickle.load(f)
        return meta.get("desc") or meta.get("label_name") or key
    except Exception:
        return key


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


def _infer_features_file(cols: list[str]) -> str:
    """沒有 `features_file` 欄位的舊 bundle → 從欄名回推特徵檔。

    ⚠️ 這個回退存在的理由：`features_file` 是 2026-08-24 才加進 bundle 的，
    在那之前訓練的五個 bundle（models/bundle_*.pkl）裡沒有這個欄位，而重訓
    要好幾個小時、還會讓門檻全部作廢（CLAUDE.md 規則 7：重訓必須重挑門檻）。
    所以用欄名判斷而不是要求重訓。

    判準用 v3 轉換的後綴而不是欄數 —— 欄數會隨資料漂移（CLAUDE.md 提到 v3
    重建後是 518 欄而不是文件上的 509），後綴則是 v3 轉換的定義本身，不會變。
    """
    if any(c.endswith(V3_COL_SUFFIXES) for c in cols):
        return V3_FEATURES_FILE
    return BASE_FEATURES_FILE


def features_file(bundle: dict) -> str:
    """這個 bundle 要用哪一份特徵檔（檔名，不是絕對路徑）。

    存檔名而非絕對路徑：bundle 搬到別台機器、或 repo 換位置時仍然對得到。
    """
    return bundle.get("features_file") or _infer_features_file(bundle["cols"])


def features_path(bundle: dict) -> Path:
    """這個 bundle 要用的特徵檔完整路徑（走 paths.py，CLAUDE.md 規則 13）。"""
    return DATA_DIR / features_file(bundle)


def features_file_for_key(key: str) -> str:
    """模型代號 → 特徵檔名。"""
    return features_file(load_by_key(key))


# ── 推論 ──────────────────────────────────────────────────────────────────────

# 缺欄比例超過這個門檻就視為「餵錯特徵檔」而不是「特徵還沒暖機」。
# 少量缺欄是合理的（例如新上市股票某些長週期特徵還算不出來），照舊補中位數；
# 餵錯檔案則是每天產出整份錯誤推薦名單，必須擋下來。
MISSING_COLS_TOLERANCE = 0.01


def _check_columns(bundle: dict, feat: pd.DataFrame) -> list[str]:
    """檢查特徵表有沒有 bundle 要的欄位，缺太多就 raise。

    為什麼要擋：`reindex` 把缺的欄位變成 NaN，`apply_stats()` 再用訓練期中位數
    補上 —— 不報錯、不警告，模型照樣吐得出分數，只是那些分數毫無意義。
    """
    cols = bundle["cols"]
    missing = [c for c in cols if c not in feat.columns]
    if not missing:
        return missing

    ratio = len(missing) / len(cols)
    if ratio > MISSING_COLS_TOLERANCE:
        name = bundle.get("desc") or bundle.get("label_name") or "（未命名）"
        raise ValueError(
            f"特徵檔不對：模型「{name}」需要 {features_file(bundle)}（{len(cols)} 欄），"
            f"但收到的特徵表只有 {len(feat.columns)} 欄，其中 {len(missing)} 欄"
            f"（{ratio:.1%}）缺漏，例如 {missing[:5]}。"
            f"缺的欄位會被訓練期中位數填掉，推論結果沒有意義。"
            f"請依 bundle 指定的特徵檔載入（engine.models.bundle.features_path()）。")

    # 少量缺欄：仍照舊補中位數，但要留下痕跡，不然又是一次靜默降級
    logger.warning(
        f"特徵表缺 {len(missing)} 欄（{ratio:.2%}，容忍範圍內），"
        f"這些欄位改用訓練期中位數：{missing[:5]}")
    return missing


def _tabular_matrix(bundle: dict, feat_day: pd.DataFrame):
    """單日特徵 → 標準化矩陣。缺欄一律補成 NaN 再走訓練期中位數補值。"""
    cols = bundle["cols"]
    frame = feat_day.reindex(columns=cols)
    return apply_stats(frame, cols, bundle["stats"])


def score_single(bundle: dict, feat: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """單一模型對某一天的全市場推論 → DataFrame[stock_id, score]。"""
    _check_columns(bundle, feat)
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
    # 2026-08-26 使用者在 val_sel 曲線（dedup=False 訊號層級）上挑定。
    # 這是 P1 重建後的新模型，先前那組（0.60/0.58）是舊模型的值，已作廢。
    #
    # 各門檻在驗證期（2024H2）的表現：
    #   模型              門檻    訊號數  佔比    勝率    平均      中位
    #   m1_base_up20      0.77     630   0.29%  80.3%  +9.29%   +8.55%
    #   m2_nomkt_up20     0.74     333   0.15%  78.4%  +10.29%  +8.34%
    #   m3_v3_up20        0.70   1,598   0.73%  71.9%  +5.67%   +6.79%
    #   m6_base_nobear    0.625    259   0.20%  62.2%  +4.77%   +5.37%
    #   m8_v3_nobear      0.60     313   0.24%  57.5%  +1.27%   +4.30%
    #
    # ⚠️ m2 / m6 / m8 在驗證期只有兩三百筆訊號，統計上偏薄。測試期長三倍，
    #    實際筆數會多一些，但看回測結果時要記得這件事。
    # ⚠️ 每次重訓都必須重挑（規則 7）—— 分數分布會變，絕對值搬不動。
    "m1_base_up20": 0.77,
    "m2_nomkt_up20": 0.74,
    "m3_v3_up20": 0.70,
    "m6_base_nobear": 0.625,
    "m8_v3_nobear": 0.60,
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
