"""SWING 模型：只分辨「上漲-起 vs 下跌-起」（2026-09-05 新增）。
輸出：data/score_swing_{split}.parquet   欄位 date / stock_id / score

跟既有 2 個模型的差別：
- **標的不同**：`label_swing_up`（build_labels_swing.py），只有波段起點的兩類
  有值，其餘七類是 NA 不進訓練。既有模型的 `label_up20` 是全樣本都有值。
- **模型類型不同**：這支用 LightGBM。實測 RandomForest 只差 0.005
  （test AUC 0.747 vs 0.752）但慢 3 倍；兩者接近正好說明訊號不是靠單一模型
  鑽出來的。要換回 RF 只需改 `_fit`。

為什麼值得多這一個模型（2026-09-05 實測）：
  九類混在一起訓練 test AUC 只有 0.53~0.55，縮到「起」的兩類是 **0.75**。
  差別在於中段/末段/盤整本來就沒有方向性，混進去只是稀釋訊號。
  分數越高效果越好且單調：門檻 0.90 時「先漲 5%」勝率 63.9%（不篩 44.7%），
  買到的樣本有 40.5% 真的是上漲-起（全體只有 12.8%）。

⚠️ 切分的 embargo 比既有模型長很多。`label_up20` 只前瞻 20 個交易日，一個月
embargo 就夠；本標的的 ZigZag 轉折確認延遲中位數 6 天但**最長到 445 天**，
所以 train→val 與 val→test 各留 6 個月。這是刻意的保守，不是筆誤。

⚠️ 特徵集是 `submodel_config.feature_cols("SWING", ...)`，跟 UP20 同一組但**不含
mkt_***：放進去模型會改去猜大盤方向，實測第一輪就過擬合早停（train/val/test 的
正例率分別 42.3%/48.8%/40.0%，大盤循環主宰了目標）。

**分數檔一律是全市場口徑**。label 只存在於「起」的那兩類，而那是事後才知道的；
推論時看不到未來的 label，只對有 label 的列打分等於偷看答案，`threshold_curve` /
`backtest.simulate()` 也會拿到錯的母體。所以 `score_swing_{split}.parquet` 與
`score_live_swing.parquet` 都是全市場，**AUC 才在有 label 的子集上算**。

用法：
  python -m engine.models.train_swing              # 訓練 + 產訓練期分數 + 全市場分數
  python -m engine.models.train_swing --no-live    # 只產訓練期分數
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 專案路徑一律走 engine/paths.py（唯一來源），不要各檔自行推導
from engine.paths import DATA_DIR, MODEL_DIR  # noqa: E402
from engine.models.submodel_config import feature_cols, label_col  # noqa: E402
from engine.models.train_single import apply_stats, fit_stats  # noqa: E402

MODEL_ID = "SWING"
LABEL = label_col(MODEL_ID)
TAG = "swing"

# embargo：train→val_es 六個月、val_es→val_sel 兩個月、val_sel→test 六個月。
# val_es 是早停用、val_sel 是挑門檻用，兩個都會影響最終決策，所以中間也要留。
# 實測標籤解析延遲中位數 21 個交易日、P90 57，兩個月（約 42 天）蓋得住大半。
SPLITS: dict[str, tuple[str, str]] = {
    "train":   ("2019-01-01", "2023-06-30"),
    "val_es":  ("2024-01-01", "2024-04-30"),
    "val_sel": ("2024-07-01", "2024-12-31"),
    "test":    ("2025-07-01", "2025-12-31"),
    "test2":   ("2026-01-01", "2026-07-31"),
}
EVAL_SPLITS = ("train", "val_es", "val_sel", "test", "test2")

LGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.02, num_leaves=63,
    min_child_samples=200, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.5, reg_lambda=10.0, n_jobs=-1, verbose=-1,
)
EARLY_STOPPING_ROUNDS = 150


def _feature_path() -> "object":
    p = DATA_DIR / "features.parquet"
    if not p.exists():
        raise RuntimeError("features.parquet 不存在，請先跑 make features")
    return p


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    """逐年讀特徵再跟 label 內連接，避免一次把 1.8GB 的 features 全載進記憶體。"""
    labels = pd.read_parquet(DATA_DIR / "labels_swing.parquet")
    labels["date"] = pd.to_datetime(labels["date"])
    labels = labels.dropna(subset=[LABEL])
    if labels.empty:
        raise RuntimeError("labels_swing.parquet 沒有可訓練樣本，請先跑 build_labels_swing")
    logger.info(f"可訓練 label {len(labels):,} 筆，正例率 {labels[LABEL].mean():.3f}")

    path = _feature_path()
    years = sorted(labels["date"].dt.year.unique())
    frames = []
    for year in years:
        feat = pd.read_parquet(path, filters=[
            ("date", ">=", pd.Timestamp(f"{year}-01-01")),
            ("date", "<=", pd.Timestamp(f"{year}-12-31")),
        ])
        if feat.empty:
            continue
        feat["date"] = pd.to_datetime(feat["date"])
        for col in feat.columns:
            if col not in ("date", "stock_id") and feat[col].dtype == "float64":
                feat[col] = feat[col].astype("float32")
        frames.append(labels.merge(feat, on=["date", "stock_id"], how="inner"))
        logger.info(f"  {year}: 併入 {len(frames[-1]):,} 筆")
    if not frames:
        raise RuntimeError("label 與 features 沒有任何交集，檢查兩邊的日期範圍")
    df = pd.concat(frames, ignore_index=True)
    # 特徵集走 submodel_config.feature_cols（唯一入口）。它已經擋掉 revenue_year /
    # revenue_month 這類日曆索引——那兩欄在 train 與 test 的分布零重疊（KS=1.0），
    # 樹只要切一刀所有測試列就落進同一分支，是結構性的泛化失效（見 EXCLUDE_COLS）。
    meta = {"date", "stock_id", "swing_state", "swing_phase", "swing_amp", LABEL}
    cols = feature_cols(MODEL_ID, [c for c in df.columns if c not in meta])
    logger.info(f"訓練資料 {len(df):,} 筆 × 特徵 {len(cols)}")
    return df, cols


def split_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = SPLITS[name]
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask]


def _fit(train: pd.DataFrame, val: pd.DataFrame, cols: list[str], stats: dict):
    """要換模型類型只改這裡。RF 版本見檔頭說明（差 0.005，慢 3 倍）。

    ⚠️ 前處理走 `train_single.fit_stats/apply_stats`（中位數補值 + 標準化 +
    ±10σ winsorize），統計量只從訓練期算。LightGBM 本身能吃 NaN，但**推論端
    `bundle.score_single()` 一律會套 apply_stats**——訓練不套的話，前端算出來的
    分數會跟訓練期完全對不上，而且不會報錯。
    """
    import lightgbm as lgb
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(apply_stats(train, cols, stats), train[LABEL].astype("int8"),
              eval_set=[(apply_stats(val, cols, stats), val[LABEL].astype("int8"))],
              eval_metric="auc",
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    logger.info(f"早停於第 {model.best_iteration_} 輪")
    return model


def run(live: bool = True) -> dict[str, float]:
    df, cols = load_dataset()
    frames = {name: split_frame(df, name) for name in SPLITS}
    for name, part in frames.items():
        if part.empty:
            logger.warning(f"切分 {name} 沒有資料（{SPLITS[name]}）")
    if frames["train"].empty or frames["val_es"].empty:
        raise RuntimeError("train 或 val_es 是空的，無法訓練")

    stats = fit_stats(frames["train"], cols)
    model = _fit(frames["train"], frames["val_es"], cols, stats)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 全市場推論一次，各切分的分數檔都從這裡切（口徑正確才餵得進 threshold_curve）
    live_scores = score_market(model, cols, stats) if live else pd.DataFrame()
    aucs: dict[str, float] = {}
    print(f"\n{'split':10s} {'n(有label)':>12s} {'n(全市場)':>11s} {'base':>7s} "
          f"{'AUC':>8s} {'PR-AUC':>8s} {'lift':>7s}")
    for name in EVAL_SPLITS:
        part = frames[name]
        if part.empty:
            continue
        y = part[LABEL].astype("int8").to_numpy()
        score = model.predict_proba(apply_stats(part, cols, stats))[:, 1]
        base = float(y.mean())
        auc = roc_auc_score(y, score) if len(np.unique(y)) > 1 else float("nan")
        pr = average_precision_score(y, score) if len(np.unique(y)) > 1 else float("nan")
        aucs[name] = auc
        start, end = SPLITS[name]
        # 全市場推論只涵蓋評估區間（train 期不在裡面，也不需要——它不進回測）
        m = (live_scores["date"] >= start) & (live_scores["date"] <= end) \
            if not live_scores.empty else None
        if m is not None and bool(m.any()):
            out = live_scores.loc[m]
            _assert_same_scores(part, score, out, name)
        elif name == "train":
            # train 期不在推論區間，也不進回測；它的分數檔只給診斷用，明確標名
            out = pd.DataFrame({"date": part["date"].to_numpy(),
                                "stock_id": part["stock_id"].to_numpy(), "score": score})
        else:
            # 只有 label 的列是事後口徑，餵進 threshold_curve/simulate 母體會少 85%。
            # 寧可不產檔，也不要產一個同名同 schema 但口徑錯掉的檔案。
            raise RuntimeError(
                f"{name}：沒有全市場分數可寫。score_{TAG}_{name}.parquet 必須是全市場"
                "口徑，不要用 --no-live 產它")
        print(f"{name:10s} {len(part):>12,} {len(out):>11,} {base:>7.3f} {auc:>8.4f} "
              f"{pr:>8.4f} {pr / base if base else float('nan'):>7.2f}")
        out.to_parquet(DATA_DIR / f"score_{TAG}_{name}.parquet", index=False,
                       engine="pyarrow")

    save_bundle(model, cols, stats, frames)

    imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    print("\n特徵重要度 Top 15：")
    for k, v in imp.head(15).items():
        print(f"  {k:<34}{v:>8.0f}")
    return aucs


def save_bundle(model, cols: list[str], stats: dict, frames: dict) -> None:
    """存成 models/bundle_swing.pkl —— 前端是掃這個目錄列選單的（bundle.py:116）。

    多存一個 `exit_rule`：既有兩個模型的出場是 backtest.CURRENT_EXIT_RULES（移動
    停利/停損/均線），swing 的出場是**分數跌破門檻**，走 backtest/score_exit.py。
    舊 bundle 沒有這一欄，讀取端用 .get() 取，預設就是既有行為。
    """
    import pickle
    from datetime import datetime

    val = frames["val_sel"]
    score = model.predict_proba(apply_stats(val, cols, stats))[:, 1] if len(val) else np.array([0.0, 1.0])
    bundle = {
        "family": "lgbm", "round": None, "cols": cols,
        "features_file": "features.parquet",
        "stats": stats, "model": model, "arch": None, "params": dict(LGB_PARAMS),
        "config_round": None, "config_source_key": None,
        "score_min": float(score.min()), "score_max": float(score.max()),
        "val_split": "val_sel", "n_train": int(len(frames["train"])),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "label_name": LABEL, "desc": "波段起漲vs起跌",
        "dropped_prefixes": [], "dropped_cols": [],
        # 這個模型專屬：出場不走 CURRENT_EXIT_RULES，走分數門檻
        "exit_rule": {"type": "score", "sell_threshold": 0.20,
                      "module": "engine.backtest.score_exit"},
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"bundle_{TAG}.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"bundle → {path}")


def _assert_same_scores(labelled: pd.DataFrame, direct: np.ndarray,
                        market: pd.DataFrame, name: str) -> None:
    """全市場分數在有 label 的那些列，必須跟直接推論逐列相同。

    兩條路徑的特徵取法不同（一條走 label join、一條走全表），欄位順序或補值只要
    有一點漂移，分數就會靜默偏掉而不報錯。這裡把它釘死。
    """
    ref = pd.DataFrame({"date": labelled["date"].to_numpy(),
                        "stock_id": labelled["stock_id"].to_numpy(), "s": direct})
    merged = ref.merge(market, on=["date", "stock_id"], how="inner")
    if merged.empty:
        raise RuntimeError(f"{name}：全市場分數與有 label 的列沒有交集")
    diff = float((merged["s"] - merged["score"]).abs().max())
    if diff > 1e-9:
        raise RuntimeError(f"{name}：兩條推論路徑不一致（max|diff|={diff:.3e}）")
    logger.info(f"  {name}：逐列一致 {len(merged):,}/{len(ref):,} 列（max|diff|={diff:.1e}）")


def score_market(model, cols: list[str], stats: dict) -> pd.DataFrame:
    """對評估區間的**每一列**打分（不看 label），另存 score_live_{TAG}.parquet。"""
    start = min(SPLITS[n][0] for n in EVAL_SPLITS if n != "train")
    end = max(SPLITS[n][1] for n in EVAL_SPLITS)
    logger.info(f"全市場推論 {start} ~ {end}")
    frames = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        lo = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(start))
        hi = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(end))
        feat = pd.read_parquet(_feature_path(),
                               filters=[("date", ">=", lo), ("date", "<=", hi)])
        if feat.empty:
            continue
        feat["date"] = pd.to_datetime(feat["date"])
        missing = [c for c in cols if c not in feat.columns]
        if missing:
            raise RuntimeError(f"特徵檔缺欄，分數會是錯的：{missing[:5]}")
        frames.append(pd.DataFrame({
            "date": feat["date"].to_numpy(), "stock_id": feat["stock_id"].to_numpy(),
            "score": model.predict_proba(apply_stats(feat, cols, stats))[:, 1]}))
        logger.info(f"  {year}: {len(frames[-1]):,} 筆")
    live = pd.concat(frames, ignore_index=True)
    out = DATA_DIR / f"score_live_{TAG}.parquet"
    live.to_parquet(out, index=False, engine="pyarrow")
    logger.info(f"{out.name} 寫入 {len(live):,} 筆（全市場口徑）")
    return live


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-live", action="store_true",
                        help="只產訓練期分數，不做全市場推論")
    args = parser.parse_args()
    run(live=not args.no_live)
