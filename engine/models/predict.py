"""每日推薦名單產生。

2026-08-12 改寫：委員會 + Meta stacking 已拆除（實測遠差於單一 RF，也差於隨機
對照），改成「單一 ground truth `label_up20` + 可選模型」。
2026-08-14：MLP / LSTM / 兩種集成也一併移除，只剩 RF × 2 個訓練期（Round 1/2）。
模型代號與推論邏輯集中在 `bundle.py`，前端與這一支共用。

用法：
  python -m engine.models.predict --model m1_base_up20
  python -m engine.models.predict --model m1_mdd10 --date 2026-06-26
  python -m engine.models.predict --model m1_mdd10 --top 30   # Top-N 優先於門檻
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


from engine.models.bundle import (available_keys, features_file, features_path,
                                  load_by_key, load_sigcurve, model_label,
                                  score_single, sigcurve_stats_at)
from engine.paths import DATA_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TOP_N = 20

# 建議購買價格區間 / 觀察天數（僅供參考，非精確買賣點）：
# 價格區間 = 訊號當天收盤價 ± BUY_BAND_PCT。
# 觀察天數用 ground truth 自己的 horizon：`label_up20` 是「未來 20 個交易日裡
# 上漲天數 >= 10」，所以是 20 天（舊版是 label_meta 的 5 天，已不適用）。
BUY_BAND_PCT = 0.02
OBSERVE_DAYS = 20


def _attach_buy_hint(scores: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """加入當天收盤價、建議購買價格區間（收盤價±2%）跟觀察天數。"""
    price_day = pd.read_parquet(DATA_DIR / "price.parquet",
                                columns=["date", "stock_id", "close"])
    price_day["date"] = pd.to_datetime(price_day["date"])
    price_day = price_day[price_day["date"] == date][["stock_id", "close"]]

    out = scores.merge(price_day, on="stock_id", how="left")
    out["buy_price_low"] = out["close"] * (1 - BUY_BAND_PCT)
    out["buy_price_high"] = out["close"] * (1 + BUY_BAND_PCT)
    out["observe_days"] = OBSERVE_DAYS
    return out


def attach_stock_info(df: pd.DataFrame) -> pd.DataFrame:
    """補上股票名稱與產業（stock_list.parquet 不存在時原樣回傳）。"""
    path = DATA_DIR / "stock_list.parquet"
    if not path.exists() or df.empty:
        return df
    sl = pd.read_parquet(path, columns=["stock_id", "stock_name", "industry"])
    return df.merge(sl, on="stock_id", how="left")


def load_features(path: Path | None = None) -> pd.DataFrame:
    """載入特徵表。預設是 base 特徵檔；要用哪一份一律問 `bundle.features_path()`。

    ⚠️ 不要在這裡寫死 features.parquet 當「唯一的特徵檔」—— 2026-09-02 移除 v3
    家族之前，m3/m8 是用 features_v3.parquet 訓練的，餵錯檔案不會報錯（缺欄被
    中位數補掉），只會每天安靜地產出錯誤的推薦名單。現在雖然只剩一份特徵檔，
    這個參數仍然由 bundle 決定，不要改回寫死。
    """
    feat = pd.read_parquet(DATA_DIR / "features.parquet" if path is None else path)
    feat["date"] = pd.to_datetime(feat["date"])
    return feat


def load_features_for(model_key: str) -> pd.DataFrame:
    """該模型訓練時用的那一份特徵表。"""
    return load_features(features_path(load_by_key(model_key)))


def get_scores_for_date(model_key: str, target_date: str | None = None,
                        feat: pd.DataFrame | None = None,
                        model: dict | None = None) -> pd.DataFrame:
    """指定模型、指定日期（不傳則用最新一天）全市場的分數，**不做任何篩選**。

    `feat` 不傳的話會依 bundle 指定的特徵檔載入；要自己傳就必須傳對那一份
    （傳錯會被 `bundle.score_single()` 擋下並 raise）。

    回傳欄位：stock_id / score / close / buy_price_low / buy_price_high /
    observe_days，供前端自行套用門檻、股價區間等條件。
    """
    # `model` 是已載入的 bundle（呼叫端已經載過就傳進來，省一次 150MB 的 unpickle）
    model = load_by_key(model_key) if model is None else model
    feat = load_features(features_path(model)) if feat is None else feat
    date = pd.Timestamp(target_date) if target_date else feat["date"].max()

    scores = score_single(model, feat, date)
    if scores.empty:
        return pd.DataFrame()
    scores = scores.sort_values("score", ascending=False).reset_index(drop=True)
    return _attach_buy_hint(scores, date)


def run(model_key: str, target_date: str | None = None, top_n: int | None = None,
        threshold: float | None = None) -> pd.DataFrame:
    model = load_by_key(model_key)
    feat = load_features(features_path(model))
    date = pd.Timestamp(target_date) if target_date else feat["date"].max()
    logger.info(f"模型：{model_label(model_key)}（{model_key}）"
                f"，特徵檔：{features_file(model)}（{len(feat.columns)} 欄）")

    scores = get_scores_for_date(model_key, str(date.date()), feat, model)
    if scores.empty:
        logger.error(f"{date.date()} 無特徵資料或模型推論不出結果")
        return pd.DataFrame()
    logger.info(f"預測日期：{date.date()}，股票數：{len(scores)}")

    if top_n is not None:
        result, label = scores.head(top_n), f"Top-{top_n}"
    elif threshold is not None:
        result, label = scores[scores["score"] >= threshold], f"門檻={threshold:.4f}"
        stats = sigcurve_stats_at(load_sigcurve(model_key), threshold)
        if stats:
            logger.info(
                f"該門檻在驗證期（val_sel）：{stats['n']:,} 筆訊號、"
                f"勝率 {stats['win_rate']:.1%}、平均報酬 {stats['avg_return']:+.2%}")
    else:
        result, label = scores.head(DEFAULT_TOP_N), f"Top-{DEFAULT_TOP_N}（未指定門檻）"

    result = attach_stock_info(result)
    display_cols = [c for c in ["stock_id", "stock_name", "industry", "score", "close",
                                "buy_price_low", "buy_price_high", "observe_days"]
                    if c in result.columns]
    logger.info(f"推薦名單（{label}，共 {len(result)} 支）：")
    if not result.empty:
        print(result[display_cols].to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="m1_base_up20")
    parser.add_argument("--date", default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    if args.model not in available_keys():
        raise SystemExit(
            f"{args.model} 的 bundle 還沒訓練，可用的有：{available_keys()}\n"
            f"請先執行 `make train`")
    run(args.model, args.date, args.top, args.threshold)
