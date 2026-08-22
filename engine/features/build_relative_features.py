"""
自我正規化特徵（2026-07-29 新增）：把「絕對值」轉成「相對這檔股票自己的歷史分布」。

## 動機（doc/AUDIT_20260728.md §D-5，使用者 2026-07-29 提出）

系統目前有兩種正規化，缺第三種：
  (a) 跨股票、同一天          → `rank_*` 前綴（build_features.py:164）✅
  (b) 相對自己的價格水位      → `close_ma20_ratio`、`vol_ratio` 等 ✅
  (c) **相對自己的歷史分布**  → 幾乎沒有 ❌  ← 本模組補這一塊

具體問題舉例：
- 外資買超 1000 張，對台積電是雜訊、對小型股是重大事件，模型看到的卻是同一個數字
- `return_20d = 30%` 對低波動股是極端事件、對投機股是家常便飯
- `vol_ratio` 的基準只有 5 日均量，**且用平均數**——成交量分布極度右偏，
  一天暴量就把基準拉高、之後好幾天的比值都被壓下去。而且 5 日基準本身已經被
  當下的爆量污染（本策略 73% 進場點在剛漲停的隔天）

## 設計決定

**用中位數不用平均數**：成交量/金額近似對數常態，平均數會被單日暴量主導，
中位數才代表「平常的量」。

**用金額（amount）不用張數（volume）**：金額已含價格，跨時間比較才對等
（股價漲一倍、張數不變，實際資金流入是兩倍）。

**窗口選擇**：`rel_amt_*` 依使用者指定掃 W ∈ {10,20,30,60,120,250} 全部；
其餘特徵族用 {60, 250}（短/長）。理由：BACKTEST_LOG #17 實測證明，當基準窗口
與被正規化指標自己的窗口重疊時（該次是 10 日中位數 vs `natr_14` 的 14 日窗口），
比值會緊貼 1、退化成常數，**沒有資訊**。W=30/120 與 60/250 高度相關，先不重複。

⚠️ **本專案過去加特徵的成功率是 0**（KMeans 價量形態重要度排 236~347/347；
Meta 加原始特徵報酬變差 6.26% vs 8.99%）。訓練資料只有 892k 列 / 2 年單一 regime，
新特徵很可能只是雜訊。**驗收必須用對照組重訓**（見 BACKTEST_LOG 任務 #6）。

輸出：`data/relative_features.parquet`
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 專案路徑一律走 code/paths.py（唯一來源），不要各檔自行推導
from engine.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402


# 使用者指定的量能窗口（全掃）
AMT_WINDOWS = [10, 20, 30, 60, 120, 250]
# 其餘特徵族的窗口（短/長各一，避免高度相關的中間值灌水）
STD_WINDOWS = [60, 250]

# 比值類的 clip 範圍：避免暖機期或除以極小值產生的荒謬倍數主導樹的分裂點
RATIO_CLIP = (0.0, 20.0)
Z_CLIP = (-10.0, 10.0)

INDEX_IDS = {"TWII"}


def _min_periods(w: int) -> int:
    """暖機期要求：至少窗口的 1/3，避免前幾天用 2~3 個點就算出極端基準值。"""
    return max(5, w // 3)


def _rel_amount(price: pd.DataFrame) -> pd.DataFrame:
    """量能相對自己：amount ÷ 該股自己過去 W 日 amount 中位數。"""
    out = price[["date", "stock_id"]].copy()
    g = price.groupby("stock_id")["amount"]
    for w in AMT_WINDOWS:
        base = g.transform(lambda s, w=w: s.rolling(w, min_periods=_min_periods(w)).median())
        out[f"rel_amt_{w}"] = (price["amount"] / base.replace(0, np.nan)) \
            .clip(*RATIO_CLIP).astype("float32")
    return out


def _rel_risk_adj_momentum(price: pd.DataFrame) -> pd.DataFrame:
    """
    風險調整動能：H 日報酬 ÷ 該股自己的報酬波動度（× sqrt(H) 做期間對齊）。
    等同資訊比率——把「漲 30%」換算成「相對自己平常波動漲了幾個標準差」。
    """
    out = price[["date", "stock_id"]].copy()
    g = price.groupby("stock_id")["close"]
    ret1 = g.transform(lambda s: s.pct_change())
    gr = ret1.groupby(price["stock_id"])

    for w in STD_WINDOWS:
        sd = gr.transform(lambda s, w=w: s.rolling(w, min_periods=_min_periods(w)).std())
        for h in (20, 60):
            rh = g.transform(lambda s, h=h: s / s.shift(h) - 1.0)
            denom = (sd * np.sqrt(h)).replace(0, np.nan)
            out[f"rel_ramom_{h}_{w}"] = (rh / denom).clip(*Z_CLIP).astype("float32")
    return out


def _rel_dist_atr(feat: pd.DataFrame) -> pd.DataFrame:
    """
    距離用 ATR 計價：離 N 日高/低點的距離換算成「幾個日波動」。
    「離高點 5%」對日波動 1% 的股票是 5 天的路程，對日波動 8% 的只是半天。
    """
    out = feat[["date", "stock_id"]].copy()
    natr = (feat["natr_14"] / 100.0).replace(0, np.nan)
    for col in ("dist_high_60", "dist_low_60", "dist_high_240", "dist_low_240"):
        if col in feat.columns:
            out[f"rel_{col}_atr"] = (feat[col] / natr).clip(*Z_CLIP).astype("float32")
    return out


def _rel_self_percentile(feat: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    指標的自身歷史百分位：該欄在**這檔股票自己**過去 W 日中排第幾（0~1）。

    注意跟既有 `*_rank` / `rank_*` 的差別：那些是**跨股票同日**排名（軸 a），
    這裡是**同一檔跨時間**排名（軸 c）。有些股票 RSI 很少超過 70、有些常上 90，
    「RSI=75」對兩者意義完全不同。
    """
    out = feat[["date", "stock_id"]].copy()
    for col in cols:
        if col not in feat.columns:
            logger.warning(f"  找不到欄位 {col}，跳過")
            continue
        g = feat.groupby("stock_id")[col]
        for w in STD_WINDOWS:
            out[f"rel_pct_{col}_{w}"] = g.transform(
                lambda s, w=w: s.rolling(w, min_periods=_min_periods(w))
                                .rank(pct=True)).astype("float32")
    return out


def _rel_chip_z(chip: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    籌碼相對自己：買賣超比率 ÷ 該股自己過去 W 日該比率的標準差。

    現有的 `foreign_net_ratio` 等已經對成交量做過正規化（軸 b），但仍然沒有回答
    「這個買超量對**這檔股票而言**算不算異常」——有些股票長期就有穩定外資流入，
    有些平常根本沒有法人參與。除以自身歷史標準差才是「幾個標準差的異常買盤」。
    """
    out = chip[["date", "stock_id"]].copy()
    for col in cols:
        if col not in chip.columns:
            logger.warning(f"  找不到欄位 {col}，跳過")
            continue
        g = chip.groupby("stock_id")[col]
        for w in STD_WINDOWS:
            sd = g.transform(lambda s, w=w: s.rolling(w, min_periods=_min_periods(w)).std())
            out[f"rel_chipz_{col}_{w}"] = (chip[col] / sd.replace(0, np.nan)) \
                .clip(*Z_CLIP).astype("float32")
    return out


def build() -> pd.DataFrame:
    logger.info("讀取來源資料…")
    price = pd.read_parquet(DATA_DIR / "price.parquet",
                            columns=["date", "stock_id", "close", "amount"])
    price["date"] = pd.to_datetime(price["date"])
    price = price[~price["stock_id"].isin(INDEX_IDS)]
    price = price.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # natr_14 只在 talib_features；rsi_14 / k_value / bb_width_pct / dist_* 都在
    # price_features（2026-07-29 踩過：誤以為 rsi/dist 在 talib，讀取直接失敗）
    tal = pd.read_parquet(DATA_DIR / "talib_features.parquet",
                          columns=["date", "stock_id", "natr_14"])
    tal["date"] = pd.to_datetime(tal["date"])

    pf = pd.read_parquet(DATA_DIR / "price_features.parquet",
                         columns=["date", "stock_id", "rsi_14", "k_value", "bb_width_pct",
                                  "dist_high_60", "dist_low_60",
                                  "dist_high_240", "dist_low_240"])
    pf["date"] = pd.to_datetime(pf["date"])
    pf = pf.merge(tal, on=["date", "stock_id"], how="left")
    pf = pf.sort_values(["stock_id", "date"]).reset_index(drop=True)

    chip = pd.read_parquet(DATA_DIR / "chip_features.parquet",
                           columns=["date", "stock_id", "foreign_net_ratio",
                                    "trust_net_ratio", "dealer_net_ratio"])
    chip["date"] = pd.to_datetime(chip["date"])
    chip = chip.sort_values(["stock_id", "date"]).reset_index(drop=True)

    logger.info("量能相對自己…")
    parts = [_rel_amount(price)]

    logger.info("風險調整動能…")
    parts.append(_rel_risk_adj_momentum(price))

    logger.info("距離 ATR 計價…")
    parts.append(_rel_dist_atr(pf))

    logger.info("指標自身歷史百分位…")
    parts.append(_rel_self_percentile(pf, ["rsi_14", "k_value", "bb_width_pct"]))

    logger.info("籌碼相對自己…")
    parts.append(_rel_chip_z(chip, ["foreign_net_ratio", "trust_net_ratio",
                                    "dealer_net_ratio"]))

    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["date", "stock_id"], how="outer")

    out = out.sort_values(["stock_id", "date"]).reset_index(drop=True)
    n_feat = len([c for c in out.columns if c not in ("date", "stock_id")])
    logger.info(f"完成：{len(out)} 列 × {n_feat} 個新特徵")
    return out


def main() -> None:
    out = build()
    path = DATA_DIR / "relative_features.parquet"
    out.to_parquet(path, index=False)
    logger.info(f"已寫入 {path}")

    na = out.drop(columns=["date", "stock_id"]).isna().mean().sort_values(ascending=False)
    logger.info("缺失率前 10：\n" + na.head(10).round(3).to_string())


if __name__ == "__main__":
    main()
