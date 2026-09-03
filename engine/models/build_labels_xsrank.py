"""`label_xsrank20`：當日全市場中，風險調整後的未來報酬排名前 20%。

## 為什麼是這個定義（2026-09-03，第二輪修正）

第一輪的 `label_steady20` 修好了選股偏差卻整組出局（見 doc/BACKTEST_LOG.md #31）。
死因不是偏差，是**正例率隨市場環境擺盪 5.3%~14.5%（2.75 倍）** —— 在 val_sel 挑的
門檻搬到 test2 根本不成立，而且基準率只有 10%，RF 的分數上限被壓到 0.443，
連前端滑桿的下限 0.50 都碰不到。

橫斷面排名從定義上解決這件事：**每天都取前 20%，正例率被釘死在 20.0%**，
不管那天是大漲還是崩盤。實測跨切分擺盪 1.00 倍。

## 為什麼要「風險調整」而不是純報酬排名（這一步是關鍵）

    候選                        崩跌/持平  跨切分擺盪  正例中緩漲%  正例報酬中位
    label_up20（現行）              1.84     1.30x     53.6%     +5.2%
    X1 純報酬排名前 20%              1.85     1.00x     52.4%    +12.9%
    X3 純報酬排名前 10%              2.30     1.01x     49.6%    +20.8%
    X4 排名前 20% 且站上 20 日線      1.61     1.14x     53.7%    +14.0%
    ★ 風險調整報酬排名前 20%          1.12     1.00x     59.6%    +11.3%

（全市場基準：緩漲 57.4%、未來 20 日報酬中位 -0.2%）

**單純的報酬排名完全修不掉偏差** —— X1 的 1.85 跟現行的 1.84 一樣糟。因為大跌
反彈的股票報酬本來就是真的高，排名排得上去。修掉偏差的是**除以自身波動**那一步：
崩跌股波動大，同樣的報酬除下來就不突出了。這與第一輪的結論一致
（見 `build_labels_steady.py`：波動標準化有效，路徑/趨勢條件無效）。

所以這個標的是兩個洞見的結合：
  * **風險調整**（÷ 進場前 20 日波動）壓掉大跌反彈的偏差
  * **橫斷面排名**把正例率釘死，讓門檻搬得動

## 規則（與 `build_labels.py` 同為收盤制）

    sigma          = 進場前 20 個交易日的日報酬標準差 × sqrt(20)
    risk_adj       = (收盤_t+20 / 收盤_t - 1) / sigma
    label_xsrank20 = 1  ⟺  risk_adj 在「當日全市場」的百分位 > 1 - TOP_PCT

⚠️ **分母沒有洩漏**：`sigma` 只用進場前的資料。

⚠️ **橫斷面排名用到同一天其他股票的未來報酬，這是刻意的，不是洩漏。**
標籤本來就允許看未來（那正是標籤）；要防的是**跨切分污染**，而這裡不會 ——
切分是按日期切的，某一列標籤用到的全部資訊都落在它自己那一天的前瞻視窗內。
副作用是同一天的標籤互相不獨立（每天恰好 20% 是正例），那正是這個設計的目的。

⚠️ `MIN_STOCKS_PER_DAY`：當天有效樣本太少就整天作廢。早期上市家數少的日子，
前 20% 只有幾檔，排名不穩定；用半截母體算出來的百分位會把雜訊灌成正例。

## 基準率 20% 對「分數上限」的意義

`label_up20` 基準率 38.2% → m1 分數上限 0.971；`label_steady20` 10% → 0.443
（碰不到滑桿下限 0.50，直接出局）。這個標的是 20%，落在兩者之間，**分數上限
仍有可能低於 0.50** —— 訓練完第一件事就是查 `score_*.parquet` 的最大值，
不要等到挑門檻時才發現。

用法：
  python -m engine.models.build_labels_xsrank
  python -m engine.models.build_labels_xsrank --top-pct 0.10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from engine.paths import DATA_DIR

logger = logging.getLogger(__name__)

OUTPUT_LABEL = "label_xsrank20"
HOLD_BARS = 20
LOOKBACK_BARS = 20
# 前 20%：基準率剛好 20.0%，偏差 1.12。前 10% 的偏差會惡化到 2.30
# （排名越嚴格，越只剩下崩跌後的大反彈擠得進去），實測過，不要往那邊調。
DEFAULT_TOP_PCT = 0.20
# 當天有效樣本少於這個數就整天作廢，理由見模組 docstring。
MIN_STOCKS_PER_DAY = 200


def trailing_sigma(close: pd.Series, group: pd.Series,
                   bars: int = LOOKBACK_BARS) -> pd.Series:
    """進場前 `bars` 個交易日的日報酬標準差 × sqrt(bars)。

    ⚠️ 只用進場當下已知的資料。`min_periods=bars` 是刻意的：暖機期不足時回 NaN，
    讓那些列被剔除，而不是拿半截樣本算出一個偏小的 sigma —— 偏小的 sigma 會把
    risk_adj 灌大，讓那一列假性擠進前 20%。
    """
    daily = close.groupby(group).pct_change(fill_method=None)
    return (daily.groupby(group)
                 .rolling(bars, min_periods=bars).std()
                 .reset_index(level=0, drop=True) * np.sqrt(bars))


def build(price: pd.DataFrame, top_pct: float = DEFAULT_TOP_PCT,
          min_stocks: int = MIN_STOCKS_PER_DAY) -> pd.DataFrame:
    frame = price[["date", "stock_id", "close"]].sort_values(
        ["stock_id", "date"]).reset_index(drop=True)
    close, sid = frame["close"], frame["stock_id"]

    fwd_ret = close.groupby(sid).shift(-HOLD_BARS) / close - 1
    sigma = trailing_sigma(close, sid)
    # sigma 為 0（整段完全沒動）會讓 risk_adj 變成 inf —— 當成算不出來
    frame["risk_adj"] = (fwd_ret / sigma.where(sigma > 0)).replace(
        [np.inf, -np.inf], np.nan)

    valid = frame["risk_adj"].notna()
    day_n = valid.groupby(frame["date"]).transform("sum")
    usable = valid & (day_n >= min_stocks)

    # 百分位只在「當天可用的那些列」之間算 —— 把算不出來的列一起丟進去排名，
    # 會讓母體大小隨暖機狀況變動，前 20% 的實際口徑就跟著漂。
    pct = frame.loc[usable].groupby("date")["risk_adj"].rank(pct=True)
    frame[OUTPUT_LABEL] = 0
    frame.loc[pct.index[pct > 1 - top_pct], OUTPUT_LABEL] = 1
    frame[OUTPUT_LABEL] = frame[OUTPUT_LABEL].astype("int8")

    out = frame.loc[usable, ["date", "stock_id", OUTPUT_LABEL]].reset_index(drop=True)
    dropped_days = int(frame.loc[valid, "date"].nunique() - out["date"].nunique())
    logger.info(
        f"{len(out):,} 列：正例率 {out[OUTPUT_LABEL].mean():.1%}"
        f"（前 {top_pct:.0%}；樣本不足而整天剔除 {dropped_days} 天）")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price", type=Path, default=DATA_DIR / "price.parquet")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "labels_xsrank20.parquet")
    parser.add_argument("--top-pct", type=float, default=DEFAULT_TOP_PCT)
    parser.add_argument("--min-stocks", type=int, default=MIN_STOCKS_PER_DAY)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    price = pd.read_parquet(args.price, columns=["date", "stock_id", "close"])
    price["date"] = pd.to_datetime(price["date"])

    out = build(price, args.top_pct, args.min_stocks)
    out.to_parquet(args.out, index=False)
    logger.info(f"{len(out):,} 列 → {args.out}")


if __name__ == "__main__":
    main()
