"""
漲停條件基準（2026-07-29 新增，doc/AUDIT_20260728.md §C-13）。

## 為什麼需要這個

現有指標（平均報酬 / 中位數 / Sharpe / 勝率 / PR-AUC）都只描述「這批交易賺了
多少」，**沒有一個能區分「選股能力」與「大盤 beta」**。2026 大盤漲 48.7%，
把平均報酬從 10% 推到 16% 有可能只是多吃了 beta。

而本策略 **73% 的訊號股在訊號日當天就已漲停**。拿漲停股去跟「包含大量平盤/
下跌股的全市場平均」比，本來就會贏 —— 贏的是「漲停股隔天通常較強」這個
人盡皆知的動能效應，不是模型的選股能力。

**正確的基準是「同一個訊號日、同樣漲停的其他股票」**，並且對它們套用
**完全相同的出場規則**（所以本模組呼叫 `backtest._run_exit()`，不自己複製一份）。

## 已知的量測結果（doc/AUDIT_20260728.md §C-13）

|                          | 2025    | 2026    |
|--------------------------|---------|---------|
| 策略                     | +6.91%  | +10.05% |
| 同日漲停 peer            | +2.96%  | +3.24%  |
| alpha（平均）            | +3.96pp | +6.81pp |
| **配對差中位數**         | **-6.19pp** | **-3.64pp** |
| 勝過 peer 平均的比例     | 40.9%   | 46.1%   |

平均為正、中位數為負 → 典型的一筆交易表現**不如**當天隨便一檔漲停股，
正 alpha 完全由每年 3~5 張彩票撐起（去掉最大 5 筆，兩年都翻負）。

## 用法

    from backtest import simulate
    from benchmark import attach_peer_benchmark, alpha_summary

    trades, price = simulate(split="meta_test", threshold=0.775)
    trades = attach_peer_benchmark(trades, price)
    print(alpha_summary(trades))

驗收新想法時，**看 `alpha_median` 有沒有變正，比看 `avg_return` 有沒有變高更重要**。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from engine.backtest.backtest import _run_exit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 專案路徑一律走 code/paths.py（唯一來源），不要各檔自行推導
from engine.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402


# 台股漲停 10%，容忍 tick 進位與還原股價的小數誤差
LIMIT_UP = 0.095

# 大盤指數不是可交易個股，一律排除在 peer 池外
INDEX_IDS = {"TWII"}


def _build_store(price: pd.DataFrame, ma_col: str) -> dict:
    """把逐股票的價格/均線攤成 numpy，供大量 peer 模擬快速查詢。"""
    store = {}
    for sid, grp in price.groupby("stock_id", sort=False):
        store[sid] = {
            "dates": grp["date"].values,
            "close": grp["close"].to_numpy(float),
            "open":  grp["open"].to_numpy(float),
            ma_col:  grp[ma_col].to_numpy(float),
        }
    return store


def _peer_return(store: dict, sid: str, signal_dt, ma_col: str, stop_ma: int,
                 **exit_kw) -> float:
    """
    對單一 peer 套用跟策略完全相同的進出場規則。

    進場慣例跟 `simulate()` 一致：訊號日的**下一個交易日開盤**買進。
    """
    s = store.get(sid)
    if s is None:
        return np.nan

    dates = s["dates"]
    idx_buy = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(signal_dt)) +
                                  np.timedelta64(1, "D")))
    if idx_buy >= len(dates):
        return np.nan

    buy_price = s["open"][idx_buy]
    if not np.isfinite(buy_price) or buy_price <= 0:
        return np.nan

    def row_getter(d):
        k = int(np.searchsorted(dates, d))
        if k >= len(dates) or dates[k] != d:
            return None
        return {"close": s["close"][k], ma_col: s[ma_col][k]}

    _, sell_price, _ = _run_exit(
        list(dates), idx_buy, buy_price, row_getter, ma_col, stop_ma, **exit_kw
    )
    if sell_price is None:
        return np.nan
    return float((sell_price - buy_price) / buy_price)


def attach_peer_benchmark(trades: pd.DataFrame, price: pd.DataFrame,
                          stop_ma: int = 20, max_peers: int | None = None,
                          seed: int = 42, **exit_kw) -> pd.DataFrame:
    """
    對每筆交易算出「同訊號日漲停 peer」的平均報酬與配對差。

    參數
    ----
    trades   : `simulate()` 回傳的交易明細（需有 signal_date / stock_id / return）
    price    : `simulate()` 一併回傳的價格表（已含 ma{stop_ma} 欄）
    stop_ma  : 停損均線，必須跟產生 trades 時用的值一致
    max_peers: 每個訊號日最多取幾檔 peer（None = 全取）。設定時以固定亂數種子
               抽樣，避免 peer 數量爆炸拖慢速度
    exit_kw  : 直接透傳給 `_run_exit()`（take_profit / trail_trigger / trail_pct）。
               **必須跟產生 trades 時的設定完全相同**，否則比較無效。

    回傳
    ----
    trades 的複本，新增三欄：
      peer_n       該訊號日的漲停 peer 數
      peer_return  peer 的平均報酬（套用相同出場規則）
      paired_diff  策略報酬 − peer 平均報酬
    """
    if trades.empty:
        return trades.copy()

    ma_col = f"ma{stop_ma}"
    if ma_col not in price.columns:
        raise ValueError(f"price 缺少 {ma_col} 欄，請傳入 simulate() 回傳的 price")

    px = price[~price["stock_id"].isin(INDEX_IDS)].copy()
    px = px.sort_values(["stock_id", "date"]).reset_index(drop=True)
    px["ret1"] = px.groupby("stock_id")["close"].transform(lambda s: s / s.shift(1) - 1.0)

    store = _build_store(px, ma_col)

    # 每個交易日的漲停名單
    lu = px[px["ret1"] >= LIMIT_UP]
    lu_by_date = {d: grp["stock_id"].tolist() for d, grp in lu.groupby("date")}

    rng = np.random.default_rng(seed)
    cache: dict[tuple, float] = {}
    rows = []

    for _, tr in trades.iterrows():
        sig_dt = pd.Timestamp(tr["signal_date"])
        peers = [p for p in lu_by_date.get(sig_dt, []) if p != tr["stock_id"]]

        if max_peers is not None and len(peers) > max_peers:
            peers = list(rng.choice(peers, size=max_peers, replace=False))

        rets = []
        for p in peers:
            key = (p, sig_dt)
            if key not in cache:
                cache[key] = _peer_return(store, p, sig_dt, ma_col, stop_ma, **exit_kw)
            r = cache[key]
            if np.isfinite(r):
                rets.append(r)

        peer_ret = float(np.mean(rets)) if rets else np.nan
        rows.append({"peer_n": len(rets), "peer_return": peer_ret,
                     "paired_diff": tr["return"] - peer_ret})

    out = trades.copy().reset_index(drop=True)
    return pd.concat([out, pd.DataFrame(rows)], axis=1)


def alpha_summary(trades: pd.DataFrame) -> dict:
    """
    彙總 alpha 指標。**驗收新想法時以 `alpha_median` 為第一順位**
    （doc/AUDIT_20260728.md §D-0）：平均會被少數彩票撐高，中位數才反映
    「典型的一筆交易有沒有比隨便買一檔當天漲停股好」。
    """
    if trades.empty or "paired_diff" not in trades.columns:
        return {}

    d = trades["paired_diff"].dropna()
    if d.empty:
        return {"n_paired": 0}

    # 去掉配對差最大的 N 筆後 alpha 還剩多少 —— 檢查 alpha 是不是彩票撐起來的
    ranked = d.sort_values(ascending=False)
    drop_stats = {f"alpha_mean_drop{n}": round(float(ranked.iloc[n:].mean()), 4)
                  for n in (1, 3, 5) if len(ranked) > n}

    return {
        "n_paired":       int(len(d)),
        "strategy_mean":  round(float(trades.loc[d.index, "return"].mean()), 4),
        "peer_mean":      round(float(trades.loc[d.index, "peer_return"].mean()), 4),
        "alpha_mean":     round(float(d.mean()), 4),
        "alpha_median":   round(float(d.median()), 4),
        "win_vs_peer":    round(float((d > 0).mean()), 4),
        "median_peer_n":  int(trades["peer_n"].median()),
        **drop_stats,
    }
