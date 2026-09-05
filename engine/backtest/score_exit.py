"""分數出場的回測（2026-09-05 新增，swing 模型專用）。

跟 `backtest.simulate()` 的差別只有**出場**：那支是移動停利／停損／均線，這支是
**分數跌破門檻就賣**。進場、成交價、去重、回傳欄位全部一致，所以下游
（`performance()`、streamlit app、門檻曲線）可以直接接。

為什麼要另開一支而不是在 `simulate()` 加參數：`simulate()` 是 m1_base_up20 與
m1_mdd10 共用的唯一回測路徑，CLAUDE.md 的最高原則是不能讓那兩個模型變得無法
重建。分數出場只有 swing 用得到，塞進去只會讓共用路徑多一條分支。

規則（swing 模型的正式回測口徑）：
  進場：分數 >= buy_threshold → 隔日開盤買
  出場：分數 <= sell_threshold → 隔日開盤賣
  兜底：max_hold_bars 到期、或資料結束（sell_reason 會標明）

⚠️ 這支用的分數檔必須是**全市場口徑**（`score_swing_*.parquet` 與
`score_live_swing.parquet` 都是）。只含有 label 的列會讓母體少八成，
而且那是事後才知道的資訊。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from engine.paths import DATA_DIR  # noqa: E402

TRADE_COLS = ["signal_date", "stock_id", "score", "buy_date", "buy_price",
              "sell_date", "sell_price", "return", "sell_reason"]
DEFAULT_MAX_HOLD = 250          # 兜底：分數一直沒跌破時最多抱這麼久


def _load_price() -> pd.DataFrame:
    price = pd.read_parquet(DATA_DIR / "price.parquet",
                            columns=["date", "stock_id", "open", "close"])
    price["date"] = pd.to_datetime(price["date"])
    return price.sort_values(["stock_id", "date"]).reset_index(drop=True)


def _one_stock(dates: np.ndarray, open_: np.ndarray, score: np.ndarray,
               buy_th: float, sell_th: float, max_hold: int,
               dedup: bool) -> list[dict]:
    """單檔的進出場。分數是 NaN 的日子既不進場也不觸發出場。"""
    n = len(dates)
    rows: list[dict] = []
    busy_until = -1
    for i in range(n - 2):                       # 要留隔日開盤才買得到
        if not (np.isfinite(score[i]) and score[i] >= buy_th):
            continue
        if dedup and i <= busy_until:
            continue
        buy_i = i + 1
        if not np.isfinite(open_[buy_i]) or open_[buy_i] <= 0:
            continue
        sell_i, reason = None, ""
        limit = min(buy_i + max_hold, n - 2)
        for j in range(buy_i, limit + 1):
            if np.isfinite(score[j]) and score[j] <= sell_th:
                sell_i, reason = j + 1, "score_exit"
                break
        if sell_i is None:
            sell_i = min(limit + 1, n - 1)
            reason = "max_hold" if limit == buy_i + max_hold else "data_end"
        if not np.isfinite(open_[sell_i]) or open_[sell_i] <= 0:
            continue
        rows.append({"signal_date": dates[i], "stock_id": None, "score": float(score[i]),
                     "buy_date": dates[buy_i], "buy_price": float(open_[buy_i]),
                     "sell_date": dates[sell_i], "sell_price": float(open_[sell_i]),
                     "return": float(open_[sell_i] / open_[buy_i] - 1),
                     "sell_reason": reason})
        busy_until = sell_i
    return rows


def simulate_score_exit(score_path: Path, buy_threshold: float, sell_threshold: float,
                        dedup: bool = False, max_hold_bars: int = DEFAULT_MAX_HOLD,
                        date_start: str | None = None, date_end: str | None = None,
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """回傳 (trades, price)，欄位與 `backtest.simulate()` 相同。

    `return` 是**未扣成本**的毛報酬，跟 `simulate()` 一致 —— 成本由下游統一處理，
    兩邊口徑才比得起來。
    """
    # buy_threshold == 0 是合法的：`summary.py` 的「訊號數對齊」口徑會先把分數檔
    # 篩成每日前 1.5%，母體本身就是進場清單，門檻自然是 0。除此之外進場門檻低於
    # 出場門檻沒有意義（買進當下就滿足賣出條件），擋下來。
    if 0 < buy_threshold <= sell_threshold:
        raise ValueError(f"進場門檻({buy_threshold})必須高於出場門檻({sell_threshold})")
    score = pd.read_parquet(score_path)
    score["date"] = pd.to_datetime(score["date"])
    price = _load_price()
    df = price.merge(score, on=["date", "stock_id"], how="left")
    if date_start:
        df = df[df["date"] >= date_start]
    if date_end:
        df = df[df["date"] <= date_end]

    rows: list[dict] = []
    for sid, g in df.groupby("stock_id", sort=False):
        s = g["score"].to_numpy("float64")
        if not np.isfinite(s).any():
            continue
        got = _one_stock(g["date"].to_numpy(), g["open"].to_numpy("float64"), s,
                         buy_threshold, sell_threshold, max_hold_bars, dedup)
        for r in got:
            r["stock_id"] = sid
        rows.extend(got)

    trades = (pd.DataFrame(rows, columns=TRADE_COLS)
              .sort_values(["signal_date", "stock_id"]).reset_index(drop=True))
    logger.info(f"分數出場回測：買>={buy_threshold} 賣<={sell_threshold} "
                f"→ {len(trades)} 筆（dedup={dedup}）")
    return trades, price
