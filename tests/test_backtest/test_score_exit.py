"""`score_exit.py` 的回歸測試（2026-09-05 新增）。

這支是 swing 模型唯一的回測路徑，而且刻意不走 `simulate()`（那是 m1 兩個模型
共用的，不能為了一個實驗模型多加分支）。兩邊的欄位與成交價規則必須一致，
否則下游 `performance()` 與 streamlit app 會靜默拿到對不起來的東西。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.backtest.score_exit import TRADE_COLS, _one_stock, simulate_score_exit


def _dates(n: int) -> np.ndarray:
    return pd.date_range("2024-01-01", periods=n, freq="B").to_numpy()


def test_buys_next_open_and_sells_next_open_after_score_drops():
    o = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    s = np.array([0.99, 0.90, 0.10, 0.10, 0.10, 0.10])   # 第 2 天分數就跌破
    rows = _one_stock(_dates(6), o, s, 0.95, 0.20, 250, dedup=False)
    assert len(rows) == 1
    r = rows[0]
    assert r["buy_price"] == 11.0        # 訊號日 index0 → 隔日開盤 index1
    assert r["sell_price"] == 13.0       # 分數 index2 跌破 → 隔日開盤 index3
    assert r["sell_reason"] == "score_exit"
    assert r["return"] == pytest.approx(13.0 / 11.0 - 1)


def test_no_entry_when_score_below_threshold():
    o = np.arange(10.0, 16.0)
    s = np.full(6, 0.5)
    assert _one_stock(_dates(6), o, s, 0.95, 0.20, 250, dedup=False) == []


def test_nan_score_neither_enters_nor_exits():
    """分數是 NaN 代表那天沒有推論結果，不能當成「跌破門檻」。"""
    o = np.arange(10.0, 18.0)
    s = np.array([0.99, np.nan, np.nan, np.nan, 0.50, 0.50, 0.10, 0.10])
    rows = _one_stock(_dates(8), o, s, 0.95, 0.20, 250, dedup=False)
    assert len(rows) == 1
    # NaN 那幾天不觸發出場，要等 index6 真的跌破
    assert rows[0]["sell_date"] == _dates(8)[7]


def test_max_hold_caps_the_position():
    o = np.arange(10.0, 30.0)
    s = np.concatenate([[0.99], np.full(19, 0.90)])      # 分數一直沒跌破
    rows = _one_stock(_dates(20), o, s, 0.95, 0.20, max_hold=5, dedup=False)
    assert len(rows) == 1
    assert rows[0]["sell_reason"] == "max_hold"
    assert rows[0]["sell_date"] == _dates(20)[1 + 5 + 1]


def test_dedup_blocks_reentry_while_holding():
    o = np.arange(10.0, 20.0)
    s = np.array([0.99, 0.99, 0.99, 0.99, 0.10, 0.99, 0.99, 0.10, 0.5, 0.5])
    many = _one_stock(_dates(10), o, s, 0.95, 0.20, 250, dedup=False)
    one = _one_stock(_dates(10), o, s, 0.95, 0.20, 250, dedup=True)
    assert len(many) > len(one), "dedup=False 應該每個訊號各算一筆"
    # 去重後不得有任何一筆的買進日落在前一筆的持有期間內
    for a, b in zip(one, one[1:]):
        assert b["buy_date"] >= a["sell_date"]


def test_rejects_sell_threshold_above_buy_threshold():
    with pytest.raises(ValueError):
        simulate_score_exit(__import__("pathlib").Path("x.parquet"),
                            buy_threshold=0.2, sell_threshold=0.9)


def test_zero_buy_threshold_is_allowed_for_prefiltered_universe():
    """`summary.py` 的訊號數對齊口徑會先把分數檔篩成每日前 1.5%，
    母體本身就是進場清單，門檻是 0 —— 不能被防呆擋掉（2026-09-05 匯出時踩到）。"""
    import pathlib
    with pytest.raises(FileNotFoundError):        # 過了門檻檢查才會去讀檔
        simulate_score_exit(pathlib.Path("does_not_exist.parquet"),
                            buy_threshold=0.0, sell_threshold=0.20)


def test_trade_columns_match_simulate_schema():
    """欄位必須跟 backtest.simulate() 逐字相同，下游才接得起來。"""
    from engine.backtest import backtest as bt
    src = (bt.__file__ and open(bt.__file__).read()) or ""
    for col in TRADE_COLS:
        assert f'"{col}"' in src, f"simulate() 沒有 {col} 這一欄，schema 對不起來"


def test_zero_or_missing_open_price_is_skipped():
    o = np.array([10.0, 0.0, 12.0, 13.0, 14.0])          # 隔日開盤是 0（停牌）
    s = np.array([0.99, 0.99, 0.10, 0.10, 0.10])
    rows = _one_stock(_dates(5), o, s, 0.95, 0.20, 250, dedup=False)
    assert all(r["buy_price"] > 0 for r in rows)
