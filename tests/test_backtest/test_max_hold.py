"""持有期上限，以及「兩種都產」的契約。

## 為什麼有這個參數

2026-08-25 的資料稽核追出：val_sel（2024H2）的全買基準是 −7.93%，而舊記錄是
+3.21%。根因不是 bug —— `simulate()` **沒有持有期上限**，2024H2 的訊號平均抱
108 根 bar、一路抱到 2025/4 的關稅崩盤（41.6% 的停損發生在那個月）。也就是說
「val_sel 的回測」實際量的是「2024H2 進場 + 最長兩年持有」，橫跨三種市場狀態。
把價格截在 2025-03-31 強制平倉，平均從 −7.87% 變 −4.07%。

使用者決定**兩種都產**：不限上限（與 norf 歷史記錄同口徑）與 20 日上限
（與 `label_up20` 的 horizon 一致）。並列才看得出「績效有多少來自持有期拉長、
有多少來自選股本身」。
"""
from __future__ import annotations

import inspect

import pandas as pd
import pytest

from engine.backtest.backtest import _run_exit, simulate
from engine.backtest.summary import HOLD_VARIANTS


class TestMaxHoldBars:
    def test_simulate_exposes_the_parameter(self):
        # Act / Assert
        assert "max_hold_bars" in inspect.signature(simulate).parameters
        assert simulate.__signature__.parameters["max_hold_bars"].default is None \
            if hasattr(simulate, "__signature__") \
            else inspect.signature(simulate).parameters["max_hold_bars"].default is None

    def test_default_is_unlimited(self):
        """預設不限 —— 既有行為不能因為新參數而改變。"""
        # Act
        param = inspect.signature(_run_exit).parameters["max_hold_bars"]

        # Assert
        assert param.default is None

    def test_exit_happens_at_the_limit(self):
        """持有到上限那天就出場，即使沒有觸發任何停損停利。"""
        # Arrange：價格完全不動，不會觸發 trail / stop
        days = pd.date_range("2024-01-02", periods=40, freq="B").tolist()
        row = {"open": 100.0, "close": 100.0, "ma20": 100.0}

        # Act
        sell_date, _, reason = _run_exit(
            days, 0, 100.0, lambda d: row, "ma20", 20, max_hold_bars=5,
            take_profit=0.20, trail_trigger=0.15, trail_pct=0.10, stop_loss=0.20)

        # Assert：第 5 根 bar 出場，而不是撐到資料結束
        assert sell_date == days[5], f"應在第 5 根 bar 出場，實際 {sell_date}"
        assert reason == "max_hold", f"出場原因應為 max_hold，實際 {reason}"

    def test_unlimited_holds_to_data_end(self):
        # Arrange
        days = pd.date_range("2024-01-02", periods=40, freq="B").tolist()
        row = {"open": 100.0, "close": 100.0, "ma20": 100.0}

        # Act
        sell_date, _, reason = _run_exit(
            days, 0, 100.0, lambda d: row, "ma20", 20, max_hold_bars=None,
            take_profit=0.20, trail_trigger=0.15, trail_pct=0.10, stop_loss=0.20)

        # Assert
        assert sell_date == days[-1]
        assert reason == "data_end"


class TestBothVariantsAreProduced:
    def test_summary_covers_unlimited_and_twenty(self):
        """規則 9 的精神：只給一種口徑的比較沒有意義。"""
        # Assert
        assert None in HOLD_VARIANTS, "少了『不限上限』—— 那是與歷史記錄可比的口徑"
        assert 20 in HOLD_VARIANTS, "少了 20 日上限 —— 那是與 label horizon 一致的口徑"
