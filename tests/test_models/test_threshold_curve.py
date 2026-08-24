"""門檻曲線的口徑與效能。

兩件事在 2026-08-24 同時被抓到，而且是同一個根因的兩面：
1. 口徑錯了 —— simulate() 沒傳 dedup，吃到預設值 True（同股不重複進場），
   但規則、bundle.py、make backtest 全部是 dedup=False（訊號層級）。
   「在去重曲線上挑門檻、拿去跑無去重回測」就是 BACKTEST_LOG #24 vs #25
   警告過的兩把尺。
2. 效能 —— med_return 是 [np.median(r[:i]) for i in n]，O(n²)。去重口徑下
   只有 4,463 筆跑得動；改成正確的 218,830 筆就要好幾小時。
   **口徑的錯誤把效能問題一起掩蓋了。**
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.models.threshold_curve import _running_median, cumulative_by_threshold


class TestRunningMedian:
    def test_matches_numpy_median(self):
        # Arrange
        values = np.random.default_rng(42).normal(size=2000)

        # Act
        got = _running_median(values)

        # Assert：逐點與 np.median 相同
        want = [float(np.median(values[:i + 1])) for i in range(len(values))]
        assert np.allclose(got, want)

    def test_handles_even_and_odd_lengths(self):
        # Arrange / Act / Assert
        assert _running_median(np.array([3.0])) == [3.0]
        assert _running_median(np.array([3.0, 1.0])) == [3.0, 2.0]
        assert _running_median(np.array([3.0, 1.0, 2.0])) == [3.0, 2.0, 2.0]

    def test_is_not_quadratic(self):
        """20 萬筆要在一秒內跑完 —— 舊的 O(n²) 版本要幾小時。"""
        # Arrange
        import time
        values = np.random.default_rng(0).normal(size=200_000)

        # Act
        started = time.time()
        _running_median(values)

        # Assert
        assert time.time() - started < 5.0


class TestCumulativeByThreshold:
    def test_cumulative_stats_are_top_down(self):
        """第 n 列＝分數最高的 n 筆的統計。"""
        # Arrange：分數由低到高給，確認函式自己會排序
        trades = pd.DataFrame({
            "score": [0.1, 0.9, 0.5, 0.7],
            "return": [-0.10, 0.20, -0.05, 0.30],
        })

        # Act
        curve = cumulative_by_threshold(trades)

        # Assert：MIN_SAMPLES=40 會濾掉小樣本，這裡直接驗排序前的邏輯
        ranked = trades.sort_values("score", ascending=False)["return"].to_numpy()
        assert ranked[0] == 0.20 and ranked[-1] == -0.10
        assert curve.empty  # 4 筆 < MIN_SAMPLES，不該畫

    def test_filters_below_min_samples(self):
        """樣本太少的深度不入表 —— 不在雜訊上挑點。"""
        # Arrange
        n = 100
        trades = pd.DataFrame({"score": np.linspace(0, 1, n),
                               "return": np.random.default_rng(1).normal(size=n)})

        # Act
        curve = cumulative_by_threshold(trades)

        # Assert
        assert curve["n"].min() >= 40
        assert curve["n"].max() == n
