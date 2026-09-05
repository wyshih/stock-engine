"""swing 面板純函式的測試（2026-09-05）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.app.frontend.swing_panel import (DENSITY_HIGH, DENSITY_LOW,
                                             daily_density, density_verdict,
                                             holdings_status)


def _scores(rows):
    return pd.DataFrame(rows, columns=["date", "stock_id", "score"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def test_density_counts_signals_over_total():
    s = _scores([("2024-01-02", "A", 0.99), ("2024-01-02", "B", 0.10),
                 ("2024-01-02", "C", 0.98), ("2024-01-03", "A", 0.10)])
    d = daily_density(s, 0.97).set_index("date")
    assert d.loc["2024-01-02", "n_signal"] == 2
    assert d.loc["2024-01-02", "density"] == pytest.approx(2 / 3)
    assert d.loc["2024-01-03", "density"] == 0.0


def test_density_on_empty_input_returns_empty_frame():
    out = daily_density(pd.DataFrame(columns=["date", "stock_id", "score"]), 0.97)
    assert out.empty and list(out.columns) == ["date", "n_signal", "n_total", "density"]


def test_verdict_flags_scarce_signals_as_do_not_trade():
    """2025H2 密度 0.04%，那半年輸給大盤 —— 這條是那個教訓的程式化。"""
    label, note = density_verdict(0.0004)
    assert "稀少" in label and "不該勉強出手" in note


def test_verdict_boundaries():
    assert "稀少" in density_verdict(DENSITY_LOW - 1e-6)[0]
    assert "一般" in density_verdict(DENSITY_LOW)[0]
    assert "一般" in density_verdict(DENSITY_HIGH)[0]
    assert "偏多" in density_verdict(DENSITY_HIGH + 1e-6)[0]


def test_verdict_handles_missing_density():
    assert "無資料" in density_verdict(None)[0]
    assert "無資料" in density_verdict(float("nan"))[0]


def test_holdings_uses_latest_score_per_stock():
    s = _scores([("2024-01-02", "A", 0.90), ("2024-01-03", "A", 0.15)])
    out = holdings_status([{"stock_id": "A"}], s, 0.20)
    assert out.loc[0, "score"] == pytest.approx(0.15)
    assert "該賣" in out.loc[0, "status"]


def test_holdings_shows_distance_to_exit_when_still_held():
    s = _scores([("2024-01-02", "A", 0.55)])
    out = holdings_status([{"stock_id": "A"}], s, 0.20)
    assert "離出場門檻還有 0.350" in out.loc[0, "status"]


def test_holdings_handles_stock_without_scores():
    s = _scores([("2024-01-02", "A", 0.55)])
    out = holdings_status([{"stock_id": "Z"}], s, 0.20)
    assert "沒有分數" in out.loc[0, "status"]
    assert np.isnan(out.loc[0, "score"])


def test_holdings_on_empty_watchlist():
    assert holdings_status([], _scores([("2024-01-02", "A", 0.5)]), 0.20).empty
