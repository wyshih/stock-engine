"""買賣點規則面板的純函式測試（2026-09-06）。"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.app.frontend import trade_rule_panel as trp


def _hits(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    return df


def test_half_year_label_splits_at_june():
    assert trp.half_year_label(pd.Timestamp("2024-06-30")) == "2024H1"
    assert trp.half_year_label(pd.Timestamp("2024-07-01")) == "2024H2"


def test_period_summary_marks_periods_below_target_as_not_passing():
    # Arrange：2024H1 五筆全贏、2024H2 五筆只贏兩筆
    rows = [{"signal_date": "2024-02-01", "ret": 0.03, "hold": 5} for _ in range(5)]
    rows += [{"signal_date": "2024-08-01", "ret": r, "hold": 5}
             for r in (0.03, 0.03, -0.1, -0.1, -0.1)]

    # Act
    out = trp.period_summary(_hits(rows))

    # Assert
    by_period = out.set_index("期間")
    assert bool(by_period.loc["2024H1", "達標"]) is True
    assert bool(by_period.loc["2024H2", "達標"]) is False
    assert by_period.loc["2024H2", "勝率"] == pytest.approx(0.4)


def test_period_summary_returns_empty_frame_with_columns_when_no_hits():
    out = trp.period_summary(pd.DataFrame())
    assert out.empty
    assert "達標" in out.columns


def test_overall_stats_counts_passing_periods():
    rows = [{"signal_date": "2024-02-01", "ret": 0.03, "hold": 5} for _ in range(5)]
    rows += [{"signal_date": "2024-08-01", "ret": -0.1, "hold": 20} for _ in range(5)]

    out = trp.overall_stats(_hits(rows))

    assert out["交易數"] == 10
    assert out["達標期數"] == 1
    assert out["總期數"] == 2


def test_overall_stats_on_empty_returns_zeros_not_nan():
    out = trp.overall_stats(pd.DataFrame())
    assert out["交易數"] == 0
    assert out["勝率"] == 0.0


def test_open_positions_can_filter_to_watchlist():
    rows = [{"signal_date": "2024-03-01", "stock_id": "2330", "ret": -0.4,
             "hold": 20, "resolved": False},
            {"signal_date": "2024-04-01", "stock_id": "1101", "ret": -0.3,
             "hold": 20, "resolved": False}]

    out = trp.open_positions(_hits(rows), watch=["1101"])

    assert list(out["stock_id"]) == ["1101"]


def test_open_positions_is_empty_when_column_missing():
    """舊資料沒有 resolved 欄 —— 一律視為已結束，不該憑空生出未平倉部位。"""
    rows = [{"signal_date": "2024-03-01", "stock_id": "2330", "ret": -0.4, "hold": 20}]

    assert trp.open_positions(_hits(rows)).empty


def test_recent_trades_puts_newest_first_and_respects_limit():
    rows = [{"signal_date": f"2024-0{i}-01", "ret": 0.03, "hold": 5} for i in range(1, 6)]

    out = trp.recent_trades(_hits(rows), limit=2)

    assert len(out) == 2
    assert out["signal_date"].iloc[0] == pd.Timestamp("2024-05-01")


def test_loaders_return_empty_when_file_missing(tmp_path):
    assert trp.load_registry(tmp_path / "nope.yaml") == []
    assert trp.load_hits(tmp_path / "nope.csv").empty
    assert trp.load_pending(tmp_path / "nope.csv").empty


# ── 未結束交易（2026-09-06）──────────────────────────────────────────────────

def test_period_summary_excludes_unresolved_from_win_rate_but_reports_the_count():
    # Arrange：2026H1 三筆已結束全贏、兩筆還開著
    rows = [{"signal_date": "2026-02-01", "ret": 0.03, "hold": 5, "resolved": True}
            for _ in range(3)]
    rows += [{"signal_date": "2026-03-01", "ret": -0.4, "hold": 20, "resolved": False}
             for _ in range(2)]

    # Act
    out = trp.period_summary(_hits(rows)).set_index("期間")

    # Assert：勝率不被還沒認賠的部位拉低，但未結束筆數要看得到
    assert out.loc["2026H1", "勝率"] == 1.0
    assert out.loc["2026H1", "交易數"] == 3
    assert out.loc["2026H1", "未結束"] == 2


def test_period_with_only_unresolved_trades_still_appears():
    # Arrange：整期都還沒結束 —— 不能整期消失，不然使用者以為沒訊號
    rows = [{"signal_date": "2026-08-01", "ret": -0.1, "hold": 10, "resolved": False}]

    # Act
    out = trp.period_summary(_hits(rows)).set_index("期間")

    # Assert
    assert out.loc["2026H2", "未結束"] == 1
    assert out.loc["2026H2", "交易數"] == 0


def test_hits_without_resolved_column_are_treated_as_finished():
    rows = [{"signal_date": "2024-02-01", "ret": 0.03, "hold": 5}]

    out = trp.period_summary(_hits(rows)).set_index("期間")

    assert out.loc["2024H1", "交易數"] == 1
    assert out.loc["2024H1", "未結束"] == 0


def test_overall_stats_reports_open_positions_separately():
    rows = [{"signal_date": "2024-02-01", "ret": 0.03, "hold": 5, "resolved": True},
            {"signal_date": "2024-03-01", "ret": -0.4, "hold": 20, "resolved": False}]

    out = trp.overall_stats(_hits(rows))

    assert out["交易數"] == 1
    assert out["未結束"] == 1
    assert out["勝率"] == 1.0


def test_open_positions_returns_unresolved_trades():
    rows = [{"signal_date": "2024-02-01", "stock_id": "1101", "ret": 0.03,
             "hold": 5, "resolved": True},
            {"signal_date": "2024-03-01", "stock_id": "2330", "ret": -0.4,
             "hold": 20, "resolved": False}]

    out = trp.open_positions(_hits(rows))

    assert list(out["stock_id"]) == ["2330"]


# ── 每日買賣點（2026-09-06）──────────────────────────────────────────────────

def _trades(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in ("signal_date", "buy_date", "sell_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def test_buys_on_returns_positions_bought_that_morning():
    # Arrange：訊號 1/02 → 1/03 開盤買 → 1/10 賣
    t = _trades([{"signal_date": "2024-01-02", "buy_date": "2024-01-03",
                  "sell_date": "2024-01-10", "stock_id": "2330", "ret": 0.03,
                  "hold": 5, "resolved": True}])

    # Act / Assert
    assert list(trp.buys_on(t, "2024-01-03")["stock_id"]) == ["2330"]
    assert trp.buys_on(t, "2024-01-02").empty      # 訊號日還沒買


def test_sells_on_ignores_unresolved_trades():
    # Arrange：未結束的那筆 sell_date 是資料尾端，不是真的賣出
    t = _trades([{"signal_date": "2024-01-02", "buy_date": "2024-01-03",
                  "sell_date": "2024-01-10", "stock_id": "A", "ret": 0.03,
                  "hold": 5, "resolved": True},
                 {"signal_date": "2024-01-02", "buy_date": "2024-01-03",
                  "sell_date": "2024-01-10", "stock_id": "B", "ret": -0.2,
                  "hold": 5, "resolved": False}])

    # Act
    out = trp.sells_on(t, "2024-01-10")

    # Assert
    assert list(out["stock_id"]) == ["A"]


def test_new_signals_merges_pending_picks_not_yet_traded():
    # Arrange：當天兩檔訊號，其中 B 還沒成交（隔日開盤價還沒出來）
    t = _trades([{"signal_date": "2026-09-01", "buy_date": "2026-09-02",
                  "sell_date": "2026-09-05", "stock_id": "A", "ret": 0.03,
                  "hold": 3, "resolved": True}])
    pending = pd.DataFrame({"date": pd.to_datetime(["2026-09-01"]),
                            "stock_id": ["B"], "rank": [1], "score": [0.99]})

    # Act
    out = trp.new_signals_on(t, pending, "2026-09-01")

    # Assert
    assert set(out["stock_id"]) == {"A", "B"}


def test_new_signals_does_not_duplicate_a_pick_present_in_both_sources():
    t = _trades([{"signal_date": "2026-09-01", "buy_date": "2026-09-02",
                  "sell_date": "2026-09-05", "stock_id": "A", "ret": 0.03,
                  "hold": 3, "resolved": True}])
    pending = pd.DataFrame({"date": pd.to_datetime(["2026-09-01"]), "stock_id": ["A"]})

    out = trp.new_signals_on(t, pending, "2026-09-01")

    assert len(out) == 1


def test_holding_on_covers_the_days_between_buy_and_sell():
    t = _trades([{"signal_date": "2024-01-02", "buy_date": "2024-01-03",
                  "sell_date": "2024-01-10", "stock_id": "A", "ret": 0.03,
                  "hold": 5, "resolved": True}])

    assert list(trp.holding_on(t, "2024-01-05")["stock_id"]) == ["A"]
    assert trp.holding_on(t, "2024-01-02").empty   # 還沒買
    assert trp.holding_on(t, "2024-01-10").empty   # 當天賣掉，不算持有


def test_trading_days_are_newest_first_and_include_pending():
    t = _trades([{"signal_date": "2024-01-02", "buy_date": "2024-01-03",
                  "sell_date": "2024-01-10", "stock_id": "A", "ret": 0.0,
                  "hold": 5, "resolved": True}])
    pending = pd.DataFrame({"date": pd.to_datetime(["2026-09-01"]), "stock_id": ["B"]})

    days = trp.trading_days(t, pending)

    assert days[0] == pd.Timestamp("2026-09-01")
    assert days == sorted(days, reverse=True)
