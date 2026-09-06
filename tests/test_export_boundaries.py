"""匯出邊界的契約測試 —— 這些邏輯 2026-08-27 前完全沒有測試護著。"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.export.build_public_bundle import (LOOKAHEAD_TRADING_DAYS, TEST_END,
                                               price_end_date)


def _days(n, start="2026-08-01"):
    return pd.Series(pd.bdate_range(start, periods=n))


def test_no_days_after_test_end_falls_back_to_test_end():
    only_before = pd.Series(pd.to_datetime(["2026-07-30", "2026-07-31"]))
    assert price_end_date(only_before) == pd.Timestamp(TEST_END)


def test_fewer_days_than_lookahead_gives_all_of_them():
    """資料不足 20 天時給到底，不可以拋錯也不可以少給。"""
    after = _days(3)
    assert price_end_date(after) == after.iloc[-1]


def test_price_runs_to_the_last_available_day():
    """2026-09-06 起契約改了：訊號會延伸到最新，價格就不能卡在 TEST_END+20，
    否則最新那批訊號在站上只看得到中途 —— 正是 lookahead 當初要避免的事。"""
    after = _days(40)
    assert price_end_date(after) == after.iloc[-1]


def test_price_end_is_not_capped_at_the_old_twenty_day_limit():
    after = _days(40)
    assert price_end_date(after) != after.iloc[LOOKAHEAD_TRADING_DAYS - 1]


def test_days_before_test_end_are_ignored():
    mixed = pd.concat([pd.Series(pd.to_datetime(["2025-06-01"])), _days(40)])
    assert price_end_date(mixed) == _days(40).iloc[-1]


def test_lookahead_note_tells_the_truth_when_short():
    """不足 20 天時 manifest 不可以照抄 20 —— 稽核抓到的不實宣告。"""
    from engine.export.build_public_bundle import _lookahead_note, lookahead_days

    short = pd.DataFrame({"date": pd.concat([
        pd.Series(pd.to_datetime(["2026-07-31"])), _days(3)])})
    assert lookahead_days(short["date"]) == 3
    note = _lookahead_note(short)
    assert "多帶 3 個交易日" in note
    assert "還差 17" in note

    full = pd.DataFrame({"date": pd.concat([
        pd.Series(pd.to_datetime(["2026-07-31"])), _days(40)])})
    assert lookahead_days(full["date"]) == 40
    assert "還差" not in _lookahead_note(full)


def test_superseded_outputs_are_removed(tmp_path):
    """契約改過後的舊產物要被清掉，不能只靠 dashboard 測試事後擋。"""
    from engine.export.build_public_bundle import (SUPERSEDED_OUTPUTS,
                                                   clear_superseded)

    stale = tmp_path / SUPERSEDED_OUTPUTS[0]
    stale.write_bytes(b"x")
    keep = tmp_path / "price_test.parquet"
    keep.write_bytes(b"y")

    removed = clear_superseded(tmp_path)

    assert removed == [SUPERSEDED_OUTPUTS[0]]
    assert not stale.exists()
    assert keep.exists(), "不該碰到有效的產出"
