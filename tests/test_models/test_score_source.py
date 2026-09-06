"""分數組裝的規矩（2026-09-06）。

`forward_scores()` 只往後延伸 live 分數。這條規矩存在的理由：`score_live_*.parquet`
是對全歷史算的，包含訓練期；把那段當成驗證資料等於用樣本內分數自我證明。
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.models import score_source as ss


@pytest.fixture
def fake_scores(tmp_path, monkeypatch):
    """訓練期切分 2025-01~2025-02，live 涵蓋 2024-01~2025-04（含訓練期之前）。"""
    split = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-02", "2025-02-03"]),
        "stock_id": ["1101", "1101"], "score": [0.5, 0.6]})
    live = pd.DataFrame({
        "date": pd.to_datetime(["2024-06-03", "2025-01-02", "2025-03-03", "2025-04-01"]),
        "stock_id": ["1101"] * 4, "score": [0.9, 0.1, 0.7, 0.8]})
    split_path = tmp_path / "score_swing_test.parquet"
    live_path = tmp_path / "score_live_swing.parquet"
    split.to_parquet(split_path); live.to_parquet(live_path)
    monkeypatch.setattr(ss, "score_path", lambda k, s: split_path)
    monkeypatch.setattr(ss, "live_score_path", lambda k: live_path)
    return split, live


def test_live_extends_only_after_the_last_split_date(fake_scores):
    out = ss.forward_scores("swing", ("test",))

    dates = sorted(out["date"].dt.date.astype(str))
    assert dates == ["2025-01-02", "2025-02-03", "2025-03-03", "2025-04-01"]


def test_live_rows_before_the_splits_are_dropped(fake_scores):
    """2024-06 那筆在訓練期內，收進來就是拿樣本內分數當驗證。"""
    out = ss.forward_scores("swing", ("test",))

    assert pd.Timestamp("2024-06-03") not in set(out["date"])


def test_split_score_wins_when_a_date_exists_in_both(fake_scores):
    """2025-01-02 兩邊都有 —— 以訓練期分數檔為準，不被 live 蓋掉。"""
    out = ss.forward_scores("swing", ("test",))

    row = out[out["date"] == pd.Timestamp("2025-01-02")]
    assert row["score"].iloc[0] == 0.5


def test_missing_split_files_return_empty_frame_with_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "score_path", lambda k, s: tmp_path / "nope.parquet")
    monkeypatch.setattr(ss, "live_score_path", lambda k: tmp_path / "nope2.parquet")

    out = ss.forward_scores("swing", ("test",))

    assert out.empty
    assert list(out.columns) == ["date", "stock_id", "score"]
