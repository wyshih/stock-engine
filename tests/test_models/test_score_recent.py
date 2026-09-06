"""分數落後的偵測（2026-09-06）。

背景：swing 訓練完之後沒人跑過 `make update`，它的 live 分數卡在 2026-07-31，
而其他模型都到 09-03。公開站的八月買賣點整個消失，沒有任何地方報錯。
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.models import score_recent as sr


@pytest.fixture
def live_dates(monkeypatch):
    """{模型: 它的 live 分數日期集合}，由測試自己填。"""
    table: dict[str, set] = {}
    monkeypatch.setattr(sr, "existing_dates", lambda key: table.get(key, set()))
    return table


def test_model_behind_the_latest_date_is_reported(live_dates):
    # Arrange：swing 落後兩天
    live_dates["m1"] = {pd.Timestamp("2026-09-03")}
    live_dates["swing"] = {pd.Timestamp("2026-07-31")}

    # Act
    stale = sr.stale_models(["m1", "swing"], pd.Timestamp("2026-09-03"))

    # Assert
    assert set(stale) == {"swing"}
    assert stale["swing"] == pd.Timestamp("2026-07-31")


def test_model_with_no_scores_at_all_is_reported_with_none(live_dates):
    live_dates["m1"] = {pd.Timestamp("2026-09-03")}

    stale = sr.stale_models(["m1", "brand_new"], pd.Timestamp("2026-09-03"))

    assert stale == {"brand_new": None}


def test_all_models_up_to_date_reports_nothing(live_dates):
    live_dates["m1"] = {pd.Timestamp("2026-09-02"), pd.Timestamp("2026-09-03")}
    live_dates["swing"] = {pd.Timestamp("2026-09-03")}

    assert sr.stale_models(["m1", "swing"], pd.Timestamp("2026-09-03")) == {}


def test_scores_ahead_of_the_latest_date_are_not_flagged(live_dates):
    """分數比特徵檔新不是落後（重跑過特徵檔時會出現），不該報警。"""
    live_dates["m1"] = {pd.Timestamp("2026-09-10")}

    assert sr.stale_models(["m1"], pd.Timestamp("2026-09-03")) == {}
