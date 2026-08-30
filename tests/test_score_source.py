"""分數來源的契約測試。

2026-08-27：匯出程式把 live 分數直接併進來，破壞了 nobear 家族的股票宇宙，
而回測那條路徑沒跟著改 —— 同一模型同一門檻，網站兩頁差 5~14%。
這裡把當時全部沒被擋住的行為釘死。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from engine.models import score_source as ss


def _frame(dates, ids, score):
    return pd.DataFrame({"date": pd.to_datetime(dates), "stock_id": ids,
                         "score": score})


@pytest.fixture
def fake_sources(monkeypatch, tmp_path):
    """訓練期分數：D1~D2 各兩檔；live：D1~D3 各三檔（多一檔、多一天）。"""
    trained = _frame(["2026-01-01"] * 2 + ["2026-01-02"] * 2,
                     ["1101", "1102"] * 2, [0.9] * 4)
    live = _frame(["2026-01-01"] * 3 + ["2026-01-02"] * 3 + ["2026-01-03"] * 3,
                  ["1101", "1102", "9999"] * 3, [0.1] * 9)
    tp, lp = tmp_path / "trained.parquet", tmp_path / "live.parquet"
    trained.to_parquet(tp)
    live.to_parquet(lp)
    monkeypatch.setattr(ss, "score_path", lambda k, s: tp)
    monkeypatch.setattr(ss, "live_score_path", lambda k: lp)
    monkeypatch.setattr(ss, "_is_nobear", lambda k: False)


def test_live_only_fills_dates_the_training_file_lacks(fake_sources):
    """live 不可以往訓練期既有的日子塞股票 —— 那些是被過濾掉的，不是漏算的。"""
    out = ss.combined_scores("m1", ("test",))
    per_day = out.groupby(out["date"].dt.date)["stock_id"].nunique().to_dict()
    assert per_day[dt.date(2026, 1, 1)] == 2, "訓練期既有日期被 live 灌入額外股票"
    assert per_day[dt.date(2026, 1, 2)] == 2
    assert per_day[dt.date(2026, 1, 3)] == 3, "訓練期沒有的日期應該由 live 補上"
    assert "9999" not in set(out[out["date"] == "2026-01-01"]["stock_id"])


def test_training_scores_win_on_overlapping_rows(fake_sources):
    """重疊列一律以訓練期為準（目前兩者數值相同，但不可以靠這個巧合）。"""
    out = ss.combined_scores("m1", ("test",))
    overlap = out[out["date"] <= "2026-01-02"]
    assert (overlap["score"] == 0.9).all(), "live 蓋掉了訓練期的分數"


def test_nobear_models_filter_live_rows(fake_sources, monkeypatch):
    """nobear 家族的 live 分數必須套上同一份去空頭過濾。"""
    monkeypatch.setattr(ss, "_is_nobear", lambda k: True)
    monkeypatch.setattr(ss, "_load_bear_flags", lambda: pd.DataFrame({
        "date": pd.to_datetime(["2026-01-03"] * 3),
        "stock_id": ["1101", "1102", "9999"],
        ss.BEAR_FLAG: [0.0, 1.0, 0.0],          # 1102 是空頭排列
    }))
    out = ss.combined_scores("m6", ("test",))
    day3 = out[out["date"] == "2026-01-03"]
    assert set(day3["stock_id"]) == {"1101", "9999"}, "空頭排列的股票沒被排除"


def test_nobear_detection_reads_bundle_not_the_model_name(monkeypatch):
    """是不是 nobear 要看 bundle 記的 label_name，不能從代號猜。"""
    monkeypatch.setattr(ss, "load_by_key", lambda k: {"label_name": "label_nobear"})
    assert ss._is_nobear("randomly_named_model") is True
    monkeypatch.setattr(ss, "load_by_key", lambda k: {"label_name": "label_up20"})
    assert ss._is_nobear("m6_base_nobear") is False


def test_missing_score_files_abort_with_actionable_message(monkeypatch, tmp_path):
    monkeypatch.setattr(ss, "score_path", lambda k, s: tmp_path / "nope.parquet")
    monkeypatch.setattr(ss, "live_score_path", lambda k: tmp_path / "nope.parquet")
    with pytest.raises(SystemExit, match="make train"):
        ss.combined_scores("m1", ("test",))


def test_backtest_and_export_share_one_score_source():
    """兩邊各組一份就是 2026-08-27 那個 bug 的成因 —— 釘住共用。"""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "engine"
    for rel in ("backtest/summary.py", "export/build_public_bundle.py"):
        code = (root / rel).read_text(encoding="utf-8")
        assert "combined_scores" in code, f"{rel} 沒有走共用的分數來源"
        assert "live_score_path" not in code, f"{rel} 又自己讀 live 分數了"


def test_earlier_split_wins_when_two_splits_overlap(monkeypatch, tmp_path):
    """切分之間若有重疊日期，以先列出的那個為準（keep="first"）。

    live 改成只補缺日之後就不再產生重疊列，所以 keep= 只剩這條路徑會走到 ——
    沒有這個測試，keep 被改成 "last" 不會有任何測試變紅。
    """
    first = _frame(["2026-01-01"], ["1101"], [0.9])
    second = _frame(["2026-01-01"], ["1101"], [0.1])
    fp, sp = tmp_path / "a.parquet", tmp_path / "b.parquet"
    first.to_parquet(fp)
    second.to_parquet(sp)
    monkeypatch.setattr(ss, "score_path", lambda k, s: {"test": fp, "test2": sp}[s])
    monkeypatch.setattr(ss, "live_score_path", lambda k: tmp_path / "none.parquet")
    monkeypatch.setattr(ss, "_is_nobear", lambda k: False)
    out = ss.combined_scores("m1", ("test", "test2"))
    assert len(out) == 1
    assert out["score"].iloc[0] == 0.9, "後面的切分蓋掉了前面的"
