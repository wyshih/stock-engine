"""相依圖與過期偵測的測試。

2026-08-29 的教訓：chip_features 修好之後，build_features 看到那幾天日期已存在
就跳過，修正沒有往下傳，而且沒有任何錯誤訊息 —— 得靠人自己去查才發現。
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.features import pipeline_graph as pg


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(pg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pg, "STAMP_FILE", tmp_path / ".pipeline_stamps.json")
    return tmp_path


def _write(tmp_path, name, dates, value=1.0):
    pd.DataFrame({"date": pd.to_datetime(dates),
                  "stock_id": ["1101"] * len(dates),
                  "v": [value] * len(dates)}).to_parquet(
        tmp_path / f"{name}.parquet", index=False)


def test_dependency_graph_has_no_dangling_names():
    """每個上游若本身也是衍生檔，必須在圖裡有定義。"""
    derived = set(pg.DEPENDENCIES)
    sources = {u for ups in pg.DEPENDENCIES.values() for u in ups}
    dangling = {u for u in sources & derived if u not in pg.DEPENDENCIES}
    assert not dangling


def test_features_depends_on_chip_features():
    """這條相依就是 2026-08-29 那次沒被追蹤的關係。"""
    assert "chip_features" in pg.DEPENDENCIES["features"]
    assert "features" in pg.DEPENDENCIES["features_v3"]


def test_no_stamp_means_not_stale(data_dir):
    """第一次跑不該把所有東西都報成過期。"""
    _write(data_dir, "chip", ["2026-01-01"])
    _write(data_dir, "chip_features", ["2026-01-01"])
    assert pg.stale("chip_features") == []


def test_upstream_change_is_detected(data_dir):
    _write(data_dir, "chip", ["2026-01-01"])
    _write(data_dir, "price", ["2026-01-01"])
    _write(data_dir, "stock_list", ["2026-01-01"])
    _write(data_dir, "chip_features", ["2026-01-01"])
    pg.stamp("chip_features")
    assert pg.stale("chip_features") == []

    _write(data_dir, "chip", ["2026-01-01", "2026-01-02"], value=2.0)   # 上游被改
    assert pg.stale("chip_features") == ["chip"]


def test_require_fresh_aborts_with_actionable_message(data_dir):
    _write(data_dir, "price", ["2026-01-01"])
    _write(data_dir, "labels", ["2026-01-01"])
    pg.stamp("labels")
    _write(data_dir, "price", ["2026-01-01", "2026-01-02"], value=9.0)

    with pytest.raises(SystemExit, match="invalidate"):
        pg.require_fresh("labels")


def test_invalidate_removes_rows_from_the_cutoff(data_dir):
    dates = ["2026-01-01", "2026-01-02", "2026-01-05"]
    _write(data_dir, "features", dates)
    _write(data_dir, "features_v3", dates)

    removed = pg.invalidate_from("2026-01-02", names=("features", "features_v3"))

    assert removed == {"features": 2, "features_v3": 2}
    left = pd.read_parquet(data_dir / "features.parquet")
    assert pd.to_datetime(left["date"]).max() == pd.Timestamp("2026-01-01")


def test_invalidate_skips_files_without_a_date_column(data_dir):
    pd.DataFrame({"stock_id": ["1101"], "name": ["台泥"]}).to_parquet(
        data_dir / "stock_list.parquet", index=False)
    assert pg.invalidate_from("2026-01-01", names=("stock_list",)) == {}
