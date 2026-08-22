"""`build_public_bundle.py` 的回歸測試。

存在的理由：這支決定了**什麼東西會被推上 public repo**。推錯就收不回來，
所以要釘住兩件事：

  1. 期間與模型清單不可以被改壞（10 個模型、2025-02-01 ~ 2026-07-31）
  2. 體積護欄真的會擋 —— 產出一包推不上 GitHub 的東西比失敗還糟
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from engine.export import build_public_bundle as bpb
from engine.models.bundle import CHOSEN_THRESHOLDS


class TestScope:
    def test_covers_exactly_ten_models(self):
        assert len(bpb.MODEL_KEYS) == 10

    def test_every_model_has_a_chosen_threshold(self):
        """門檻讀 bundle.CHOSEN_THRESHOLDS，不硬編在 export 裡（CLAUDE.md 規則 7）。"""
        assert set(bpb.MODEL_KEYS) == set(CHOSEN_THRESHOLDS)

    def test_period_is_round4_test_span(self):
        assert bpb.TEST_START == "2025-02-01"
        assert bpb.TEST_END == "2026-07-31"

    def test_matched_variant_is_daily_top_1p5_pct(self):
        """訊號數對齊版是每日前 1.5%（BACKTEST_LOG #28）。"""
        assert bpb.MATCHED_TOP_PCT == pytest.approx(0.015)


class TestSizeGuard:
    def test_passes_when_small(self, tmp_path):
        # Arrange
        (tmp_path / "a.parquet").write_bytes(b"x" * 1024)

        # Act / Assert：不該丟例外
        bpb.check_size(tmp_path)

    def test_aborts_when_single_file_too_big(self, tmp_path, monkeypatch):
        # Arrange
        monkeypatch.setattr(bpb, "MAX_FILE_MB", 0.001)
        (tmp_path / "big.parquet").write_bytes(b"x" * 200_000)

        # Act / Assert
        with pytest.raises(SystemExit, match="體積護欄"):
            bpb.check_size(tmp_path)

    def test_aborts_when_total_too_big(self, tmp_path, monkeypatch):
        # Arrange
        monkeypatch.setattr(bpb, "MAX_TOTAL_MB", 0.01)
        for i in range(5):
            (tmp_path / f"f{i}.parquet").write_bytes(b"x" * 200_000)

        # Act / Assert
        with pytest.raises(SystemExit, match="體積護欄"):
            bpb.check_size(tmp_path)


class TestManifest:
    def test_records_period_models_and_disclaimer(self, tmp_path):
        # Arrange
        price = pd.DataFrame({
            "date": pd.to_datetime(["2025-02-03", "2025-02-04"]),
            "stock_id": ["2330", "2330"],
        })
        scores = pd.DataFrame({"date": price["date"], "stock_id": price["stock_id"]})

        # Act
        bpb.write_manifest(tmp_path, price, scores)
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

        # Assert
        assert manifest["period"] == {"start": "2025-02-01", "end": "2026-07-31",
                                      "trading_days": 2}
        assert [m["key"] for m in manifest["models"]] == list(bpb.MODEL_KEYS)
        assert manifest["backtest"]["dedup"] is False
        assert manifest["backtest"]["exit_rules"] == {
            "trail_trigger": 0.15, "trail_pct": 0.10, "stop_loss": 0.20}
        assert "不構成任何投資建議" in manifest["disclaimer"]
        assert manifest["caveats"], "限制說明不可以是空的"

    def test_thresholds_come_from_bundle(self, tmp_path):
        # Arrange
        price = pd.DataFrame({"date": pd.to_datetime(["2025-02-03"]), "stock_id": ["2330"]})

        # Act
        bpb.write_manifest(tmp_path, price, price)
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

        # Assert
        assert {m["key"]: m["threshold"] for m in manifest["models"]} == CHOSEN_THRESHOLDS


class TestMissingInputs:
    def test_build_scores_fails_loudly_when_untrained(self, tmp_path, monkeypatch):
        """分數檔還沒產生時要明講「先 make train」，不可以默默產出半包。"""
        # Arrange
        monkeypatch.setattr(bpb, "score_path", lambda key, split: tmp_path / f"{key}_{split}.parquet")

        # Act / Assert
        with pytest.raises(SystemExit, match="make train"):
            bpb.build_scores(tmp_path)
