"""`build_public_bundle.py` 的回歸測試。

存在的理由：這支決定了**什麼東西會被推上 public repo**。推錯就收不回來，
所以要釘住兩件事：

  1. 期間與模型清單不可以被改壞（5 個模型、2025-02-01 ~ 2026-07-31）
  2. 體積護欄真的會擋 —— 產出一包推不上 GitHub 的東西比失敗還糟
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from engine.export import build_public_bundle as bpb
from engine.models.bundle import CHOSEN_THRESHOLDS


class TestScope:
    def test_caveats_do_not_claim_a_stale_model_count(self):
        """CAVEATS 會進 manifest.json，被 dashboard 的「關於」頁逐條顯示給外人看。

        2026-08-24 稽核抓到：模型從十個縮到五個之後，這裡仍寫著「10 個模型全部是
        Round 4 切分…」——**那是會出現在公開網站上的事實錯誤**。文案裡的數字必須
        跟 MODEL_KEYS 對得起來。
        """
        # Arrange
        import re
        from engine.export import build_public_bundle as bpb
        actual = len(bpb.MODEL_KEYS)

        # Act：抓出 CAVEATS 裡所有「N 個模型」的宣稱
        claimed = [int(n) for text in bpb.CAVEATS
                   for n in re.findall(r"(\d+)\s*個模型", text)]

        # Assert
        for n in claimed:
            assert n == actual, (
                f"CAVEATS 宣稱 {n} 個模型，實際是 {actual} 個。"
                f"這條字串會顯示在公開網站上")

    def test_covers_exactly_the_shipped_models(self):
        """出貨清單。2026-09-02 只留 m1 兩個，2026-09-05 使用者要求加上 swing。

        寫死代號而不是只比個數 —— 個數對得上但換了模型，公開站就換了內容而
        沒有任何測試變紅。

        swing 的出場規則跟 m1 不同（分數跌破門檻就賣），能進共用出貨路徑的前提是
        `summary._run_one()` 與 `threshold_curve.backtest_trades()` 都已依 bundle 的
        `exit_rule` 分流。改動那兩處時要記得這條依賴。
        """
        assert bpb.MODEL_KEYS == ("m1_base_up20", "m1_mdd10", "swing")

    def test_every_model_has_a_chosen_threshold(self):
        """門檻讀 bundle.CHOSEN_THRESHOLDS，不硬編在 export 裡（CLAUDE.md 規則 7）。"""
        from engine.models.bundle import EXPERIMENTAL_KEYS
        assert set(bpb.MODEL_KEYS) == set(CHOSEN_THRESHOLDS) - EXPERIMENTAL_KEYS

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

        # Assert：end 跟著分數的最後一天走，不是照抄 TEST_END —— 分數會往後延伸到
        # 最新，宣告卡在 TEST_END 就是不實宣告（2026-09-06 改）。
        assert manifest["period"]["start"] == "2025-02-01"
        assert manifest["period"]["end"] == "2025-02-04"
        assert manifest["period"]["trading_days"] == 2
        assert "樣本外" in manifest["period"]["note"]
        assert all("start" in m for m in manifest["models"]), "各模型起點不同，要寫出來"
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
        from engine.models.bundle import EXPERIMENTAL_KEYS
        shipped = {k: v for k, v in CHOSEN_THRESHOLDS.items()
                   if k not in EXPERIMENTAL_KEYS}
        assert {m["key"]: m["threshold"] for m in manifest["models"]} == shipped


class TestMissingInputs:
    def test_build_scores_fails_loudly_when_untrained(self, tmp_path, monkeypatch):
        """分數檔還沒產生時要明講「先 make train」，不可以默默產出半包。"""
        # Arrange —— 分數來源已抽到 score_source，兩個消費端共用同一支。
        from engine.models import score_source as ss
        monkeypatch.setattr(ss, "score_path",
                            lambda key, split: tmp_path / f"{key}_{split}.parquet")
        monkeypatch.setattr(ss, "live_score_path", lambda key: tmp_path / f"{key}_live.parquet")

        # Act / Assert
        with pytest.raises(SystemExit, match="make train"):
            bpb.build_scores(tmp_path)
