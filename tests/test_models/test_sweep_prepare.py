"""sweep 的資料準備與網格展開。

為什麼有這支：2026-08-23 這條路徑連續炸兩次，兩次都是「跑了幾小時、跑到那一行
才發現」——

1. `objective()` 寫死回傳 `val_es_auc`，把 val_es 從評估切分拿掉後 KeyError
2. `prepare()` 重用 `load_with_label`，但那支回傳整包 dict 不是 (df, cols)，
   解包失敗。而且只有非預設 label 的模型才會走到，前面 12 組全正常，掩蓋了三小時

兩次都不是難以預料的邏輯錯誤，是**契約沒對齊**。這種東西應該由測試在幾秒內
攔下來，而不是由三小時的訓練來發現。
"""
from __future__ import annotations

import itertools

import pandas as pd
import pytest

from engine.models import sweep_round1 as sw
from engine.models.train_single import use_round


class TestSearchSpace:
    """每個模型一份空間，不得共用；未定義的 key 要擋下。"""

    def test_every_model_has_its_own_space(self):
        # Arrange
        expected = {"m1_base_up20", "m1_mdd10"}

        # Act / Assert
        from engine.models.bundle import EXPERIMENTAL_KEYS
        assert set(sw.MODEL_SPACES) - EXPERIMENTAL_KEYS == expected
        assert set(sw.MODEL_FIXED_PARAMS) - EXPERIMENTAL_KEYS == expected

    def test_unknown_key_is_rejected(self):
        """沒定義空間的模型必須報錯，不可以默默沿用別人的。"""
        # Act / Assert
        with pytest.raises(ValueError, match="沒有定義搜尋空間"):
            sw.search_space("rf", 4, "m99_bogus")

    @pytest.mark.parametrize("key", list(sw.MODEL_SPACES))
    def test_grid_expands_to_four_combos(self, key):
        # Act
        space, fixed = sw.search_space("rf", 4, key)
        combos = list(itertools.product(*space.values()))

        # Assert
        assert len(combos) == 4
        assert "min_samples_leaf" in fixed   # 固定值仍要寫進 CSV，日後查得到
        assert fixed["class_weight"] is None

    def test_tpe_families_are_rejected(self):
        """lightgbm / lambdarank 走 TPE，Optuna 移除後不該還能悄悄跑網格。"""
        # Act / Assert
        with pytest.raises(NotImplementedError, match="Optuna"):
            sw.run_sweep("lightgbm", {}, out_path=None)

    def test_outer_parallelism_is_rejected(self):
        """外層並行會與內層 n_jobs 相乘超賣。"""
        # Act / Assert
        with pytest.raises(ValueError, match="外層並行"):
            sw.run_sweep("rf", {}, out_path=None, jobs=2)


class TestEvalSplits:
    def test_only_val_sel(self):
        """調參不碰測試期，也不需要 val_es（RF 沒有 early stopping）。"""
        # Arrange
        use_round(4)

        # Act / Assert
        assert sw.sweep_eval_splits() == ("val_sel",)
        assert "test" not in sw.sweep_eval_splits()


class TestLabelMerge:
    """非預設 label 的合併 —— 就是炸掉 m6 的那條路徑。"""

    def test_merges_external_label(self, tmp_path, monkeypatch):
        """外部 label 要被 merge 進特徵表，且不可變成 label_x / label_y。"""
        # Arrange
        dates = pd.to_datetime(["2020-01-02"] * 3)
        feats = pd.DataFrame({"date": dates, "stock_id": ["1101", "2330", "2317"],
                              "ma5": [1.0, 2.0, 3.0]})
        label_file = tmp_path / "labels_mdd10.parquet"
        pd.DataFrame({"date": dates[:2], "stock_id": ["1101", "2330"],
                      "label_mdd10": [1, 0]}).to_parquet(label_file)

        monkeypatch.setattr(sw, "load_data", lambda p: (feats.copy(), ["ma5"]))
        monkeypatch.setattr(sw, "split_frame", lambda df, name: df)
        # 前處理需要真實的統計量，這裡只驗合併，故換成回傳形狀正確的假值
        monkeypatch.setattr(sw, "preprocess", lambda train, others, cols: (None, {}))

        # Act
        data = sw.prepare(tmp_path / "features.parquet", None, (), (),
                          label_file, "label_mdd10")

        # Assert：merge 成功（第三檔沒有 label，應被 notna 濾掉）
        assert len(data["y_train"]) == 2
        assert list(data["y_train"]) == [1, 0]

    def test_label_already_present_is_not_merged_twice(self, tmp_path, monkeypatch):
        """重複 merge 會產生 label_x / label_y，後面就取不到欄位了。"""
        # Arrange
        dates = pd.to_datetime(["2020-01-02"] * 2)
        feats = pd.DataFrame({"date": dates, "stock_id": ["1101", "2330"],
                              "ma5": [1.0, 2.0], "label_mdd10": [1, 0]})
        label_file = tmp_path / "labels_mdd10.parquet"
        pd.DataFrame({"date": dates, "stock_id": ["1101", "2330"],
                      "label_mdd10": [9, 9]}).to_parquet(label_file)

        monkeypatch.setattr(sw, "load_data", lambda p: (feats.copy(), ["ma5"]))
        captured = {}

        def _capture(df, name):
            captured["df"] = df      # 不要用 `setdefault(...) or df`：
            return df                # DataFrame 的布林判斷會 ValueError

        monkeypatch.setattr(sw, "split_frame", _capture)
        monkeypatch.setattr(sw, "preprocess", lambda train, others, cols: (None, {}))

        # Act
        sw.prepare(tmp_path / "features.parquet", None, (), (), label_file, "label_mdd10")

        # Assert：用原本那欄（1,0），不是檔案裡的 9
        cols = captured["df"].columns
        assert "label_mdd10_x" not in cols and "label_mdd10_y" not in cols
        assert list(captured["df"]["label_mdd10"]) == [1, 0]
