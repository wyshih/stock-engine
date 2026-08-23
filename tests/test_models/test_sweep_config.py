"""從各模型自己的 sweep CSV 取最佳組態。

為什麼有這支：2026-08-24 這裡是**同一個根因的第三次失敗** —— 調參的 CSV 欄位
集合改了（不再評估測試期），但 best_config() 有一行給人看的 log 仍用
eval_splits() 去讀 test_auc，KeyError 中止。選組態的邏輯本身完全正常。

所以這裡釘住的是**契約**：best_config 只能依賴 val_sel_auc 與參數欄，
CSV 有沒有其他 AUC 欄都不該影響它。
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.models import sweep_config as sc


def _csv(tmp_path, rows, name="sweep_m1_base_up20_rf.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sc, "CONFIG_DIR", tmp_path / "_nonexistent")
    return tmp_path


class TestBestConfig:
    def test_picks_highest_val_sel(self, data_dir):
        # Arrange
        _csv(data_dir, [
            {"model": "rf", "max_features": 15, "max_depth": 10,
             "min_samples_leaf": 200, "class_weight": None,
             "seconds": 100.0, "val_sel_auc": 0.6000, "val_sel_lift": 1.1},
            {"model": "rf", "max_features": 15, "max_depth": 20,
             "min_samples_leaf": 200, "class_weight": None,
             "seconds": 200.0, "val_sel_auc": 0.6088, "val_sel_lift": 1.2},
        ])

        # Act
        params = sc.best_config("rf", key="m1_base_up20")

        # Assert
        assert params["max_depth"] == 20
        assert params["max_features"] == 15
        assert params["min_samples_leaf"] == 200
        # metrics 與計時不是超參數，不可以混進去餵給模型
        for junk in ("val_sel_auc", "val_sel_lift", "seconds", "model"):
            assert junk not in params

    def test_works_without_test_columns(self, data_dir):
        """調參只評估 val_sel —— CSV 沒有 test_auc 時不得爆炸（就是踩過的那次）。"""
        # Arrange
        _csv(data_dir, [{"model": "rf", "max_features": 20, "max_depth": 10,
                         "min_samples_leaf": 200, "class_weight": None,
                         "seconds": 100.0, "val_sel_auc": 0.6, "val_sel_lift": 1.1}])

        # Act / Assert：不丟例外即為通過
        assert sc.best_config("rf", key="m1_base_up20")["max_features"] == 20

    def test_extra_auc_columns_are_tolerated(self, data_dir):
        """反向情況：CSV 若含舊格式的 test_auc，也要能讀，且不混進超參數。"""
        # Arrange
        _csv(data_dir, [{"model": "rf", "max_features": 20, "max_depth": 10,
                         "min_samples_leaf": 200, "class_weight": None,
                         "seconds": 100.0, "val_sel_auc": 0.6, "val_sel_lift": 1.1,
                         "val_es_auc": 0.59, "test_auc": 0.62, "test2_auc": 0.61}])

        # Act
        params = sc.best_config("rf", key="m1_base_up20")

        # Assert
        assert params["max_features"] == 20
        assert not any(k.endswith(("_auc", "_lift")) for k in params)

    def test_class_weight_nan_becomes_none(self, data_dir):
        """CSV 的空值讀回來是 NaN，要還原成 None 才餵得進 sklearn。"""
        # Arrange
        _csv(data_dir, [{"model": "rf", "max_features": 15, "max_depth": 10,
                         "min_samples_leaf": 200, "class_weight": None,
                         "seconds": 100.0, "val_sel_auc": 0.6, "val_sel_lift": 1.1}])

        # Act / Assert
        assert sc.best_config("rf", key="m1_base_up20")["class_weight"] is None

    def test_missing_csv_names_both_locations(self, data_dir):
        """缺檔的錯誤訊息要講清楚找過哪裡，否則沒人知道該去哪補。"""
        # Act / Assert
        with pytest.raises(FileNotFoundError, match="sweep_m8_v3_nobear_rf.csv"):
            sc.best_config("rf", key="m8_v3_nobear")
