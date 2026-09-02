"""bundle 的「該用哪一份特徵檔」推導，以及餵錯特徵檔時的擋人邏輯。

背景（2026-08-24 修）：當時五個模型分兩群訓練 —— m1/m2/m6 用 features.parquet
（378 欄）、m3/m8 用 features_v3.parquet（518 欄），但推論路徑全部寫死讀
features.parquet。`_tabular_matrix()` 的 `reindex` 會把缺的欄位變成 NaN，
再被 `apply_stats()` 用訓練期中位數補上 —— 不報錯、不警告。實測 m3 有 41%
的欄位被中位數取代，Top-20 只有 3/20 與正確推論重疊。

2026-09-02 移除 v3 家族後只剩一份特徵檔，`_infer_features_file()` 的欄名回推
也跟著刪了。**擋人邏輯與 `features_file` 這層間接刻意留著並繼續測** ——
它擋的是「bundle 要的欄位不在收到的特徵表裡」，與有幾份特徵檔無關；
下次再開第二份特徵集時，這道防護必須已經在。

這裡全部用**假 bundle 與假 DataFrame**，不載入真模型（真 bundle 150MB，
而且測試不該依賴 models/ 底下有東西）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.models import bundle as bundle_mod


# ── 假資料 ────────────────────────────────────────────────────────────────────

class _FakeModel:
    """predict_proba 只回定值，這裡測的是特徵組裝不是模型本身。"""

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([np.zeros(len(x)), np.full(len(x), 0.42)])


def _make_bundle(cols: list[str], features_file: str | None = None,
                 desc: str = "①原特徵·上漲天數") -> dict:
    stats = {
        "median": pd.Series(0.0, index=cols),
        "mean": pd.Series(0.0, index=cols),
        "std": pd.Series(1.0, index=cols),
    }
    bundle = {"cols": cols, "stats": stats, "model": _FakeModel(), "desc": desc}
    if features_file is not None:
        bundle["features_file"] = features_file
    return bundle


def _make_features(cols: list[str], n_rows: int = 3) -> pd.DataFrame:
    keys = pd.DataFrame({
        "date": pd.Timestamp("2026-08-21"),
        "stock_id": [f"{2330 + i}" for i in range(n_rows)],
    })
    values = pd.DataFrame({c: float(i) for i, c in enumerate(cols)}, index=keys.index)
    return pd.concat([keys, values], axis=1)


BASE_COLS = [f"rsi_{i}" for i in range(200)]
# 「bundle 要的欄位比收到的特徵表多」的情境。名稱沿用已移除的 v3 轉換後綴只是
# 為了與 2026-08-24 那個 bug 的現場一致，測的是缺欄比例不是後綴本身。
WIDE_COLS = BASE_COLS + [f"natr_{i}_sz" for i in range(50)] + [f"cci_{i}_szx" for i in range(50)]


# ── 1. 特徵檔的推導 ───────────────────────────────────────────────────────────

class TestFeaturesFile:

    def test_uses_stored_field_when_present(self):
        # Arrange
        bundle = _make_bundle(BASE_COLS, features_file="features_other.parquet")
        # Act / Assert：以 bundle 自己記的那一份為準，不從欄名猜
        assert bundle_mod.features_file(bundle) == "features_other.parquet"

    def test_falls_back_to_base_when_field_missing(self):
        """2026-08-24 之前訓練的 bundle 沒有 features_file 欄 → 退回 base。"""
        assert bundle_mod.features_file(_make_bundle(BASE_COLS)) == "features.parquet"

    def test_features_path_goes_through_paths_module(self):
        from engine.paths import DATA_DIR
        path = bundle_mod.features_path(_make_bundle(BASE_COLS))
        assert path == DATA_DIR / "features.parquet"

    def test_real_bundles_resolve_correctly(self):
        """兩個真模型的推導結果（bundle 不在就 skip，CI 上沒有 models/）。"""
        expected = {
            "m1_base_up20": "features.parquet",
            "m1_mdd10": "features.parquet",
        }
        available = set(bundle_mod.available_keys())
        if not expected.keys() <= available:
            pytest.skip("models/ 底下沒有這兩個 bundle")
        for key, want in expected.items():
            assert bundle_mod.features_file_for_key(key) == want, key


# ── 2. 餵錯特徵檔要 raise ─────────────────────────────────────────────────────

class TestMissingColumnGuard:

    def test_bundle_fed_a_narrower_feature_table_raises(self):
        """bundle 要 300 欄、只收到 200 欄（缺 33%）→ 必須 raise，不可靜默補值。"""
        # Arrange
        bundle = _make_bundle(WIDE_COLS, features_file="features_wide.parquet")
        feat = _make_features(BASE_COLS)     # 少了 100 欄（33%）

        # Act / Assert
        with pytest.raises(ValueError) as err:
            bundle_mod.score_single(bundle, feat, pd.Timestamp("2026-08-21"))
        msg = str(err.value)
        assert "features_wide.parquet" in msg    # 要講清楚該用哪一份
        assert "①原特徵·上漲天數" in msg          # 以及是哪個模型
        assert str(len(feat.columns)) in msg     # 以及收到的表有幾欄

    def test_small_gap_still_fills_with_median(self):
        """少量缺欄（新上市股票某些特徵還沒暖機）仍走既有的中位數補值。"""
        # Arrange：200 欄裡缺 1 欄 = 0.5%，在 1% 容忍範圍內
        bundle = _make_bundle(BASE_COLS)
        feat = _make_features(BASE_COLS[:-1])

        # Act
        out = bundle_mod.score_single(bundle, feat, pd.Timestamp("2026-08-21"))

        # Assert
        assert len(out) == 3
        assert out["score"].eq(np.float32(0.42)).all()

    def test_exactly_at_tolerance_is_allowed(self):
        """剛好等於門檻不算超過（> 才擋），避免邊界值被誤殺。"""
        cols = [f"f{i}" for i in range(100)]
        bundle = _make_bundle(cols)
        feat = _make_features(cols[:-1])          # 缺 1/100 = 1.0%
        assert bundle_mod.MISSING_COLS_TOLERANCE == 0.01
        assert len(bundle_mod.score_single(bundle, feat, pd.Timestamp("2026-08-21"))) == 3

    def test_complete_features_are_untouched(self):
        """欄位齊全時行為完全不變（沒有把正常路徑弄壞）。"""
        bundle = _make_bundle(BASE_COLS)
        feat = _make_features(BASE_COLS)
        out = bundle_mod.score_single(bundle, feat, pd.Timestamp("2026-08-21"))
        assert list(out.columns) == ["stock_id", "score"]
        assert len(out) == 3

    def test_warns_but_does_not_raise_on_small_gap(self, caplog):
        bundle = _make_bundle(BASE_COLS)
        feat = _make_features(BASE_COLS[:-1])
        with caplog.at_level("WARNING", logger="engine.models.bundle"):
            bundle_mod.score_single(bundle, feat, pd.Timestamp("2026-08-21"))
        assert any("中位數" in r.message for r in caplog.records)

    def test_empty_day_still_checked_first(self):
        """該日期沒有資料時也要先擋特徵檔 —— 否則餵錯檔會靜悄悄回空表。"""
        bundle = _make_bundle(WIDE_COLS)
        feat = _make_features(BASE_COLS)
        with pytest.raises(ValueError):
            bundle_mod.score_single(bundle, feat, pd.Timestamp("1999-01-01"))
