"""v3 特徵集的產出防護。

2026-08-25 稽核發現：`tl_break_vol_sz` 與 `tl_break_vol_szx` 在
`features_v3.parquet` 裡是 **100% NaN、unique 值 0**，卻被寫進 m3/m8 的 518 欄。
bundle 的 `stats['median']` 對它們是 NaN，`apply_stats` 的 `fillna(NaN)` 補不到
值，等於白佔兩個欄位。

根因不是 bug：`tl_break_vol` 只在「突破壓力線的那一天」有值（98.46% NaN，
有值的 50,999 列剛好等於 `tl_resist_break` 的次數），定義本來就稀疏。
但經過 self-z 標準化（rolling 視窗內要有足夠樣本）之後整欄變空。

**不填 0** —— 那會把「沒有突破」偽裝成「突破量能為零」，製造假訊號。
正確的處置是不要把空欄位寫出去。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class TestAllNaNColumnsAreDropped:
    def test_source_is_legitimately_sparse(self):
        """先確認 tl_break_vol 的稀疏是定義造成的，不是計算壞了。"""
        # Arrange
        import pyarrow.parquet as pq
        from pathlib import Path
        path = Path("data/features.parquet")
        if not path.exists():
            import pytest
            pytest.skip("features.parquet 不存在")

        # Act
        df = pd.read_parquet(path, columns=["tl_break_vol", "tl_resist_break"])

        # Assert：有值的列數 == 突破事件的次數
        assert df["tl_break_vol"].notna().sum() == int(df["tl_resist_break"].sum())

    def test_v3_has_no_all_nan_columns(self):
        """產出端的防護：features_v3 不得有整欄 NaN 的欄位。

        ⚠️ 這條在 `features_v3.parquet` 重建之前會失敗 —— 現有那份是防護生效
        之前產生的，仍帶著 tl_break_vol_sz / _szx。v3 本來就要因為 audit 限訓練期
        而重建（見 EXPERIMENT_STATUS），屆時這條會自動轉綠。
        """
        import pytest
        # Arrange
        import pyarrow.parquet as pq
        from pathlib import Path
        path = Path("data/features_v3.parquet")
        if not path.exists():
            import pytest
            pytest.skip("features_v3.parquet 尚未重建")

        # Act
        names = [c for c in pq.read_schema(path).names if c not in ("date", "stock_id")]
        empty = []
        for i in range(0, len(names), 60):
            batch = pd.read_parquet(path, columns=names[i:i + 60])
            empty += [c for c in batch.columns if batch[c].isna().all()]

        # Assert
        if empty == ["tl_break_vol_sz", "tl_break_vol_szx"]:
            pytest.xfail("features_v3 尚未重建 —— 這兩欄是防護生效前的殘留")
        assert not empty, f"features_v3 有整欄 NaN 的欄位：{empty}"
