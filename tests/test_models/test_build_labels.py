"""
`build_labels.py` 的邊界條件回歸測試（2026-07-30 新增）。

存在的理由（doc/AUDIT_20260728.md §A-1 / §C-10 / §C-11）：
`_triple_barrier` 的「資料尾端沒有未來價格時應標 NaN 而非 0」這個 bug，
**2026-07-26 已經為 persist 系列修過一次，卻在 tb 系列（含 label_meta）復發**。
直接原因就是沒有回歸測試把它釘住。CLAUDE.md 也明訂「所有金融計算必須有
unit test 驗證邊界條件」。

pandas 的陷阱：`NaN >= 0.07` 回傳 `False` 而不是 NaN，所以尾端「還不知道」
會被靜默寫成「沒中」，下游 `dropna()` 完全攔不到。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.models.build_labels import _shift_forward, _triple_barrier


def _make_price(n_days: int = 30, stocks=("1101", "2330")) -> pd.DataFrame:
    """造一段平盤走勢的假價格，方便個別測試自行覆寫特定日的收盤價。"""
    rows = []
    for sid in stocks:
        for i in range(n_days):
            rows.append({"date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                         "stock_id": sid, "close": 100.0})
    return pd.DataFrame(rows).sort_values(["stock_id", "date"]).reset_index(drop=True)


class TestShiftForward:
    def test_does_not_leak_across_stocks(self):
        """_shift_forward 必須逐股票獨立，不可把下一檔股票的價格當成本檔的未來。"""
        # Arrange：兩檔股票各 3 天，價格刻意不同
        price = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "stock_id": ["1101"] * 3 + ["2330"] * 3,
            "close": [10.0, 11.0, 12.0, 500.0, 510.0, 520.0],
        })

        # Act
        fwd = _shift_forward(price, "close", 1)

        # Assert：每檔的最後一天必須是 NaN，而不是接到下一檔的第一天
        assert pd.isna(fwd.iloc[2]), "1101 最後一天應為 NaN，不可接到 2330"
        assert pd.isna(fwd.iloc[5]), "2330 最後一天應為 NaN"
        assert fwd.iloc[0] == 11.0
        assert fwd.iloc[3] == 510.0


class TestTripleBarrierTailBoundary:
    """
    尾端邊界：未來資料不足以判定勝負的列，必須是 NaN（未定），不能是 0（沒中）。
    """

    def test_tail_rows_are_nan_not_zero(self):
        # Arrange：30 天平盤，horizon=5
        price = _make_price(n_days=30, stocks=("1101",))
        entry = price["close"]
        ma = price["close"] * 0.9          # 均線壓在下方，不會觸發停損
        horizon = 5

        # Act
        label = _triple_barrier(price, entry, ma, horizon=horizon, profit=0.10)

        # Assert：最後 horizon-1 天沒有足夠的未來收盤價，必須是 NaN
        tail = label.iloc[-(horizon - 1):]
        assert tail.isna().all(), (
            f"尾端 {horizon - 1} 天應為 NaN（未定），實際為 {tail.tolist()}。"
            "這正是 2026-07-26 在 persist 系列修過、卻在 tb 系列復發的 bug。"
        )

    def test_resolved_rows_are_not_nan(self):
        """有足夠未來資料、且明確觸及停利的列，必須是 1 而不是 NaN。"""
        # Arrange：第 3 天起大漲，前段列可在 horizon 內判定停利
        price = _make_price(n_days=30, stocks=("1101",))
        price.loc[price.index >= 3, "close"] = 200.0
        entry = pd.Series(100.0, index=price.index)
        ma = pd.Series(1.0, index=price.index)

        # Act
        label = _triple_barrier(price, entry, ma, horizon=5, profit=0.10)

        # Assert
        assert label.iloc[0] == 1, "明確達標的列應為 1"
        assert not pd.isna(label.iloc[0])

    def test_tail_nan_survives_dropna(self):
        """尾端 NaN 必須能被 dropna() 濾掉——這是下游訓練/評估防呆的關鍵。"""
        # Arrange
        price = _make_price(n_days=20, stocks=("1101",))
        label = _triple_barrier(price, price["close"], price["close"] * 0.9,
                                horizon=5, profit=0.10)

        # Act
        kept = pd.DataFrame({"y": label}).dropna(subset=["y"])

        # Assert
        assert len(kept) < len(label), (
            "dropna() 應該要濾掉尾端未定列。若長度相同，代表尾端被寫成 0 而非 NaN，"
            "假陰性會被當成負樣本吃進訓練與評估。"
        )

    def test_each_stock_has_own_tail(self):
        """多檔股票時，每一檔都要有自己的尾端 NaN，不能只有最後一檔有。"""
        # Arrange
        price = _make_price(n_days=20, stocks=("1101", "2330"))
        horizon = 5

        # Act
        label = _triple_barrier(price, price["close"], price["close"] * 0.9,
                                horizon=horizon, profit=0.10)
        price = price.assign(_y=label)

        # Assert
        for sid, grp in price.groupby("stock_id"):
            tail = grp["_y"].iloc[-(horizon - 1):]
            assert tail.isna().all(), f"{sid} 的尾端應為 NaN，實際 {tail.tolist()}"


class TestFeatureCols:
    """特徵選取不可混入標籤或洩漏欄位。"""

    def test_no_label_or_hitrate_columns(self):
        # Arrange
        from engine.models.submodel_config import ALL_MODEL_IDS, feature_cols

        all_cols = [
            "return_5d", "ma20", "rsi_14", "foreign_net_ratio", "mkt_return_20d",
            # 這些絕對不可以被選進特徵：
            "label_meta", "label_buy5_tb_ma20",
            "hitrate_buy5_tb_ma20",   # 用到 label(t-1)，需要未來 horizon 天的收盤價
        ]

        # Act / Assert
        for mid in ALL_MODEL_IDS:
            cols = feature_cols(mid, all_cols)
            bad = [c for c in cols if c.startswith(("label_", "hitrate_"))]
            assert not bad, f"{mid} 選到了標籤/洩漏欄位：{bad}"


# ⚠️ 原本這裡有 `TestGapCutoff`，測 `train_submodels.py::_gap_cutoff`。
# `train_submodels.py` 是已停用的舊委員會系統（檔頭自述），2026-08-22 已封存、
# 沒有搬進本 repo，這個測試因此一併移除。舊 repo 裡它其實也早就是壞的
# （`train_submodels.py` 已不在 `code/models/`，import 直接 ModuleNotFoundError）。
# 現行流程的 embargo 是寫死在 `train_single.ROUND_SPLITS`（兩個交界各一個月），
# 不再用交易日 gap 計算。


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
