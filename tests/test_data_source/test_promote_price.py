"""`promote_price.py` 的回歸測試。

存在的理由：`price_official.parquet → price.parquet` 這一步當初是**手動**做的，
repo 裡沒有程式，整條流程因此重建不出來。規則是從兩份現有檔案反推的，
必須有測試把它釘住，否則下次有人「順手」改篩選條件就再也沒人發現。

釘住三件事：
  1. ETF（代號 "00" 開頭）要被篩掉
  2. 大盤指數 TWII 一定要留（build_market_features.py 要它）
  3. 冪等：同樣的輸入跑幾次結果都一樣
  4. 預設不用 stock_list 當篩子：已下市但仍有歷史的個股不可以被刪掉
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.data_source.promote_price import (
    OUTPUT_COLUMNS,
    allowed_ids,
    promote,
)


def _official(ids: list[str]) -> pd.DataFrame:
    """造一份 price_official 形狀的資料（含 ex_flag）。"""
    rows = []
    for sid in ids:
        for i in range(3):
            rows.append({
                "date": pd.Timestamp("2026-08-20") + pd.Timedelta(days=i),
                "stock_id": sid, "open": 10.0, "high": 11.0, "low": 9.0,
                "close": 10.5, "volume": 1000.0, "amount": 10500.0, "ex_flag": 0,
            })
    return pd.DataFrame(rows)


def _stock_list(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"stock_id": ids})


class TestPromote:
    def test_drops_etf_not_in_stock_list(self):
        # Arrange：0050 是 ETF，不在 stock_list 裡
        official = _official(["2330", "0050", "TWII"])
        keep = allowed_ids(official, _stock_list(["2330"]), prune=False)

        # Act
        out = promote(official, keep)

        # Assert
        assert set(out["stock_id"]) == {"2330", "TWII"}

    def test_keeps_market_index(self):
        """TWII 不在 stock_list 裡，但一定要留 —— 大盤特徵靠它。"""
        # Arrange
        official = _official(["2330", "TWII"])
        keep = allowed_ids(official, _stock_list([]), prune=False)

        # Act / Assert
        assert "TWII" in set(promote(official, keep)["stock_id"])

    def test_drops_ex_flag_and_fixes_column_order(self):
        # Arrange
        official = _official(["2330"])[["ex_flag", "close", "stock_id", "date",
                                        "open", "high", "low", "volume", "amount"]]
        keep = allowed_ids(official, _stock_list(["2330"]), prune=False)

        # Act
        out = promote(official, keep)

        # Assert
        assert list(out.columns) == OUTPUT_COLUMNS
        assert "ex_flag" not in out.columns

    def test_date_is_nanosecond_dtype(self):
        """price_official 是 datetime64[us]，產出必須統一成舊 price.parquet 的 ns。

        數值一樣但 parquet schema 不同，下游 merge 或 columns= 讀取會踩到。
        """
        # Arrange
        official = _official(["2330"])
        official["date"] = official["date"].astype("datetime64[us]")
        keep = allowed_ids(official, _stock_list(["2330"]), prune=False)

        # Act
        out = promote(official, keep)

        # Assert
        assert out["date"].dtype == "datetime64[ns]"

    def test_is_idempotent(self):
        """跑兩次結果必須完全一樣。"""
        # Arrange
        official = _official(["2330", "0050", "TWII"])
        keep = allowed_ids(official, _stock_list(["2330"]), prune=False)

        # Act
        first, second = promote(official, keep), promote(official, keep)

        # Assert
        pd.testing.assert_frame_equal(first, second)

    def test_missing_column_raises(self):
        # Arrange：少了 amount
        official = _official(["2330"]).drop(columns=["amount"])

        # Act / Assert
        with pytest.raises(ValueError, match="缺欄位"):
            promote(official, {"2330"})

    def test_output_is_sorted_by_date_then_stock(self):
        # Arrange
        official = _official(["2454", "1101"]).sample(frac=1.0, random_state=0)
        keep = allowed_ids(official, _stock_list(["2454", "1101"]), prune=False)

        # Act
        out = promote(official, keep)

        # Assert
        assert out.equals(out.sort_values(["date", "stock_id"]).reset_index(drop=True))


class TestAllowedIds:
    def test_keeps_delisted_ids_missing_from_stock_list(self):
        """已下市的個股不在 stock_list 裡，但仍有歷史，不可以被刪掉。

        刪掉的話 price 會縮短，而下游特徵檔是 upsert 寫入、舊列會殘留，
        兩邊對不上（CLAUDE.md 規則 5 踩過的那種不一致）。
        """
        # Arrange：1333 已下市、不在 stock_list
        official = _official(["2330", "1333", "0050"])

        # Act
        keep = allowed_ids(official, _stock_list(["2330"]), prune=False)

        # Assert
        assert {"2330", "1333", "TWII"} <= keep
        assert "0050" not in keep

    def test_prune_applies_stock_list_filter(self):
        # Arrange
        official = _official(["2330", "1333"])

        # Act
        keep = allowed_ids(official, _stock_list(["2330"]), prune=True)

        # Assert
        assert "1333" not in keep
        assert {"2330", "TWII"} <= keep

    def test_etf_detection(self):
        from engine.data_source.promote_price import is_etf
        assert is_etf("0050") and is_etf("0056")
        assert not is_etf("2330") and not is_etf("TWII") and not is_etf("1101")
