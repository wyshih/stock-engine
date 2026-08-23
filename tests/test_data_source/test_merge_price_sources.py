"""合併分開抓的行情檔。

釘住的重點：缺來源要出聲但不中止、鍵重複要去重、date 統一成 ns、
以及「全部缺」時要明確中止而不是寫出一個空檔。
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.data_source import merge_price_sources as mps

COLUMNS = ["date", "stock_id", "open", "high", "low", "close", "volume", "amount"]


def _frame(stock_ids: list[str], day: str = "2026-08-21") -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime([day] * len(stock_ids)),
        "stock_id": stock_ids,
        "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5,
        "volume": 1000.0, "amount": 10500.0,
    })[COLUMNS]


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mps, "DATA_DIR", tmp_path)
    return tmp_path


class TestMerge:
    def test_merges_all_sources(self, data_dir):
        # Arrange
        _frame(["2330", "2317"]).to_parquet(data_dir / "price_official_twse.parquet")
        _frame(["6510"]).to_parquet(data_dir / "price_official_tpex.parquet")
        _frame(["TWII"]).to_parquet(data_dir / "price_official_index.parquet")

        # Act
        out = mps.merge()

        # Assert
        assert len(out) == 4
        assert set(out["stock_id"]) == {"2330", "2317", "6510", "TWII"}

    def test_missing_source_warns_but_continues(self, data_dir, caplog):
        """只有上市也要能合 —— 但必須留下警告，否則少一個來源沒人會發現。"""
        # Arrange
        _frame(["2330"]).to_parquet(data_dir / "price_official_twse.parquet")

        # Act
        out = mps.merge()

        # Assert
        assert len(out) == 1
        assert "缺少來源" in caplog.text

    def test_all_missing_aborts(self, data_dir):
        # Act / Assert
        with pytest.raises(SystemExit):
            mps.merge()

    def test_duplicate_keys_are_deduped(self, data_dir):
        """來源理論上不重疊；真的重疊時要去重而不是留兩列同鍵。"""
        # Arrange
        _frame(["2330"]).to_parquet(data_dir / "price_official_twse.parquet")
        _frame(["2330"]).to_parquet(data_dir / "price_official_tpex.parquet")

        # Act
        out = mps.merge()

        # Assert
        assert len(out) == 1

    def test_date_is_nanosecond_dtype(self, data_dir):
        # Arrange
        frame = _frame(["2330"])
        frame["date"] = frame["date"].astype("datetime64[us]")
        frame.to_parquet(data_dir / "price_official_twse.parquet")

        # Act / Assert
        assert mps.merge()["date"].dtype == "datetime64[ns]"

    def test_output_is_sorted_by_date_then_stock(self, data_dir):
        # Arrange
        _frame(["2330"], "2026-08-21").to_parquet(data_dir / "price_official_twse.parquet")
        _frame(["1101"], "2026-08-20").to_parquet(data_dir / "price_official_tpex.parquet")

        # Act
        out = mps.merge()

        # Assert
        assert out["stock_id"].tolist() == ["1101", "2330"]
