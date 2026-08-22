"""
`validate_data.py` 的 OHLC 不變式回歸測試（2026-08-04 新增）。

存在的理由：`data/price.parquet` 曾發現 7,009 列（0.205%）違反
`low <= min(open, close) <= max(open, close) <= high` 的不變式，集中在
成交量稀薄的上櫃股票（見 舊 repo code/data_collection/repair_ohlc.py 的說明、
doc/BACKTEST_LOG.md 2026-08-04）。追查根因發現這是 Yahoo Finance 自家
原始資料本身的瑕疵（即使 auto_adjust=False 也重現），不是我們抓取／
合併邏輯造成的。既然抓取端無法從根本修正，就必須在 validate_data.py
當場攔截，不能等三年後才在 KD／CMF 值裡發現。

pandas 陷阱：混用 `>` / `<` 比較浮點數時要留一點 epsilon，避免真正相等
（例如 open == high）被誤判為違反。
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from engine.data_source import validate_data


TARGET_DATE = date(2024, 1, 2)


def _make_day(rows: list[dict], n_pad: int = 60) -> pd.DataFrame:
    """
    造一天份的價格資料。除了 rows 指定的列之外，補上 n_pad 筆正常列，
    確保通過 `total < 50` 的假日偵測門檻，只單獨測試 OHLC 不變式那條路徑。
    """
    base = []
    for i in range(n_pad):
        base.append({
            "date": pd.Timestamp(TARGET_DATE), "stock_id": f"PAD{i:04d}",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
            "volume": 1000.0, "amount": 100500.0,
        })
    for r in rows:
        row = {"date": pd.Timestamp(TARGET_DATE), "volume": 1000.0, "amount": 0.0}
        row.update(r)
        base.append(row)
    return pd.DataFrame(base)


class TestOhlcInvariant:
    """low <= min(open, close) 且 high >= max(open, close) 且 high >= low。"""

    def test_valid_day_passes(self, monkeypatch):
        """全部列都合法時，驗證應該通過。"""
        # Arrange
        day_df = _make_day([
            {"stock_id": "1101", "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2},
        ])
        monkeypatch.setattr(validate_data, "read_parquet", lambda name: day_df)

        # Act
        ok = validate_data.validate_price(TARGET_DATE)

        # Assert
        assert ok is True

    def test_close_above_high_fails(self, monkeypatch):
        """典型病灶：open=high=low 鎖死，close 卻高出區間（如 3067 在 2020-08-28 的真實案例）。"""
        # Arrange
        day_df = _make_day([
            {"stock_id": "3067", "open": 37.0797, "high": 37.0797, "low": 37.0797, "close": 40.7804},
        ])
        monkeypatch.setattr(validate_data, "read_parquet", lambda name: day_df)

        # Act
        ok = validate_data.validate_price(TARGET_DATE)

        # Assert
        assert ok is False, "close 落在 high 之外必須被攔下，不能悄悄過關"

    def test_open_below_low_fails(self, monkeypatch):
        # Arrange
        day_df = _make_day([
            {"stock_id": "2330", "open": 8.0, "high": 10.0, "low": 9.0, "close": 9.5},
        ])
        monkeypatch.setattr(validate_data, "read_parquet", lambda name: day_df)

        # Act
        ok = validate_data.validate_price(TARGET_DATE)

        # Assert
        assert ok is False, "open 落在 low 之外必須被攔下"

    def test_high_less_than_low_fails(self, monkeypatch):
        # Arrange
        day_df = _make_day([
            {"stock_id": "2454", "open": 9.5, "high": 9.0, "low": 10.0, "close": 9.5},
        ])
        monkeypatch.setattr(validate_data, "read_parquet", lambda name: day_df)

        # Act
        ok = validate_data.validate_price(TARGET_DATE)

        # Assert
        assert ok is False, "high < low 代表區間本身壞了，必須被攔下"

    def test_open_equals_high_is_not_a_false_positive(self, monkeypatch):
        """open 剛好貼在 high 上（無漲跌）是合法情況，不該被 epsilon 誤判。"""
        # Arrange
        day_df = _make_day([
            {"stock_id": "2603", "open": 10.0, "high": 10.0, "low": 9.7, "close": 9.9},
        ])
        monkeypatch.setattr(validate_data, "read_parquet", lambda name: day_df)

        # Act
        ok = validate_data.validate_price(TARGET_DATE)

        # Assert
        assert ok is True, "open == high 是合法邊界，不應誤判為違反"

    def test_multiple_violations_all_reported(self, monkeypatch, caplog):
        """多檔股票同時違反時，日誌應該列出全部受影響的 stock_id，不能只報第一筆。"""
        # Arrange
        day_df = _make_day([
            {"stock_id": "3067", "open": 37.0, "high": 37.0, "low": 37.0, "close": 40.7},
            {"stock_id": "9999", "open": 5.0, "high": 6.0, "low": 5.5, "close": 4.0},
        ])
        monkeypatch.setattr(validate_data, "read_parquet", lambda name: day_df)

        # Act
        with caplog.at_level("ERROR"):
            ok = validate_data.validate_price(TARGET_DATE)

        # Assert
        assert ok is False
        assert "2" in caplog.text  # n_violation == 2 出現在錯誤訊息裡


def _dispatch(price_df: pd.DataFrame, chip_df: pd.DataFrame):
    """模擬 read_parquet：依名稱回傳不同的表，用來測 price/chip 交叉檢查。"""
    def _read(name: str) -> pd.DataFrame:
        return {"price": price_df, "chip": chip_df}.get(name, pd.DataFrame())
    return _read


def _chip_day(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame([
        {"date": pd.Timestamp(TARGET_DATE), "stock_id": f"C{i:04d}"}
        for i in range(n_rows)
    ])


class TestOhlcMissingRate:
    """
    2026-07-31 的教訓：全市場 1,943 列裡 OHLC 幾乎全是 NaN，卻通過了所有既有檢查。
    根因是 pandas 的 `NaN > NaN` 回傳 False，不變式那組布林比較整片變 False，
    列數檢查又只看 len()——NaN 列照樣算一列。缺值率必須單獨查。
    """

    def test_high_missing_rate_fails(self, monkeypatch):
        # Arrange：60 筆正常 + 20 筆 OHLC 全 NaN → 缺值率 25% > 5%
        nan_rows = [
            {"stock_id": f"NAN{i:04d}", "open": float("nan"), "high": float("nan"),
             "low": float("nan"), "close": float("nan")}
            for i in range(20)
        ]
        price_df = _make_day(nan_rows)
        monkeypatch.setattr(
            validate_data, "read_parquet", _dispatch(price_df, _chip_day(len(price_df)))
        )

        # Act
        ok = validate_data.validate_price(TARGET_DATE)

        # Assert
        assert ok is False, "OHLC 大量缺值必須被攔下，不能因為 NaN 比較恆為 False 就過關"

    def test_low_missing_rate_passes(self, monkeypatch):
        # Arrange：60 筆正常 + 2 筆 NaN → 缺值率 3.2% < 5%
        nan_rows = [
            {"stock_id": f"NAN{i:04d}", "open": float("nan"), "high": float("nan"),
             "low": float("nan"), "close": float("nan")}
            for i in range(2)
        ]
        price_df = _make_day(nan_rows)
        monkeypatch.setattr(
            validate_data, "read_parquet", _dispatch(price_df, _chip_day(len(price_df)))
        )

        # Act
        ok = validate_data.validate_price(TARGET_DATE)

        # Assert
        assert ok is True, "零星缺值屬正常（個股停牌），不該誤判為整日失敗"


class TestPriceChipRowRatio:
    """
    2025-08-01 的教訓：price 只有 15 列、chip 有 1,953 列。price 與 chip 各自
    獨立回報，沒有人比對兩者，矛盾就這樣溜過去。抓取覆蓋率必須交叉檢查。
    """

    def test_price_far_fewer_rows_than_chip_fails(self, monkeypatch):
        # Arrange：price 60 列、chip 1000 列 → 比值 0.06 < 0.5
        price_df = _make_day([])
        monkeypatch.setattr(
            validate_data, "read_parquet", _dispatch(price_df, _chip_day(1000))
        )

        # Act
        ok = validate_data.validate_price(TARGET_DATE)

        # Assert
        assert ok is False, "price 覆蓋率遠低於 chip 代表整批抓取失敗，必須被攔下"

    def test_comparable_row_counts_pass(self, monkeypatch):
        # Arrange：price 60 列、chip 62 列 → 比值 0.97
        price_df = _make_day([])
        monkeypatch.setattr(
            validate_data, "read_parquet", _dispatch(price_df, _chip_day(62))
        )

        # Act
        ok = validate_data.validate_price(TARGET_DATE)

        # Assert
        assert ok is True

    def test_missing_chip_day_does_not_fail_price(self, monkeypatch):
        """chip 當天沒資料時無從比對，交叉檢查應跳過而不是誤判 price 失敗。"""
        # Arrange
        price_df = _make_day([])
        monkeypatch.setattr(
            validate_data, "read_parquet", _dispatch(price_df, pd.DataFrame())
        )

        # Act
        ok = validate_data.validate_price(TARGET_DATE)

        # Assert
        assert ok is True


GAP_DATE = date(2025, 8, 1)


def _day_at(target: date, n_rows: int, rows: list[dict] | None = None) -> pd.DataFrame:
    """造指定日期的價格資料（n_rows 筆正常列 + 可選的自訂列）。"""
    base = [
        {
            "date": pd.Timestamp(target), "stock_id": f"PAD{i:04d}",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
            "volume": 1000.0, "amount": 100500.0,
        }
        for i in range(n_rows)
    ]
    for r in rows or []:
        row = {"date": pd.Timestamp(target), "volume": 1000.0, "amount": 0.0}
        row.update(r)
        base.append(row)
    return pd.DataFrame(base)


def _chip_day_at(target: date, n_rows: int) -> pd.DataFrame:
    return pd.DataFrame([
        {"date": pd.Timestamp(target), "stock_id": f"C{i:04d}"}
        for i in range(n_rows)
    ])


class TestKnownUpstreamGaps:
    """
    2026-08-07：2019-09-09 / 2021-04-06 / 2025-08-01 這三天經查證是 Yahoo Finance
    上游本身就沒有個股報價（單檔整段 request 與窄區間重抓都重現，全市場重抓後
    列數毫無變化），而 chip 有 1,681 / 1,749 / 1,953 列 —— 它們是真交易日。
    決策是接受缺口不補值，所以這三天必須降級成 WARNING，否則每次驗證都固定 FAIL。

    但降級只能針對「已查證」的日期，規則本身不能關掉——2025-08-01 當初正是被
    price/chip 比值這條規則抓出來的，關掉就再也攔不到下一個新缺口。
    """

    def test_known_gap_date_degrades_to_warning(self, monkeypatch, caplog):
        # Arrange：重現 2025-08-01 的實況（price 15 列 vs chip 1,953 列）
        price_df = _day_at(GAP_DATE, 15)
        monkeypatch.setattr(
            validate_data, "read_parquet",
            _dispatch(price_df, _chip_day_at(GAP_DATE, 1953)),
        )

        # Act
        with caplog.at_level(logging.WARNING):
            ok = validate_data.validate_price(GAP_DATE)

        # Assert
        assert ok is True, "已知上游缺口不該讓每日驗證固定 FAIL"
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR], \
            "已知缺口應降級成 WARNING，不該再記 ERROR"
        assert any(validate_data.KNOWN_GAP_NOTE in r.getMessage()
                   for r in caplog.records), "WARNING 訊息要寫明原因與出處"

    def test_non_gap_date_still_fails(self, monkeypatch, caplog):
        """同樣的 price/chip 覆蓋率落差，發生在清單外的日期就必須 FAIL。"""
        # Arrange：price 60 列 vs chip 1000 列，日期不在例外清單
        assert TARGET_DATE not in validate_data.KNOWN_UPSTREAM_GAPS
        price_df = _day_at(TARGET_DATE, 60)
        monkeypatch.setattr(
            validate_data, "read_parquet",
            _dispatch(price_df, _chip_day_at(TARGET_DATE, 1000)),
        )

        # Act
        with caplog.at_level(logging.WARNING):
            ok = validate_data.validate_price(TARGET_DATE)

        # Assert
        assert ok is False, "例外清單不能把整條規則關掉，新缺口仍要攔下"
        assert not any(validate_data.KNOWN_GAP_NOTE in r.getMessage()
                       for r in caplog.records)

    def test_known_gap_date_still_fails_on_bad_ohlc(self, monkeypatch):
        """降級只針對覆蓋率，資料正確性（OHLC 不變式）在例外日期照樣要 FAIL。"""
        # Arrange
        price_df = _day_at(GAP_DATE, 14, rows=[
            {"stock_id": "9999", "open": 10.0, "high": 10.5, "low": 9.8, "close": 11.9},
        ])
        monkeypatch.setattr(
            validate_data, "read_parquet",
            _dispatch(price_df, _chip_day_at(GAP_DATE, 1953)),
        )

        # Act
        ok = validate_data.validate_price(GAP_DATE)

        # Assert
        assert ok is False, "close 落在 [low, high] 之外是資料錯誤，不在缺口豁免範圍內"
