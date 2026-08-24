"""
`fetch_revenue.py` 批次模式的 URL 契約 + 月營收覆蓋率回歸測試（2026-08-24 新增）。

存在的理由：MOPS 整月營收彙總頁 t21sc03 依「公司別」拆成兩個檔案——
  _0 = 國內公司、_1 = 國外公司（第一上市/上櫃的 -KY 與 -DR 存託憑證）。
批次模式原本把結尾寫死成 `_0`，導致 121 檔仍在市交易的外國公司
（120 檔 -KY + 9105 泰金寶-DR）在 revenue.parquet 裡一列都沒有，
2025 年起這批股票的 revenue_yoy 是 100% NaN，等於整個 -KY 族群
在月營收類特徵上被靜默排除。

這裡釘兩件事：
1. URL 組成必須把 suffix 當參數，而且 run_bulk 掃過的公司別要涵蓋 _0 與 _1。
2. 端到端的資料契約：現仍在市交易的個股，月營收覆蓋率不得掉回修復前的水準。
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.data_source import fetch_revenue
from engine.paths import DATA_DIR


# ── URL 契約（純單元測試，不連外）─────────────────────────────────────────────

def test_bulk_url_includes_company_type_suffix():
    # Arrange / Act
    domestic = fetch_revenue.bulk_url("sii", 114, 7, "0")
    foreign = fetch_revenue.bulk_url("sii", 114, 7, "1")

    # Assert
    assert domestic == (
        "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_114_7_0.html"
    )
    assert foreign == (
        "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_114_7_1.html"
    )


def test_bulk_suffixes_cover_domestic_and_foreign():
    """_1（外國公司）漏抓正是本次事故的根因，不能再退回只掃 _0。"""
    assert "0" in fetch_revenue.MOPS_BULK_SUFFIXES
    assert "1" in fetch_revenue.MOPS_BULK_SUFFIXES


def test_run_bulk_requests_every_market_and_suffix(monkeypatch):
    """一個月份應該打 (sii, otc) × (_0, _1) 共 4 次請求。"""
    # Arrange
    calls: list[tuple] = []

    def _fake_fetch(market, roc_year, month, suffix="0"):
        calls.append((market, roc_year, month, suffix))
        return pd.DataFrame(columns=["stock_id", "revenue"])

    monkeypatch.setattr(fetch_revenue, "fetch_bulk_month", _fake_fetch)
    monkeypatch.setattr(fetch_revenue.time, "sleep", lambda *_: None)

    # Act
    fetch_revenue.run_bulk("2025-07", "2025-07", dry_run=True)

    # Assert
    assert sorted(calls) == sorted([
        ("sii", 114, 7, "0"), ("sii", 114, 7, "1"),
        ("otc", 114, 7, "0"), ("otc", 114, 7, "1"),
    ])


def test_fetch_bulk_month_treats_404_as_empty(monkeypatch):
    """缺檔的月份要當成空結果，不能觸發 retry 空轉或炸掉整批。"""
    # Arrange
    class _Resp:
        status_code = 404

        def raise_for_status(self):  # pragma: no cover - 不該被呼叫
            raise AssertionError("404 不該進到 raise_for_status")

    monkeypatch.setattr(fetch_revenue.requests, "get", lambda *a, **k: _Resp())

    # Act
    df = fetch_revenue.fetch_bulk_month("sii", 108, 1, "1")

    # Assert
    assert df.empty
    assert list(df.columns) == ["stock_id", "revenue"]


# ── 資料契約：在市個股的月營收覆蓋率 ───────────────────────────────────────────

# 覆蓋率門檻。修復前為 93.9%（121 檔外國公司完全空白），修復後為 99.9%。
# 取 99% 是「還能容忍幾檔剛上市/剛暫停交易的個股」但擋得住整個族群消失。
MIN_REVENUE_COVERAGE = 0.99
# 判定「仍在市交易」用的視窗：最後一個交易日往前 7 天內有價量即算在市。
RECENT_TRADING_DAYS = 7


@pytest.mark.skipif(
    not all((DATA_DIR / f"{n}.parquet").exists()
            for n in ("revenue", "price", "stock_list")),
    reason="需要 data/revenue.parquet、price.parquet 與 stock_list.parquet",
)
def test_active_stocks_have_revenue_coverage():
    # Arrange
    price = pd.read_parquet(DATA_DIR / "price.parquet", columns=["date", "stock_id"])
    revenue = pd.read_parquet(DATA_DIR / "revenue.parquet", columns=["stock_id"])
    # price.parquet 也放了 TWII 之類的指數，只看 stock_list 認得的個股。
    listed = set(pd.read_parquet(DATA_DIR / "stock_list.parquet",
                                 columns=["stock_id"])["stock_id"])
    last_day = price["date"].max()
    traded = set(
        price.loc[
            price["date"] >= last_day - pd.Timedelta(days=RECENT_TRADING_DAYS),
            "stock_id",
        ]
    ) & listed

    # Act
    covered = traded & set(revenue["stock_id"])
    coverage = len(covered) / len(traded)

    # Assert
    missing = sorted(traded - covered)
    assert coverage >= MIN_REVENUE_COVERAGE, (
        f"在市個股月營收覆蓋率 {coverage:.2%} 低於門檻 "
        f"{MIN_REVENUE_COVERAGE:.0%}，缺 {len(missing)} 檔：{missing[:20]}"
    )
