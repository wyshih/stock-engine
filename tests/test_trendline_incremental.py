"""趨勢線特徵增量路徑的契約測試。

2026-08-30 之前這支沒有增量路徑，每次 `make update` 都把 2,069 檔 × 全部歷史
重算一遍（佔整個更新流程四分之一的時間）。加了增量之後，**最重要的不是快，
是算出來的值要跟全量一模一樣** —— 暖身視窗不夠的話會像籌碼特徵那次一樣，
安靜地吐出一堆 NaN。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.features import build_trendline_features as tl


def _fake_price(n_days: int = 400, n_stocks: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for i in range(n_stocks):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        rows.append(pd.DataFrame({
            "date": dates,
            "stock_id": f"{1000 + i}",
            "high": close * (1 + abs(rng.normal(0, 0.01, n_days))),
            "low": close * (1 - abs(rng.normal(0, 0.01, n_days))),
            "close": close,
            "volume": rng.integers(1000, 50000, n_days).astype(float),
        }))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def patched(monkeypatch, tmp_path):
    price = _fake_price()
    price.to_parquet(tmp_path / "price.parquet", index=False)
    monkeypatch.setattr(tl, "DATA_DIR", tmp_path)
    return price, tmp_path


def test_warmup_covers_the_longest_dependency():
    """暖身視窗必須大於最長的相依 —— 不夠就會安靜吐 NaN。"""
    longest = tl.LOOKBACK + tl.PIVOT_WINDOW + tl.CONFIRM_LAG
    assert tl.WARMUP_BARS > longest, (
        f"WARMUP_BARS={tl.WARMUP_BARS} 不足以覆蓋 {longest} 根的相依")


def test_incremental_matches_full_recompute(patched):
    """增量算出來的值必須與全量逐列相同 —— 這是整條增量路徑的意義所在。"""
    price, tmp_path = patched
    full = tl.build(full=True, jobs=1).sort_values(["stock_id", "date"]).reset_index(drop=True)

    # 假裝只做到倒數第 5 天，把後面的砍掉當成「已存在的舊檔」
    cutoff = sorted(price["date"].unique())[-6]
    full[full["date"] <= cutoff].to_parquet(
        tmp_path / "trendline_features.parquet", index=False)

    incr = tl.build(jobs=1)
    assert not incr.empty, "增量應該要算出新日期"
    assert (incr["date"] > cutoff).all(), "增量不該回頭寫舊日期"

    expected = full[full["date"] > cutoff].reset_index(drop=True)
    got = incr.sort_values(["stock_id", "date"]).reset_index(drop=True)
    assert len(got) == len(expected)
    num = [c for c in expected.columns if c not in ("date", "stock_id")]
    pd.testing.assert_frame_equal(
        got[["date", "stock_id"]], expected[["date", "stock_id"]])
    for c in num:
        pd.testing.assert_series_equal(got[c], expected[c], check_names=False,
                                       rtol=1e-9, atol=1e-9)


def test_incremental_does_not_add_nan_the_full_run_would_not(patched):
    """暖身不足的症狀是整欄 NaN（籌碼特徵那次就是這樣壞的）。

    基準必須是**全量在同一批列上**的結果，不是歷史區段 —— 短短幾天內某些
    事件型特徵（例如 tl_break_vol）本來就可能全部是 NaN，那不是 bug。
    """
    price, tmp_path = patched
    full = tl.build(full=True, jobs=1)
    cutoff = sorted(price["date"].unique())[-6]
    full[full["date"] <= cutoff].to_parquet(
        tmp_path / "trendline_features.parquet", index=False)

    incr = tl.build(jobs=1)
    same_rows = full[full["date"] > cutoff]
    body, base = incr.drop(columns=["date", "stock_id"]), same_rows.drop(columns=["date", "stock_id"])
    extra = {c for c in body.columns
             if body[c].isna().all() and not base[c].isna().all()}
    assert not extra, f"這些欄只在增量時變成全 NaN：{extra}"

    worse = {c for c in body.columns if body[c].isna().sum() > base[c].isna().sum()}
    assert not worse, f"這些欄的缺失比全量還多：{worse}"


def test_no_new_dates_returns_empty(patched):
    price, tmp_path = patched
    tl.build(full=True, jobs=1).to_parquet(
        tmp_path / "trendline_features.parquet", index=False)
    assert tl.build(jobs=1).empty


def test_only_resistance_break_resets_the_counter():
    """`tl_days_since_break` 只在壓力線突破時歸零，支撐跌破不算。

    主迴圈裡 `last_break_i = i` 寫在 resist 分支內。增量的 reseed 初版把
    support 也算進去，2026-08-30 用真實資料逐列比對才抓到（174 列不符，
    49 檔）—— 合成資料因為突破夠頻繁，剛好都落在暖身視窗裡，測不出來。
    """
    n = 6
    out = {
        "tl_resist_break": np.zeros(n, "int8"),
        "tl_support_break": np.zeros(n, "int8"),
        "tl_days_since_break": np.full(n, np.nan, "float32"),
    }
    out["tl_support_break"][3] = 1          # 只有支撐跌破
    out["tl_resist_break"][5] = 1           # 壓力突破
    dates = pd.bdate_range("2026-01-01", periods=n).to_numpy()

    tl._reseed_days_since_break(out, dates, pd.Timestamp(dates[1]), gap=2)

    got = out["tl_days_since_break"]
    assert got[1] == pytest.approx(np.log1p(2)), "錨點的值要等於 log1p(gap)"
    assert got[3] == pytest.approx(np.log1p(4)), "支撐跌破不可以重置計數"
    assert got[4] == pytest.approx(np.log1p(5))
    assert got[5] == pytest.approx(0.0), "壓力突破當天要歸零"


def test_reseed_skips_stocks_without_data_on_the_anchor_day():
    """停牌的股票在錨點那天沒有資料 —— 不可以硬套，維持原值。"""
    n = 4
    out = {
        "tl_resist_break": np.zeros(n, "int8"),
        "tl_support_break": np.zeros(n, "int8"),
        "tl_days_since_break": np.full(n, 1.5, "float32"),
    }
    dates = pd.bdate_range("2026-01-01", periods=n).to_numpy()
    tl._reseed_days_since_break(out, dates, pd.Timestamp("2026-06-01"), gap=3)
    assert (out["tl_days_since_break"] == np.float32(1.5)).all()
