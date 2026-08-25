"""無法用公司行動解釋的跨日跳空 —— 偵測與遮蔽。

## 背景

上櫃的分割與減資沒有任何官方歷史來源（2026-08-25 查證：TWSE 的 exright 三個端點
只涵蓋上市；price_official 的 ex_flag 補得到上櫃除權息但不含分割減資；TPEx
OpenAPI 的 /tpex_exright_daily 實測只回當日快照 2 天）。所以只能偵測不能查表。

## 這支釘住的四個判準

一筆跳空要被列為「無法解釋」，四個條件必須同時成立 —— 少一個就會誤判：

1. |報酬| 超過門檻
2. 相鄰交易日（跨停牌的價格落差不是當日報酬）
3. 上市已滿 5 個交易日（新股前五日無漲跌幅限制，實測有 +46% 的真實報酬）
4. exright 與 ex_flag 兩份白名單都沒有事件

⚠️ 稽核者最初回報「208 筆、73 筆穿過濾網」是高估的 —— 沒有扣掉 ex_flag、
新上市、跨停牌三類。實際只有 8 筆。5314 在 2026-08-14 的 −73.6% 也被誤判為
無法解釋，實際上 ex_flag 標記了「除權」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.data_source.suspect_jumps import find_unexplained

EMPTY_EX = pd.DataFrame(columns=["date", "stock_id"])
EMPTY_OFFICIAL = pd.DataFrame(columns=["date", "stock_id", "ex_flag"])


def _price(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": pd.Timestamp(d), "stock_id": s, "close": c} for d, s, c in rows])


def _series(stock: str, closes: list[float], start: str = "2024-01-02") -> pd.DataFrame:
    days = pd.bdate_range(start, periods=len(closes))
    return _price([(str(d.date()), stock, c) for d, c in zip(days, closes)])


class TestFindUnexplained:
    def test_flags_unexplained_adjacent_day_crash(self):
        """判準的正例：相鄰日腰斬、非新股、無事件。"""
        # Arrange：前 8 天平穩讓它「上市已久」，第 9 天腰斬
        px = _series("3293", [100] * 8 + [50])

        # Act
        found = find_unexplained(px, EMPTY_EX, EMPTY_OFFICIAL)

        # Assert
        assert len(found) == 1
        assert found.iloc[0]["stock_id"] == "3293"
        assert found.iloc[0]["ret"] == -0.5

    def test_exright_event_explains_it(self):
        # Arrange
        px = _series("1101", [100] * 8 + [50])
        ex = pd.DataFrame([{"date": px["date"].iloc[-1], "stock_id": "1101"}])

        # Act / Assert
        assert find_unexplained(px, ex, EMPTY_OFFICIAL).empty

    def test_ex_flag_explains_it(self):
        """上櫃的除權息只有 ex_flag 標記得到 —— 這是 5314 那個誤判的來源。"""
        # Arrange
        px = _series("5314", [100] * 8 + [26])
        official = pd.DataFrame([{"date": px["date"].iloc[-1],
                                  "stock_id": "5314", "ex_flag": "除權"}])

        # Act / Assert
        assert find_unexplained(px, EMPTY_EX, official).empty

    def test_ignores_new_listing(self):
        """新股前五日無漲跌幅限制，+46% 是真實報酬不是錯誤。"""
        # Arrange：第 2 個交易日就大漲
        px = _series("6921", [100, 146, 150, 150])

        # Act / Assert
        assert find_unexplained(px, EMPTY_EX, EMPTY_OFFICIAL).empty

    def test_ignores_non_adjacent_days(self):
        """跨停牌的價格落差不是當日報酬。

        ⚠️ 交易日曆是從 price 自己的日期推出來的，所以測試資料必須有**另一檔**
        持續交易的股票來撐出日曆 —— 只放停牌那一檔的話，它復牌那天在日曆上
        就成了「下一個交易日」，gap 會是 1（2026-08-25 寫測試時踩到）。
        """
        # Arrange：2330 天天交易撐出日曆；5314 停牌一個月後復牌腰斬
        calendar = pd.bdate_range("2024-01-02", periods=35)
        other = _price([(str(d.date()), "2330", 100.0) for d in calendar])
        halted = _price([(str(d.date()), "5314", 100.0) for d in calendar[:9]]
                        + [(str(calendar[-1].date()), "5314", 50.0)])
        px = pd.concat([other, halted], ignore_index=True)

        # Act / Assert
        assert find_unexplained(px, EMPTY_EX, EMPTY_OFFICIAL).empty

    def test_small_moves_are_not_flagged(self):
        # Arrange
        px = _series("2330", [100] * 8 + [105])

        # Act / Assert
        assert find_unexplained(px, EMPTY_EX, EMPTY_OFFICIAL).empty


class TestMasking:
    def test_masked_returns_become_nan(self, monkeypatch):
        """被列入清單的那天，報酬類特徵要變 NaN 而不是留假值。"""
        # Arrange
        from engine.features import build_price_features as bpf
        day = pd.Timestamp("2024-07-26")
        monkeypatch.setattr(bpf, "_SUSPECT_JUMPS", {(day, "3293")})
        out = {
            "date": pd.Series([pd.Timestamp("2024-07-25"), day]),
            "stock_id": pd.Series(["3293", "3293"]),
            "return_1d": pd.Series([0.01, -0.46]),
            "rs_5d": pd.Series([0.02, -0.40]),
            "ma5": pd.Series([100.0, 90.0]),      # 非報酬類，不該被動到
        }

        # Act
        bpf._mask_suspect_jumps(out)

        # Assert
        assert np.isnan(out["return_1d"].iloc[1]) and out["return_1d"].iloc[0] == 0.01
        assert np.isnan(out["rs_5d"].iloc[1])
        assert out["ma5"].iloc[1] == 90.0, "均線不是報酬，不該被遮蔽"

    def test_no_list_means_no_op(self, monkeypatch):
        """清單不存在時什麼都不做 —— 這是選用防護。"""
        # Arrange
        from engine.features import build_price_features as bpf
        monkeypatch.setattr(bpf, "_SUSPECT_JUMPS", set())
        out = {"date": pd.Series([pd.Timestamp("2024-07-26")]),
               "stock_id": pd.Series(["3293"]),
               "return_1d": pd.Series([-0.46])}

        # Act
        bpf._mask_suspect_jumps(out)

        # Assert
        assert out["return_1d"].iloc[0] == -0.46
