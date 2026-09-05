"""`build_labels_swing.py` 的邊界條件回歸測試（2026-09-05 新增）。

存在的理由：這支 label 用到未來資料（ZigZag 要反向走 5% 才確認轉折、效率比是
置中窗口、起/中/末要等整段結束），所以「還不知道」的情況比 build_labels.py 更多。
CLAUDE.md 明訂所有金融計算必須有 unit test 驗證邊界條件，而 doc/AUDIT_20260728.md
§A-1 記錄過同一類 bug（尾端未定被靜默寫成 0，下游 dropna 攔不到）在修過一次之後
還復發，直接原因就是沒有回歸測試釘住。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.models.build_labels_swing import (
    ER_FLAT, PHASE_EARLY, PHASE_LATE, STATE_DOWN, STATE_FLAT, STATE_UP,
    adjust_factor, efficiency_ratio, label_one_stock, zigzag,
)


# ── zigzag ────────────────────────────────────────────────────────────────────

def test_zigzag_finds_bottom_then_top_of_a_v_shape():
    # 先跌 20% 再漲 20%。價格是先跌的，所以第一個確認的轉折是「起點那個頂」，
    # 真正的底部要等後面漲超過 5% 才會被確認成 +1。
    close = np.concatenate([np.linspace(100, 80, 20), np.linspace(80, 100, 20)])
    idx, kind = zigzag(close)
    assert kind[0] == -1                     # 起點被確認成頂
    bottoms = idx[kind == 1]
    assert len(bottoms) == 1
    assert close[bottoms[0]] == pytest.approx(close.min(), rel=1e-6)


def test_zigzag_ignores_moves_smaller_than_threshold():
    # 全程振幅只有 2%，遠小於 5% 門檻 → 一個轉折都不該產生
    close = 100 + np.sin(np.linspace(0, 6 * np.pi, 200))
    idx, _ = zigzag(close)
    assert len(idx) == 0


def test_zigzag_last_leg_is_never_confirmed():
    """最後一段一定沒有轉折點——反向 5% 還沒發生，方向就還不知道。"""
    close = np.concatenate([np.linspace(100, 80, 30), np.linspace(80, 120, 30)])
    idx, _ = zigzag(close)
    # 最後一個轉折必定早於序列結尾（結尾那段仍在進行中）
    assert idx[-1] < len(close) - 1


# ── efficiency ratio ──────────────────────────────────────────────────────────

def test_efficiency_ratio_is_one_for_a_straight_line():
    er = efficiency_ratio(np.linspace(100, 200, 60))
    mid = er[~np.isnan(er)]
    assert mid.min() > 0.99                  # 單向直線，效率比應該是 1


def test_efficiency_ratio_is_near_zero_for_pure_oscillation():
    close = 100 + np.tile([0, 2], 60)        # 原地來回，淨移動 0
    er = efficiency_ratio(close)
    mid = er[~np.isnan(er)]
    assert mid.max() < ER_FLAT


def test_efficiency_ratio_leaves_tail_undefined():
    """置中窗口要看未來 10 天，資料尾端必須是 NaN 而不是硬算。"""
    er = efficiency_ratio(np.linspace(100, 200, 60))
    assert np.isnan(er[-1])
    assert np.isnan(er[-5])


# ── 還原權值 ──────────────────────────────────────────────────────────────────

def test_adjust_factor_scales_only_prices_before_the_event():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).to_numpy()
    events = pd.DataFrame({"date": pd.to_datetime(["2024-01-03"]), "ratio": [0.1]})
    f = adjust_factor(dates, events)
    assert f[0] == pytest.approx(0.1)        # 事件日之前要縮
    assert f[1] == pytest.approx(0.1)
    assert f[2] == pytest.approx(1.0)        # 事件日當天以後不動


def test_adjust_factor_compounds_multiple_events():
    dates = pd.to_datetime(["2024-01-01", "2024-06-01", "2025-01-01"]).to_numpy()
    events = pd.DataFrame({"date": pd.to_datetime(["2024-06-01", "2025-01-01"]),
                           "ratio": [0.5, 0.5]})
    f = adjust_factor(dates, events)
    assert f[0] == pytest.approx(0.25)       # 兩個事件連乘
    assert f[1] == pytest.approx(0.5)
    assert f[2] == pytest.approx(1.0)


def test_adjust_factor_is_identity_without_events():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"]).to_numpy()
    assert np.all(adjust_factor(dates, None) == 1.0)
    assert np.all(adjust_factor(dates, pd.DataFrame()) == 1.0)


def test_adjust_factor_ignores_bad_ratio():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"]).to_numpy()
    events = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "ratio": [0.0]})
    assert np.all(adjust_factor(dates, events) == 1.0)   # ratio<=0 是壞資料，不能乘


# ── label_one_stock：整合行為與「還不知道」 ────────────────────────────────────

def _zigzag_series(n_legs: int = 6, leg: int = 60, amp: float = 0.25) -> np.ndarray:
    """造一段乾淨的鋸齒走勢，每段漲跌 amp，確保超過 5% 門檻。"""
    out = [100.0]
    for k in range(n_legs):
        target = out[-1] * (1 + amp) if k % 2 == 0 else out[-1] * (1 - amp)
        out.extend(np.linspace(out[-1], target, leg)[1:])
    return np.asarray(out)


def test_short_history_produces_nothing():
    state, phase = label_one_stock(np.linspace(100, 120, 50))
    assert (state == "").all() and (phase == "").all()


def test_bad_price_only_breaks_its_own_segment_not_the_whole_stock():
    """壞值只切斷該處。舊版一票否決整檔，實測害 2,071 檔裡 799 檔整檔消失
    （376 檔只有 1~5 根壞值），而被砍掉的正是停牌／低流動性那些——跟結果高度
    相關，等於在 label 階段做了選擇偏誤。"""
    clean = _zigzag_series(n_legs=12, leg=60)     # 夠長，切成兩段後每段仍 >= MIN_BARS
    n_clean = (label_one_stock(clean)[0] != "").sum()
    assert n_clean > 0

    broken = clean.copy()
    broken[len(broken) // 2] = np.nan             # 正中間插一根壞值
    state, phase = label_one_stock(broken)
    assert (state != "").sum() > 0, "壞值不該讓整檔消失"
    assert state[len(broken) // 2] == "", "壞值那一根本身必須是未定"
    assert ((state == "") == (phase == "")).all()


def test_segment_shorter_than_min_bars_is_skipped():
    """壞值切出來的短段落不夠長就跳過，不能硬算。"""
    close = _zigzag_series(n_legs=12, leg=60)
    close[50] = np.nan                            # 前面只剩 50 根，遠小於 MIN_BARS
    state, _ = label_one_stock(close)
    assert (state[:50] == "").all()
    assert (state[51:] != "").any()               # 後面那段夠長，照常標


def test_labels_cover_up_and_down_and_all_three_phases():
    state, phase = label_one_stock(_zigzag_series())
    assert {STATE_UP, STATE_DOWN} <= set(state[state != ""])
    assert {PHASE_EARLY, PHASE_LATE} <= set(phase[phase != ""])


def test_state_and_phase_are_blank_together():
    """phase 標不出來時 state 也不能輸出，否則下游會拿到半套標籤。"""
    state, phase = label_one_stock(_zigzag_series())
    assert ((state == "") == (phase == "")).all()


def test_tail_is_undefined_not_guessed():
    """尾端方向未定，必須留空——這是 §A-1 那一類 bug 的防線。"""
    state, phase = label_one_stock(_zigzag_series())
    assert state[-1] == "" and phase[-1] == ""


def test_flat_stretch_is_labelled_flat():
    """波段中間插一段橫盤，那段要被效率比覆寫成 flat。"""
    up = np.linspace(100, 150, 80)
    flat = 150 + np.tile([0.0, 0.3], 60)          # 120 天原地震盪
    down = np.linspace(150, 100, 80)
    up2 = np.linspace(100, 160, 80)
    close = np.concatenate([up, flat, down, up2])
    assert len(close) >= 300                      # 低於 MIN_BARS 會整檔跳過
    state, _ = label_one_stock(close)
    assert (state == STATE_FLAT).any()


def test_phase_ordering_within_a_run_is_monotonic():
    """同一段連續區間內，起一定在中之前、中一定在末之前。"""
    state, phase = label_one_stock(_zigzag_series())
    order = {PHASE_EARLY: 0, "mid": 1, PHASE_LATE: 2}
    # 相鄰兩段狀態不同時中間沒有空白，所以切段要同時看 state 有沒有換
    runs, cur, cur_state = [], [], None
    for s, p in zip(state, phase):
        if p == "":
            if cur: runs.append(cur)
            cur, cur_state = [], None
            continue
        if s != cur_state:
            if cur: runs.append(cur)
            cur, cur_state = [], s
        cur.append(order[p])
    if cur: runs.append(cur)
    assert len(runs) >= 2
    for run in runs:
        assert run == sorted(run)
