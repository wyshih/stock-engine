"""型態規則總表：趨勢線／均線／指標，並組裝其他兩組規則。

設計上的關鍵：每條規則只寫一次判定式 `predicate(df) -> bool Series`。
    - 敘述：把**這一檔股票這一天**的單列 DataFrame 丟進去，True 就顯示敘述。
    - 統計：把**全市場所有歷史列**丟進去，得到樣本再算後續報酬。
兩邊共用同一個函式，畫面上的話術與旁邊的勝率數字才保證是同一個條件。
分開寫兩份的話，判定式一旦微調就會出現「敘述說突破、統計算的卻是別的東西」。

欄位語意來源：`features_v2/price_trend.py`、`price_flow.py`、`price_oscillator.py`、
`build_trendline.py`、`build_swing.py` 的檔頭說明。
- `bb_width_rank`：布林帶寬在過去窗口的百分位（0~1），越小越壓縮。
- `ma_squeeze`：短中長三條均線的 max/min − 1，越小越糾結。
- 旗標欄（`tl_*`、`macd_top_div`…）是可為空的 Int8，NA 代表「算不出來」，
  一律當成 False，不能當成 0（見 `build_trendline.py` 檔頭）。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from engine.app.frontend.pattern_base import (
    BEARISH,
    BULLISH,
    NEUTRAL,
    Pattern,
    flag,
    fmt_int,
    num,
)
from engine.app.frontend.patterns_chip import CHIP_AND_RELATIVE
from engine.app.frontend.patterns_talib import TALIB_PATTERNS

TREND_GROUP = "技術型態"

# 帶量的定義：突破當日量能達 20 日均量的倍數
VOLUME_CONFIRM_MULT = 1.5
# 布林壓縮的百分位門檻
SQUEEZE_RANK = 0.15
# 均線糾結：三線 max/min − 1 的上限
MA_SQUEEZE_MAX = 0.02
# 通道位置的上下緣
CHANNEL_HIGH, CHANNEL_LOW = 0.8, 0.2
# 假突破次數門檻
FALSE_BREAK_MIN = 2


TREND_PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        key="triangle_dry",
        name="三角收斂 + 量縮",
        tone=NEUTRAL,
        columns=("tl_is_triangle", "tl_triangle_vol_dry", "tl_apex_bars"),
        predicate=lambda df: flag(df, "tl_is_triangle") & flag(df, "tl_triangle_vol_dry"),
        text="壓力線下彎、支撐線上揚，兩線正在收斂，而且量能同步萎縮 —— "
             "教科書上的「三角收斂末端」，通常在頂點前後選邊表態。",
        group=TREND_GROUP,
        detail=lambda row: fmt_int(row, "tl_apex_bars", "兩線約在 {} 根 K 之後交會"),
    ),
    Pattern(
        key="triangle",
        name="三角收斂",
        tone=NEUTRAL,
        columns=("tl_is_triangle", "tl_apex_bars"),
        predicate=lambda df: flag(df, "tl_is_triangle"),
        text="上下軌正在收斂，區間越走越窄，方向還沒表態。",
        group=TREND_GROUP,
        detail=lambda row: fmt_int(row, "tl_apex_bars", "兩線約在 {} 根 K 之後交會"),
    ),
    Pattern(
        key="resist_break_vol",
        name="帶量突破壓力線",
        tone=BULLISH,
        columns=("tl_resist_break", "tl_break_vol"),
        predicate=lambda df: (flag(df, "tl_resist_break")
                              & num(df, "tl_break_vol").ge(VOLUME_CONFIRM_MULT)),
        text="今天站上壓力線，而且量能放大到 20 日均量的 1.5 倍以上 —— 帶量突破。",
        group=TREND_GROUP,
        detail=lambda row: (None if pd.isna(row.get("tl_break_vol"))
                            else f"今日量能是均量的 {float(row['tl_break_vol']):.1f} 倍"),
    ),
    Pattern(
        key="resist_break_novol",
        name="無量突破壓力線",
        tone=NEUTRAL,
        columns=("tl_resist_break", "tl_break_vol"),
        predicate=lambda df: (flag(df, "tl_resist_break")
                              & ~num(df, "tl_break_vol").ge(VOLUME_CONFIRM_MULT)),
        text="站上壓力線，但量沒跟上 —— 這種突破常見的下場是隔幾天就跌回線內。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="support_break",
        name="跌破支撐線",
        tone=BEARISH,
        columns=("tl_support_break",),
        predicate=lambda df: flag(df, "tl_support_break"),
        text="收盤跌破下軌支撐線，原本的上升結構被破壞。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="false_breaks",
        name="假突破頻繁",
        tone=BEARISH,
        columns=("tl_false_breaks_20d",),
        predicate=lambda df: num(df, "tl_false_breaks_20d").ge(FALSE_BREAK_MIN),
        text="過去 20 根 K 之內出現 2 次以上「衝出去又縮回來」的假突破，"
             "這種盤先出手的通常兩面挨巴掌。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="above_vh1",
        name="站上大量套牢區",
        tone=BULLISH,
        columns=("is_above_vh1", "vh1_strength"),
        predicate=lambda df: flag(df, "is_above_vh1"),
        text="現價已經站上最近一次「大量高點」—— 那一天套在上面的人現在解套了，"
             "上方的賣壓被消化掉一層。",
        group=TREND_GROUP,
        detail=lambda row: (None if pd.isna(row.get("vh1_strength"))
                            else f"該高點當日量能是 60 日均量的 {float(row['vh1_strength']):.1f} 倍"),
    ),
    Pattern(
        key="below_vl1",
        name="跌破大量底部",
        tone=BEARISH,
        columns=("is_below_vl1", "vl1_strength"),
        predicate=lambda df: flag(df, "is_below_vl1"),
        text="現價跌破最近一次「大量低點」—— 當初在那裡進場的人全部套牢，"
             "反彈上去會遇到解套賣壓。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="channel_top",
        name="貼近通道上緣",
        tone=NEUTRAL,
        columns=("tl_channel_pos",),
        predicate=lambda df: num(df, "tl_channel_pos").between(CHANNEL_HIGH, 1.05),
        text="價格走到通道上緣，離壓力線很近 —— 不是突破就是回頭。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="channel_bottom",
        name="貼近通道下緣",
        tone=NEUTRAL,
        columns=("tl_channel_pos",),
        predicate=lambda df: num(df, "tl_channel_pos").between(-0.05, CHANNEL_LOW),
        text="價格滑到通道下緣，正在測試支撐線。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="bull_ma",
        name="均線多頭排列",
        tone=BULLISH,
        columns=("bull_3ma_1d",),
        predicate=lambda df: flag(df, "bull_3ma_1d"),
        text="短中長期均線由上而下排好（多頭排列），趨勢方向偏多。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="bear_ma",
        name="均線空頭排列",
        tone=BEARISH,
        columns=("bear_3ma_1d",),
        predicate=lambda df: flag(df, "bear_3ma_1d"),
        text="均線呈空頭排列，反彈碰到均線就是壓力。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="ma_squeeze",
        name="均線糾結",
        tone=NEUTRAL,
        columns=("ma_squeeze",),
        predicate=lambda df: num(df, "ma_squeeze").le(MA_SQUEEZE_MAX),
        text="三條均線黏在 2% 以內（均線糾結），多空都沒優勢，等變盤。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="bb_squeeze",
        name="布林通道壓縮",
        tone=NEUTRAL,
        columns=("bb_width_rank",),
        predicate=lambda df: num(df, "bb_width_rank").le(SQUEEZE_RANK),
        text="布林帶寬壓到過去窗口的最低 15%，波動被壓縮 —— 這種狀態不會維持太久。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="bb_upper_break",
        name="突破布林上軌",
        tone=BULLISH,
        columns=("bb_upper_break_1d",),
        predicate=lambda df: flag(df, "bb_upper_break_1d"),
        text="收盤衝出布林上軌，短線強勢但也偏過熱。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="macd_top_div",
        name="MACD 頂背離",
        tone=BEARISH,
        columns=("macd_top_div",),
        predicate=lambda df: flag(df, "macd_top_div"),
        text="價格創新高但 MACD 沒跟上（頂背離）—— 上攻的力道正在衰竭。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="macd_bot_div",
        name="MACD 底背離",
        tone=BULLISH,
        columns=("macd_bot_div",),
        predicate=lambda df: flag(df, "macd_bot_div"),
        text="價格破底但 MACD 沒破（底背離）—— 賣壓在減弱。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="kd_golden_oversold",
        name="低檔 KD 黃金交叉",
        tone=BULLISH,
        columns=("kd_golden_1d", "k_oversold"),
        predicate=lambda df: flag(df, "kd_golden_1d") & flag(df, "k_oversold"),
        text="KD 在超賣區黃金交叉，短線的反彈訊號。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="rsi_overbought",
        name="RSI 超買",
        tone=BEARISH,
        columns=("rsi_overbought_1d",),
        predicate=lambda df: flag(df, "rsi_overbought_1d"),
        text="RSI 進入超買區，追高的風險報酬比開始變差。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="rsi_oversold_rebound",
        name="RSI 超賣反彈",
        tone=BULLISH,
        columns=("rsi_oversold_rebound",),
        predicate=lambda df: flag(df, "rsi_oversold_rebound"),
        text="RSI 從超賣區翻上來，跌深反彈的型態。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="vol_breakout",
        name="爆量紅 K",
        tone=BULLISH,
        columns=("vol_breakout",),
        predicate=lambda df: flag(df, "vol_breakout"),
        text="今天爆量收紅（量能達 5 日均量的倍數以上）—— 有人在買。",
        group=TREND_GROUP,
    ),
    Pattern(
        key="new_high_60",
        name="創 60 日新高",
        tone=BULLISH,
        columns=("is_new_high_60",),
        predicate=lambda df: flag(df, "is_new_high_60"),
        text="收盤創 60 個交易日新高，上方沒有近期套牢賣壓。",
        group=TREND_GROUP,
    ),
)


PATTERNS: tuple[Pattern, ...] = TREND_PATTERNS + TALIB_PATTERNS + CHIP_AND_RELATIVE
PATTERNS_BY_KEY: dict[str, Pattern] = {p.key: p for p in PATTERNS}

# 畫面上的分組順序：先講結構（支撐壓力、趨勢），再講 K 棒，最後講籌碼
GROUP_ORDER = ("技術型態", "K 棒型態", "籌碼面", "相對強弱")


def all_columns() -> tuple[str, ...]:
    """所有規則需要的欄位聯集 —— 供前端一次把該股的特徵撈齊。"""
    seen: dict[str, None] = {}
    for pattern in PATTERNS:
        for column in pattern.columns:
            seen.setdefault(column, None)
    return tuple(seen)


def matched(row_frame: pd.DataFrame) -> list[Pattern]:
    """單列（某股某日）DataFrame → 今天成立的所有型態。

    傳入 DataFrame 而不是 Series 是刻意的：判定式必須是餵給全市場統計的**同一個**
    函式，而那個函式的輸入是 DataFrame。
    """
    if len(row_frame) != 1:
        raise ValueError(f"matched() 只接受單列，收到 {len(row_frame)} 列")
    hits: list[Pattern] = []
    for pattern in PATTERNS:
        try:
            result = pattern.predicate(row_frame)
        except (KeyError, TypeError, ValueError):
            continue    # 缺欄位的規則直接跳過，不能讓一條規則弄掛整頁
        if bool(result.iloc[0]):
            hits.append(pattern)
    return hits


def describe(pattern: Pattern, row: pd.Series) -> str:
    """型態敘述，可用的話補上具體數字。"""
    if pattern.detail is None:
        return pattern.text
    extra: Optional[str]
    try:
        extra = pattern.detail(row)
    except (KeyError, TypeError, ValueError):
        extra = None
    return f"{pattern.text}（{extra}）" if extra else pattern.text
