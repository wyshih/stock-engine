"""籌碼面與相對強弱的說法。

欄位語意（來源：`features_v2/build_chip.py` 與 `build_relative.py` 的檔頭）：
- `foreign_net_ratio` 等法人欄位：買賣超**佔當日成交量的百分數**（`5.0` = 5%）。
- `institutional_10d`：三大法人 10 日買賣超總和 ÷ **一日**均量，同樣是百分數，
  所以 100 代表「十天累積買掉一整天的量」。
- `absorb_stall_10d`：法人買超佔比 >= 60% 但 10 日報酬在 ±3% 以內 —— 吸籌不漲。
- `margin_slope_5d_raw`：融資餘額 5 日斜率的原始值，只用正負號（尺度隨個股而異，
  不設絕對門檻）。
- `rel_ramom_20_60`：20 日報酬 ÷（日報酬標準差 × √20），再相對自身 60 日分布，
  是**風險調整後**的動能 z 分數。
- `rel_amt_20`：成交金額相對自身近期中位數的倍數。
- `rel_pct_rsi_14_250`：RSI 在這檔股票自己過去 250 日分布中的百分位（0~1）。
- `rel_dist_high_60_atr`：距 60 日高點幾個 ATR（負值代表在高點下方）。

⚠️ `margin_ratio` 與 `margin_short_ratio` 的尺度在個股之間差異極大（見 build_chip
檔頭關於分母下限的說明），這裡刻意不對它們設絕對門檻，只用融資的方向。
"""

from __future__ import annotations

from engine.app.frontend.pattern_base import (
    BEARISH,
    BULLISH,
    NEUTRAL,
    Pattern,
    flag,
    fmt_float,
    num,
)

CHIP_GROUP = "籌碼面"
RELATIVE_GROUP = "相對強弱"

# 十日累積買超達一日均量的百分數
INST_HEAVY = 100.0
# 風險調整後動能的強弱門檻（z 分數）
MOMENTUM_STRONG, MOMENTUM_WEAK = 1.0, -1.0
# 成交金額相對自身中位數的倍數
AMOUNT_SURGE = 2.0
# RSI 相對自身歷史分布的百分位
RSI_PCT_HIGH, RSI_PCT_LOW = 0.9, 0.1
# 距 60 日高點幾個 ATR 之內算「貼近」
NEAR_HIGH_ATR = 1.0


CHIP_PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        key="foreign_consec_buy",
        name="外資連續買超",
        tone=BULLISH,
        columns=("foreign_consec_buy_1d", "foreign_net_ratio"),
        predicate=lambda df: flag(df, "foreign_consec_buy_1d"),
        text="外資連續買超中 —— 這是最常被拿來當「有人在收」的依據。",
        group=CHIP_GROUP,
        detail=lambda row: fmt_float(row, "foreign_net_ratio",
                                     "今日外資買賣超佔成交量 {:.1f}%"),
    ),
    Pattern(
        key="foreign_trust_align",
        name="外資投信同步",
        tone=BULLISH,
        columns=("foreign_trust_align_1d",),
        predicate=lambda df: flag(df, "foreign_trust_align_1d"),
        text="外資與投信同一天站在買方 —— 兩大法人同向，籌碼面的說法會說「有共識」。",
        group=CHIP_GROUP,
    ),
    Pattern(
        key="inst_heavy_buy",
        name="法人十日大買",
        tone=BULLISH,
        columns=("institutional_10d",),
        predicate=lambda df: num(df, "institutional_10d").ge(INST_HEAVY),
        text="三大法人 10 日累積買超已經超過一整天的成交量 —— 籌碼在往法人手上集中。",
        group=CHIP_GROUP,
        detail=lambda row: fmt_float(row, "institutional_10d",
                                     "10 日累積買賣超相當於 {:.0f}% 的日均量"),
    ),
    Pattern(
        key="inst_heavy_sell",
        name="法人十日大賣",
        tone=BEARISH,
        columns=("institutional_10d",),
        predicate=lambda df: num(df, "institutional_10d").le(-INST_HEAVY),
        text="三大法人 10 日累積賣超超過一整天的量 —— 法人在出貨。",
        group=CHIP_GROUP,
        detail=lambda row: fmt_float(row, "institutional_10d",
                                     "10 日累積買賣超相當於 {:.0f}% 的日均量"),
    ),
    Pattern(
        key="absorb_stall",
        name="吸籌不漲",
        tone=BULLISH,
        columns=("absorb_stall_10d",),
        predicate=lambda df: flag(df, "absorb_stall_10d"),
        text="法人買超佔了成交量六成以上，股價 10 天卻幾乎沒動 —— "
             "「有人在低檔默默收貨」的典型說法。",
        group=CHIP_GROUP,
    ),
    Pattern(
        key="margin_rising",
        name="融資增加",
        tone=BEARISH,
        columns=("margin_slope_5d_raw",),
        predicate=lambda df: num(df, "margin_slope_5d_raw").gt(0),
        text="融資餘額最近 5 天在增加 —— 散戶在追價，籌碼面的傳統看法是變凌亂。",
        group=CHIP_GROUP,
    ),
    Pattern(
        key="margin_falling",
        name="融資減少",
        tone=BULLISH,
        columns=("margin_slope_5d_raw",),
        predicate=lambda df: num(df, "margin_slope_5d_raw").lt(0),
        text="融資餘額最近 5 天在減少 —— 浮額被洗掉，籌碼相對安定。",
        group=CHIP_GROUP,
    ),
)


RELATIVE_PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        key="rel_momentum_strong",
        name="風險調整後動能強",
        tone=BULLISH,
        columns=("rel_ramom_20_60",),
        predicate=lambda df: num(df, "rel_ramom_20_60").ge(MOMENTUM_STRONG),
        text="用自己的波動度校正之後，最近 20 天的漲勢仍明顯高於這檔股票的常態 —— 強勢股。",
        group=RELATIVE_GROUP,
    ),
    Pattern(
        key="rel_momentum_weak",
        name="風險調整後動能弱",
        tone=BEARISH,
        columns=("rel_ramom_20_60",),
        predicate=lambda df: num(df, "rel_ramom_20_60").le(MOMENTUM_WEAK),
        text="以自身波動度校正後仍屬明顯落後 —— 弱勢整理，等落後補漲要有耐心。",
        group=RELATIVE_GROUP,
    ),
    Pattern(
        key="rel_amount_surge",
        name="量能異常放大",
        tone=NEUTRAL,
        columns=("rel_amt_20",),
        predicate=lambda df: num(df, "rel_amt_20").ge(AMOUNT_SURGE),
        text="成交金額是自己近期常態的兩倍以上 —— 有事情在發生，方向要配合價格看。",
        group=RELATIVE_GROUP,
        detail=lambda row: fmt_float(row, "rel_amt_20", "目前是常態量的 {:.1f} 倍"),
    ),
    Pattern(
        key="rel_rsi_extreme_high",
        name="RSI 站上自身高位",
        tone=BEARISH,
        columns=("rel_pct_rsi_14_250",),
        predicate=lambda df: num(df, "rel_pct_rsi_14_250").ge(RSI_PCT_HIGH),
        text="RSI 站上這檔股票過去一年分布的前 10% —— 用它自己的標準看已經偏熱"
             "（比固定的 70/30 門檻合理：有些股票長年 RSI 都在 60 以上）。",
        group=RELATIVE_GROUP,
    ),
    Pattern(
        key="rel_rsi_extreme_low",
        name="RSI 落到自身低位",
        tone=BULLISH,
        columns=("rel_pct_rsi_14_250",),
        predicate=lambda df: num(df, "rel_pct_rsi_14_250").le(RSI_PCT_LOW),
        text="RSI 落到這檔股票過去一年分布的後 10% —— 以它自己的標準算是深度回檔。",
        group=RELATIVE_GROUP,
    ),
    Pattern(
        key="rel_near_high",
        name="貼近 60 日高點",
        tone=BULLISH,
        columns=("rel_dist_high_60_atr",),
        predicate=lambda df: num(df, "rel_dist_high_60_atr").ge(-NEAR_HIGH_ATR),
        text="離 60 日高點不到一個 ATR（一天的正常波動）—— 一根像樣的紅 K 就能創高。",
        group=RELATIVE_GROUP,
    ),
)


CHIP_AND_RELATIVE: tuple[Pattern, ...] = CHIP_PATTERNS + RELATIVE_PATTERNS
