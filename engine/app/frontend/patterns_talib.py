"""TA-Lib 的 61 個 K 棒型態 → 中文說法。

`data/talib_features.parquet` 的 `cdl*` 欄位是 TA-Lib 原始輸出：
    +100 / +80  看多方向成立（80 是 TA-Lib 的次級確認）
    -100 / -80  看空方向成立
       0        沒有出現
所以每個欄位展開成「看多版」與「看空版」兩條規則。單向的型態（例如錘子線只會回
+100）那一邊永遠不會成立，不必特別排除 —— 讓資料自己決定，比人工維護哪個型態是
單向的更不容易錯。

⚠️ 這些型態是「傳統說法」，不是本專案驗證過的東西。每一條在畫面上都會掛全市場
歷史統計，讓使用者自己看它到底有沒有用；樣本不足的一律標示出來。
"""

from __future__ import annotations

import pandas as pd

from engine.app.frontend.pattern_base import BEARISH, BULLISH, NEUTRAL, Pattern, num

GROUP = "K 棒型態"

# TA-Lib 的正負號不是每個型態都代表方向。十字星、紡錘線這類本質上「方向未定」的
# 型態一律回 +100，照著展開成「看多版」會產生「十字線（看多）」這種錯誤敘述；
# 墓碑十字同樣只回 +100，但傳統上是看空。這兩類要單獨處理。
NEUTRAL_ONLY = frozenset({
    "cdldoji", "cdldojistar", "cdlhighwave", "cdllongleggeddoji",
    "cdlrickshawman", "cdlspinningtop", "cdlshortline", "cdllongline",
})
BEARISH_ONLY = frozenset({"cdlgravestonedoji"})

# (欄位, 中文名, 傳統說法補充)
CANDLES: tuple[tuple[str, str, str], ...] = (
    ("cdl2crows", "兩隻烏鴉", "高檔連兩根黑 K，傳統上視為反轉訊號"),
    ("cdl3blackcrows", "三隻烏鴉", "連三根長黑，典型的頭部訊號"),
    ("cdl3inside", "三內部型態", "母子線後出現確認 K 棒"),
    ("cdl3linestrike", "三線打擊", "連三根同向後被一根反向長 K 全數吞掉"),
    ("cdl3outside", "三外部型態", "吞噬型態後出現確認 K 棒"),
    ("cdl3starsinsouth", "南方三星", "低檔連三根實體遞減的黑 K，賣壓衰竭"),
    ("cdl3whitesoldiers", "紅三兵", "連三根長紅逐步墊高，經典的攻擊型態"),
    ("cdlabandonedbaby", "棄嬰", "跳空孤島的十字，反轉力道強但很少見"),
    ("cdladvanceblock", "大敵當前", "連三紅但上影線變長、實體縮短，追價力道在減弱"),
    ("cdlbelthold", "捉腰帶線", "開盤即最低（或最高）的長實體 K 棒"),
    ("cdlbreakaway", "脫離型態", "跳空後連續同向，缺口沒有回補"),
    ("cdlclosingmarubozu", "收盤光頭光腳", "收在最高（或最低），當日方向完全沒有遲疑"),
    ("cdlconcealbabyswall", "藏嬰吞沒", "極罕見的低檔反轉型態"),
    ("cdlcounterattack", "反擊線", "兩根反向長 K 收在同一價位，多空短兵相接"),
    ("cdldarkcloudcover", "烏雲罩頂", "長紅後跳高開低走並深入前一根實體，高檔反轉"),
    ("cdldoji", "十字線", "開收幾乎同價，多空拉鋸、方向未定"),
    ("cdldojistar", "十字星", "跳空後的十字，前一段走勢可能停頓"),
    ("cdldragonflydoji", "蜻蜓十字", "長下影的十字，下方有承接"),
    ("cdlengulfing", "吞噬型態", "後一根 K 棒的實體完全吃掉前一根"),
    ("cdleveningdojistar", "十字夜星", "夜星的十字版本，高檔反轉訊號更強"),
    ("cdleveningstar", "夜星（黃昏之星）", "高檔三根 K 棒的反轉型態"),
    ("cdlgapsidesidewhite", "缺口並列白線", "跳空後兩根同向紅 K，趨勢延續"),
    ("cdlgravestonedoji", "墓碑十字", "長上影的十字，上方賣壓沉重"),
    ("cdlhammer", "錘子線", "低檔長下影，跌深後有人承接"),
    ("cdlhangingman", "吊人線", "高檔長下影，形狀同錘子但位置在高點，是警訊"),
    ("cdlharami", "母子線", "後一根被前一根完全包住，動能停頓"),
    ("cdlharamicross", "十字母子線", "母子線的十字版本，停頓訊號更明確"),
    ("cdlhighwave", "風高浪大", "上下影線都很長，當天多空劇烈拉扯"),
    ("cdlhikkake", "Hikkake 陷阱", "假突破後反向的陷阱型態"),
    ("cdlhikkakemod", "修正版 Hikkake", "Hikkake 的嚴格版本"),
    ("cdlhomingpigeon", "家鴿", "低檔兩根黑 K 但第二根被包住，賣壓收斂"),
    ("cdlidentical3crows", "三胞胎烏鴉", "三根開盤即前一根收盤的長黑，殺盤沉重"),
    ("cdlinneck", "頸內線", "反彈力道不足，下跌延續"),
    ("cdlinvertedhammer", "倒錘子線", "低檔長上影，有人嘗試往上攻"),
    ("cdlkicking", "反衝型態", "兩根光頭光腳中間跳空，方向瞬間翻轉"),
    ("cdlkickingbylength", "長線反衝", "反衝型態，以較長的那根實體定方向"),
    ("cdlladderbottom", "梯底", "連續下跌後的低檔反轉型態"),
    ("cdllongleggeddoji", "長腳十字", "上下影都長的十字，變盤前兆"),
    ("cdllongline", "長實體 K 棒", "實體明顯大於近期平均，當天方向明確"),
    ("cdlmarubozu", "光頭光腳", "沒有上下影線的長實體，一路走到底"),
    ("cdlmatchinglow", "相同低價", "兩根黑 K 收在同一低點，支撐浮現"),
    ("cdlmathold", "鋪墊", "上升途中的小幅整理後續攻"),
    ("cdlmorningdojistar", "十字晨星", "晨星的十字版本，底部訊號更強"),
    ("cdlmorningstar", "晨星（早晨之星）", "低檔三根 K 棒的反轉型態"),
    ("cdlonneck", "頸上線", "反彈只到前一根低點，下跌未止"),
    ("cdlpiercing", "貫穿線", "長黑後跳空開低但收回一半以上，底部訊號"),
    ("cdlrickshawman", "黃包車夫", "實體極小且位於長影線中央，方向完全未定"),
    ("cdlrisefall3methods", "上升／下降三法", "趨勢中的小幅整理，之後延續原方向"),
    ("cdlseparatinglines", "分離線", "與前一根反向但開在同一價位，原趨勢延續"),
    ("cdlshootingstar", "流星線", "高檔長上影，追高的人被套在上面"),
    ("cdlshortline", "短實體 K 棒", "實體很小，當天沒有方向"),
    ("cdlspinningtop", "紡錘線", "小實體加上下影線，多空僵持"),
    ("cdlstalledpattern", "停頓型態", "連紅後最後一根力道明顯轉弱"),
    ("cdlsticksandwich", "條形三明治", "兩根黑 K 夾一根紅 K 且收在同價，支撐成立"),
    ("cdltakuri", "探水竿", "下影特別長的蜻蜓十字，探底後拉回"),
    ("cdltasukigap", "跳空並列", "跳空後的回補失敗，趨勢延續"),
    ("cdlthrusting", "插入線", "反彈未過前一根中點，力道不足"),
    ("cdltristar", "三星", "連續三根十字，極罕見的變盤訊號"),
    ("cdlunique3river", "奇特三河床", "低檔的罕見反轉型態"),
    ("cdlupsidegap2crows", "向上跳空兩隻烏鴉", "跳空後連兩根黑 K，攻擊失敗"),
    ("cdlxsidegap3methods", "跳空三法", "跳空後的整理，之後延續原方向"),
)


def _directional(column: str, name: str, blurb: str, bullish: bool) -> Pattern:
    """正負號代表方向的型態：展開成看多版與看空版兩條。"""
    side = "看多" if bullish else "看空"
    if bullish:
        def predicate(df: pd.DataFrame) -> pd.Series:
            return num(df, column).gt(0)
    else:
        def predicate(df: pd.DataFrame) -> pd.Series:
            return num(df, column).lt(0)
    return Pattern(
        key=f"{column}_{'bull' if bullish else 'bear'}",
        name=f"{name}（{side}）",
        tone=BULLISH if bullish else BEARISH,
        columns=(column,),
        predicate=predicate,
        text=f"K 棒走出「{name}」的{side}型態 —— {blurb}。",
        group=GROUP,
    )


def _single(column: str, name: str, blurb: str, tone: str) -> Pattern:
    """正負號不代表方向的型態：只出一條，方向由傳統說法決定。"""
    return Pattern(
        key=column,
        name=name,
        tone=tone,
        columns=(column,),
        predicate=lambda df: num(df, column).ne(0) & num(df, column).notna(),
        text=f"K 棒走出「{name}」—— {blurb}。",
        group=GROUP,
    )


def _build() -> tuple[Pattern, ...]:
    out: list[Pattern] = []
    for column, name, blurb in CANDLES:
        if column in NEUTRAL_ONLY:
            out.append(_single(column, name, blurb, NEUTRAL))
        elif column in BEARISH_ONLY:
            out.append(_single(column, name, blurb, BEARISH))
        else:
            out.extend(_directional(column, name, blurb, bullish)
                       for bullish in (True, False))
    return tuple(out)


TALIB_PATTERNS: tuple[Pattern, ...] = _build()
