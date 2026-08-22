"""綜合結論：這種盤面歷史上勝率多少、該不該進場。

三個刻意的設計：

1. **勝率用聯合條件算，不平均**。今天成立的十條說法彼此高度重疊（均線多頭排列
   與創 60 日新高幾乎同時發生），把各自的勝率平均等於把同一件事重複計數。這裡
   改成「歷史上同時符合這幾個條件的日子」，那才是今天的真實對照組。

2. **貪婪加條件，樣本撐不住就停**。條件愈多樣本愈少，全部套下去通常只剩個位數。
   逐條嘗試，加進去後樣本仍達 `MIN_JOINT_SAMPLES` 才保留；結論會明講「用了其中
   幾條」，不假裝十條都納入了。

   ⚠️ 加入的順序**刻意不看歷史勝率**。第一版是照「與基準的差距」由大到小排，
   那等於專挑歷史上看起來最極端的條件組合，是標準的選擇偏誤，算出來的勝率會
   系統性偏高（台積電那天因此得到 58.9%）。改成先照型態分組的固定順序、同組
   內樣本多的優先 —— 排序完全不看結果，數字才是誠實的。

3. **不給買賣建議，給的是條件與代價**。停損位、目標位、風險報酬比是可以從支撐
   壓力算出來的事實；「該不該買」不是。畫面上只說歷史統計站在哪一邊。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from engine.app.frontend import conditional_stats as stats_mod
from engine.app.frontend.levels import Level, nearest
from engine.app.frontend.pattern_base import BEARISH, BULLISH, NEUTRAL, Pattern

# 聯合條件至少要留下這麼多樣本才算數。單一型態用 30 就夠（`MIN_SAMPLES`），
# 但綜合結論會被人拿來當進出場依據，門檻拉高。
MIN_JOINT_SAMPLES = 100
# 勝率與基準差多少才算「有差別」／「明顯有差別」
EDGE_NONE, EDGE_STRONG = 0.01, 0.03
# 風險報酬比的及格線（賺賠比 1.5:1 是常見的最低要求）
GOOD_RR = 1.5


@dataclass(frozen=True)
class Conclusion:
    """頁首那張結論卡的內容。"""

    bullish_count: int
    bearish_count: int
    stats: Optional[dict]           # 聯合條件的歷史統計
    used: tuple[Pattern, ...]       # 實際納入聯合條件的說法
    baseline: dict
    stop_price: Optional[float]
    target_price: Optional[float]
    close: float

    @property
    def is_conclusive(self) -> bool:
        return bool(self.stats) and (self.stats.get("n20") or 0) >= MIN_JOINT_SAMPLES

    @property
    def edge(self) -> Optional[float]:
        """20 日勝率與隨機進場的差距。"""
        if not self.is_conclusive:
            return None
        return self.stats["win20"] - self.baseline["win20"]

    @property
    def risk_reward(self) -> Optional[float]:
        """(目標 − 現價) / (現價 − 停損)。"""
        if self.stop_price is None or self.target_price is None:
            return None
        risk = self.close - self.stop_price
        if risk <= 0:
            return None
        return (self.target_price - self.close) / risk

    @property
    def tone(self) -> str:
        """整體傾向。以歷史勝率為準，說法條數只在沒有統計時當備案。"""
        edge = self.edge
        if edge is None:
            if self.bullish_count > self.bearish_count:
                return BULLISH
            if self.bearish_count > self.bullish_count:
                return BEARISH
            return NEUTRAL
        if edge >= EDGE_NONE:
            return BULLISH
        if edge <= -EDGE_NONE:
            return BEARISH
        return NEUTRAL

    @property
    def headline(self) -> str:
        """一句話結論。"""
        if not self.is_conclusive:
            return ("歷史上找不到夠多「長得像今天」的日子"
                    f"（不足 {MIN_JOINT_SAMPLES} 天），這個盤面無法用統計下結論")
        win = self.stats["win20"]
        edge = self.edge
        if edge >= EDGE_STRONG:
            return f"歷史上這種盤面的 20 日勝率 {win:.1%}，明顯優於隨機進場"
        if edge >= EDGE_NONE:
            return f"歷史上這種盤面的 20 日勝率 {win:.1%}，略優於隨機進場"
        if edge <= -EDGE_STRONG:
            return f"歷史上這種盤面的 20 日勝率 {win:.1%}，明顯差於隨機進場"
        if edge <= -EDGE_NONE:
            return f"歷史上這種盤面的 20 日勝率 {win:.1%}，略差於隨機進場"
        return f"歷史上這種盤面的 20 日勝率 {win:.1%}，與隨機進場沒有差別"

    @property
    def entry_note(self) -> str:
        """能不能進場 —— 只講統計站在哪一邊，以及風險報酬比合不合格。"""
        rr = self.risk_reward
        rr_text = (f"以最近的支撐當停損、最近的壓力當目標，風險報酬比 {rr:.1f}:1"
                   if rr is not None
                   else "附近缺少支撐或壓力，算不出風險報酬比")
        if not self.is_conclusive:
            return f"{rr_text}。沒有統計依據，這時候進場等於純賭方向。"
        edge = self.edge
        if edge <= -EDGE_NONE:
            return (f"{rr_text}。但歷史統計站在反方向 —— 這種盤面過去表現比隨機還差，"
                    "沒有統計上的進場理由。")
        if edge < EDGE_NONE:
            return (f"{rr_text}。歷史統計顯示這種盤面與隨機無異，"
                    "進場的期望值只能來自賺賠比，不是勝率。")
        if rr is not None and rr < GOOD_RR:
            return (f"{rr_text}。勝率雖然高於隨機，但賺賠比不到 {GOOD_RR}:1，"
                    "上方空間不夠划算。")
        return (f"{rr_text}，且歷史勝率高於隨機 —— 統計上站得住腳，"
                "但這是全市場的平均，不保證這一檔。")


def _select_conditions(hits: list[Pattern],
                       base: dict) -> tuple[tuple[Pattern, ...], Optional[dict]]:
    """貪婪挑選：依固定順序加條件，加到樣本撐不住為止。

    排序只用「型態分組的固定優先序」與「單獨出現的樣本數」，兩者都與後續報酬無關。
    任何用勝率或報酬來排序的做法都會把結論推向極端值（見檔頭第 2 點）。
    """
    from engine.app.frontend.patterns import GROUP_ORDER

    scored: list[tuple[int, int, Pattern]] = []
    for pattern in hits:
        single = stats_mod.pattern_stats(pattern)
        if not stats_mod.is_conclusive(single):
            continue
        group_rank = (GROUP_ORDER.index(pattern.group)
                      if pattern.group in GROUP_ORDER else len(GROUP_ORDER))
        scored.append((group_rank, -int(single["n20"]), pattern))
    scored.sort(key=lambda item: (item[0], item[1]))

    selected: list[Pattern] = []
    current: Optional[dict] = None
    for _, _, pattern in scored:
        trial = (*(p.key for p in selected), pattern.key)
        candidate = stats_mod.joint_stats(trial)
        if candidate and (candidate.get("n20") or 0) >= MIN_JOINT_SAMPLES:
            selected.append(pattern)
            current = candidate
    return tuple(selected), current


def build(hits: list[Pattern], all_levels: list[Level], close: float,
          base: dict) -> Conclusion:
    used, stats = _select_conditions(hits, base)
    support = nearest(all_levels, close, above=False)
    resistance = nearest(all_levels, close, above=True)
    return Conclusion(
        bullish_count=sum(1 for p in hits if p.tone == BULLISH),
        bearish_count=sum(1 for p in hits if p.tone == BEARISH),
        stats=stats,
        used=used,
        baseline=base,
        stop_price=support.price if support else None,
        target_price=resistance.price if resistance else None,
        close=close,
    )
