"""技術面分析頁：選一檔股票，講它的壓力區、支撐區、破了會怎樣、有什麼型態說法。

這頁刻意**不碰模型分數** —— 模型的部分在「今日推薦」與「個股歷史預測」兩頁，
混在一起會讓人分不清哪句話是模型講的、哪句是技術分析的傳統說法。

每一條敘述旁邊都掛全市場歷史統計（樣本數、勝率、與隨機進場的差距）。
沒有統計就只是話術；有統計但樣本不足時，畫面直接說樣本不足，不給結論。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from engine.app.frontend import conditional_stats as stats_mod
from engine.app.frontend import levels as levels_mod
from engine.app.frontend import patterns as patterns_mod
from engine.app.frontend import technical_chart
from engine.app.frontend import verdict as verdict_mod
from engine.app.frontend.pattern_base import TONE_ICON, TONE_NAME

DATA_DIR = stats_mod.DATA_DIR
RANGE_CHOICES = {"最近 3 個月": 60, "最近半年": 120, "最近一年": 250, "最近兩年": 500}
DEFAULT_RANGE = "最近一年"

# 副圖用的指標序列。MACD 家族在特徵檔裡是「除以收盤價」的比值，要乘回來
RATIO_COLUMNS = {"dif_ratio": "dif", "macd_ratio": "macd", "hist_ratio": "hist"}
DIRECT_COLUMNS = ("k_value", "d_value", "rsi_14",
                  "foreign_net_ratio", "trust_net_ratio", "dealer_net_ratio")

# levels.py 反推價位需要的欄位（規則本身不一定用得到，要另外撈）
LEVEL_COLUMNS = tuple(
    [f"{prefix}{rank}_{stat}"
     for prefix in ("high", "low", "vh", "vl")
     for rank in (1, 2, 3)
     for stat in ("dist", "days")]
    + ["vh1_strength", "vl1_strength",
       "tl_resist_dist", "tl_resist_slope", "tl_resist_r2", "tl_resist_touches",
       "tl_support_dist", "tl_support_slope", "tl_support_r2", "tl_support_touches",
       "tl_apex_bars", "tl_channel_pos", "tl_channel_width"]
    + [f"close_ma{w}_ratio" for w in levels_mod.MA_WINDOWS]
)


@st.cache_data(ttl=300, show_spinner=False)
def _stock_ohlc(stock_id: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        DATA_DIR / "price.parquet",
        columns=["date", "stock_id", "open", "high", "low", "close", "volume"],
        filters=[("stock_id", "=", stock_id)])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def _features(stock_id: str, columns: tuple[str, ...]) -> pd.DataFrame:
    """把散在六個特徵檔的欄位，撈出這檔股票的完整歷史（單股，量很小）。"""
    index = stats_mod._column_index()
    by_file: dict[str, list[str]] = {}
    for column in columns:
        source = index.get(column)
        if source is not None:
            by_file.setdefault(source, []).append(column)

    merged: Optional[pd.DataFrame] = None
    for name, cols in by_file.items():
        part = pd.read_parquet(DATA_DIR / f"{name}.parquet",
                               columns=["date", "stock_id", *cols],
                               filters=[("stock_id", "=", stock_id)])
        part["date"] = pd.to_datetime(part["date"])
        merged = (part if merged is None
                  else merged.merge(part.drop(columns=["stock_id"]),
                                    on="date", how="outer"))
    if merged is None:
        return pd.DataFrame()
    return merged.sort_values("date").reset_index(drop=True)


def _chart_extras(features: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
    """副圖用的指標，對齊到 K 線視窗。缺的欄位就不畫那一段。"""
    wanted = [c for c in (*RATIO_COLUMNS, *DIRECT_COLUMNS) if c in features.columns]
    if not wanted:
        return pd.DataFrame()
    aligned = ohlc[["date", "close"]].merge(
        features[["date", *wanted]], on="date", how="left")
    extras = pd.DataFrame({"date": aligned["date"]})
    for ratio_column, name in RATIO_COLUMNS.items():
        if ratio_column in aligned.columns:
            extras[name] = aligned[ratio_column] * aligned["close"]
    for column in DIRECT_COLUMNS:
        if column in aligned.columns:
            extras[column] = aligned[column]
    return extras


def _stat_line(stats: Optional[dict], base: dict, days: int = 20) -> str:
    if not stats or not stats.get("n"):
        return "└ 歷史統計：這個條件在資料裡沒有出現過"
    if not stats_mod.is_conclusive(stats, days):
        return (f"└ 歷史統計：全市場只出現 {stats.get(f'n{days}', 0)} 次，"
                f"樣本不足（低於 {stats_mod.MIN_SAMPLES} 次），不下結論")
    return (f"└ 歷史統計：全市場出現 {stats[f'n{days}']:,} 次，"
            f"之後 {days} 日勝率 {stats[f'win{days}']:.1%}、"
            f"中位數報酬 {stats[f'median{days}']:+.2%}"
            f"（隨機進場 {base[f'win{days}']:.1%} / {base[f'median{days}']:+.2%}，"
            f"{stats_mod.verdict(stats, base, days)}）")


def _render_conclusion(conclusion: verdict_mod.Conclusion, place: str) -> None:
    """結論卡。頁首、頁尾各出現一次 —— 使用者要的是「一打開就看到結論」。"""
    icon = TONE_ICON[conclusion.tone]
    stance = {"bullish": "偏多", "bearish": "偏空", "neutral": "中性"}[conclusion.tone]
    st.markdown(f"### {icon} 結論：{stance}")
    st.markdown(f"**{conclusion.headline}**")

    if conclusion.is_conclusive:
        stats, base = conclusion.stats, conclusion.baseline
        cols = st.columns(4)
        cols[0].metric("20 日勝率", f"{stats['win20']:.1%}",
                       f"{conclusion.edge:+.1%} vs 隨機")
        cols[1].metric("20 日中位數報酬", f"{stats['median20']:+.2%}",
                       f"{stats['median20'] - base['median20']:+.2%} vs 隨機")
        cols[2].metric("5 日勝率", f"{stats['win5']:.1%}")
        cols[3].metric("歷史樣本", f"{stats['n20']:,} 天")
        used = "、".join(p.name for p in conclusion.used)
        st.caption(f"條件：歷史上同時符合「{used}」的日子。"
                   f"（今天成立 {conclusion.bullish_count + conclusion.bearish_count} "
                   f"條有方向的說法，全部套下去樣本會不足，"
                   f"只納入差距最大且樣本撐得住的 {len(conclusion.used)} 條）")
    else:
        cols = st.columns(3)
        cols[0].metric("看多說法", f"{conclusion.bullish_count} 條")
        cols[1].metric("看空說法", f"{conclusion.bearish_count} 條")
        cols[2].metric("隨機進場勝率", f"{conclusion.baseline['win20']:.1%}")

    st.markdown(f"**能不能進場？** {conclusion.entry_note}")
    if conclusion.stop_price and conclusion.target_price:
        st.caption(f"停損參考 {conclusion.stop_price:.2f}"
                   f"（{(conclusion.stop_price - conclusion.close) / conclusion.close:.1%}）"
                   f"　目標參考 {conclusion.target_price:.2f}"
                   f"（+{(conclusion.target_price - conclusion.close) / conclusion.close:.1%}）")
    if place == "top":
        st.caption("↓ 以下是這個結論的依據：圖、支撐壓力、各條說法的個別統計")


def _render_levels(resistance, support, close: float) -> None:
    col_r, col_s = st.columns(2)
    with col_r:
        st.markdown("#### 🔺 上方壓力區")
        if not resistance:
            st.write("附近沒有明顯壓力 —— 上方是真空區。")
        for group in reversed(levels_mod.cluster(resistance)):
            price = sum(lv.price for lv in group) / len(group)
            st.write(f"**{price:.2f}**（+{(price - close) / close:.1%}）　"
                     + "、".join(lv.label for lv in group))
    with col_s:
        st.markdown("#### 🔻 下方支撐區")
        if not support:
            st.write("下方沒有明顯支撐 —— 破了直接看更下面的位置。")
        for group in reversed(levels_mod.cluster(support)):
            price = sum(lv.price for lv in group) / len(group)
            st.write(f"**{price:.2f}**（{(price - close) / close:.1%}）　"
                     + "、".join(lv.label for lv in group))


def _render_scenario(stock_id: str, close: float, all_levels, base: dict) -> None:
    st.markdown("#### 🎯 破了會怎樣")
    below = levels_mod.nearest(all_levels, close, above=False)
    above = levels_mod.nearest(all_levels, close, above=True)
    if below:
        st.write(f"最近的支撐是 **{below.price:.2f}**（{below.label}，"
                 f"距現價 {(below.price - close) / close:.1%}）。")
    if above:
        st.write(f"最近的壓力是 **{above.price:.2f}**（{above.label}，"
                 f"距現價 +{(above.price - close) / close:.1%}）。")

    default_price = float(below.price) if below else float(close)
    target = st.number_input(
        "試算價位（想知道破了或站上這個價會怎樣）", min_value=0.01,
        value=round(default_price, 2), step=0.05, key="scenario_price")

    downward = target < close
    direction = "跌破" if downward else "站上"
    st.write(f"**{target:.2f}** 在現價{'下方' if downward else '上方'} "
             f"{abs(target - close) / close:.1%}。")

    beyond = [lv for lv in all_levels
              if (lv.price < target if downward else lv.price > target)]
    if beyond:
        nxt = (max(beyond, key=lambda lv: lv.price) if downward
               else min(beyond, key=lambda lv: lv.price))
        st.write(f"{direction}之後的下一站是 **{nxt.price:.2f}**（{nxt.label}），"
                 f"從 {target:.2f} 再走 {(nxt.price - target) / target:+.1%}。")
    else:
        st.write(f"{direction}之後，附近沒有下一個{'支撐' if downward else '壓力'}了。")

    history = stats_mod.crossing_history(stock_id, float(target), downward)
    if stats_mod.is_conclusive(history):
        st.write(f"這檔股票歷史上收盤{direction} {target:.2f} 共 {history['n20']} 次，"
                 f"之後 20 日勝率 {history['win20']:.1%}、"
                 f"中位數報酬 {history['median20']:+.2%}"
                 f"（隨機進場 {base['win20']:.1%} / {base['median20']:+.2%}）。")
    else:
        st.write(f"這檔股票歷史上收盤{direction}這個價位只有 "
                 f"{history.get('n20', 0)} 次，樣本不足以下結論。")
    st.caption("穿越事件的定義是「前一天收盤在價位的另一側、當天收盤穿過去」，"
               "不是「收盤在價位下方的所有日子」—— 後者會把同一段行情重複計入。")


def _render_patterns(row_frame: pd.DataFrame, base: dict, show_stats: bool) -> None:
    hits = patterns_mod.matched(row_frame)
    st.markdown(f"#### 🗣️ 今天成立的說法（{len(hits)} 條）")
    if not hits:
        st.write("今天沒有任何型態成立 —— 就是一根普通的 K 棒。")
        return

    row = row_frame.iloc[0]
    for group_name in patterns_mod.GROUP_ORDER:
        group_hits = [p for p in hits if p.group == group_name]
        if not group_hits:
            continue
        st.markdown(f"**{group_name}**")
        for pattern in group_hits:
            st.markdown(f"{TONE_ICON[pattern.tone]} **{pattern.name}**"
                        f"（{TONE_NAME[pattern.tone]}）　"
                        f"{patterns_mod.describe(pattern, row)}")
            if show_stats:
                st.caption(_stat_line(stats_mod.pattern_stats(pattern), base))


def render(load_stock_list) -> None:
    """由 `streamlit_app.py` 呼叫；股票清單的讀取沿用主檔既有的快取函式。"""
    st.title("📐 技術面分析")
    st.caption("壓力區、支撐區、破了會怎樣、有哪些傳統說法 —— "
               "每條說法都掛全市場歷史統計。這頁不使用預測模型。")

    stock_list = load_stock_list()
    if stock_list.empty:
        st.warning("找不到 data/stock_list.parquet")
        return

    ids = sorted(stock_list["stock_id"].astype(str).unique().tolist())
    name_of = dict(zip(stock_list["stock_id"].astype(str), stock_list["stock_name"]))
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        stock_id = st.selectbox("股票代號", ids,
                                format_func=lambda s: f"{s} {name_of.get(s, '')}",
                                key="tech_stock")
    ohlc_all = _stock_ohlc(stock_id)
    if ohlc_all.empty:
        st.warning("這檔股票沒有價格資料")
        return
    available = sorted(ohlc_all["date"].dt.date.unique())
    with col_b:
        as_of = st.date_input("分析日期", value=available[-1], min_value=available[0],
                              max_value=available[-1], key="tech_date")
    with col_c:
        span = st.selectbox("顯示範圍", list(RANGE_CHOICES),
                            index=list(RANGE_CHOICES).index(DEFAULT_RANGE),
                            key="tech_range")

    as_of_ts = pd.Timestamp(as_of)
    window = ohlc_all[ohlc_all["date"] <= as_of_ts].tail(RANGE_CHOICES[span])
    if window.empty:
        st.warning("這個日期沒有資料")
        return
    close = float(window["close"].iloc[-1])
    actual_date = window["date"].iloc[-1]
    if actual_date != as_of_ts:
        st.caption(f"{as_of} 沒有交易，改用最近的交易日 {actual_date.date()}")

    wanted = tuple(dict.fromkeys(
        patterns_mod.all_columns() + LEVEL_COLUMNS
        + tuple(RATIO_COLUMNS) + DIRECT_COLUMNS))
    features = _features(stock_id, wanted)
    row_frame = (features[features["date"] == actual_date]
                 if not features.empty else pd.DataFrame())
    if row_frame.empty:
        st.warning("這一天沒有特徵資料（新上市、或歷史不足 260 個交易日的股票不會產出）")
        return
    row = row_frame.iloc[0]

    previous = window["close"].iloc[-2] if len(window) > 1 else close
    hits = patterns_mod.matched(row_frame)
    metric_cols = st.columns(4)
    metric_cols[0].metric("收盤價", f"{close:.2f}", f"{(close - previous) / previous:+.2%}")
    metric_cols[1].metric("成交量", f"{window['volume'].iloc[-1]:,.0f}")
    metric_cols[2].metric("看多說法", f"{sum(1 for p in hits if p.tone == 'bullish')} 條")
    metric_cols[3].metric("看空說法", f"{sum(1 for p in hits if p.tone == 'bearish')} 條")

    all_levels_early = levels_mod.extract_levels(row, close)
    base = stats_mod.baseline()
    with st.spinner("計算歷史上「長得像今天」的日子…"):
        conclusion = verdict_mod.build(hits, all_levels_early, close, base)
    _render_conclusion(conclusion, place="top")
    st.divider()

    layers = st.multiselect("圖層", list(technical_chart.ALL_LAYERS),
                            default=list(technical_chart.DEFAULT_LAYERS),
                            key="tech_layers")

    all_levels = all_levels_early
    trendlines = levels_mod.extract_trendlines(row, close)
    resistance, support = levels_mod.classify(all_levels, close)
    apex = row.get("tl_apex_bars")

    figure = technical_chart.build_figure(
        window, close, levels_mod.cluster(all_levels), trendlines,
        apex_bars=apex if pd.notna(apex) else None,
        extras=_chart_extras(features, window), layers=layers)
    scenario_price = st.session_state.get("scenario_price")
    if scenario_price:
        technical_chart.add_scenario_line(figure, float(scenario_price), "試算價位")

    chart_col, profile_col = st.columns([4, 1])
    with chart_col:
        st.plotly_chart(figure, use_container_width=True)
    with profile_col:
        st.plotly_chart(technical_chart.build_volume_profile(window, close),
                        use_container_width=True)

    if not trendlines:
        st.caption("這一天畫不出趨勢線：已確認的樞紐點少於 3 個"
                   "（見 build_trendline.py 規格 §8.0 第 3 點）。")

    _render_levels(resistance, support, close)
    st.divider()
    _render_scenario(stock_id, close, all_levels, base)
    st.divider()
    show_stats = st.checkbox("顯示每條說法的歷史統計佐證（第一次載入需要數十秒）",
                             value=True, key="tech_show_stats")
    _render_patterns(row_frame, base, show_stats)

    st.divider()
    _render_conclusion(conclusion, place="bottom")

    st.divider()
    st.caption(
        "⚠️ 統計的限制：(1) `data/price.parquet` 不含已下市股票，數字帶生存偏差、"
        "偏樂觀；(2) 是全市場統計，沒有分產業、分市值、分多空年份；"
        "(3) 進場價設為訊號日**隔天**的收盤，未計手續費與交易稅；"
        "(4) 這些型態是技術分析的傳統說法，不是本專案驗證過的策略。")
