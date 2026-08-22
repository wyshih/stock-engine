"""技術面分析頁的看盤圖。

版面照台股看盤軟體的慣例由上而下排：
    價（K 線、均線群、布林通道、VWAP、支撐壓力區、趨勢線、三角收斂、缺口、
        費波那契回撤、樞紐點）
    量（成交量、5/20 日均量）
    MACD（DIF、MACD、柱狀）
    KD / RSI
    三大法人買賣超（佔當日成交量的百分比）
量價分布（套牢區）因為是橫向直方圖，另外畫一張並排的小圖，不擠進主圖的 x 軸。

趨勢線的座標怎麼來（見 `levels.py` 檔頭）：用「分析日的線價 + 每根 K 的絕對斜率」
表達，往回推 k 根就是 `price_now - k * slope`。線的位置因此與特徵完全一致，不是
前端另外配適出來的第二條線。

外推到未來只畫到收斂頂點（上限 `MAX_FUTURE_BARS`）。線性外推走太遠會變成負值，
`build_trendline.py` 也是因為這樣才設了預測價下限。
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine.app.frontend import indicators
from engine.app.frontend.levels import Level, TrendLine

RESIST_COLOUR = "#d64545"
SUPPORT_COLOUR = "#2f9e44"
MA_COLOUR = "#868e96"
TRIANGLE_COLOUR = "#7048e8"
VWAP_COLOUR = "#0ca678"
FIB_COLOUR = "#f08c00"
PIVOT_COLOUR = "#1098ad"
GAP_UP_COLOUR, GAP_DOWN_COLOUR = "#d64545", "#2f9e44"
UP_COLOUR, DOWN_COLOUR = "#d64545", "#2f9e44"
MA_PALETTE = {5: "#f03e3e", 10: "#f76707", 20: "#f59f00",
              60: "#37b24d", 120: "#1c7ed6", 240: "#7048e8"}
CHIP_COLOURS = {"foreign_net_ratio": "#1c7ed6", "trust_net_ratio": "#f76707",
                "dealer_net_ratio": "#7048e8"}
CHIP_NAMES = {"foreign_net_ratio": "外資", "trust_net_ratio": "投信",
              "dealer_net_ratio": "自營商"}

MAX_FUTURE_BARS = 60
MIN_BAND_FRAC = 0.001   # 單一價位的「區」給的最小厚度，否則帶狀退化成看不見的線

# 圖層名稱（頁面用它做開關；預設全開）
ALL_LAYERS = ("均線", "布林通道", "VWAP", "支撐壓力區", "趨勢線", "三角收斂",
              "跳空缺口", "費波那契", "樞紐點", "MACD", "KD / RSI", "三大法人")
DEFAULT_LAYERS = ("均線", "布林通道", "支撐壓力區", "趨勢線", "三角收斂",
                  "跳空缺口", "MACD", "KD / RSI", "三大法人")


def _future_dates(last_date: pd.Timestamp, bars: int) -> list[pd.Timestamp]:
    """外推用的未來日期。用工作日近似，不扣國定假日 —— 只影響 x 軸刻度位置。"""
    if bars <= 0:
        return []
    return list(pd.bdate_range(last_date, periods=bars + 1)[1:])


def _has(frame: Optional[pd.DataFrame], *columns: str) -> bool:
    return (frame is not None and not frame.empty
            and all(c in frame.columns and frame[c].notna().any() for c in columns))


def _row_plan(layers: Iterable[str], extras: Optional[pd.DataFrame]) -> list[str]:
    """依照有哪些資料、開了哪些圖層，決定副圖要畫幾列。"""
    plan = ["價", "量"]
    if "MACD" in layers and _has(extras, "dif"):
        plan.append("MACD")
    if "KD / RSI" in layers and (_has(extras, "k_value") or _has(extras, "rsi_14")):
        plan.append("KD / RSI")
    if "三大法人" in layers and _has(extras, "foreign_net_ratio"):
        plan.append("三大法人")
    return plan


def _add_price_overlays(fig: go.Figure, ohlc: pd.DataFrame, layers) -> None:
    if "均線" in layers:
        for window, series in indicators.moving_averages(ohlc).items():
            fig.add_trace(go.Scatter(
                x=ohlc["date"], y=series, mode="lines", name=f"MA{window}",
                line=dict(color=MA_PALETTE.get(window, MA_COLOUR), width=1.2),
                hovertemplate=f"MA{window} %{{y:.2f}}<extra></extra>"), row=1, col=1)

    if "布林通道" in layers:
        for name, series in indicators.bollinger(ohlc).items():
            dashed = name != "中軌"
            fig.add_trace(go.Scatter(
                x=ohlc["date"], y=series, mode="lines", name=f"布林{name}",
                line=dict(color="#adb5bd", width=1, dash="dot" if dashed else "solid"),
                hovertemplate=f"布林{name} %{{y:.2f}}<extra></extra>"), row=1, col=1)

    if "VWAP" in layers:
        fig.add_trace(go.Scatter(
            x=ohlc["date"], y=indicators.anchored_vwap(ohlc), mode="lines",
            name="VWAP（區間平均成本）",
            line=dict(color=VWAP_COLOUR, width=1.5, dash="dash"),
            hovertemplate="VWAP %{y:.2f}<extra></extra>"), row=1, col=1)

    if "跳空缺口" in layers:
        for gap in indicators.gaps(ohlc):
            colour = GAP_UP_COLOUR if gap["direction"] == "up" else GAP_DOWN_COLOUR
            fig.add_shape(type="rect", x0=gap["date"], x1=ohlc["date"].iloc[-1],
                          y0=gap["low"], y1=gap["high"], fillcolor=colour,
                          opacity=0.18, line_width=0, layer="below", row=1, col=1)

    if "費波那契" in layers:
        for name, price in indicators.fibonacci_levels(ohlc).items():
            if name in ("起點", "終點"):
                continue
            fig.add_hline(y=price, line=dict(color=FIB_COLOUR, width=0.8, dash="dot"),
                          annotation_text=f"Fib {name}", annotation_position="right",
                          annotation_font=dict(size=9, color=FIB_COLOUR), row=1, col=1)

    if "樞紐點" in layers:
        for name, price in indicators.pivot_points(ohlc).items():
            fig.add_hline(y=price, line=dict(color=PIVOT_COLOUR, width=0.8, dash="dashdot"),
                          annotation_text=name, annotation_position="left",
                          annotation_font=dict(size=9, color=PIVOT_COLOUR), row=1, col=1)


def _add_level_bands(fig: go.Figure, groups: list[list[Level]], close: float,
                     x_start, x_end) -> None:
    """每一「區」畫成一條帶狀 —— 講的是壓力區間，不是單一價位。"""
    for group in groups:
        prices = [lv.price for lv in group]
        lo, hi = min(prices), max(prices)
        colour = RESIST_COLOUR if lo > close else SUPPORT_COLOUR
        if all(lv.source == "ma" for lv in group):
            colour = MA_COLOUR
        if hi - lo < close * MIN_BAND_FRAC:
            pad = close * MIN_BAND_FRAC / 2
            lo, hi = lo - pad, hi + pad
        fig.add_shape(type="rect", x0=x_start, x1=x_end, y0=lo, y1=hi,
                      fillcolor=colour, opacity=0.13, line_width=0, layer="below",
                      row=1, col=1)
        fig.add_annotation(
            x=x_end, y=(lo + hi) / 2, xanchor="right", yanchor="middle",
            text=" / ".join(lv.label for lv in group) + f"　{(lo + hi) / 2:.1f}",
            showarrow=False, font=dict(size=10, color=colour), row=1, col=1)


def _add_subplots(fig: go.Figure, ohlc: pd.DataFrame, extras: Optional[pd.DataFrame],
                  plan: list[str]) -> None:
    dates = ohlc["date"]

    row = plan.index("量") + 1
    volume_colours = [UP_COLOUR if c >= o else DOWN_COLOUR
                      for c, o in zip(ohlc["close"], ohlc["open"])]
    fig.add_trace(go.Bar(x=dates, y=ohlc["volume"], name="成交量",
                         marker_color=volume_colours, marker_line_width=0,
                         opacity=0.55, showlegend=False), row=row, col=1)
    for window, series in indicators.volume_averages(ohlc).items():
        fig.add_trace(go.Scatter(x=dates, y=series, mode="lines", name=f"{window} 日均量",
                                 line=dict(color=MA_PALETTE.get(window, MA_COLOUR),
                                           width=1)), row=row, col=1)

    if "MACD" in plan:
        row = plan.index("MACD") + 1
        colours = [UP_COLOUR if v >= 0 else DOWN_COLOUR
                   for v in extras["hist"].fillna(0)]
        fig.add_trace(go.Bar(x=dates, y=extras["hist"], name="MACD 柱",
                             marker_color=colours, marker_line_width=0,
                             showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(x=dates, y=extras["dif"], mode="lines", name="DIF",
                                 line=dict(color="#1c7ed6", width=1.2)), row=row, col=1)
        if "macd" in extras.columns:
            fig.add_trace(go.Scatter(x=dates, y=extras["macd"], mode="lines", name="MACD",
                                     line=dict(color="#f76707", width=1.2)), row=row, col=1)
        fig.add_hline(y=0, line=dict(color="#adb5bd", width=0.8), row=row, col=1)

    if "KD / RSI" in plan:
        row = plan.index("KD / RSI") + 1
        for column, name, colour in (("k_value", "K", "#1c7ed6"),
                                     ("d_value", "D", "#f76707"),
                                     ("rsi_14", "RSI(14)", "#7048e8")):
            if column in extras.columns and extras[column].notna().any():
                fig.add_trace(go.Scatter(x=dates, y=extras[column], mode="lines",
                                         name=name, line=dict(color=colour, width=1.2)),
                              row=row, col=1)
        for level in (20, 80):
            fig.add_hline(y=level, line=dict(color="#adb5bd", width=0.8, dash="dot"),
                          row=row, col=1)

    if "三大法人" in plan:
        row = plan.index("三大法人") + 1
        for column, colour in CHIP_COLOURS.items():
            if column in extras.columns and extras[column].notna().any():
                fig.add_trace(go.Bar(x=dates, y=extras[column], name=CHIP_NAMES[column],
                                     marker_color=colour, marker_line_width=0,
                                     opacity=0.8), row=row, col=1)
        fig.add_hline(y=0, line=dict(color="#adb5bd", width=0.8), row=row, col=1)


def build_figure(ohlc: pd.DataFrame, close: float, groups: list[list[Level]],
                 trendlines: list[TrendLine], apex_bars: Optional[float] = None,
                 extras: Optional[pd.DataFrame] = None,
                 layers: Iterable[str] = DEFAULT_LAYERS) -> go.Figure:
    """`ohlc` 需含 date/open/high/low/close/volume，且**最後一列就是分析日**。

    `extras` 是與 `ohlc` 對齊的指標序列（dif/macd/hist/k_value/d_value/rsi_14/
    三大法人買賣超比），來源是特徵檔，缺哪個就不畫哪一段。
    """
    layers = set(layers)
    plan = _row_plan(layers, extras)
    heights = {"價": 0.46, "量": 0.14, "MACD": 0.14, "KD / RSI": 0.14, "三大法人": 0.12}
    row_heights = [heights[name] for name in plan]
    total = sum(row_heights)
    fig = make_subplots(rows=len(plan), cols=1, shared_xaxes=True,
                        vertical_spacing=0.025,
                        row_heights=[h / total for h in row_heights])

    fig.add_trace(go.Candlestick(
        x=ohlc["date"], open=ohlc["open"], high=ohlc["high"],
        low=ohlc["low"], close=ohlc["close"],
        increasing_line_color=UP_COLOUR, decreasing_line_color=DOWN_COLOUR,
        name="股價", showlegend=False), row=1, col=1)

    _add_price_overlays(fig, ohlc, layers)
    _add_subplots(fig, ohlc, extras, plan)

    # 趨勢線往回覆蓋整個視窗，往未來畫到收斂頂點
    n_past = len(ohlc)
    last_date = pd.Timestamp(ohlc["date"].iloc[-1])
    future_bars = 0
    if "三角收斂" in layers and apex_bars is not None and pd.notna(apex_bars):
        future_bars = int(min(max(round(float(apex_bars)), 0), MAX_FUTURE_BARS))
    future = _future_dates(last_date, future_bars)
    dates = list(ohlc["date"]) + future
    offsets = list(range(-(n_past - 1), 1)) + list(range(1, future_bars + 1))

    if "趨勢線" in layers:
        for line in trendlines:
            colour = RESIST_COLOUR if "壓力" in line.label else SUPPORT_COLOUR
            label = line.label
            if line.touches is not None and line.r2 is not None:
                label = f"{label}（觸線 {int(line.touches)} 次 · r² {line.r2:.2f}）"
            fig.add_trace(go.Scatter(
                x=dates, y=[line.price_at(offset) for offset in offsets],
                mode="lines", name=label, line=dict(color=colour, width=2),
                hovertemplate=f"{line.label} %{{y:.2f}}<extra></extra>"), row=1, col=1)

    if future_bars > 0 and len(trendlines) == 2:
        apex_date = future[-1]
        fig.add_vrect(x0=last_date, x1=apex_date, fillcolor=TRIANGLE_COLOUR,
                      opacity=0.07, line_width=0, layer="below", row=1, col=1)
        fig.add_vline(x=apex_date, line=dict(color=TRIANGLE_COLOUR, width=1, dash="dot"),
                      row=1, col=1)
        fig.add_annotation(x=apex_date, y=1.0, yref="y domain", yanchor="bottom",
                           text=f"收斂頂點（約 {future_bars} 根 K 後）", showarrow=False,
                           font=dict(size=11, color=TRIANGLE_COLOUR), row=1, col=1)

    if "支撐壓力區" in layers:
        _add_level_bands(fig, groups, close, ohlc["date"].iloc[0], dates[-1])

    fig.add_hline(y=close, line=dict(color="#1c1c1c", width=1, dash="dash"),
                  annotation_text=f"現價 {close:.2f}", annotation_position="top left",
                  row=1, col=1)

    fig.update_layout(height=260 + 180 * len(plan), bargap=0,
                      margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_rangeslider_visible=False, barmode="relative",
                      legend=dict(orientation="h", y=1.04, font=dict(size=10)))
    for position, name in enumerate(plan, start=1):
        fig.update_yaxes(title_text=name, row=position, col=1)
        # 未來那段沒有 K 棒，x 軸要明確拉到外推終點才看得到收斂頂點
        fig.update_xaxes(range=[ohlc["date"].iloc[0], dates[-1]], row=position, col=1)
    return fig


def build_volume_profile(ohlc: pd.DataFrame, close: float) -> go.Figure:
    """量價分布（套牢區）：橫向直方圖，價格在 y 軸，與主圖的 y 軸對得起來。"""
    profile = indicators.volume_profile(ohlc)
    fig = go.Figure()
    if profile.empty:
        return fig
    colours = [RESIST_COLOUR if price > close else SUPPORT_COLOUR
               for price in profile["price"]]
    fig.add_trace(go.Bar(x=profile["volume"], y=profile["price"], orientation="h",
                         marker_color=colours, marker_line_width=0, opacity=0.65,
                         hovertemplate="%{y:.2f} 元　累積量 %{x:,.0f}<extra></extra>",
                         showlegend=False))
    fig.add_hline(y=close, line=dict(color="#1c1c1c", width=1, dash="dash"))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                      title=dict(text="量價分布（套牢區）", font=dict(size=13)),
                      xaxis_title="累積成交量", yaxis_title="價格")
    return fig


def add_scenario_line(fig: go.Figure, price: float, label: str) -> go.Figure:
    """把使用者試算的價位畫上去。"""
    fig.add_hline(y=price, line=dict(color=TRIANGLE_COLOUR, width=2, dash="dashdot"),
                  annotation_text=f"{label} {price:.2f}",
                  annotation_position="bottom left", row=1, col=1)
    return fig
