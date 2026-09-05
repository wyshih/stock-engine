"""swing 模型的專屬面板邏輯（2026-09-05）。

為什麼獨立一頁而不是併進共用頁面：swing 的出場是**分數規則**，共用回測頁上
超過一半的控制項（移動停利、停損均線、固定停利、型態、連續達標天數）對它
完全不生效。先前用分流補在共用頁面上，三個分流點漏掉一個都不會報錯 ——
實際上第一版就漏了參數搜尋頁。

這裡只放**純函式**，畫面留在 streamlit_app.py，方便測試。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 訊號密度＝當日「分數 >= 門檻」的檔數 ÷ 當日有分數的檔數。
# 2026-09-05 實測：2024H2 0.14%、2025H2 0.04%、2026H1 0.26%，而 2025H2 正是
# swing 唯一輸給大盤的半年（大盤緩漲、最大回檔只有 6%）。密度低本身就是
# 「現在沒有跌完的轉折可買」的信號，而且**當天就看得到**，不必等結果。
DENSITY_LOW = 0.0008     # 低於此值視為「這個模型現在沒舞台」
DENSITY_HIGH = 0.0020    # 高於此值視為「機會很多」


def daily_density(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """逐日訊號密度。回傳 date / n_signal / n_total / density。"""
    if scores.empty:
        return pd.DataFrame(columns=["date", "n_signal", "n_total", "density"])
    g = scores.assign(hit=scores["score"] >= threshold).groupby("date")
    out = g.agg(n_signal=("hit", "sum"), n_total=("hit", "size")).reset_index()
    out["density"] = out["n_signal"] / out["n_total"].replace(0, np.nan)
    return out


def density_verdict(density: float | None) -> tuple[str, str]:
    """密度 → (圖示標籤, 白話說明)。給面板頂部的狀態列用。"""
    if density is None or not np.isfinite(density):
        return "❔ 無資料", "這一天沒有分數資料。"
    if density < DENSITY_LOW:
        return "🔴 訊號稀少", (
            f"今天只有 {density:.3%} 的股票過門檻（低於 {DENSITY_LOW:.2%}）。"
            "這個模型抓的是跌完之後的轉折，市場沒有明顯回檔時它沒有舞台 —— "
            "2025 下半年就是這個狀態，那半年它輸給大盤。**這種時候不該勉強出手。**")
    if density > DENSITY_HIGH:
        return "🟢 機會偏多", (
            f"今天有 {density:.3%} 的股票過門檻（高於 {DENSITY_HIGH:.2%}）。"
            "通常出現在大盤剛回檔完，是這個模型最有效的環境。")
    return "🟡 一般", f"今天有 {density:.3%} 的股票過門檻，屬於常見水準。"


def holdings_status(watch: list[dict], scores: pd.DataFrame,
                    sell_threshold: float) -> pd.DataFrame:
    """關注清單每一檔目前的分數與離出場門檻的距離。

    這個模型的「該不該賣」只看分數，跟價格規則無關 —— 對它顯示移動停利是誤導。
    """
    if not watch or scores.empty:
        return pd.DataFrame(columns=["stock_id", "date", "score", "status"])
    latest = scores.sort_values("date").groupby("stock_id").tail(1)
    rows = []
    for item in watch:
        sid = item["stock_id"]
        mine = latest[latest["stock_id"] == sid]
        if mine.empty:
            rows.append({"stock_id": sid, "date": pd.NaT, "score": np.nan,
                         "status": "－（沒有分數）"})
            continue
        r = mine.iloc[0]
        if r["score"] <= sell_threshold:
            status = f"🔴 分數 {r['score']:.3f} 已跌破 {sell_threshold:.2f} → 該賣"
        else:
            gap = r["score"] - sell_threshold
            status = f"🟢 分數 {r['score']:.3f}，離出場門檻還有 {gap:.3f}"
        rows.append({"stock_id": sid, "date": r["date"],
                     "score": float(r["score"]), "status": status})
    return pd.DataFrame(rows)
