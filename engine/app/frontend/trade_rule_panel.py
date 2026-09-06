"""買賣點規則頁的純邏輯（2026-09-06）。

跟「型態規則」頁分開的理由：那一頁的資料是「訊號日 + 20 天後的結果」，這一頁是
**成對的買點與賣點**（買進日/買價/賣出日/賣價/持有天數）。硬塞進同一份 CSV 會讓
`r_end` / `label` 這些欄位在不同列代表不同東西 —— 這種欄位語意漂移最難察覺。

資料只讀不寫，由 fpm 專案的 `src/target_rule.py` 產出後複製進
`data/fpm_rules/trade_rules_*`。要新增規則只要多寫一筆進 registry 與 hitlist，
這頁不用改程式。

這裡只放純函式，畫面在 streamlit_app.py。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from engine.paths import DATA_DIR

RULES_DIR = DATA_DIR / "fpm_rules"
REGISTRY_PATH = RULES_DIR / "trade_rules_registry.yaml"
HITLIST_PATH = RULES_DIR / "trade_rules_hitlist.csv"
PENDING_PATH = RULES_DIR / "trade_rules_pending.csv"

DATE_COLS = ("signal_date", "buy_date", "sell_date")

# 逐期一定要看：全期平均會被單一時段主導（這個專案吃過這個虧）。
WIN_RATE_TARGET = 0.80


def load_registry(path: Path = REGISTRY_PATH) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def load_hits(path: Path = HITLIST_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def load_pending(path: Path = PENDING_PATH) -> pd.DataFrame:
    """還沒成交的最新訊號（隔日開盤價還沒出來）。"""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def half_year_label(ts: pd.Timestamp) -> str:
    return f"{ts.year}H{1 if ts.month <= 6 else 2}"


COLUMNS = ["期間", "交易數", "勝率", "平均報酬", "中位報酬", "平均持有", "未結束", "達標"]


def resolved_only(hits: pd.DataFrame) -> pd.DataFrame:
    """只留已結束的交易。沒有 `resolved` 欄的舊資料一律視為已結束。"""
    if hits.empty or "resolved" not in hits.columns:
        return hits
    return hits[hits["resolved"].astype(bool)]


def period_summary(hits: pd.DataFrame) -> pd.DataFrame:
    """逐半年期績效。「達標」＝該期勝率 >= `WIN_RATE_TARGET`，這是使用者定的驗收線。

    ⚠️ 勝率只算**已結束**的交易，但「未結束」欄一定要跟著出去：達標的部位會先
    結束、沒達標的還開著，所以最近幾期的勝率天生偏高。未結束筆數多的那幾期
    不能當定論看 —— 這個專案就是被這個偏誤騙過一次（近期勝率虛報到 99%）。
    """
    if hits.empty:
        return pd.DataFrame(columns=COLUMNS)
    d = hits.assign(期間=hits["signal_date"].map(half_year_label))
    if "resolved" not in d.columns:
        d["resolved"] = True
    d["resolved"] = d["resolved"].astype(bool)
    done = d[d["resolved"]]
    out = done.groupby("期間").agg(
        交易數=("ret", "size"),
        勝率=("ret", lambda s: (s > 0).mean()),
        平均報酬=("ret", "mean"),
        中位報酬=("ret", "median"),
        平均持有=("hold", "mean"),
    ).reset_index()
    open_n = d[~d["resolved"]].groupby("期間").size().rename("未結束")
    out = out.merge(open_n, on="期間", how="outer")
    out["未結束"] = out["未結束"].fillna(0).astype(int)
    out["交易數"] = out["交易數"].fillna(0).astype(int)
    out["達標"] = out["勝率"] >= WIN_RATE_TARGET
    return out[COLUMNS].sort_values("期間").reset_index(drop=True)


def overall_stats(hits: pd.DataFrame) -> dict:
    """全期摘要。空表回傳 0 而不是 NaN —— 前端要直接顯示。"""
    if hits.empty:
        return {"交易數": 0, "勝率": 0.0, "平均報酬": 0.0, "中位持有": 0.0,
                "達標期數": 0, "總期數": 0, "未結束": 0}
    per = period_summary(hits)
    done = resolved_only(hits)
    if done.empty:
        return {"交易數": 0, "勝率": 0.0, "平均報酬": 0.0, "中位持有": 0.0,
                "達標期數": 0, "總期數": int(len(per)), "未結束": int(len(hits))}
    return {
        "交易數": int(len(done)),
        "勝率": float((done["ret"] > 0).mean()),
        "平均報酬": float(done["ret"].mean()),
        "中位持有": float(done["hold"].median()),
        "達標期數": int(per["達標"].sum()),
        "總期數": int(len(per)),
        "未結束": int(len(hits) - len(done)),
    }


def recent_trades(hits: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    """最近的成交紀錄，最新在最前面。"""
    if hits.empty:
        return hits
    return hits.sort_values("signal_date", ascending=False).head(limit).reset_index(drop=True)


def open_positions(hits: pd.DataFrame, watch: list[str] | None = None) -> pd.DataFrame:
    """還沒賣掉的部位（`resolved` 為 False）—— 買了但還沒達標、也還沒抱到上限。

    不設停損的代價全在這裡：這些單子帳面可能已經虧很多，只是還沒認列。
    """
    if hits.empty or "resolved" not in hits.columns:
        return pd.DataFrame()
    stuck = hits[~hits["resolved"].astype(bool)]
    if watch:
        stuck = stuck[stuck["stock_id"].astype(str).isin([str(w) for w in watch])]
    return stuck.sort_values("signal_date", ascending=False).reset_index(drop=True)


# ── 每日買賣點 ────────────────────────────────────────────────────────────────
# 一天會有三種東西，混在一張表裡看不懂，所以分三個函式：
#   訊號日 D 收盤後決策 → D+1 開盤買進 → 達標日 K 收盤 → K+1 開盤賣出
# 所以「今天要買的」是昨天的訊號，「今天出現的訊號」要明天才買得到。

def trading_days(hits: pd.DataFrame, pending: pd.DataFrame | None = None) -> list:
    """有東西可看的日期（買進日 / 賣出日 / 訊號日的聯集），由新到舊。"""
    days: set = set()
    if not hits.empty:
        for c in ("buy_date", "sell_date", "signal_date"):
            if c in hits.columns:
                days |= set(hits[c].dropna())
    if pending is not None and not pending.empty and "date" in pending.columns:
        days |= set(pending["date"].dropna())
    return sorted(days, reverse=True)


def _on_date(df: pd.DataFrame, col: str, day) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame()
    return df[df[col] == pd.Timestamp(day)].reset_index(drop=True)


def buys_on(hits: pd.DataFrame, day) -> pd.DataFrame:
    """這一天**開盤買進**的部位（訊號是前一個交易日發出的）。"""
    return _on_date(hits, "buy_date", day)


def sells_on(hits: pd.DataFrame, day) -> pd.DataFrame:
    """這一天**開盤賣出**的部位。只有已結束的交易才算數 —— 未結束的那筆
    `sell_date` 是資料尾端，不是真的賣出。"""
    done = resolved_only(hits)
    return _on_date(done, "sell_date", day)


def new_signals_on(hits: pd.DataFrame, pending: pd.DataFrame | None, day) -> pd.DataFrame:
    """這一天收盤後選出的股票 —— **明天開盤才買得到**。

    已經成交的從 hits 取（帶得出後來的結果），還沒成交的從 pending 取。
    """
    fired = _on_date(hits, "signal_date", day)
    if pending is None or pending.empty:
        return fired
    waiting = _on_date(pending.rename(columns={"date": "signal_date"}), "signal_date", day)
    if waiting.empty:
        return fired
    if fired.empty:
        return waiting
    known = set(fired["stock_id"].astype(str))
    waiting = waiting[~waiting["stock_id"].astype(str).isin(known)]
    return pd.concat([fired, waiting], ignore_index=True)


def holding_on(hits: pd.DataFrame, day) -> pd.DataFrame:
    """這一天**手上還抱著**的部位：買進日 <= 當天 < 賣出日。

    未結束的交易 `sell_date` 是資料尾端，當天之後一律算還抱著。
    """
    if hits.empty or "buy_date" not in hits.columns:
        return pd.DataFrame()
    d = pd.Timestamp(day)
    held = hits[(hits["buy_date"] <= d) & (hits["sell_date"] > d)]
    return held.sort_values("buy_date").reset_index(drop=True)
