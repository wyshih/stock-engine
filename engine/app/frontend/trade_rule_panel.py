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
