"""
台股智能預測系統 - Streamlit 介面

執行：streamlit run streamlit_app.py

2026-08-12 改版：委員會 + Meta stacking 已拆除，改成「單一 ground truth
`label_up20` + 可選模型」。
2026-08-14：MLP / LSTM / 兩種集成移除，只剩 RF × 2 個訓練期（Round 1 / Round 2）。
側欄選模型，各頁共用。

分數門檻的滑桿範圍**依所選模型自動調整**：RF 的分數擠在基準率附近（最高只到
0.6 上下），固定 0~100% 的滑桿不能用。
範圍與滑桿旁的「驗證期勝率 / 平均報酬 / 訊號數」都讀 data/sigcurve_*_val_sel.csv。
"""
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from engine.models import bundle as bundle_mod  # （模型代號、bundle 載入、單日推論）
from engine.paths import DATA_DIR, MODEL_DIR

WATCHLIST_PATH = DATA_DIR / "watchlist.json"

st.set_page_config(page_title="台股預測系統", page_icon="📈", layout="wide")


# ── 快取資料讀取 ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_parquet(name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    p = DATA_DIR / f"{name}.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p, columns=columns) if columns else pd.read_parquet(p)


@st.cache_data(ttl=300)
def load_stock_list() -> pd.DataFrame:
    return load_parquet("stock_list")


HISTORY_SPLITS = ("val_es", "val_sel", "test", "test2", "test3")


@st.cache_data(ttl=300)
def load_score_history(model_key: str) -> pd.DataFrame:
    """該模型在**驗證期／測試期**每天每股的分數（訓練時就存好的分數 parquet）。

    用途：個股歷史預測頁的曲線，以及今日推薦頁的「累積達標天數」。
    ⚠️ 訓練期與 embargo 月份沒有分數，中間會有斷點。測試期之後的新資料則由
    `score_recent.py` 產生的 score_live 檔補上（`make update` 會跑），沒跑過
    的話曲線就會停在測試期最後一天。
    """
    parts = []
    for split in HISTORY_SPLITS:
        p = bundle_mod.score_path(model_key, split)
        if p.exists():
            parts.append(pd.read_parquet(p))

    live_path = bundle_mod.live_score_path(model_key)
    if live_path.exists():
        parts.append(pd.read_parquet(live_path))
    if not parts:
        return pd.DataFrame(columns=["date", "stock_id", "score"])

    hist = pd.concat(parts, ignore_index=True)
    hist["date"] = pd.to_datetime(hist["date"])
    # score_live 與訓練期分數檔的日期可能重疊（切分末端），去重後以 live 為準
    hist = hist.drop_duplicates(subset=["date", "stock_id"], keep="last")
    return hist.sort_values(["stock_id", "date"]).reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def features_stem(model_key: str) -> str:
    """該模型訓練時用的特徵檔（去掉 .parquet，給 `load_parquet()` 用）。

    ⚠️ 一律問 bundle 自己要哪一份，不要寫死 features.parquet。2026-09-02 移除
    v3 家族之前，m3/m8 是用 features_v3.parquet 訓練的；當時這裡對每個模型都載
    features.parquet，那兩個模型有 41% 的欄位被訓練期中位數填掉，前端每天給出的
    是錯的推薦名單 —— 而且不報錯。現在只剩一份特徵檔，這層間接仍然保留。
    """
    return bundle_mod.features_file_for_key(model_key).removesuffix(".parquet")


@st.cache_data(ttl=1800, show_spinner=False)
def cached_scores(model_key: str, date_str: str, features_name: str) -> pd.DataFrame:
    """某模型在某一天的全市場分數，依 (模型, 日期, 特徵檔) 快取。

    切換模型會觸發整頁重跑、需數秒。沒有快取的話每次
    切回看過的模型都要重算一次，使用者會以為點了沒反應而重複點擊。
    features 在函式內載入（它本身也是快取的），不當參數傳 —— 當參數的話
    st.cache_data 要對 3.4M 列的 DataFrame 算雜湊，比重算還慢。
    ⚠️ `features_name` 一定要進快取鍵：不同模型用不同特徵檔，漏掉的話切模型時
    會拿到上一個模型那份特徵算出來的東西。（雖然 model_key 已經決定了特徵檔，
    但明寫出來才不會在之後改動時又被拿掉。）
    """
    from engine.models.predict import get_scores_for_date

    feat = load_parquet(features_name)
    if feat.empty:
        return pd.DataFrame()
    feat["date"] = pd.to_datetime(feat["date"])
    return get_scores_for_date(model_key, date_str, feat)


def scores_for(model_key: str, date_str: str) -> pd.DataFrame:
    """`cached_scores()` 的呼叫端捷徑：特徵檔由 bundle 自己決定。"""
    return cached_scores(model_key, date_str, features_stem(model_key))


@st.cache_data(ttl=1800, show_spinner=False)
def combined_test_scores(model_key: str, splits: tuple[str, ...]) -> Path:
    """把多個測試切分的分數接成一份暫存 parquet，回傳路徑。

    `simulate()` 一次只讀一個分數檔，但 Round 4 的 test 只有 11 個月、test2 只有
    7 個月，分開看區間太短。合併後才能一次回測完整的樣本外期間。
    寫到系統暫存目錄，不碰 data/。
    """
    import tempfile

    frames = []
    for sp in splits:
        p = bundle_mod.score_path(model_key, sp)
        if p.exists():
            frames.append(pd.read_parquet(p))
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(subset=["date", "stock_id"]).sort_values(["date", "stock_id"])

    path = Path(tempfile.gettempdir()) / f"score_alltests_{model_key}.parquet"
    out.to_parquet(path, index=False)
    return path


@st.cache_data(ttl=300)
def live_day_max(model_key: str, date) -> float | None:
    """該模型在某一天的全市場最高分（讀 score_live，沒有就回 None）。

    用來把門檻滑桿的上限撐到當天實際分數之上 —— 驗證期的分數分布未必涵蓋今天。
    """
    p = bundle_mod.live_score_path(model_key)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    day = df[df["date"] == pd.Timestamp(date)]
    return float(day["score"].max()) if not day.empty else None


@st.cache_data(ttl=300)
def load_sigcurve(model_key: str) -> pd.DataFrame:
    """驗證期（val_sel）的門檻曲線：每個門檻點的筆數 / 勝率 / 平均報酬。"""
    return bundle_mod.load_sigcurve(model_key)


@st.cache_data(ttl=300)
def load_liquidity_history() -> pd.DataFrame:
    """每檔股票每天的近20日平均成交金額，用來篩掉流動率太低的股票（今日推薦頁）。"""
    p = load_parquet("price", ["date", "stock_id", "amount"])
    if p.empty:
        return p
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["stock_id", "date"])
    p["avg_amount_20"] = p.groupby("stock_id")["amount"].transform(
        lambda s: s.rolling(20, min_periods=1).mean())
    return p[["date", "stock_id", "avg_amount_20"]]


def compute_threshold_streaks(hist: pd.DataFrame, up_to: pd.Timestamp, thr: float) -> pd.Series:
    """對每檔股票，計算到 up_to 為止，score 連續 >= thr 的天數（中斷就重算）。"""
    h = hist[hist["date"] <= up_to]
    if h.empty:
        return pd.Series(dtype="int64")
    h = h.assign(hit=h["score"] >= thr)

    def _trailing_streak(hits: np.ndarray) -> int:
        n = 0
        for v in hits[::-1]:
            if not v:
                break
            n += 1
        return n

    return h.groupby("stock_id")["hit"].apply(lambda s: _trailing_streak(s.to_numpy()))


# ── 關注股票的停損停利追蹤 ────────────────────────────────────────────────────
# 規則與出場判定都走 backtest.py：回測、對照組、這裡三邊同一份程式，
# 不在前端另外實作一套（見 backtest.py `track_position` 的說明）。
from engine.backtest.backtest import CURRENT_EXIT_RULES as EXIT_RULES, track_position  # noqa: E402


def _model_exit_rule(key: str) -> dict | None:
    """這個模型的出場規則；None＝走 EXIT_RULES（移動停利/停損），既有行為。

    2026-09-05 新增 swing 模型，它的出場是「分數跌回門檻以下就賣」而不是價格
    規則。讀 bundle 的 `exit_rule` 欄位，舊 bundle 沒有這一欄就回 None。
    """
    try:
        from engine.models.bundle import exit_rule, load_by_key
        return exit_rule(load_by_key(key))
    except Exception:
        return None


def _run_backtest(key: str, split: str, score_path, threshold: float, **kw):
    """依模型的出場規則挑回測路徑。回傳 (trades, price)，欄位兩邊相同。

    swing 走 score_exit.simulate_score_exit（分數出場），其餘走 backtest.simulate。
    刻意不在 simulate() 裡加分支 —— 那是 m1 兩個模型共用的唯一路徑。
    """
    rule = _model_exit_rule(key)
    if rule and rule.get("type") == "score":
        from engine.backtest.score_exit import simulate_score_exit
        return simulate_score_exit(
            score_path=score_path, buy_threshold=threshold,
            sell_threshold=rule.get("sell_threshold", 0.20),
            dedup=kw.get("dedup", False),
            date_start=kw.get("date_start"), date_end=kw.get("date_end"))
    from engine.backtest.backtest import simulate
    return simulate(split, score_path=score_path, threshold=threshold, **kw)

_REASON_LABEL = {
    "trail_stop": "移動停利", "take_profit": "停利",
    "stop_loss": "停損", "ma10_stop": "跌破MA10", "ma20_stop": "跌破MA20",
}


@st.cache_data(ttl=300)
def _track(sid: str, buy_date: str, buy_price: float, as_of) -> dict | None:
    return track_position(sid, buy_date, buy_price=buy_price, as_of=as_of)


def _score_exit_status(item: dict, as_of, key: str, sell_th: float) -> str:
    """分數出場模型的持倉狀態（swing 用）。

    這類模型不看價格規則，只看分數：買進之後第一次出現「分數 <= 出場門檻」就賣。
    對它顯示移動停利／停損是誤導 —— 那些規則在它的回測裡根本沒生效。
    """
    sid, buy_date = item["stock_id"], pd.Timestamp(item["buy_date"])
    hist = load_score_history(key)
    if hist.empty:
        return "－（沒有分數資料）"
    mine = hist[(hist["stock_id"] == sid) & (hist["date"] >= buy_date) &
                (hist["date"] <= pd.Timestamp(as_of))].sort_values("date")
    if mine.empty:
        return "－（買進後沒有分數）"
    # 兜底上限跟回測同一個常數 —— 分數一直沒跌破時，回測會在這裡出場，
    # 顯示端不套的話兩邊會對不起來（實測有部位分數 15 個月都沒跌破）。
    from engine.backtest.score_exit import DEFAULT_MAX_HOLD
    if len(mine) > DEFAULT_MAX_HOLD:
        return (f"🔴 抱滿 {DEFAULT_MAX_HOLD} 個交易日上限 "
                f"（{mine.iloc[DEFAULT_MAX_HOLD]['date'].date()}）→ 隔日開盤賣出")
    hit = mine[mine["score"] <= sell_th]
    if not hit.empty:
        row = hit.iloc[0]
        return (f"🔴 分數跌破 {sell_th:.2f}（{row['date'].date()} 分數 "
                f"{row['score']:.3f}）→ 隔日開盤賣出")
    last = mine.iloc[-1]
    return (f"🟢 持倉中・分數 {last['score']:.3f}（{last['date'].date()}），"
            f"跌到 {sell_th:.2f} 以下才賣")


def _exit_status(item: dict, as_of, key: str | None = None) -> str:
    """關注清單某一筆的出場狀態，給損益表當一欄用。

    2026-09-05：模型如果有自己的出場規則（bundle 的 `exit_rule`），就走它的；
    沒有的話維持原本的移動停利／停損追蹤。
    """
    if not item.get("buy_date"):
        return "－（沒填購入日期）"
    rule = _model_exit_rule(key) if key else None
    if rule and rule.get("type") == "score":
        try:
            return _score_exit_status(item, as_of, key, rule.get("sell_threshold", 0.20))
        except Exception as e:                   # 單一筆算不出來不能讓整頁掛掉
            return f"追蹤失敗：{e}"
    try:
        r = _track(item["stock_id"], item["buy_date"], item["buy_price"], as_of)
    except Exception as e:                       # 前端不能因為單一筆算不出來就整頁掛掉
        return f"追蹤失敗：{e}"
    if r is None:
        return "－（查無價格資料）"

    if r["status"] == "exited":
        reason = _REASON_LABEL.get(r["sell_reason"], r["sell_reason"])
        return (f"🔴 {reason} {r['sell_date'].date()} "
                f"@{r['sell_price']:.2f}（{r['return']:+.2%}）")

    if r["trail_armed"]:
        return (f"🟢 持倉中・移動停利已啟動（最高 {r['peak_return']:+.2%}，"
                f"跌破 {r['trail_stop_price']:.2f} 出場）")
    return (f"🟢 持倉中（最高 {r['peak_return']:+.2%}，"
            f"停損價 {r['stop_loss_price']:.2f}）")


# ── 關注股票清單（新增/移除，可選填購買價，存成 json，跨 session 保留）─────────
# 每筆格式：{"stock_id": "2330", "buy_price": 500.0 或 None}

def load_watchlist() -> list[dict]:
    if not WATCHLIST_PATH.exists():
        return []
    raw = json.loads(WATCHLIST_PATH.read_text())
    # 相容舊格式（純字串清單）
    return [{"stock_id": r, "buy_price": None} if isinstance(r, str) else r for r in raw]


def save_watchlist(items: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dedup = {it["stock_id"]: it for it in items}  # 同代號只留最後一筆
    WATCHLIST_PATH.write_text(json.dumps(sorted(dedup.values(), key=lambda x: x["stock_id"]),
                                         ensure_ascii=False, indent=2))


def watchlist_ids() -> list[str]:
    return [it["stock_id"] for it in st.session_state.watchlist]


if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()


# ── 側欄導覽 ──────────────────────────────────────────────────────────────────

# ── 模型選擇（RF × 2 個訓練期）──────────────────────────────────────
# 放在頁面導覽「上方」是刻意的：切模型會觸發數秒的重跑，這段期間下拉選單已經
# 關閉但畫面還沒更新，使用者常會再點一下 —— 選單如果在導覽按鈕正上方，那一下
# 就會穿透到導覽、整個頁面跳走。把它移到最上面，穿透的點擊落不到導覽上。
st.sidebar.subheader("🧠 預測模型")
_ready_keys = bundle_mod.available_keys()
if _ready_keys:
    model_key = st.sidebar.selectbox(
        "選擇模型", _ready_keys, format_func=bundle_mod.model_label, key="model_key")
    st.sidebar.caption(f"代號 `{model_key}`")
else:
    model_key = None
    st.sidebar.error("找不到任何 model bundle，請先執行\n"
                     "`make train`")
st.sidebar.divider()

PAGES = ["今日推薦", "訊號清單", "個股歷史預測", "技術面分析", "型態規則", "買賣點規則",
         "資料預覽", "特徵預覽", "回測結果", "模型成效", "波段模型"]
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]
for _p in PAGES:
    if st.sidebar.button(_p, use_container_width=True,
                         type="primary" if st.session_state.page == _p else "secondary"):
        st.session_state.page = _p
        st.rerun()
page = st.session_state.page
st.sidebar.divider()

st.sidebar.subheader("⭐ 關注股票")
sl_for_watch = load_stock_list()
all_stock_ids = sorted(sl_for_watch["stock_id"].unique().tolist()) if not sl_for_watch.empty else []

def _stock_label(sid: str) -> str:
    if sl_for_watch.empty:
        return sid
    name = sl_for_watch.loc[sl_for_watch["stock_id"] == sid, "stock_name"]
    return f"{sid} {name.values[0]}" if len(name) else sid

add_candidates = [s for s in all_stock_ids if s not in watchlist_ids()]
if add_candidates:
    pick_add = st.sidebar.selectbox("新增股票代號", add_candidates, format_func=_stock_label,
                                    key="watch_add_pick")
    has_price = st.sidebar.checkbox("填入購入價格（選填）", key="watch_add_has_price")
    buy_price_input = (st.sidebar.number_input("購入價格", min_value=0.0, step=0.1,
                                               key="watch_add_price")
                        if has_price else None)
    # 有進場日才有辦法往後跑出場規則，所以停損停利追蹤只對有填日期的部位顯示
    buy_date_input = (st.sidebar.date_input("購入日期（填了才會追蹤停損停利）",
                                            value=None, key="watch_add_date")
                      if has_price else None)
    if st.sidebar.button("➕ 加入關注", use_container_width=True):
        st.session_state.watchlist = st.session_state.watchlist + [
            {"stock_id": pick_add, "buy_price": buy_price_input,
             "buy_date": str(buy_date_input) if buy_date_input else None}]
        save_watchlist(st.session_state.watchlist)
        st.rerun()

if st.session_state.watchlist:
    st.sidebar.caption(f"目前關注 {len(st.session_state.watchlist)} 檔：")
    for it in list(st.session_state.watchlist):
        sid = it["stock_id"]
        c1, c2 = st.sidebar.columns([4, 1])
        label = _stock_label(sid)
        if it.get("buy_price"):
            label += f"（購入 {it['buy_price']:.2f}"
            label += f" @ {it['buy_date']}）" if it.get("buy_date") else "）"
        c1.write(label)
        if c2.button("✕", key=f"watch_remove_{sid}"):
            st.session_state.watchlist = [x for x in st.session_state.watchlist if x["stock_id"] != sid]
            save_watchlist(st.session_state.watchlist)
            st.rerun()

    # ── 損益查詢：只對有填購入價的關注股票顯示 ─────────────────────────────
    priced = [it for it in st.session_state.watchlist if it.get("buy_price")]
    if priced:
        with st.sidebar.expander("📊 查看損益"):
            _price_all = load_parquet("price", ["date", "stock_id", "close"])
            _price_all["date"] = pd.to_datetime(_price_all["date"])
            _avail_d = sorted(_price_all["date"].dt.date.unique())
            pnl_date = st.date_input("計算到哪一天", value=_avail_d[-1],
                                     min_value=_avail_d[0], max_value=_avail_d[-1],
                                     key="watch_pnl_date")
            rows = []
            for it in priced:
                sid, buy_price = it["stock_id"], it["buy_price"]
                p = _price_all[(_price_all["stock_id"] == sid) &
                               (_price_all["date"] <= pd.Timestamp(pnl_date))]
                if p.empty:
                    continue
                cur_close = p.sort_values("date").iloc[-1]["close"]
                pnl_pct = (cur_close - buy_price) / buy_price
                rows.append({"股票": _stock_label(sid), "購入價": buy_price,
                            "現價": cur_close, "損益%": pnl_pct,
                            "出場狀態": _exit_status(it, pnl_date, model_key)})
            if rows:
                pnl_df = pd.DataFrame(rows)
                pnl_df["損益%"] = pnl_df["損益%"].map("{:+.2%}".format)
                st.dataframe(pnl_df, use_container_width=True, hide_index=True)
                _r = _model_exit_rule(model_key) if model_key else None
                if _r and _r.get("type") == "score":
                    st.caption(
                        f"出場規則：**分數跌回 {_r.get('sell_threshold', 0.20):.2f} 以下就賣**"
                        f"（{bundle_mod.model_label(model_key)} 專用，與回測頁同一套）。"
                        "移動停利／停損對這個模型不生效。沒填購入日期的不會追蹤。")
                else:
                    st.caption(
                        f"出場規則：移動停利 獲利 {EXIT_RULES['trail_trigger']:.0%} 觸發、"
                        f"回落 {EXIT_RULES['trail_pct']:.0%} 出場；固定停損 "
                        f"{EXIT_RULES['stop_loss']:.0%}（與回測頁預設同一套）。"
                        "沒填購入日期的不會追蹤。")
else:
    st.sidebar.caption("目前沒有關注股票")

st.sidebar.divider()
st.sidebar.caption("⚠️ 回測未納入手續費及已下市股票，存在生存偏差")


# ═══════════════════════════════════════════════════════════════════════════════
# 頁面 1：今日推薦
# ═══════════════════════════════════════════════════════════════════════════════

if page == "今日推薦":
    st.title("📈 今日推薦名單")

    # 這一頁只需要日期清單與兩個型態濾網欄位。推論需要的完整特徵在
    # cached_scores() 內部才載入 —— 整張表 394 欄 2GB，每次重跑都拉進來的話，
    # 光是 st.cache_data 回傳前的複製就要 2~3 秒，切模型會明顯卡住
    feat = load_parquet("features", ["date", "stock_id", "bull_3ma_1d", "bear_3ma_3d"])
    if feat.empty:
        st.error("features.parquet 不存在，請先執行 build_features.py")
        st.stop()

    feat["date"] = pd.to_datetime(feat["date"])
    latest = feat["date"].max()
    st.caption(f"最新資料日期：{latest.date()}")

    if model_key is None:
        st.warning("模型尚未訓練，請先執行 `make train`")
        st.stop()

    try:
        from engine.models.predict import get_scores_for_date

        curve = load_sigcurve(model_key)
        # 滑桿範圍依所選模型自動調整：用該模型在驗證期（val_sel）實際出現過的
        # 分數範圍（門檻曲線的兩端）。不同模型的分數尺度差很多——RF 最高只到
        # 0.60、LSTM 到 0.89、排序平均到 1.0——固定 0~100% 的滑桿會不能用。
        thr_lo = float(curve["threshold"].min()) if not curve.empty else 0.0
        thr_hi = float(curve["threshold"].max()) if not curve.empty else 1.0
        default_thr = min(max(bundle_mod.default_threshold(model_key, curve), thr_lo), thr_hi)

        avail_dates = sorted(feat["date"].dt.date.unique())
        st.caption(f"模型：**{bundle_mod.model_label(model_key)}**"
                   f"（ground truth = label_up20，見 doc/PLAN.md）")

        col0, col1, col2 = st.columns([2, 1, 3])
        with col0:
            picked = st.date_input("預測日期", value=avail_dates[-1],
                                   min_value=avail_dates[0], max_value=avail_dates[-1])
            # 月曆可能選到非交易日（假日），自動取「不晚於選定日」的最近交易日
            candidates = [d for d in avail_dates if d <= picked]
            pick_date = candidates[-1] if candidates else avail_dates[-1]
            if pick_date != picked:
                st.caption(f"{picked} 非交易日，改用最近交易日 {pick_date}")
        # 當天實際分數可能整體高於驗證期（分布會漂移），這時滑桿上限若停在驗證期
        # 的最高點，就會出現「拉到底仍有幾百檔在門檻之上」而篩不動。上限改成
        # 兩者取大，超出驗證期的那一段沒有勝率/報酬可參考，會另外標示。
        day_hi = live_day_max(model_key, pick_date)
        slider_hi = max(thr_hi, day_hi) if day_hi else thr_hi
        with col1:
            thr = st.slider("分數門檻", thr_lo, slider_hi, default_thr,
                            (slider_hi - thr_lo) / 200 or 0.001, format="%.4f")
            top_n = st.number_input("最多顯示幾支", 5, 200, 30)
            stats = bundle_mod.sigcurve_stats_at(curve, thr)
            if thr > thr_hi:
                st.caption(f"⚠️ 此門檻高於驗證期最高分（{thr_hi:.4f}），"
                           "沒有驗證統計可參考")
            elif stats:
                st.caption(
                    f"此門檻在**驗證期**：{stats['n']:,} 筆訊號、"
                    f"勝率 {stats['win_rate']:.1%}、平均報酬 {stats['avg_return']:+.2%}")
            else:
                st.caption("此門檻在驗證期沒有任何訊號")

        price_df = load_parquet("price", ["date", "stock_id", "close"])
        price_df["date"] = pd.to_datetime(price_df["date"])
        day_price = price_df[price_df["date"] == pd.Timestamp(pick_date)][["stock_id", "close"]]
        max_close = float(day_price["close"].max()) if not day_price.empty else 2000.0

        with col2:
            price_range = st.slider("股價區間（元）", 0.0, max(max_close, 10.0),
                                    (0.0, max(max_close, 10.0)), 0.5)
            watch_only = st.checkbox(f"只看關注清單（{len(st.session_state.watchlist)}檔）",
                                     disabled=not st.session_state.watchlist)
            min_liquidity_wan = st.slider("最低近20日均成交金額（萬元）", 0, 5000, 500, 50)
            fresh_only = st.checkbox("只看突破三天（連續達標剛好3天）")
            not_bear_recent = st.checkbox("前三天不是空頭排列（近3個交易日不是連續MA5<MA10<MA20）")
            bull_today = st.checkbox("今天是多頭排列（MA5>MA10>MA20）")

        with st.spinner(f"{bundle_mod.model_label(model_key)} 計算中..."):
            # 先拿到全部候選（不篩門檻），過濾股價後才依門檻/顯示數量截取，
            # 避免「先截斷再過濾」導致篩選後剩沒幾支
            all_probs = scores_for(model_key, str(pick_date))
            if all_probs.empty:
                st.warning(f"{pick_date} 這一天算不出分數（無特徵資料）")
                st.stop()
            result = all_probs[all_probs["score"] >= thr].copy()
            if not result.empty:
                result = result.merge(
                    load_stock_list()[["stock_id", "stock_name", "industry"]],
                    on="stock_id", how="left")

        if not result.empty and "close" in result.columns:
            # get_probs_for_date() 已經內附當天收盤價（用來算建議價格區間），
            # 不用再跟 day_price 重複 merge
            result = result[result["close"].between(price_range[0], price_range[1])]

        if watch_only and not result.empty:
            result = result[result["stock_id"].isin(watchlist_ids())]

        if min_liquidity_wan > 0 and not result.empty:
            liq_hist = load_liquidity_history()
            if not liq_hist.empty:
                liq_asof = (liq_hist[liq_hist["date"] <= pd.Timestamp(pick_date)]
                            .sort_values("date").groupby("stock_id").tail(1)
                            .set_index("stock_id")["avg_amount_20"])
                liq_vals = result["stock_id"].map(liq_asof)
                # price.parquet 的 amount(成交金額)欄位只從2026-06-29才開始有值，
                # 更早的日期 liq_vals 會是 NaN——這代表「沒有流動性資料」，不是
                # 「流動性等於0」，NaN一律放行(不篩掉)，避免歷史日期被誤篩到只剩0檔。
                no_liq_data = liq_vals.isna().all() and not result.empty
                result = result[liq_vals.isna() | (liq_vals >= min_liquidity_wan * 10_000)]
                if no_liq_data:
                    st.caption("⚠️ 這個日期沒有成交金額歷史資料（只從2026-06-29起才有），"
                               "流動性篩選本次未生效")

        if bull_today and not result.empty:
            # 今天是多頭排列：MA5>MA10>MA20（bull_3ma_1d==1）
            bull_flag = (feat[feat["date"] == pd.Timestamp(pick_date)]
                         .set_index("stock_id")["bull_3ma_1d"])
            result = result[result["stock_id"].map(bull_flag).fillna(0).astype(bool)]

        if not_bear_recent and not result.empty:
            # 前三天不是空頭排列：近3個交易日(t-1,t-2,t-3)不是連續MA5<MA10<MA20。
            # bear_3ma_3d 在 t-1 那天=1 代表「以 t-1 為終點，已連續空頭排列>=3天」，
            # 也就是 t-1/t-2/t-3 這三天都是空頭排列，我們要排除這種情況。
            prev_candidates = [d for d in avail_dates if d < pick_date]
            if prev_candidates:
                prev_date = prev_candidates[-1]
                bear3_flag = (feat[feat["date"] == pd.Timestamp(prev_date)]
                              .set_index("stock_id")["bear_3ma_3d"])
                result = result[~result["stock_id"].map(bear3_flag).fillna(0).astype(bool)]

        streak_available = False
        if not result.empty:
            # 累積達標天數用「已存檔的分數歷史」算。分數檔只涵蓋驗證期／測試期，
            # 最新一段（2026-07 起）沒有分數檔，那些日期一律顯示 0 並標註。
            hist = load_score_history(model_key)
            # 必須是「所選日期當天有分數」，不能只看歷史最大日期 —— 分數歷史中間
            # 是有斷層的（訓練期、embargo 月份都沒有分數），用 max() 判斷會讓
            # 落在斷層裡的日期被誤判成有資料，連續達標天數全變 0 卻沒有任何提示
            streak_available = (not hist.empty
                                and (hist["date"] == pd.Timestamp(pick_date)).any())
            if streak_available:
                streaks = compute_threshold_streaks(hist, pd.Timestamp(pick_date), thr)
                result["streak_days"] = result["stock_id"].map(streaks).fillna(0).astype(int)
                if fresh_only:
                    result = result[result["stock_id"].map(streaks).fillna(0) == 3]
            else:
                result["streak_days"] = 0
                # 沒有分數歷史就算不出連續達標天數，這個濾網等於沒作用。
                # 之前只在表格下方提「天數顯示 0」，不會告訴使用者勾選失效了
                if fresh_only:
                    st.warning("「只看突破三天」這次沒有生效：這個日期還沒有存檔的"
                               "分數歷史，算不出連續達標天數。請先執行 "
                               "`make update`（會跑 score_recent.py 補分數）")
            result = result.sort_values("score", ascending=False)

        n_matched = len(result)          # 截斷前的實際符合檔數
        result = result.head(top_n) if not result.empty else result

        if result.empty:
            st.info(f"門檻 {thr:.4f}、股價 {price_range[0]:.0f}~{price_range[1]:.0f} 元 下無推薦股票")
        else:
            # 只報顯示筆數的話，門檻在低檔移動時數字會卡在 top_n 不動，
            # 看起來像滑桿沒作用（實際候選可能從幾百變幾十）
            if n_matched > len(result):
                st.success(f"共 {n_matched} 支符合條件，顯示分數最高的 {len(result)} 支"
                           "（可調整右上「最多顯示幾支」）")
            else:
                st.success(f"共推薦 {len(result)} 支股票")
            # 共識標記：實測「兩個模型都看好」的標的品質明顯較高
            # （BACKTEST_LOG #27：共識 +7.87% > 單一模型獨有 +2.04~4.79% > 全買 +3.21%）
            others = [k for k in _ready_keys if k != model_key]
            if others:
                peer = st.selectbox("對照模型（標記雙方都看好的標的）", ["（不對照）"] + others,
                                    format_func=lambda k: k if k == "（不對照）"
                                    else bundle_mod.model_label(k), key="peer_model")
                if peer != "（不對照）":
                    peer_scores = scores_for(peer, str(pick_date))
                    peer_thr = bundle_mod.default_threshold(peer, load_sigcurve(peer))
                    ok = set(peer_scores[peer_scores["score"] >= peer_thr]["stock_id"])
                    result["共識"] = result["stock_id"].map(lambda s: "✅" if s in ok else "")
                    n_both = int((result["共識"] == "✅").sum())
                    st.caption(f"對照 **{bundle_mod.model_label(peer)}**（門檻 {peer_thr:.4f}）："
                               f"顯示的 {len(result)} 支中有 **{n_both}** 支雙方都看好。"
                               f"實測共識標的報酬明顯較高（見 doc/BACKTEST_LOG.md #27）")

            if "buy_price_low" in result.columns and "buy_price_high" in result.columns:
                result["buy_range"] = result.apply(
                    lambda r: f"{r['buy_price_low']:.2f} ~ {r['buy_price_high']:.2f}"
                    if pd.notna(r["buy_price_low"]) else "N/A", axis=1)
            show_cols = ["共識", "stock_id", "stock_name", "industry", "close", "score",
                         "buy_range", "observe_days", "streak_days"]
            show_cols = [c for c in show_cols if c in result.columns]

            result_show = result[show_cols].copy()
            result_show["score"] = result_show["score"].map("{:.4f}".format)
            rename_map = {
                "close": "股價", "streak_days": "累積達標天數", "score": "分數",
                "buy_range": "建議買進區間", "observe_days": "觀察天數",
            }
            result_show = result_show.rename(columns=rename_map)

            # 表格可直接點列跳轉（Streamlit 1.35+ 的 on_select）。
            # ⚠️ 這條路徑無法用 AppTest 驗證（Dataframe 測試物件不支援模擬選取），
            # 所以下面另外保留一組 selectbox + 按鈕作為保底，兩者都能跳。
            _ev = st.dataframe(result_show, use_container_width=True, hide_index=True,
                               on_select="rerun", selection_mode="single-row",
                               key="reco_table")
            try:
                _rows = list(_ev.selection.rows)
            except Exception:
                _rows = []
            if _rows:
                _sid = result.iloc[_rows[0]]["stock_id"]
                # 防止跳轉後回到本頁時舊選取又觸發一次
                if st.session_state.get("hist_jump") != (_sid, pick_date):
                    st.session_state["hist_jump"] = (_sid, pick_date)
                    st.session_state["page"] = "個股歷史預測"
                    st.rerun()

            # 保底：表格點不動時用這組
            jc1, jc2 = st.columns([3, 1])
            _opts = list(result["stock_id"])
            _nm = dict(zip(result["stock_id"], result.get("stock_name", result["stock_id"])))
            with jc1:
                _pick = st.selectbox("看走勢圖", _opts,
                                     format_func=lambda x: f"{x} {_nm.get(x, '')}",
                                     key="reco_jump_pick")
            with jc2:
                st.write("")
                if st.button("跳轉 →", key="reco_jump_btn", use_container_width=True):
                    st.session_state["hist_jump"] = (_pick, pick_date)
                    st.session_state["page"] = "個股歷史預測"
                    st.rerun()
            st.caption("建議買進區間 = 訊號當天收盤價 ±2%（僅供參考，非精確買賣點）；"
                       "觀察天數 = ground truth `label_up20` 的判定視窗（20 個交易日），"
                       "意思是這個分數是「未來 20 個交易日」的預測，不代表要抱滿 20 天")
            if not streak_available:
                st.caption("⚠️ 這個日期還沒有存檔的分數歷史（分數檔只到各測試期結束），"
                           "「累積達標天數」一律顯示 0")

    except Exception as e:
        st.error(f"預測失敗：{e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 頁面 2：個股歷史預測
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 頁面：訊號清單（所有超過門檻的紀錄，可點擊跳到該股票該時期）
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "訊號清單":
    st.title("📋 訊號清單")
    st.caption("該模型歷史上所有「分數 ≥ 門檻」的紀錄。點選任一列可跳到那檔股票的那段走勢。")

    if model_key is None:
        st.warning("模型尚未訓練")
        st.stop()

    hist = load_score_history(model_key)
    if hist.empty:
        st.warning(f"{bundle_mod.model_label(model_key)} 沒有存檔的分數歷史")
        st.stop()

    curve = load_sigcurve(model_key)
    thr_default = bundle_mod.default_threshold(model_key, curve)
    lo, hi = float(hist["score"].min()), float(hist["score"].max())
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        thr = st.slider("分數門檻", lo, hi, min(max(thr_default, lo), hi),
                        (hi - lo) / 200 or 0.001, format="%.4f", key="siglist_thr")
    avail = sorted(hist["date"].dt.date.unique())
    with c2:
        d_from = st.date_input("起", value=avail[max(0, len(avail) - 250)],
                               min_value=avail[0], max_value=avail[-1], key="siglist_from")
    with c3:
        d_to = st.date_input("迄", value=avail[-1],
                             min_value=avail[0], max_value=avail[-1], key="siglist_to")

    sig = hist[(hist["score"] >= thr) &
               (hist["date"] >= pd.Timestamp(d_from)) &
               (hist["date"] <= pd.Timestamp(d_to))].copy()
    if sig.empty:
        st.info(f"門檻 {thr:.4f} 在此期間沒有任何訊號")
        st.stop()

    # 附上股名與訊號當天收盤價
    sl = load_stock_list()
    if not sl.empty:
        sig = sig.merge(sl[["stock_id", "stock_name", "industry"]], on="stock_id", how="left")
    price = load_parquet("price", ["date", "stock_id", "close"])
    price["date"] = pd.to_datetime(price["date"])
    sig = sig.merge(price, on=["date", "stock_id"], how="left")

    sig = sig.sort_values(["date", "score"], ascending=[False, False]).reset_index(drop=True)
    st.success(f"共 {len(sig):,} 筆訊號　"
               f"（{sig['date'].min().date()} ~ {sig['date'].max().date()}，"
               f"{sig['stock_id'].nunique():,} 檔股票）")

    show = sig.copy()
    show["日期"] = show["date"].dt.date
    show["分數"] = show["score"].map("{:.4f}".format)
    cols = [c for c in ["日期", "stock_id", "stock_name", "industry", "close", "分數"]
            if c in show.columns]
    view = show[cols].rename(columns={"stock_id": "代號", "stock_name": "名稱",
                                      "industry": "產業", "close": "收盤"})

    # 列選取 → 跳到個股歷史預測頁的那段期間
    LIMIT = 3000        # 一次最多渲染這麼多列，避免瀏覽器卡住
    if len(view) > LIMIT:
        st.caption(f"⚠️ 只顯示最新 {LIMIT:,} 筆（縮小日期區間或提高門檻可看到更早的）")
    # 表格可直接點列跳轉；無法自動驗證，故下方另有 selectbox + 按鈕保底
    _ev = st.dataframe(view.head(LIMIT), use_container_width=True, hide_index=True,
                       on_select="rerun", selection_mode="single-row", key="siglist_table")
    try:
        _rows = list(_ev.selection.rows)
    except Exception:
        _rows = []
    if _rows:
        _r = sig.iloc[_rows[0]]
        _tgt = (_r["stock_id"], _r["date"].date())
        if st.session_state.get("hist_jump") != _tgt:
            st.session_state["hist_jump"] = _tgt
            st.session_state["page"] = "個股歷史預測"
            st.rerun()

    # 保底：selectbox 可打字搜尋，選項是「日期 代號 名稱」
    PICK_N = 500
    head = sig.head(PICK_N)
    labels = [f"{r['date'].date()}　{r['stock_id']} "
              f"{r.get('stock_name', '')}　分數 {r['score']:.4f}" for _, r in head.iterrows()]
    jc1, jc2 = st.columns([3, 1])
    with jc1:
        idx = st.selectbox(f"看走勢圖（最新 {min(PICK_N, len(sig))} 筆，可打字搜尋）",
                           range(len(labels)), format_func=lambda i: labels[i],
                           key="siglist_jump_pick")
    with jc2:
        st.write("")
        if st.button("跳轉 →", key="siglist_jump_btn", use_container_width=True):
            r = head.iloc[idx]
            st.session_state["hist_jump"] = (r["stock_id"], r["date"].date())
            st.session_state["page"] = "個股歷史預測"
            st.rerun()

    st.caption("跳過去之後日期會自動對準訊號日前後各 40 個交易日，"
               "想自由瀏覽就按那裡的「清除跳轉」。")


elif page == "個股歷史預測":
    st.title("🔍 個股歷史預測")
    st.caption("查看某一檔股票在指定區間內，每天的模型分數、實際股價與 20 天後報酬")

    if model_key is None:
        st.warning("模型尚未訓練，請先執行 `make train`")
        st.stop()

    hist_all = load_score_history(model_key)
    if hist_all.empty:
        st.warning(f"{bundle_mod.model_label(model_key)} 沒有存檔的分數歷史"
                   f"（找不到 data/score_*_<split>.parquet）")
        st.stop()
    st.caption(f"模型：**{bundle_mod.model_label(model_key)}**；"
               f"分數歷史涵蓋 {hist_all['date'].min().date()} ~ "
               f"{hist_all['date'].max().date()}"
               f"（只有各驗證期／測試期有分數，訓練期與 embargo 月份沒有）")

    sl = load_stock_list()
    stocks = sorted(hist_all["stock_id"].unique().tolist())
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        # 由「訊號清單」頁點擊跳轉過來時，預先選好該檔股票
        _jump = st.session_state.get("hist_jump")
        if _jump and _jump[0] in stocks:
            st.session_state["hist_sid"] = _jump[0]
        sid = st.selectbox("股票代號", stocks, key="hist_sid",
                           format_func=lambda s: f"{s} {sl[sl['stock_id']==s]['stock_name'].values[0]}"
                           if not sl.empty and (sl['stock_id']==s).any() else s)
        if sid in watchlist_ids():
            if st.button("★ 移除關注", key="watch_toggle"):
                st.session_state.watchlist = [x for x in st.session_state.watchlist if x["stock_id"] != sid]
                save_watchlist(st.session_state.watchlist)
                st.rerun()
        else:
            if st.button("☆ 加入關注（無購入價，可到側邊欄補填）", key="watch_toggle"):
                st.session_state.watchlist = st.session_state.watchlist + [
                    {"stock_id": sid, "buy_price": None}]
                save_watchlist(st.session_state.watchlist)
                st.rerun()
    hist_sid = hist_all[hist_all["stock_id"] == sid]
    avail = sorted(hist_sid["date"].dt.date.unique())
    if not avail:
        st.warning("此股票沒有分數歷史")
        st.stop()
    # 預設看「全部」而不是最近 60 天：分數歷史涵蓋兩年多（含 2025 整年的測試期），
    # 只給最近 60 天的話一打開只看得到最近三個月，得手動改日期才找得到測試期
    RANGE_DAYS = {"最近60天": 60, "最近一年": 250, "全部": len(avail)}
    jump = st.session_state.get("hist_jump")
    if jump and jump[0] == sid:
        # 跳轉模式：把視窗對準訊號日前後各 40 個交易日，看得到訊號前的醞釀與之後的走勢
        jd = jump[1]
        idx = min(range(len(avail)), key=lambda i: abs((avail[i] - jd).days))
        lo_i, hi_i = max(0, idx - 40), min(len(avail) - 1, idx + 40)
        c1, c2 = st.columns([3, 1])
        c1.info(f"已跳轉至 **{sid}** 的訊號日 **{jd}**（視窗為前後各 40 個交易日）")
        if c2.button("清除跳轉，自由瀏覽"):
            del st.session_state["hist_jump"]
            st.rerun()
        default_start, default_end = avail[lo_i], avail[hi_i]
    else:
        span = st.radio("顯示範圍", list(RANGE_DAYS), index=2, horizontal=True,
                        key="hist_range")
        default_start = avail[max(0, len(avail) - RANGE_DAYS[span])]
        default_end = avail[-1]
    with col_b:
        d_start = st.date_input("起始日期", value=default_start,
                                min_value=avail[0], max_value=avail[-1])
    with col_c:
        d_end = st.date_input("結束日期", value=default_end,
                              min_value=avail[0], max_value=avail[-1])

    # 多選模型：把每個模型的分數線疊在同一張圖上，直接看誰在漲之前先亮。
    # 側欄換模型時要跟著重設 —— multiselect 有固定 key，Streamlit 會記住上次的
    # 選擇並忽略 default，不同步的話使用者在側欄切到某個模型、圖上卻還是舊的那條線
    if st.session_state.get("_hist_last_model") != model_key:
        st.session_state["hist_compare_models"] = [model_key]
        st.session_state["_hist_last_model"] = model_key
    compare_keys = st.multiselect(
        "要比較哪些模型（可複選）", _ready_keys, default=[model_key],
        format_func=bundle_mod.model_label, key="hist_compare_models")
    if not compare_keys:
        st.info("請至少選一個模型")
        st.stop()

    if st.button("查詢", type="primary"):
        if d_start > d_end:
            st.warning("起始日期不能晚於結束日期")
            st.stop()

        lo_ts, hi_ts = pd.Timestamp(d_start), pd.Timestamp(d_end)
        series_by_model = {}
        for mk in compare_keys:
            h = load_score_history(mk)
            h = h[(h["stock_id"] == sid) & (h["date"] >= lo_ts) & (h["date"] <= hi_ts)]
            if not h.empty:
                series_by_model[mk] = h[["date", "score"]].sort_values("date")
        if not series_by_model:
            st.info("此區間查無分數")
            st.stop()

        # 實際股價 + 20 天後報酬（買進=t+1收盤，20天後=t+21收盤，
        # 跟 ground truth `label_up20` 的 20 個交易日視窗一致）
        price_df = load_parquet("price", ["date", "stock_id", "open", "high", "low", "close"])
        price_df["date"] = pd.to_datetime(price_df["date"])
        p_sid = price_df[price_df["stock_id"] == sid].sort_values("date").reset_index(drop=True)
        p_sid["ret_20"] = (p_sid["close"].shift(-21) - p_sid["close"].shift(-1)) / p_sid["close"].shift(-1)
        p_win = p_sid[(p_sid["date"] >= lo_ts) & (p_sid["date"] <= hi_ts)]

        merged = p_win[["date", "open", "high", "low", "close", "ret_20"]].copy()
        for mk, s in series_by_model.items():
            merged = merged.merge(s.rename(columns={"score": mk}), on="date", how="left")

        st.subheader("股價（K棒）與各模型分數（雙軸疊圖）")
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=merged["date"], open=merged["open"], high=merged["high"],
            low=merged["low"], close=merged["close"],
            increasing_line_color="#d64545", decreasing_line_color="#2f9e44",
            name="股價", showlegend=False, yaxis="y",
        ))
        # 每個模型一條線，直接比「誰在漲之前先亮」
        _PALETTE = ["#4263eb", "#e8590c", "#7048e8", "#0ca678", "#c2255c"]
        lo_all, hi_all = [], []
        for i, mk in enumerate(compare_keys):
            s = series_by_model.get(mk)
            if s is None or s.empty:
                continue
            colour = _PALETTE[i % len(_PALETTE)]
            fig.add_trace(go.Scatter(
                x=s["date"], y=s["score"], mode="lines",
                line=dict(color=colour, width=2),
                name=bundle_mod.model_label(mk), yaxis="y2",
            ))
            thr_val = bundle_mod.default_threshold(mk, load_sigcurve(mk))
            fig.add_hline(y=thr_val, line_dash="dot", line_color=colour, opacity=0.5,
                          yref="y2",
                          annotation_text=f"{mk} 門檻 {thr_val:.3f}",
                          annotation_font_color=colour)
            lo_all += [float(s["score"].min()), thr_val]
            hi_all += [float(s["score"].max()), thr_val]

        # 分數軸貼合實際範圍，不要從 0 起跳 —— RF 的分數擠在基準率附近（約
        # 0.3~0.6），畫在 0~0.6 的軸上會被壓成一條平線，看不出哪天有反應
        pad = (max(hi_all) - min(lo_all)) * 0.12 or 0.01
        fig.update_layout(
            height=560, xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", y=1.05),
            yaxis=dict(title="股價", side="left"),
            yaxis2=dict(title="分數", side="right", overlaying="y",
                        range=[min(lo_all) - pad, max(hi_all) + pad], showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("明細")
        show = merged.copy()
        for mk in series_by_model:
            show[mk] = show[mk].map(lambda v: f"{v:.4f}" if pd.notna(v) else "－")
        show["ret_20"] = show["ret_20"].map(lambda v: f"{v:.2%}" if pd.notna(v) else "N/A")
        display_cols = ["date", "close", "ret_20"] + list(series_by_model)
        st.dataframe(show[[c for c in display_cols if c in show.columns]],
                     use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 頁面 4：技術面分析（純技術面，不用模型）
#
# 版面與敘述都在 code/frontend/ 底下的模組，這裡只做導覽掛載 —— 這個檔已經超過
# 1200 行，新頁面再往裡面塞會失控。
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "技術面分析":
    from engine.app.frontend import technical_page

    technical_page.render(load_stock_list)


# ═══════════════════════════════════════════════════════════════════════════════
# 頁面：型態規則（fpm 專案挖出的平盤起漲點規則，樣本外命中買點瀏覽）
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "型態規則":
    from engine.app.frontend import rule_screener

    rule_screener.render(load_stock_list)


# ═══════════════════════════════════════════════════════════════════════════════
# 頁面 5：資料預覽
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "資料預覽":
    st.title("🗄️ 資料預覽")

    tables = {
        "price":       ["date", "stock_id", "open", "high", "low", "close", "volume"],
        "chip":        ["date", "stock_id", "foreign_net", "trust_net", "dealer_net",
                        "margin_balance", "short_balance"],
        "fundamental": ["date", "stock_id", "per", "pbr", "dividend_yield"],
        "revenue":     ["announce_date", "stock_id", "revenue", "revenue_month", "revenue_year"],
        "stock_list":  None,
    }

    tab_names = list(tables.keys())
    tabs = st.tabs(tab_names)

    for tab, name in zip(tabs, tab_names):
        with tab:
            df = load_parquet(name, tables[name])
            if df.empty:
                st.warning(f"{name}.parquet 尚不存在")
                continue

            date_col = "date" if "date" in df.columns else ("announce_date" if "announce_date" in df.columns else None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])

            # 摘要
            c1, c2, c3 = st.columns(3)
            c1.metric("總筆數", f"{len(df):,}")
            if "stock_id" in df.columns:
                c2.metric("股票數", f"{df['stock_id'].nunique():,}")
            if date_col:
                c3.metric("日期範圍", f"{df[date_col].min().date()} ~ {df[date_col].max().date()}")

            # 篩選
            sl = load_stock_list()
            if not sl.empty and "stock_id" in df.columns:
                options = ["全部"] + sorted(df["stock_id"].unique().tolist())
                selected = st.selectbox(f"篩選股票 ({name})", options, key=f"sel_{name}")
                if selected != "全部":
                    df = df[df["stock_id"] == selected]

            st.dataframe(df.head(15), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 頁面 3：特徵預覽
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "特徵預覽":
    st.title("🔬 特徵預覽")

    feat = load_parquet("features")
    if feat.empty:
        st.warning("features.parquet 尚不存在")
        st.stop()

    feat["date"] = pd.to_datetime(feat["date"])

    sl = load_stock_list()
    stocks = sorted(feat["stock_id"].unique().tolist())
    sid = st.selectbox("選擇股票", stocks)
    dates = sorted(feat[feat["stock_id"] == sid]["date"].unique())
    date = st.select_slider("選擇日期", dates, value=dates[-1] if dates else None)

    row = feat[(feat["stock_id"] == sid) & (feat["date"] == date)]
    if row.empty:
        st.warning("無資料")
    else:
        row = row.iloc[0]
        # price_shape_/volume_shape_ 這 14 欄在 2026-08-05 停止計算（KMeans 群心
        # 用到未來資料），新日期一律 NaN，模型也沒用到。欄位還留在 features.parquet
        # 裡，但沒必要在這裡顯示一整排 nan。前綴定義沿用 train_single，不另外寫死。
        from engine.models.train_single import SHAPE_PREFIXES
        feat_cols = [c for c in feat.columns
                     if c not in ("date", "stock_id") and not c.startswith(SHAPE_PREFIXES)]

        # 分組顯示
        groups = {
            "均線":     [c for c in feat_cols if "ma" in c.lower()],
            "MACD":     [c for c in feat_cols if "macd" in c or "dif" in c or "hist" in c],
            "KD/RSI":   [c for c in feat_cols if "kd" in c or "rsi" in c or "_k" == c[-2:] or "_d" == c[-2:]],
            "布林":     [c for c in feat_cols if "bb_" in c],
            "量價":     [c for c in feat_cols if "vol" in c or "amount" in c or "turnover" in c],
            "籌碼":     [c for c in feat_cols if any(k in c for k in ["foreign", "trust", "dealer", "margin", "short", "institutional"])],
            "基本面":   [c for c in feat_cols if any(k in c for k in ["per", "pbr", "yield", "revenue"])],
            "其他":     [],
        }
        covered = set(c for v in groups.values() for c in v)
        groups["其他"] = [c for c in feat_cols if c not in covered]

        for gname, gcols in groups.items():
            if not gcols:
                continue
            with st.expander(f"{gname} ({len(gcols)} 個特徵)", expanded=(gname == "均線")):
                vals = {c: row[c] for c in gcols if c in row.index}
                df_show = pd.DataFrame({"特徵": list(vals.keys()), "值": list(vals.values())})
                df_show["值"] = df_show["值"].apply(
                    lambda v: f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else str(v)
                )
                st.dataframe(df_show, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 頁面 4：回測結果
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "回測結果":
    st.title("📊 回測結果")
    st.caption("出場規則：移動停利（獲利達 trigger 後追蹤最高價，回落 pct 出場）"
               "＋固定百分比停損；預設值沿用 doc/BACKTEST_LOG.md #24/#25 的設定"
               "（trail 15%/10% + 停損 20%）。")

    if model_key is None:
        st.warning("模型尚未訓練，請先執行 `make train`")
        st.stop()

    # 各 round 的測試期起訖不同（Round 2 每段剛好一年，Round 4 的 test 是 11 個月、
    # test2 只有 7 個月），所以標籤不寫死年份，直接讀分數檔的實際日期
    SPLIT_LABEL = {"val_sel": "驗證期 val_sel（挑門檻用，不是乾淨測試）",
                   "test": "test", "test2": "test2", "test3": "test3"}
    avail_splits = [s for s in SPLIT_LABEL if bundle_mod.score_path(model_key, s).exists()]
    if not avail_splits:
        st.error(f"找不到 {bundle_mod.model_label(model_key)} 的分數檔，"
                 f"無法回測（data/score_*.parquet）")
        st.stop()

    test_splits = [s for s in avail_splits if s.startswith("test")]
    ALL_TESTS = "__all_tests__"
    options = ([ALL_TESTS] if len(test_splits) > 1 else []) + avail_splits

    @st.cache_data(ttl=1800, show_spinner=False)
    def _span(key: str, sp: str) -> str:
        d = pd.read_parquet(bundle_mod.score_path(key, sp), columns=["date"])
        d = pd.to_datetime(d["date"])
        return f"{d.min().date()} ~ {d.max().date()}"

    def _fmt(s: str) -> str:
        if s == ALL_TESTS:
            return f"全部測試期（{'＋'.join(test_splits)} 接起來，共 {len(test_splits)} 段）"
        return f"{SPLIT_LABEL[s]}（{_span(model_key, s)}）"

    split_choice = st.selectbox(
        "回測區間", options,
        index=0 if ALL_TESTS in options else
              (options.index("test") if "test" in options else 0),
        format_func=_fmt)

    if split_choice == ALL_TESTS:
        # simulate() 一次只吃一個分數檔，所以把多段合併成一份暫存檔再餵進去
        score_file = combined_test_scores(model_key, tuple(test_splits))
        split = test_splits[0]      # 只用來取 split 名稱，實際讀的是上面的合併檔
    else:
        split = split_choice
        score_file = bundle_mod.score_path(model_key, split)

    # 口徑預設「無去重」——與挑門檻時看的曲線一致，也是實際用法：
    # 每天看每檔，超過門檻就買，不管買過沒有，每次進場都是獨立事件。
    dedup_mode = st.radio(
        "回測口徑", [False, True], horizontal=True,
        format_func=lambda d: "無去重（每筆訊號獨立進場，＝挑門檻的口徑）"
                              if not d else "去重（同一支持倉中不重複進場）",
    )
    st.info(f"模型 **{bundle_mod.model_label(model_key)}** · 分數檔 `{score_file.name}`。"
            f"回測用 `code/backtest/backtest.py` 的 `simulate()`／`performance()`。"
            + ("目前是**無去重**口徑，與門檻曲線同一把尺（BACKTEST_LOG #25）。"
               if not dedup_mode else
               "目前是**去重**口徑（BACKTEST_LOG #24），與門檻曲線不同尺，數字不能互相引用。"))

    test_prob = pd.read_parquet(score_file, columns=["date"])
    test_prob["date"] = pd.to_datetime(test_prob["date"])
    test_min, test_max = test_prob["date"].min().date(), test_prob["date"].max().date()

    curve_bt = load_sigcurve(model_key)
    thr_lo = float(curve_bt["threshold"].min()) if not curve_bt.empty else 0.0
    thr_hi = float(curve_bt["threshold"].max()) if not curve_bt.empty else 1.0
    thr_default = min(max(bundle_mod.default_threshold(model_key, curve_bt), thr_lo), thr_hi)

    col1, col2 = st.columns(2)
    with col1:
        use_trailing = st.checkbox("使用移動停利（建議，取代固定停利）", value=True)
        if use_trailing:
            # 預設值一律讀 backtest.CURRENT_EXIT_RULES，跟關注股票的追蹤同一份定義
            trail_trigger_pct = st.slider("移動停利觸發門檻", 5, 40,
                                          int(EXIT_RULES["trail_trigger"] * 100), 1, format="%d%%")
            trail_pct_pct = st.slider("移動停利回落幅度", 3, 30,
                                      int(EXIT_RULES["trail_pct"] * 100), 1, format="%d%%")
            take_profit = EXIT_RULES["take_profit"]  # trailing啟用時不會用到，僅保留參數位置
        else:
            take_profit_pct = st.slider("停利門檻（固定，整筆出清）", 5, 30,
                                        int(EXIT_RULES["take_profit"] * 100), 1, format="%d%%")
            take_profit = take_profit_pct / 100
        use_fixed_stop = st.checkbox("固定百分比停損（取代 MA 停損，BACKTEST_LOG #24 起的預設）",
                                     value=True)
        stop_loss = (st.slider("停損幅度", 5, 40, int(EXIT_RULES["stop_loss"] * 100), 1,
                               format="%d%%") / 100
                     if use_fixed_stop else None)
        stop_ma = st.radio("停損均線（僅在未勾固定停損時生效）", [10, 20],
                           index=[10, 20].index(EXIT_RULES["stop_ma"]),
                           horizontal=True, format_func=lambda x: f"MA{x}")
    with col2:
        signal_thr = st.slider("訊號分數門檻", thr_lo, thr_hi, thr_default,
                               (thr_hi - thr_lo) / 200 or 0.001, format="%.4f")
        _s = bundle_mod.sigcurve_stats_at(curve_bt, signal_thr)
        if _s:
            st.caption(f"此門檻在**驗證期**：{_s['n']:,} 筆訊號、勝率 {_s['win_rate']:.1%}、"
                       f"平均報酬 {_s['avg_return']:+.2%}")
        date_range = st.date_input("測試日期區間", value=(test_min, test_max),
                                   min_value=test_min, max_value=test_max)

    with st.expander("進場濾網（跟下方參數組合搜尋同一套選項，可疊加使用）"):
        f1, f2 = st.columns(2)
        with f1:
            single_streak = st.number_input("連續達標天數（1=不限制）", 1, 20, 1, key="single_streak")
            single_streak_mode = st.radio(
                "連續達標判定方式", ["at_least", "exact"], horizontal=True, key="single_streak_mode",
                format_func=lambda x: "≥N天" if x == "at_least" else "恰好第N天")
        with f2:
            single_pattern = st.selectbox(
                "均線型態", ["none", "bull3", "bull4", "bull5", "golden_ma5_20", "golden_ma10_60", "not_bear_recent"],
                key="single_pattern",
                format_func=lambda x: {
                    "none": "不限制", "bull3": "三線多排（今天多頭排列 MA5>10>20）",
                    "bull4": "四線多排（MA5>10>20>60）", "bull5": "五線多排（MA5>10>20>60>120）",
                    "golden_ma5_20": "黃金交叉（MA5穿越MA20）", "golden_ma10_60": "黃金交叉（MA10穿越MA60）",
                    "not_bear_recent": "前三天不是空頭排列（近3日不是連續MA5<10<20）",
                }[x])
            single_above_all = st.number_input("站上所有均線累積天數（0=不限制）", 0, 30, 0, key="single_above_all")

    if st.button("執行回測", type="primary"):
        if len(date_range) != 2:
            st.warning("請選擇完整的起訖日期")
            st.stop()
        try:
            from engine.backtest.backtest import performance
            trail_trigger = (trail_trigger_pct / 100) if use_trailing else None
            trail_pct = (trail_pct_pct / 100) if use_trailing else 0.10
            _rule = _model_exit_rule(model_key)
            if _rule and _rule.get("type") == "score":
                st.info(f"這個模型的出場是「分數跌回 {_rule.get('sell_threshold', 0.20):.2f} "
                        f"以下就賣」，上面的移動停利／停損／均線設定對它不生效。")
            with st.spinner("回測中..."):
                trades, price = _run_backtest(
                    model_key, split, score_file, signal_thr,
                    take_profit=take_profit, stop_ma=stop_ma, stop_loss=stop_loss,
                    date_start=str(date_range[0]), date_end=str(date_range[1]),
                    min_streak_days=single_streak, streak_mode=single_streak_mode,
                    pattern=single_pattern, min_above_all_ma_days=single_above_all,
                    trail_trigger=trail_trigger, trail_pct=trail_pct,
                    dedup=dedup_mode)
        except Exception as e:
            st.error(f"回測失敗：{e}")
            st.stop()

        if trades.empty:
            st.warning("無交易記錄")
            st.stop()

        perf = performance(trades, price)

        # 指標卡
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("勝率",     f"{perf['win_rate']:.1%}")
        c2.metric("平均報酬", f"{perf['avg_return']:.2%}")
        c3.metric("Sharpe",   f"{perf['sharpe']:.3f}")
        c4.metric("最大回撤", f"{perf['max_drawdown']:.2%}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("交易筆數", f"{perf['trades']:,}")
        c6.metric("固定停利觸發率", f"{perf['take_profit_pct']:.1%}")
        c7.metric("移動停利觸發率", f"{perf.get('trail_stop_pct', 0):.1%}")
        c8.metric(f"MA{stop_ma}停損率", f"{perf['ma_stop_pct']:.1%}")

        # 資產曲線（跟 backtest.py 的 performance() 用同一套等權重投組邏輯，
        # 避免持倉重疊時把交易硬串成一條複利鏈造成報酬率失真）
        from engine.backtest.backtest import _equity_curve
        equity = _equity_curve(trades, price)
        st.subheader("資產曲線（等權重投組，賣出獲利鎖定不再複利）")
        st.line_chart(equity)

        # 報酬分布
        st.subheader("每筆交易報酬分布")
        st.bar_chart(trades["return"].value_counts(bins=30).sort_index())

        # 明細
        with st.expander("交易明細"):
            show = trades[["signal_date", "stock_id", "buy_date", "buy_price",
                           "sell_date", "sell_price", "return", "sell_reason", "score"]].copy()
            show["return"] = show["return"].map("{:.2%}".format)
            show["score"] = show["score"].map("{:.4f}".format)
            st.dataframe(show, use_container_width=True)

        st.caption("⚠️ 未納入手續費 0.1425%、交易稅 0.3%、滑價；未納入已下市股票（生存偏差）")

    st.divider()
    st.subheader("🔍 參數組合搜尋")
    st.caption("對移動停利(觸發門檻/回落幅度) / 停損均線 / 訊號機率門檻做網格搜尋，"
               "找出報酬（或其他指標）最高的組合（2026-07-26起預設用移動停利，見 doc/BACKTEST_LOG.md）")

    with st.expander("設定搜尋範圍", expanded=True):
        g1, g2, g3 = st.columns(3)
        with g1:
            grid_tp = st.multiselect(
                "移動停利觸發門檻候選",
                [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
                default=[0.15, 0.20, 0.25, 0.30], format_func=lambda x: f"{x:.0%}")
            grid_trail_pct = st.multiselect(
                "移動停利回落幅度候選",
                [0.05, 0.08, 0.10, 0.15, 0.20],
                default=[0.08, 0.10, 0.15], format_func=lambda x: f"{x:.0%}")
        with g2:
            grid_ma = st.multiselect("停損均線候選", [10, 20], default=[20],
                                     format_func=lambda x: f"MA{x}")
        with g3:
            # 每個模型的分數尺度都不一樣（RF 最高 0.60、LSTM 0.89、排序平均 1.0），
            # 固定的百分比候選搬不動 → 候選改成「驗證期訊號數 = N 筆」對應的門檻。
            _targets = [100, 500, 2000, 10000, 50000]
            _thr_options = []
            if not curve_bt.empty:
                for t in _targets:
                    if t <= curve_bt["n"].max():
                        row = curve_bt.iloc[(curve_bt["n"] - t).abs().idxmin()]
                        _thr_options.append(round(float(row["threshold"]), 4))
                _thr_options = sorted(set(_thr_options), reverse=True)
            if thr_default not in _thr_options:
                _thr_options = sorted(set(_thr_options + [round(thr_default, 4)]), reverse=True)
            grid_thr = st.multiselect(
                "訊號分數門檻候選", _thr_options,
                default=_thr_options[:3], format_func=lambda x: f"{x:.4f}")
            st.caption("候選門檻＝驗證期訊號數約 100 / 500 / 2000 / 10000 / 50000 筆的位置，"
                       "外加目前選定的門檻（各模型分數尺度不同，不能用固定百分比）")

        g4, g5 = st.columns(2)
        with g4:
            grid_streak = st.multiselect(
                "連續達標天數候選（進場條件，1=不限制）",
                [1, 3, 5, 10], default=[1, 3])
        with g5:
            streak_mode = st.radio(
                "連續達標判定方式", ["at_least", "exact"], horizontal=True,
                format_func=lambda x: ("連續達標 ≥ N 天（每天都算訊號）" if x == "at_least"
                                       else "連續達標恰好 N 天（只在第N天當天觸發一次，之後繼續達標不重複算）"))

        g6, g7 = st.columns(2)
        with g6:
            grid_pattern = st.multiselect(
                "均線型態候選（進場條件）",
                ["none", "bull3", "bull4", "bull5", "golden_ma5_20", "golden_ma10_60", "not_bear_recent"],
                default=["none"],
                format_func=lambda x: {
                    "none": "不限制", "bull3": "三線多排（MA5>10>20，今天多頭排列）",
                    "bull4": "四線多排（MA5>10>20>60）", "bull5": "五線多排（MA5>10>20>60>120）",
                    "golden_ma5_20": "黃金交叉（MA5穿越MA20）", "golden_ma10_60": "黃金交叉（MA10穿越MA60）",
                    "not_bear_recent": "前三天不是空頭排列（近3日不是連續MA5<10<20）",
                }[x])
        with g7:
            grid_above_all = st.multiselect(
                "站上所有均線累積天數候選（0=不限制）",
                [0, 3, 5, 10], default=[0])

        rank_metric = st.selectbox("排序依據", ["total_return", "sharpe", "avg_return", "win_rate"],
                                   format_func=lambda x: {"total_return": "累計報酬", "sharpe": "Sharpe",
                                                          "avg_return": "平均報酬", "win_rate": "勝率"}[x])
        grid_date_range = st.date_input("搜尋用日期區間", value=(test_min, test_max),
                                        min_value=test_min, max_value=test_max, key="grid_date_range")

    n_combos_preview = (len(grid_tp) * len(grid_trail_pct) * len(grid_ma) * len(grid_thr)
                        * len(grid_streak or [1]) * len(grid_pattern or ["none"])
                        * len(grid_above_all or [0]))
    st.caption(f"目前設定共 {n_combos_preview} 組組合，每組約需 1~5 秒，"
              f"預估耗時 {n_combos_preview * 2 // 60} ~ {n_combos_preview * 5 // 60} 分鐘")
    if n_combos_preview > 150:
        st.warning("⚠️ 組合數偏多，長時間同步運算容易讓瀏覽器連線逾時、畫面卡住看不到結果，"
                  "建議先縮小候選範圍分批測試。")

    if st.button("開始搜尋", type="primary", key="grid_search_btn"):
        if len(grid_date_range) != 2:
            st.warning("請選擇完整的起訖日期")
            st.stop()
        combos = [(tp, tpct, ma, thr, sd, pat, aa) for tp in grid_tp for tpct in grid_trail_pct
                  for ma in grid_ma for thr in grid_thr for sd in (grid_streak or [1])
                  for pat in (grid_pattern or ["none"]) for aa in (grid_above_all or [0])]
        if not combos:
            st.warning("請至少選一個移動停利觸發門檻、回落幅度、停損均線、機率門檻")
            st.stop()

        _grid_rule = _model_exit_rule(model_key) if model_key else None
        if _grid_rule and _grid_rule.get("type") == "score":
            # 這一頁掃的全是移動停利／停損／均線參數，而這類模型的出場只看分數，
            # 掃出來的每一格都會是同一個結果。與其給一張看似有差異的假表，不如擋掉。
            st.warning(
                f"**{bundle_mod.model_label(model_key)}** 的出場是「分數跌回 "
                f"{_grid_rule.get('sell_threshold', 0.20):.2f} 以下就賣」，"
                "這一頁掃的移動停利／停損／均線參數對它全都不生效。"
                "請改用「回測結果」頁，或在那裡調整進場門檻。")
            st.stop()
        from engine.backtest.backtest import simulate, performance
        rows = []
        progress = st.progress(0.0, text=f"搜尋中... 0/{len(combos)}")
        for i, (tp, tpct, ma, thr, sd, pat, aa) in enumerate(combos):
            try:
                trades, price = simulate(split, score_path=score_file,
                                         threshold=thr, stop_ma=ma, stop_loss=stop_loss,
                                         date_start=str(grid_date_range[0]),
                                         date_end=str(grid_date_range[1]),
                                         min_streak_days=sd, streak_mode=streak_mode,
                                         pattern=pat, min_above_all_ma_days=aa,
                                         trail_trigger=tp, trail_pct=tpct,
                                         dedup=dedup_mode)
                if not trades.empty:
                    perf = performance(trades, price)
                    rows.append({"移動停利觸發": tp, "回落幅度": tpct, "停損均線": f"MA{ma}", "分數門檻": thr,
                                "連續達標天數": sd, "均線型態": pat, "站上所有均線天數": aa, **perf})
            except Exception:
                pass
            progress.progress((i + 1) / len(combos), text=f"搜尋中... {i+1}/{len(combos)}")
            # 每完成一組就存檔一次，避免跑到一半連線斷掉、重整後結果全部消失
            st.session_state["grid_search_rows"] = list(rows)
            st.session_state["grid_search_total"] = len(combos)
        progress.empty()
        st.session_state["grid_search_rows"] = rows
        st.session_state["grid_search_total"] = len(combos)

    saved_rows = st.session_state.get("grid_search_rows")
    saved_total = st.session_state.get("grid_search_total")
    if saved_rows is not None:
        if not saved_rows:
            st.warning("所有組合都沒有交易記錄，請放寬搜尋範圍")
        else:
            grid_df = pd.DataFrame(saved_rows).sort_values(rank_metric, ascending=False).reset_index(drop=True)
            best = grid_df.iloc[0]
            st.success(f"最佳組合：移動停利觸發 {best['移動停利觸發']:.0%} / 回落幅度 {best['回落幅度']:.0%} / "
                      f"{best['停損均線']} / 分數門檻 {best['分數門檻']:.4f} / 連續達標 {best['連續達標天數']}天 / "
                      f"均線型態 {best['均線型態']} / 站上所有均線 {best['站上所有均線天數']}天 → "
                      f"平均報酬 {best['avg_return']:.2%}、Sharpe {best['sharpe']:.3f}、"
                      f"勝率 {best['win_rate']:.1%}、交易 {best['trades']} 筆")
            if best["trades"] < 100:
                st.warning("⚠️ 最佳組合交易筆數偏少（<100），且全部來自同一段靜態測試窗口，"
                          "很可能是過度配適（overfitting）到這段期間的行情，統計可信度存疑"
                          "（業界經驗法則：至少100筆才算基本顯著，200-500筆才有較高信心，見doc/BACKTEST_LOG.md）。"
                          "建議：換不同日期區間重跑幾次確認方向一致，或優先選交易筆數較多、"
                          "排名接近但不是極端角落的組合。")

            show_grid = grid_df.copy()
            show_grid["移動停利觸發"] = show_grid["移動停利觸發"].map("{:.0%}".format)
            show_grid["回落幅度"] = show_grid["回落幅度"].map("{:.0%}".format)
            show_grid["分數門檻"] = show_grid["分數門檻"].map("{:.4f}".format)
            show_grid["win_rate"] = show_grid["win_rate"].map("{:.1%}".format)
            show_grid["avg_return"] = show_grid["avg_return"].map("{:.2%}".format)
            show_grid["total_return"] = show_grid["total_return"].map("{:.2%}".format)
            show_grid["max_drawdown"] = show_grid["max_drawdown"].map("{:.2%}".format)
            show_grid["take_profit_pct"] = show_grid["take_profit_pct"].map("{:.1%}".format)
            show_grid["trail_stop_pct"] = show_grid.get("trail_stop_pct", 0).map("{:.1%}".format)
            show_grid["ma_stop_pct"] = show_grid["ma_stop_pct"].map("{:.1%}".format)
            show_grid = show_grid.rename(columns={
                "trades": "交易筆數", "win_rate": "勝率", "avg_return": "平均報酬",
                "total_return": "累計報酬", "sharpe": "Sharpe", "max_drawdown": "最大回撤",
                "take_profit_pct": "固定停利觸發率", "trail_stop_pct": "移動停利觸發率",
                "ma_stop_pct": "MA停損率",
            })
            st.dataframe(show_grid, use_container_width=True, hide_index=True)
            st.caption(f"共測試 {saved_total} 組組合，{len(saved_rows)} 組有交易記錄")


# ═══════════════════════════════════════════════════════════════════════════════
# 頁面 5：模型成效
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "模型成效":
    st.title("🤖 模型成效")

    ready = bundle_mod.available_keys()
    if not ready:
        st.warning("找不到任何 model bundle，請先執行 `make train`")
        st.stop()

    # ── 1. 各模型在驗證期的門檻曲線比較 ────────────────────────────────
    st.subheader("驗證期（val_sel）門檻曲線比較")
    st.caption("每個模型都把驗證期的訊號由高分往低分累積，看「取到第 N 筆為止」的"
               "勝率與平均報酬。**無去重（訊號層級）口徑**，與挑門檻時看的曲線相同"
               "（doc/BACKTEST_LOG.md #25）。x 軸是累積訊號數，取對數比較好判讀。")

    metric = st.radio("指標", ["win_rate", "avg_return"], horizontal=True,
                      format_func=lambda x: "勝率" if x == "win_rate" else "平均報酬")
    max_n = st.select_slider("看到累積幾筆為止", [500, 2000, 10000, 50000, 200000],
                             value=50000)

    curves = {}
    for k in ready:
        c = load_sigcurve(k)
        if c.empty:
            continue
        c = c[(c["n"] >= 40) & (c["n"] <= max_n)]
        curves[bundle_mod.model_label(k)] = c.set_index("n")[metric]
    if curves:
        st.line_chart(pd.DataFrame(curves))
    else:
        st.info("找不到門檻曲線 CSV（data/sigcurve_*_val_sel.csv）")

    # ── 2. 各模型在幾個代表深度的驗證期表現 ───────────────────────────────
    st.subheader("代表深度下的驗證期表現")
    rows = []
    for k in ready:
        c = load_sigcurve(k)
        if c.empty:
            continue
        row = {"模型": bundle_mod.model_label(k)}
        for target in (100, 1000, 10000):
            if target <= c["n"].max():
                r = c.iloc[(c["n"] - target).abs().idxmin()]
                row[f"n={target} 門檻"] = f"{r['threshold']:.4f}"
                row[f"n={target} 勝率"] = f"{r['win_rate']:.1%}"
                row[f"n={target} 平均報酬"] = f"{r['avg_return']:+.2%}"
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── 3. 所選模型的 bundle 資訊與特徵重要性 ─────────────────────────────
    st.divider()
    st.subheader(f"模型細節：{bundle_mod.model_label(model_key)}")
    # 用 load_by_key 而不是 load_bundle(round, family)：實驗模型的代號不是
    # r{n}_{family} 格式，拆不出 round，硬拆會組出 bundle_r0_xxx.pkl 這種不存在的路徑
    b = bundle_mod.load_by_key(model_key)
    fam = b.get("family", "rf")
    st.markdown(f"**{fam.upper()}**：{len(b['cols'])} 欄特徵、訓練 {b['n_train']:,} 列、"
                f"驗證期分數 {b['score_min']:.4f}~{b['score_max']:.4f}、"
                f"訓練於 {b['trained_at']}")
    if b.get("label_name"):
        st.caption(f"Ground truth：`{b['label_name']}`（與其他模型的分數不可直接比較）")
    st.caption(f"超參數（取自 sweep_round{b['config_round']}_{fam}.csv 的 val_sel 最佳解）："
               f"`{b['params']}`")
    model_obj = b["model"]
    if hasattr(model_obj, "feature_importances_"):
        imp = pd.Series(model_obj.feature_importances_, index=b["cols"])
        st.bar_chart(imp.nlargest(20).sort_values())


elif page == "波段模型":
    # swing 專屬頁。獨立出來的理由見 engine/app/frontend/swing_panel.py 檔頭：
    # 它的出場是分數規則，共用回測頁上一半以上的控制項對它不生效，靠分流補在
    # 共用頁面上很容易漏（第一版就漏了參數搜尋頁）。
    from engine.app.frontend.swing_panel import (daily_density, density_verdict,
                                                 holdings_status)

    SWING_KEY = "swing"
    st.title("🌊 波段模型")
    if SWING_KEY not in bundle_mod.available_keys():
        st.warning("找不到 models/bundle_swing.pkl，請先執行 `make train-swing`")
        st.stop()

    _rule = _model_exit_rule(SWING_KEY) or {}
    _sell = _rule.get("sell_threshold", 0.20)
    _buy_default = bundle_mod.CHOSEN_THRESHOLDS.get(SWING_KEY, 0.97)
    st.caption(
        f"標的是「波段起漲 vs 起跌」（`label_swing_up`）。"
        f"**進場**分數 ≥ 門檻、**出場**分數 ≤ {_sell:.2f}，都是隔日開盤成交。"
        "這是逆勢型模型 —— 大盤有明顯回檔時有效，緩漲盤會落後。")

    swing_scores = load_score_history(SWING_KEY)
    if swing_scores.empty:
        st.error("沒有 swing 的分數檔，請先執行 `make train-swing`")
        st.stop()

    buy_th = st.slider("進場門檻", 0.50, 1.00, float(_buy_default), 0.01,
                       key="swing_buy_th")

    # ── 訊號密度：這個模型「現在該不該用」的直接指標 ──────────────────────
    dens = daily_density(swing_scores, buy_th)
    if not dens.empty:
        latest = dens.sort_values("date").iloc[-1]
        label, note = density_verdict(float(latest["density"]))
        c1, c2, c3 = st.columns(3)
        c1.metric("最新一天訊號數", f"{int(latest['n_signal']):,}")
        c2.metric("訊號密度", f"{latest['density']:.3%}")
        c3.metric("狀態", label)
        st.info(note)
        st.caption(f"資料日期 {pd.Timestamp(latest['date']).date()}。"
                   "密度＝當日過門檻檔數 ÷ 當日有分數檔數。")
        st.line_chart(dens.set_index("date")["density"], height=180)

    # ── 今日訊號 ────────────────────────────────────────────────────────────
    st.subheader("最新一天的訊號")
    if not dens.empty:
        last_day = dens.sort_values("date").iloc[-1]["date"]
        picks = swing_scores[(swing_scores["date"] == last_day) &
                             (swing_scores["score"] >= buy_th)].copy()
        picks = picks.sort_values("score", ascending=False)
        if picks.empty:
            st.info("這一天沒有任何股票過門檻 —— 依上面的說明，這種時候不該勉強出手。")
        else:
            picks["股票"] = picks["stock_id"].map(_stock_label)
            st.dataframe(picks[["股票", "score"]].rename(columns={"score": "分數"}),
                use_container_width=True, hide_index=True,
                column_config={"分數": st.column_config.NumberColumn(format="%.4f")})

    # ── 關注清單的持倉狀態（只看分數，不看價格規則）───────────────────────
    st.subheader("關注清單的分數狀態")
    _watch = st.session_state.get("watchlist", [])
    if not _watch:
        st.caption("關注清單是空的（左側可以加）。")
    else:
        hold = holdings_status(_watch, swing_scores, _sell)
        hold["股票"] = hold["stock_id"].map(_stock_label)
        st.dataframe(hold[["股票", "date", "score", "status"]].rename(
            columns={"date": "資料日", "score": "分數", "status": "狀態"}),
            use_container_width=True, hide_index=True)
        st.caption(f"這個模型的賣出只看分數：跌到 {_sell:.2f} 以下就賣，"
                   "跟移動停利／停損無關。")

    # ── 回測：只有兩個門檻，沒有價格規則可調 ────────────────────────────────
    st.subheader("回測")
    # 切分清單在這裡自己列 —— 「回測結果」頁的 SPLIT_LABEL 定義在那個頁面的
    # 區塊裡，跨頁拿不到；重複一份比把它提到模組層級改動更小。
    _swing_splits = [sp for sp in ("val_sel", "test", "test2")
                     if bundle_mod.score_path(SWING_KEY, sp).exists()]
    if not _swing_splits:
        st.warning("找不到 swing 的分數檔"); st.stop()
    bt_split = st.selectbox("切分", _swing_splits, key="swing_bt_split")
    if st.button("執行回測", type="primary", key="swing_bt_run"):
        from engine.backtest.backtest import performance
        from engine.backtest.score_exit import simulate_score_exit
        with st.spinner("回測中..."):
            trades, price = simulate_score_exit(
                score_path=bundle_mod.score_path(SWING_KEY, bt_split),
                buy_threshold=buy_th, sell_threshold=_sell, dedup=False)
        if trades.empty:
            st.warning("這個門檻在該切分下沒有任何交易")
        else:
            perf = performance(trades, price)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("交易數", f"{perf['trades']:,}")
            c2.metric("勝率", f"{perf['win_rate']:.1%}")
            c3.metric("平均報酬", f"{perf['avg_return']:+.2%}")
            c4.metric("Sharpe", f"{perf['sharpe']:.2f}")
            st.caption("報酬是毛報酬（未扣手續費與證交稅），口徑與 `make backtest` 相同。"
                       "台股來回成本約 0.585%。")
            st.dataframe(trades.tail(200), use_container_width=True, hide_index=True)


elif page == "買賣點規則":
    # 跟「型態規則」分開的理由見 engine/app/frontend/trade_rule_panel.py 檔頭：
    # 那一頁是「訊號日 + 20 天後結果」，這一頁是成對的買點與賣點。
    from engine.app.frontend import trade_rule_panel as trp

    st.title("🎯 買賣點規則")
    rules = trp.load_registry()
    hits = trp.load_hits()
    if not rules or hits.empty:
        st.error("找不到 `data/fpm_rules/trade_rules_registry.yaml` 或 "
                 "`trade_rules_hitlist.csv`。請在 fpm 專案執行 "
                 "`python src/target_rule.py`。")
        st.stop()

    rule_by_id = {r["id"]: r for r in rules}
    rule_id = st.selectbox("規則", list(rule_by_id),
                           format_func=lambda rid: rule_by_id[rid]["name"],
                           key="trade_rule_select")
    rule = rule_by_id[rule_id]
    rule_hits = hits[hits["rule_id"] == rule_id] if "rule_id" in hits.columns else hits

    st.markdown("**買點**：" + "；".join(rule.get("entry", [])))
    st.markdown("**賣點**：" + rule.get("exit", "—"))
    st.caption(f"母體：{rule.get('universe', '—')}　·　"
               f"每天最多買 {rule.get('max_picks_per_day', '—')} 檔　·　"
               f"規則挑選期間：{rule.get('selection_window', '—')}")

    with st.expander("⚠️ 這條規則的已知限制（一定要看）", expanded=True):
        for c in rule.get("caveats", []):
            st.markdown(f"- {c}")

    stats = trp.overall_stats(rule_hits)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("已結束交易", f"{stats['交易數']:,}")
    c2.metric("勝率", f"{stats['勝率']:.1%}")
    c3.metric("平均每筆", f"{stats['平均報酬']:+.2%}")
    c4.metric("逐期達標", f"{stats['達標期數']}/{stats['總期數']}",
              help=f"每個半年期勝率 >= {trp.WIN_RATE_TARGET:.0%} 的期數。"
                   "全期平均會被單一時段主導，一定要看逐期。")
    c5.metric("未結束", f"{stats['未結束']:,}",
              help="買了但還沒達標、也還沒抱到上限的部位。這些沒算進勝率。")
    st.caption(f"中位持有 {stats['中位持有']:.0f} 個交易日。"
               "報酬是**淨報酬**（已扣來回成本 0.585%），"
               "與「回測結果」頁的毛報酬口徑不同，不要並排比較。")

    st.subheader("逐半年期")
    st.caption("⚠️ 勝率只算已結束的交易。達標的部位會先結束、沒達標的還開著，"
               "所以**未結束筆數多的期間，勝率天生偏高**，不能當定論。")
    per = trp.period_summary(rule_hits)
    st.dataframe(
        per.style.format({"勝率": "{:.1%}", "平均報酬": "{:+.2%}",
                          "中位報酬": "{:+.2%}", "平均持有": "{:.0f}"}),
        use_container_width=True, hide_index=True)

    pending = trp.load_pending()
    if not pending.empty:
        pend = pending[pending["rule_id"] == rule_id] if "rule_id" in pending.columns else pending
        if not pend.empty:
            latest = pend["date"].max()
            st.subheader(f"最新買點（{latest:%Y-%m-%d} 訊號，隔日開盤買進）")
            st.dataframe(pend[pend["date"] == latest], use_container_width=True,
                         hide_index=True)

    stuck = trp.open_positions(rule_hits)
    if not stuck.empty:
        st.subheader(f"還沒賣掉的部位（{len(stuck)} 筆）")
        st.caption("買了但還沒達到 +3%、也還沒抱到上限。不設停損的代價全在這裡 —— "
                   "帳面可能已經虧很多，只是還沒認列。下面那欄報酬是**用資料最後一天"
                   "的價格試算**，不是真的賣出。")
        st.dataframe(stuck.head(50), use_container_width=True, hide_index=True)

    st.subheader("最近成交紀錄")
    st.dataframe(trp.recent_trades(rule_hits, limit=200),
                 use_container_width=True, hide_index=True)
