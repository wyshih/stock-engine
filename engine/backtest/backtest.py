"""
回測系統（PLAN.md 8，依使用者需求改為停利/停損出場邏輯）。

2026-07-26 預設值更新為移動停利版本（見 doc/BACKTEST_LOG.md #3/#4 的參數搜索
記錄）：門檻88% + 停損MA20 + 移動停利(獲利達25%後追蹤最高價，回落10%出場)，
在2026測試期驗證平均單筆報酬從0.95%提升到8.99%（約9.5倍）。**這是目前所有
回測預設採用的標準方法，不要再用舊版固定20%停利/MA10停損跑正式驗證。**

模擬規則（預設值）：
  每天看推薦名單，還沒持有該股票才在隔天開盤買進（同一支不重複進場）
  訊號機率門檻：threshold=0.88（比Meta自己校正用的78%更嚴格，是專門針對
    「交易模擬報酬」這個目標調出來的，跟 models/threshold.pkl 是不同用途）
  停損：收盤價連續兩天跌破 20 日均線（MA20）才觸發（單日正常拉回不算）
  移動停利：獲利達 trail_trigger（預設25%）後，關閉MA停損、改成追蹤進場後
    最高收盤價，價格從最高點回落超過 trail_pct（預設10%）才出場，讓真正噴出
    的股票不會被固定停利門檻提前限制漲幅
  三者都沒觸發就一直持有，直到資料結束

績效計算採等權重投資組合模擬（每筆交易買進時分配相同資金比例，
賣出後獲利鎖定為現金、不再滾入複利），避免持倉重疊時把交易硬串成
一條複利鏈造成報酬率失真。

用法：
  python backtest.py                       # 用meta_val 2025資料 + 上述預設值
  python backtest.py --split meta_test     # 最終test 2026
  python backtest.py --take-profit 0.15    # 停用移動停利、改回固定停利模式時才需要傳
"""
from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 專案路徑一律走 code/paths.py（唯一來源），不要各檔自行推導
from engine.paths import DATA_DIR, MODEL_DIR, PROJECT_ROOT  # noqa: E402


# ── 交易模擬 ──────────────────────────────────────────────────────────────────

def _load_scores(split: str, score_path: Path | None = None) -> pd.DataFrame:
    """讀單一模型的分數檔（欄位 date / stock_id / score）。

    委員會 + Meta stacking 已移除：實測委員會遠差於單一 RF，也差於隨機對照，
    且把 RF / MLP / LR / LSTM 的預測做排序平均後 AUC 單調變差
    （0.6371 → 0.6104），代表模型間的低重疊反映的是弱成員在擬合雜訊。

    LambdaRank 這類輸出分數而非機率的模型也走同一條路徑，門檻用每日分位數。
    """
    path = Path(score_path) if score_path is not None else DATA_DIR / f"score_{split}.parquet"
    if not path.exists():
        raise RuntimeError(f"{path} 不存在，請先產生該切分的模型分數")

    df = pd.read_parquet(path)
    missing = {"date", "stock_id", "score"} - set(df.columns)
    if missing:
        raise ValueError(f"分數檔缺少欄位：{sorted(missing)}")
    return df


def _get_threshold() -> float:
    """`simulate()` 沒收到 threshold 時的回退值。

    ⚠️ 2026-08-25：這條路徑已經是死的，保留只為了讓舊呼叫不會爆。
    它讀的 `threshold_trading.pkl` 由 `tune_trading_threshold.py` 產生，而那支
    在同日被刪除（它用的是委員會時代的 `meta_val` 切分，Round 4 沒有這個切分；
    且 `simulate()` 呼叫沒傳 dedup 與出場參數，違反 CLAUDE.md 規則 8）。
    該 pkl 檔目前不存在，所以這裡永遠回傳 0.775 —— 那是委員會時代 Meta 分類器
    的門檻，**與現行五個模型完全無關**。

    現行流程一律明確傳門檻：`summary.run_one()` 傳 `CHOSEN_THRESHOLDS[key]`，
    `threshold_curve` 傳 `--floor`。沒有任何生產路徑會走到這個回退值。
    """
    import pickle
    p = MODEL_DIR / "threshold_trading.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))["threshold"]
    logger.warning(
        "simulate() 沒有指定 threshold，回退到 0.775 —— 那是委員會時代的舊門檻，"
        "與現行模型無關。生產路徑應該明確傳門檻（見 CHOSEN_THRESHOLDS）。")
    return 0.775


# 目前採用的出場規則（2026-08-06 起）。前端回測頁的滑桿預設值、關注股票的
# 停損停利追蹤都讀這一份，避免「現在的定義」散在好幾個地方各自漂移。
# MA 停損已被固定百分比停損取代（stop_loss 有值時 stop_ma 不生效），
# stop_ma 保留是因為 _run_exit 仍需要一個均線欄位名。
CURRENT_EXIT_RULES = {
    "trail_trigger": 0.15,   # 獲利 15% 後啟動移動停利
    "trail_pct": 0.10,       # 從最高收盤回落 10% 出場
    "stop_loss": 0.20,       # 固定停損 20%
    "take_profit": 0.20,     # 僅在關閉移動停利時生效
    "stop_ma": 20,
}


def _run_exit(days, idx_buy: int, buy_price: float, row_getter, ma_col: str, stop_ma: int,
              max_hold_bars: int | None = None,
              take_profit: float = 0.20,
              trail_trigger: float | None = 0.25, trail_pct: float = 0.10,
              stop_loss: float | None = None):
    """
    單筆部位的出場模擬，從 `simulate()` 抽出來共用（2026-07-29）。

    抽出來的理由：`benchmark.py` 要對「同訊號日的漲停 peer」套用**完全相同**的
    出場規則才能公平比較。複製一份程式碼遲早會漂移，所以兩邊呼叫同一個函式。

    出場優先序（先到先出）：
      1. 移動停利（trail_trigger 有設定時）：獲利達 trail_trigger 後追蹤最高收盤，
         回落 trail_pct 出場。進入此模式後關閉 MA 停損。
      2. 固定停利（trail_trigger=None 時）：獲利達 take_profit 整筆出清。
      3. 停損，二選一：
         - `stop_loss` 有設定：**固定百分比停損**，收盤報酬 <= -stop_loss 就出場
           （reason="stop_loss"），**此時不使用 MA 停損**。
         - `stop_loss=None`（預設）：MA 停損 —— 收盤「連續兩天」跌破 ma_col 才觸發
           （跟 build_labels.py `_triple_barrier` 的定義一致，單日拉回不算）。
      4. 資料結束：用最後一天收盤價強制平倉（reason="data_end"，非真實出場）。

    ⚠️ 2026-08-06 新增 `stop_loss`：MA 停損與模型訊號方向相反 —— 實測 76~93% 的
    訊號股在買進當天收盤就已在 MA20 之下，導致 75~78% 的部位在 4 天內被砍，
    20 天 horizon 的 label 訊號完全沒機會兌現（見 doc/BACKTEST_LOG.md #23 與後續診斷）。

    回傳 (sell_date, sell_price, sell_reason)；取不到有效價格時回傳 (None, None, None)。
    """
    below_prev = False   # 前一天是否已跌破 MA（警戒中）
    trailing = False     # 是否已進入移動停利模式
    peak_price = None

    # max_hold_bars：持有期上限（交易日）。None＝不限，這是既有行為。
    # ⚠️ 2026-08-25 新增。不限上限時，2024H2 的訊號平均抱 108 根 bar、一路抱到
    # 2025/4 的關稅崩盤 —— 「val_sel 的回測」實際量的是「2024H2 進場 + 最長兩年
    # 持有」，橫跨三種市場狀態，不同 split 的數字在方法論上本來就不可比。
    # 使用者要求兩種都產，故用參數而非改預設。
    limit = len(days) - idx_buy
    if max_hold_bars is not None:
        limit = min(limit, max_hold_bars + 1)
    for k in range(1, limit):
        d = days[idx_buy + k]
        r = row_getter(d)
        if r is None or pd.isna(r["close"]):
            continue
        close_p, ma_stop = r["close"], r[ma_col]
        ret = (close_p - buy_price) / buy_price

        if trail_trigger is not None:
            if not trailing and ret >= trail_trigger:
                trailing = True
                peak_price = close_p
            if trailing:
                peak_price = max(peak_price, close_p)
                if close_p < peak_price * (1 - trail_pct):
                    return d, close_p, "trail_stop"
                below_prev = False   # 移動停利模式下不再判斷 MA 停損
                continue
        else:
            if ret >= take_profit:
                return d, close_p, "take_profit"

        if stop_loss is not None:
            if ret <= -stop_loss:
                return d, close_p, "stop_loss"
            continue

        below_today = (not pd.isna(ma_stop)) and close_p < ma_stop
        if below_today and below_prev:
            return d, close_p, f"ma{stop_ma}_stop"
        below_prev = below_today

    # 沒有觸發任何出場條件 —— 出場日是「持有上限那天」或「資料結束那天」，
    # 取先到的那一個。
    # ⚠️ 2026-08-25：舊版固定用 days[-1]，所以 max_hold_bars 只縮短了迴圈、
    #    最後仍會落到資料末端 —— 上限等於沒有生效（寫測試時抓到）。
    last_idx = len(days) - 1
    reason = "data_end"
    if max_hold_bars is not None and idx_buy + max_hold_bars < last_idx:
        last_idx = idx_buy + max_hold_bars
        reason = "max_hold"
    last_d = days[last_idx]
    r = row_getter(last_d)
    if r is None or pd.isna(r["close"]):
        return None, None, None
    return last_d, r["close"], reason


def _streak_of(cond: pd.Series, stock_id: pd.Series) -> pd.Series:
    """對任意布林 Series 依 stock_id 分組算連續 True 天數（中斷歸零）。
    要求 cond/stock_id 已經照 stock_id, date 排序好。"""
    run_id = (~cond).groupby(stock_id).cumsum()
    return cond.astype(int).groupby([stock_id, run_id]).cumsum()


def _add_streak_days(prob_df: pd.DataFrame, thr: float) -> pd.Series:
    """對每列算出「連續達標天數」：今天 score>=thr 且往前推算連續達標了幾天
    （中斷就歸零重算），用來做「突破門檻連續N天」這種進場條件。"""
    df = prob_df.sort_values(["stock_id", "date"])
    hit = df["score"] >= thr
    streak = _streak_of(hit, df["stock_id"])
    return streak.where(hit, 0).reindex(prob_df.index)


# 型態濾網（進場條件），2026-07-21 新增：三/四/五線多排沿用 build_price_features.py
# 已算好的 bull_Nma_1d；黃金交叉（短MA由下往上穿越長MA）用 ma5_ma20_ratio /
# ma10_ma60_ratio 前一天<=1、今天>1 判斷；「站上所有均線累積天數」用
# above_ma5~120 全部同時成立的連續天數。
PATTERN_COLS = {
    "bull3": "bull_3ma_1d", "bull4": "bull_4ma_1d", "bull5": "bull_5ma_1d",
    "golden_ma5_20": "golden_ma5_20", "golden_ma10_60": "golden_ma10_60",
    # 2026-07-25 新增，跟 streamlit_app.py「今日推薦」頁的兩個篩選條件對齊：
    # "bull3" 已經是「今天多頭排列」(MA5>MA10>MA20)，不用重複加；
    # "not_bear_recent" 是新的：前三天(t-1,t-2,t-3)不是連續空頭排列(MA5<MA10<MA20)。
    "not_bear_recent": "not_bear3_recent",
    # 2026-07-26 新增：盤整後放量突破（股票交易專家建議，見doc/PLAN.md）——
    # 「窄幅盤整後放量突破」比「已經連續急漲數日」進場風險報酬比好，追高風險低。
    "consol_breakout": "consol_breakout",
}


def _load_pattern_signals() -> pd.DataFrame:
    cols = ["date", "stock_id", "bull_3ma_1d", "bull_4ma_1d", "bull_5ma_1d",
            "bear_3ma_3d", "ma5_ma20_ratio", "ma10_ma60_ratio",
            "above_ma5", "above_ma10", "above_ma20", "above_ma60", "above_ma120",
            "bb_width_rank", "bb_upper_break_1d", "vol_ratio"]
    feat = pd.read_parquet(DATA_DIR / "features.parquet", columns=cols)
    feat["date"] = pd.to_datetime(feat["date"])
    feat = feat.sort_values(["stock_id", "date"]).reset_index(drop=True)

    same_stock = feat["stock_id"] == feat["stock_id"].shift(1)
    for short_n, long_n, ratio_col in [(5, 20, "ma5_ma20_ratio"), (10, 60, "ma10_ma60_ratio")]:
        r = feat[ratio_col]
        feat[f"golden_ma{short_n}_{long_n}"] = ((r > 1) & (r.shift(1) <= 1) & same_stock).astype("int8")

    above_all = (feat["above_ma5"].astype(bool) & feat["above_ma10"].astype(bool)
                & feat["above_ma20"].astype(bool) & feat["above_ma60"].astype(bool)
                & feat["above_ma120"].astype(bool))
    feat["above_all_ma_streak"] = _streak_of(above_all, feat["stock_id"])

    # 前三天不是空頭排列：bear_3ma_3d 在 t-1 那天=1 代表「以 t-1 為終點，已連續
    # 空頭排列>=3天」，也就是 t-1/t-2/t-3 這三天都是空頭排列（跟 streamlit
    # 「今日推薦」頁的 not_bear_recent 篩選同一套邏輯，t-1 用 shift(1) 取得）。
    bear3_prev = feat.groupby("stock_id")["bear_3ma_3d"].shift(1)
    feat["not_bear3_recent"] = (~bear3_prev.fillna(0).astype(bool)).astype("int8")

    # 盤整後放量突破：昨天布林通道寬度處於該股歷史後30%分位（=盤整、波動收斂），
    # 今天價格突破布林上軌(bb_upper_break_1d)且量能明顯放大(vol_ratio>1.5)。
    bb_width_rank_prev = feat.groupby("stock_id")["bb_width_rank"].shift(1)
    was_consolidating = bb_width_rank_prev <= 0.30
    breakout_today = feat["bb_upper_break_1d"].fillna(0).astype(bool)
    volume_confirm = feat["vol_ratio"].fillna(0) > 1.5
    feat["consol_breakout"] = (was_consolidating.fillna(False) & breakout_today & volume_confirm).astype("int8")

    return feat[["date", "stock_id", "bull_3ma_1d", "bull_4ma_1d", "bull_5ma_1d",
                "golden_ma5_20", "golden_ma10_60", "above_all_ma_streak",
                "not_bear3_recent", "consol_breakout"]]


def simulate(
    split: str = "meta_val",
    score_path: Path | None = None,
    take_profit: float = 0.20,
    threshold: float | None = 0.775,
    stop_ma: int = 20,
    date_start: str | None = None,
    date_end: str | None = None,
    min_streak_days: int = 1,
    streak_mode: str = "at_least",
    pattern: str = "none",
    min_above_all_ma_days: int = 0,
    trail_trigger: float | None = 0.25,
    trail_pct: float = 0.10,
    stop_loss: float | None = None,
    dedup: bool = True,
    max_hold_bars: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    逐日模擬：每天看訊號，隔天開盤買進。

    `dedup=True`（預設）：同一支持有中不重複進場 —— 模擬「可執行的投資組合」，
    因為同一筆資金不能重複買同一支。
    `dedup=False`：**每一筆門檻以上的訊號都獨立進場**，同一支在不同日期各算一筆、
    持有期間重疊也算。這是「訊號品質診斷」的口徑，也是使用者實際的用法
    （每天看每檔，超過門檻就買，不管買過沒有）。挑門檻用的曲線就是這個口徑，
    兩邊必須一致才比得起來（見 doc/BACKTEST_LOG.md #24 vs #25）。

    出場規則（2026-07-26 新增移動停利，見 doc/PLAN.md）：
    - trail_trigger=None（預設）：舊行為——賺 take_profit 就整筆停利，或收盤連續
      兩天跌破 stop_ma 日均線就停損，兩者先到哪個就出場。
    - trail_trigger 有設定：獲利達到 trail_trigger 之前，出場規則同上（MA停損
      持續有效）；一旦獲利 >= trail_trigger，關掉 MA 停損，改成移動停利模式：
      追蹤進場後的最高收盤價，收盤價從最高點回落超過 trail_pct 才出場
      （sell_reason="trail_stop"）。用意：業界常見做法是「不對稱出場」——
      虧損端快速止血、獲利端不設死板上限，讓少數大漲的部位貢獻遠高於固定停利
      門檻的報酬，藉此拉高整體平均單筆報酬（見 doc/BACKTEST_LOG.md 研究記錄）。

    date_start/date_end 可限制訊號日期範圍（含頭尾），不傳則用整個 split 區間。
    min_streak_days（預設1=不過濾）：進場前要求 score 連續 >= threshold 至少
    /恰好幾天，streak_mode="at_least" 表示連續達標 >= min_streak_days 天都算訊號，
    "exact" 只在連續達標「剛好第 min_streak_days 天」那天算訊號（新突破事件，不會
    每天都重複算進同一段連續達標期間）。
    pattern（預設"none"=不過濾）：額外要求當天符合均線型態，可選 "bull3"/"bull4"/"bull5"
    （三/四/五線多排：MA5>MA10>MA20[>MA60[>MA120]]）或 "golden_ma5_20"/"golden_ma10_60"
    （短MA由下往上穿越長MA，黃金交叉發生當天）。
    min_above_all_ma_days（預設0=不過濾）：要求收盤價同時站上 MA5/10/20/60/120
    已經連續至少幾天。
    回傳 (trades, price)：trades 給 performance() 算績效，price 順便回傳給後續算資產曲線用。
    """
    if stop_ma not in (10, 20):
        raise ValueError("stop_ma 只支援 10 或 20")
    if streak_mode not in ("at_least", "exact"):
        raise ValueError("streak_mode 只支援 at_least 或 exact")
    if pattern != "none" and pattern not in PATTERN_COLS:
        raise ValueError(f"pattern 只支援 none/{'/'.join(PATTERN_COLS)}")
    ma_col = f"ma{stop_ma}"

    prob_df = _load_scores(split, score_path)
    prob_df["date"] = pd.to_datetime(prob_df["date"])

    thr = threshold if threshold is not None else _get_threshold()
    mask = prob_df["score"] >= thr
    if min_streak_days > 1:
        streak = _add_streak_days(prob_df, thr)
        mask &= (streak == min_streak_days) if streak_mode == "exact" else (streak >= min_streak_days)
    signals = prob_df[mask][["date", "stock_id", "score"]].copy()

    if pattern != "none" or min_above_all_ma_days > 0:
        pat = _load_pattern_signals()
        signals = signals.merge(pat, on=["date", "stock_id"], how="left")
        if pattern != "none":
            signals = signals[signals[PATTERN_COLS[pattern]].fillna(0).astype(bool)]
        if min_above_all_ma_days > 0:
            signals = signals[signals["above_all_ma_streak"].fillna(0) >= min_above_all_ma_days]
        signals = signals[["date", "stock_id", "score"]]

    if date_start is not None:
        signals = signals[signals["date"] >= pd.Timestamp(date_start)]
    if date_end is not None:
        signals = signals[signals["date"] <= pd.Timestamp(date_end)]
    logger.info(f"[{split}] 訊號數={len(signals)}（門檻={thr:.2%}，停損=MA{stop_ma}，"
                f"連續達標={min_streak_days}天({streak_mode})，型態={pattern}，"
                f"站上所有均線>={min_above_all_ma_days}天，"
                f"日期範圍={date_start or '(不限)'}~{date_end or '(不限)'}）")

    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()

    price = pd.read_parquet(DATA_DIR / "price.parquet", columns=["date", "stock_id", "open", "close"])
    price["date"] = pd.to_datetime(price["date"])
    price = price.sort_values(["stock_id", "date"]).reset_index(drop=True)
    price[ma_col] = price.groupby("stock_id")["close"].transform(
        lambda s: s.rolling(stop_ma, min_periods=stop_ma).mean())

    price_idx = price.set_index(["stock_id", "date"])
    trading_days: dict[str, list] = {}
    for sid, grp in price.groupby("stock_id"):
        trading_days[sid] = sorted(grp["date"].unique())

    def _nth_day(sid: str, from_date: pd.Timestamp, n: int) -> pd.Timestamp | None:
        days = trading_days.get(sid, [])
        idx = next((i for i, d in enumerate(days) if d >= from_date), None)
        if idx is None or idx + n >= len(days):
            return None
        return days[idx + n]

    def _row(sid: str, date: pd.Timestamp):
        try:
            return price_idx.loc[(sid, date)]
        except KeyError:
            return None

    # 依股票分組訊號，時間排序，避免同一支股票持倉中重複進場
    signals_by_stock: dict[str, list[tuple]] = {}
    for _, row in signals.iterrows():
        signals_by_stock.setdefault(row["stock_id"], []).append((row["date"], row["score"]))
    for sid in signals_by_stock:
        signals_by_stock[sid].sort(key=lambda x: x[0])

    trades = []
    for sid, sig_list in signals_by_stock.items():
        days = trading_days.get(sid, [])
        if not days:
            continue
        held_until_idx = -1  # 持倉期間內（<= 此 index）不能再進場

        for signal_dt, score in sig_list:
            buy_date = _nth_day(sid, signal_dt + pd.Timedelta(days=1), 0)
            if buy_date is None:
                continue
            idx_buy = next((j for j, d in enumerate(days) if d == buy_date), None)
            if idx_buy is None:
                continue
            if dedup and idx_buy <= held_until_idx:
                continue  # 還在持倉中，這個訊號跳過（dedup=False 時不套用）

            buy_row = _row(sid, buy_date)
            if buy_row is None or pd.isna(buy_row["open"]) or buy_row["open"] <= 0:
                continue
            buy_price = buy_row["open"]

            sell_date, sell_price, sell_reason = _run_exit(
                days, idx_buy, buy_price, lambda d: _row(sid, d), ma_col, stop_ma,
                max_hold_bars=max_hold_bars,
                take_profit=take_profit, trail_trigger=trail_trigger, trail_pct=trail_pct,
                stop_loss=stop_loss,
            )
            if sell_date is None:
                continue

            final_return = (sell_price - buy_price) / buy_price
            trades.append({
                "signal_date": signal_dt, "stock_id": sid, "score": score,
                "buy_date": buy_date, "buy_price": buy_price,
                "sell_date": sell_date, "sell_price": sell_price,
                "return": final_return, "sell_reason": sell_reason,
            })
            held_until_idx = next((j for j, d in enumerate(days) if d == sell_date), idx_buy)

    df = pd.DataFrame(trades)
    logger.info(f"完成交易 {len(df)} 筆")
    return df, price


# ── 單筆部位追蹤（前端「關注股票」用）────────────────────────────────────────

def track_position(stock_id: str, buy_date, buy_price: float | None = None,
                   as_of=None, **rules) -> dict | None:
    """追蹤一筆已持有的部位：到 `as_of` 為止有沒有觸發停損停利。

    出場判定直接呼叫 `_run_exit()` —— 跟 `simulate()` 回測、`benchmark.py` 對照組
    同一份程式，不另外實作一套（不同實作遲早漂移，數字就對不起來）。

    `buy_price=None` 時取進場日開盤價，跟回測的成本口徑一致；使用者有填實際
    購入價就用他填的（那才是他真正的損益）。
    `rules` 可覆寫 CURRENT_EXIT_RULES 的任一項。

    回傳 dict；`status` 為 "exited"（已觸發出場）或 "holding"（到 as_of 都還沒觸發）。
    找不到該股票或進場日無有效價格時回傳 None。
    """
    params = {**CURRENT_EXIT_RULES, **rules}
    stop_ma = params["stop_ma"]
    ma_col = f"ma{stop_ma}"

    price = pd.read_parquet(DATA_DIR / "price.parquet",
                            columns=["date", "stock_id", "open", "close"])
    price = price[price["stock_id"] == stock_id].copy()
    if price.empty:
        return None
    price["date"] = pd.to_datetime(price["date"])
    price = price.sort_values("date").reset_index(drop=True)
    price[ma_col] = price["close"].rolling(stop_ma, min_periods=stop_ma).mean()

    # as_of 之後的資料要切掉，否則會拿還沒發生的價格判斷出場
    if as_of is not None:
        price = price[price["date"] <= pd.Timestamp(as_of)]

    days = list(price["date"])
    buy_dt = pd.Timestamp(buy_date)
    idx_buy = next((i for i, d in enumerate(days) if d >= buy_dt), None)
    if idx_buy is None:
        return None

    rows = price.set_index("date")
    entry = rows.loc[days[idx_buy]]
    cost = buy_price if buy_price else entry["open"]
    if cost is None or pd.isna(cost) or cost <= 0:
        return None

    sell_date, sell_price, sell_reason = _run_exit(
        days, idx_buy, cost, lambda d: rows.loc[d] if d in rows.index else None,
        ma_col, stop_ma,
        take_profit=params["take_profit"], trail_trigger=params["trail_trigger"],
        trail_pct=params["trail_pct"], stop_loss=params["stop_loss"],
    )
    if sell_date is None:
        return None

    # _run_exit 走到資料尾端沒觸發任何條件時回 "data_end"；對還在進行中的部位
    # 那不是出場，是「還持有」
    exited = sell_reason != "data_end"
    # 持有區間＝進場日到出場日（還沒出場就到 as_of 為止）。不能一路算到資料尾端，
    # 否則出場之後才發生的漲跌會被算進最高價，出現「已停損卻顯示最高 +31%」這種矛盾
    after = price.iloc[idx_buy:]
    if exited:
        after = after[after["date"] <= sell_date]
    peak_close = float(after["close"].max())
    last_close = float(after["close"].iloc[-1])
    # 移動停利是否已啟動：條件與 _run_exit 內的 `ret >= trail_trigger` 相同
    trail_armed = (params["trail_trigger"] is not None
                   and (peak_close - cost) / cost >= params["trail_trigger"])

    return {
        "stock_id": stock_id,
        "buy_date": days[idx_buy],
        "buy_price": float(cost),
        "status": "exited" if exited else "holding",
        "sell_date": sell_date if exited else None,
        "sell_price": float(sell_price) if exited else None,
        "sell_reason": sell_reason if exited else None,
        "return": (sell_price - cost) / cost if exited else (last_close - cost) / cost,
        "last_close": last_close,
        "peak_close": peak_close,
        "peak_return": (peak_close - cost) / cost,
        "trail_armed": bool(trail_armed),
        # 持倉中才有意義：還要跌到多少會被觸發
        "trail_stop_price": peak_close * (1 - params["trail_pct"]) if trail_armed else None,
        "stop_loss_price": cost * (1 - params["stop_loss"]) if params["stop_loss"] else None,
        "rules": params,
    }


# ── 績效計算 ──────────────────────────────────────────────────────────────────

def performance(trades: pd.DataFrame, price: pd.DataFrame) -> dict:
    if trades.empty:
        return {}

    r = trades["return"].values
    win_rate  = (r > 0).mean()
    avg_ret   = r.mean()

    equity = _equity_curve(trades, price)
    total_ret = equity.iloc[-1] - 1 if len(equity) else 0.0
    daily_ret = equity.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(250)) if daily_ret.std() > 0 else 0.0
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = drawdown.min() if len(drawdown) else 0.0

    # trail_stop 是移動停利模式下的出場（獲利先達 trail_trigger 才會進入這個模式），
    # 語意上是「鎖利出場」不是虧損停損，跟 ma{N}_stop（下跌停損）分開統計。
    ma_stop_mask = trades["sell_reason"].str.match(r"^ma\d+_stop$", na=False)
    trail_mask   = trades["sell_reason"] == "trail_stop"
    return {
        "trades":         len(trades),
        "win_rate":       round(float(win_rate), 4),
        "avg_return":     round(float(avg_ret), 4),
        "total_return":   round(float(total_ret), 4),
        "sharpe":         round(float(sharpe), 4),
        "max_drawdown":   round(float(max_dd), 4),
        "take_profit_pct": round(float((trades["sell_reason"] == "take_profit").mean()), 4),
        "ma_stop_pct":     round(float(ma_stop_mask.mean()), 4),
        "trail_stop_pct":  round(float(trail_mask.mean()), 4),
    }


def _equity_curve(trades: pd.DataFrame, price: pd.DataFrame) -> pd.Series:
    """
    等權重投資組合模擬：每筆交易買進時投入 1/N 資金，持有期間逐日隨股價浮動，
    賣出後獲利鎖定成現金（不再滾入複利），未進場的那份資金維持在 1.0（現金）。
    equity(day) = 平均（across 全部 N 筆交易）的每筆資金當下倍數。
    """
    n = len(trades)
    all_days = sorted(price["date"].unique())
    day_index = {d: i for i, d in enumerate(all_days)}

    close_by_stock = {sid: g.set_index("date")["close"] for sid, g in price.groupby("stock_id")}

    pnl = np.zeros((n, len(all_days)), dtype="float32")  # 每筆交易、每天的「目前報酬率」（尚未進場=0）
    # "return" 是 python 關鍵字，itertuples() 會把它改名成位置參數（例如 _2），
    # 用 .values 陣列平行迭代避免猜錯改名結果
    buy_dates  = trades["buy_date"].values
    sell_dates = trades["sell_date"].values
    stock_ids  = trades["stock_id"].values
    buy_prices = trades["buy_price"].values
    returns    = trades["return"].values

    for ti in range(n):
        buy_i = day_index.get(pd.Timestamp(buy_dates[ti]))
        if buy_i is None:
            continue
        sell_i = day_index.get(pd.Timestamp(sell_dates[ti]), len(all_days) - 1)
        stock_close = close_by_stock.get(stock_ids[ti])
        if stock_close is None:
            continue
        hold_days = all_days[buy_i:sell_i + 1]
        factors = stock_close.reindex(hold_days).ffill().values
        pnl[ti, buy_i:sell_i + 1] = (factors - buy_prices[ti]) / buy_prices[ti]
        if sell_i + 1 < len(all_days):
            pnl[ti, sell_i + 1:] = returns[ti]

    equity = 1 + pnl.mean(axis=0)
    return pd.Series(equity, index=pd.to_datetime(all_days))


def print_report(perf: dict, split: str) -> None:
    print(f"\n{'═'*40}")
    print(f" 回測結果 [{split}]")
    print(f"{'═'*40}")
    print(f"  交易筆數    : {perf.get('trades', 0):>8,}")
    print(f"  勝率        : {perf.get('win_rate', 0):>8.1%}")
    print(f"  平均報酬    : {perf.get('avg_return', 0):>8.2%}")
    print(f"  累計報酬    : {perf.get('total_return', 0):>8.2%}")
    print(f"  Sharpe      : {perf.get('sharpe', 0):>8.3f}")
    print(f"  最大回撤    : {perf.get('max_drawdown', 0):>8.2%}")
    print(f"  固定停利率  : {perf.get('take_profit_pct', 0):>8.1%}")
    print(f"  移動停利率  : {perf.get('trail_stop_pct', 0):>8.1%}")
    print(f"  MA停損率    : {perf.get('ma_stop_pct', 0):>8.1%}")
    print(f"{'═'*40}")
    print("※ 未納入手續費、交易稅；未納入已下市股票（存在生存偏差）")
    print("※ 績效採等權重投組模擬：每筆交易投入 1/N 資金，賣出獲利鎖定不再複利")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # split 名稱跟著 train_single 的切分走（val_es/val_sel/test/test2/test3）。
    # 舊的 meta_* 是委員會時代的，對應檔案早就不存在。
    parser.add_argument("--split",     default="test",
                        help="切分名稱，例如 test / test2 / test3 / val_sel")
    parser.add_argument("--tag",       default=None,
                        help="分數檔前綴，例如 final_r2_rf（＝ data/score_<tag>_<split>.parquet）")
    parser.add_argument("--score-path", default=None, help="直接指定分數檔路徑")
    parser.add_argument("--take-profit", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.775)
    parser.add_argument("--stop-ma",   type=int, default=20, choices=[10, 20])
    parser.add_argument("--stop-loss", type=float, default=None,
                        help="固定百分比停損，例如 0.20；設定時不使用 MA 停損")
    parser.add_argument("--trail-trigger", type=float, default=0.25,
                        help="移動停利觸發門檻，設 0 或負值可停用改回固定停利")
    parser.add_argument("--trail-pct", type=float, default=0.10)
    parser.add_argument("--no-dedup", action="store_true",
                        help="每筆訊號獨立進場，與門檻曲線同口徑（見 BACKTEST_LOG #25）")
    parser.add_argument("--start",     type=str, default=None, help="訊號日期起（YYYY-MM-DD）")
    parser.add_argument("--end",       type=str, default=None, help="訊號日期迄（YYYY-MM-DD，含當天）")
    parser.add_argument("--save",      action="store_true", help="儲存 trades.parquet")
    args = parser.parse_args()

    score_path = args.score_path
    if score_path is None and args.tag:
        score_path = DATA_DIR / f"score_{args.tag}_{args.split}.parquet"

    trail_trigger = args.trail_trigger if args.trail_trigger and args.trail_trigger > 0 else None
    # 一律具名傳參：simulate() 的第 2 個位置參數是 score_path，
    # 用位置呼叫會把 take_profit 綁到 score_path 而當場崩潰。
    trades, price = simulate(split=args.split, score_path=score_path,
                              take_profit=args.take_profit, threshold=args.threshold,
                              stop_ma=args.stop_ma, stop_loss=args.stop_loss,
                              date_start=args.start, date_end=args.end,
                              trail_trigger=trail_trigger, trail_pct=args.trail_pct,
                              dedup=not args.no_dedup)
    if trades.empty:
        logger.warning("無交易記錄")
        return

    perf = performance(trades, price)
    print_report(perf, args.split)

    if args.save:
        out = DATA_DIR / f"trades_{args.split}.parquet"
        trades.to_parquet(out, index=False)
        logger.info(f"trades 儲存至 {out}")


if __name__ == "__main__":
    main()
