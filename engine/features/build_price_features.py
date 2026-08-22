"""
計算 price_features.parquet（PLAN.md 5.2~5.9 節）。
Incremental：只算 price.parquet 有但 price_features.parquet 缺少的日期。
暫缺：Swing High/Low（5.10~5.11）、趨勢線（5.12）、ta-lib CDL 型態（5.13）。
用法：python build_price_features.py [--full] [--dry-run]
"""
import logging
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 專案路徑一律走 code/paths.py（唯一來源），不要各檔自行推導
from engine.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402


WARMUP = 260  # MA240 需要的最少歷史筆數


# ── I/O ──────────────────────────────────────────────────────────────────────

def _read(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _upsert(name: str, df: pd.DataFrame, keys: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read(name)
    combined = pd.concat([existing, df], ignore_index=True) if not existing.empty else df
    combined = (combined
                .drop_duplicates(subset=keys, keep="last")
                .sort_values(keys)
                .reset_index(drop=True))
    path = DATA_DIR / f"{name}.parquet"
    combined.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"寫入 {path}（{len(combined)} 筆）")


# ── 共用 helpers ──────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _streak(condition: pd.Series) -> pd.Series:
    """連續 True 的天數（False 時歸零）。"""
    g = (~condition).cumsum()
    return condition.astype(int).groupby(g).cumsum()


def _add_streak(out: dict, prefix: str, condition: pd.Series) -> None:
    """為布林條件加入 5 維 streak 特徵。"""
    s = _streak(condition)
    out[f"{prefix}_1d"] = condition.astype("int8")
    out[f"{prefix}_3d"] = (s >= 3).astype("int8")
    out[f"{prefix}_5d"] = (s >= 5).astype("int8")
    out[f"{prefix}_10d"] = (s >= 10).astype("int8")
    out[f"{prefix}_log"] = np.log1p(s).astype("float32")


# 台股漲跌幅限制 ±10%；新上市前 5 日無限制，故放寬到 ±100%。
# 超過此範圍者多半是減資、合併等公司行動造成的價格階梯 —— 那是真實的價格
# 變動，但不是投資人實際賺到的報酬，拿來當特徵只會是噪音。設 NaN。
MAX_ABS_RETURN = 1.0


def _clean_return(r: pd.Series) -> pd.Series:
    """濾掉非交易性的極端報酬（公司行動階梯、資料錯誤）。"""
    return r.where(r.abs() <= MAX_ABS_RETURN)


def _add_event(out: dict, prefix: str, event: pd.Series) -> None:
    """交叉型「事件」的特徵。

    2026-08-01 修正：原本這類條件也走 _add_streak，但交叉是瞬間事件 ——
    今天剛穿越，明天 shift(1) 的比較就反向了，條件在定義上不可能連續兩天成立，
    導致 _3d / _5d / _10d（要求連續 3/5/10 天）**永遠是 0**（實測整欄全 0）。
    事件該問的是「距上次發生幾天」，不是「連續幾天」。
    欄位名稱維持不變以免影響下游，但語意改為「過去 N 天內曾發生」。
    """
    ev = event.fillna(False).astype(bool)
    pos = np.arange(len(ev), dtype="float64")
    last = pd.Series(np.where(ev.to_numpy(), pos, np.nan), index=ev.index).ffill()
    bars_since = pd.Series(pos, index=ev.index) - last     # 未曾發生過則為 NaN

    out[f"{prefix}_1d"] = ev.astype("int8")
    out[f"{prefix}_3d"] = (bars_since <= 3).astype("int8")     # 過去 3 天內曾交叉
    out[f"{prefix}_5d"] = (bars_since <= 5).astype("int8")
    out[f"{prefix}_10d"] = (bars_since <= 10).astype("int8")
    out[f"{prefix}_log"] = np.log1p(bars_since).astype("float32")  # 距上次交叉的天數


# ── 5.2 均線 ──────────────────────────────────────────────────────────────────

def _ma_features(df: pd.DataFrame, out: dict) -> None:
    close = df["close"]
    mas = {n: close.rolling(n).mean() for n in [5, 10, 20, 60, 120, 240]}

    for n, ma in mas.items():
        out[f"close_ma{n}_ratio"] = (close / ma).astype("float32")
        out[f"above_ma{n}"] = (close > ma).astype("int8")

    out["ma5_ma20_ratio"] = (mas[5] / mas[20]).astype("float32")
    out["ma10_ma60_ratio"] = (mas[10] / mas[60]).astype("float32")
    out["ma20_ma120_ratio"] = (mas[20] / mas[120]).astype("float32")

    roll60_mean = close.rolling(60).mean()
    roll60_std = close.rolling(60).std().replace(0, np.nan)
    out["close_zscore"] = ((close - roll60_mean) / roll60_std).astype("float32")

    for n in [20, 60]:
        ma = mas[n]
        out[f"ma{n}_slope"] = ((ma - ma.shift(5)) / (5 * close)).astype("float32")

    # 最小平方法斜率（近5天，原始單位：元/天，2026-07-14 新增）
    # x=[0,1,2,3,4] 對稱於中心點2，OLS斜率公式化簡後可直接用shift線性組合算，不需rolling.apply
    def _ols_slope_5d(s: pd.Series) -> pd.Series:
        return (-2 * s.shift(4) - s.shift(3) + s.shift(1) + 2 * s) / 10

    close_ols_slope = _ols_slope_5d(close)
    ma5_ols_slope = _ols_slope_5d(mas[5])
    # 2026-08-01 修正：原本是「元/日」的絕對斜率，高價股天生數值大，模型會學到
    # 「股價高低」而非趨勢強度。除以股價變成「每日漲跌幾 %」，可跨股比較。
    out["close_ols_slope"] = (close_ols_slope / close).astype("float32")
    out["ma5_ols_slope"] = (ma5_ols_slope / close).astype("float32")

    # 三分類（下跌/盤整/上漲），固定門檻±0.2元/天，2026-07-14 新增
    TREND_THRESHOLD = 0.2
    out["close_slope_trend"] = np.select(
        [close_ols_slope > TREND_THRESHOLD, close_ols_slope < -TREND_THRESHOLD],
        [1, -1], default=0,
    ).astype("int8")
    out["ma5_slope_trend"] = np.select(
        [ma5_ols_slope > TREND_THRESHOLD, ma5_ols_slope < -TREND_THRESHOLD],
        [1, -1], default=0,
    ).astype("int8")

    stacked = pd.concat([mas[5], mas[10], mas[20]], axis=1)
    # 同一檔的 5/10/20 日均線不可能相差 10 倍以上；超過代表均線區間內有異常價格
    _mn = stacked.min(axis=1)
    out["ma_squeeze"] = ((stacked.max(axis=1) / _mn.where(_mn > 0) - 1)
                         .where(lambda x: x <= 10.0).astype("float32"))

    bull3 = (mas[5] > mas[10]) & (mas[10] > mas[20])
    bear3 = (mas[5] < mas[10]) & (mas[10] < mas[20])
    _add_streak(out, "bull_3ma", bull3)
    _add_streak(out, "bear_3ma", bear3)

    bull4 = bull3 & (mas[20] > mas[60])
    bear4 = bear3 & (mas[20] < mas[60])
    _add_streak(out, "bull_4ma", bull4)
    _add_streak(out, "bear_4ma", bear4)

    bull5 = bull4 & (mas[60] > mas[120])
    bear5 = bear4 & (mas[60] < mas[120])
    _add_streak(out, "bull_5ma", bull5)
    _add_streak(out, "bear_5ma", bear5)


# ── 5.3 MACD ─────────────────────────────────────────────────────────────────

def _macd_features(df: pd.DataFrame, out: dict) -> None:
    close = df["close"]
    dif = _ema(close, 12) - _ema(close, 26)
    sig = _ema(dif, 9)
    hist = dif - sig

    out["dif_ratio"] = (dif / close).astype("float32")
    out["macd_ratio"] = (sig / close).astype("float32")
    out["hist_ratio"] = (hist / close).where(lambda x: x.abs() <= 1.0).astype("float32")
    out["dif_positive"] = (dif > 0).astype("int8")
    # 2026-08-01 修正：DIF 會穿越零，pct_change 在零附近爆炸（實測 -6.3e14）。
    _dif_base = dif.shift(1)
    _dif_base = _dif_base.where(_dif_base.abs() >= (dif.abs().rolling(60, min_periods=10).mean() * 0.01).clip(lower=1e-8))
    out["dif_pct_change"] = (((dif - _dif_base) / _dif_base)
                             .where(lambda x: x.abs() <= 10.0).astype("float32"))
    out["dif_slope_5d"] = (((dif - dif.shift(5)) / (5 * close))
                           .where(lambda x: x.abs() <= 1.0).astype("float32"))

    golden = (dif > sig) & (dif.shift(1) <= sig.shift(1))
    _add_event(out, "macd_golden", golden)

    hist_expand = (hist.abs() > hist.shift(1).abs()) & (hist * hist.shift(1) > 0)
    _add_streak(out, "hist_expand", hist_expand)

    out["hist_3bar_same"] = (
        (np.sign(hist) == np.sign(hist.shift(1))) &
        (np.sign(hist.shift(1)) == np.sign(hist.shift(2)))
    ).astype("int8")

    N = 60
    out["macd_top_div"] = (
        (close >= close.rolling(N).max()) & (dif < dif.rolling(N).max())
    ).astype("int8")
    out["macd_bot_div"] = (
        (close <= close.rolling(N).min()) & (dif > dif.rolling(N).min())
    ).astype("int8")


# ── 5.4 布林通道 ──────────────────────────────────────────────────────────────

def _bb_features(df: pd.DataFrame, out: dict) -> None:
    close = df["close"]
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    bb_range = (upper - lower).replace(0, np.nan)

    pct_b = (close - lower) / bb_range
    out["bb_pct_b"] = pct_b.astype("float32")
    out["bb_pct_b_change"] = pct_b.diff(5).astype("float32")

    width = bb_range / mid.replace(0, np.nan)
    out["bb_width_pct"] = width.astype("float32")
    out["bb_width_rank"] = width.rolling(60).rank(pct=True).astype("float32")

    _add_streak(out, "bb_upper_break", close > upper)
    _add_streak(out, "bb_lower_break", close < lower)
    _add_streak(out, "bb_expand", std > std.shift(1))


# ── 5.5 KD ───────────────────────────────────────────────────────────────────

def _kd_features(df: pd.DataFrame, out: dict) -> None:
    close, high, low = df["close"], df["high"], df["low"]
    period = 9
    low_min = low.rolling(period).min()
    high_max = high.rolling(period).max()
    hl_range = (high_max - low_min).replace(0, np.nan)
    rsv = (close - low_min) / hl_range * 100

    # 台灣 KD：alpha = 1/3，等同 ewm(com=2)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()

    out["k_value"] = k.astype("float32")
    out["d_value"] = d.astype("float32")
    out["kd_diff"] = ((k - d) / 100).astype("float32")
    out["k_change_3d"] = (k - k.shift(3)).astype("float32")

    _add_event(out, "kd_golden", (k > d) & (k.shift(1) <= d.shift(1)))

    out["k_overbought"] = (k > 80).astype("int8")
    out["k_oversold"] = (k < 20).astype("int8")
    out["k_oversold_rebound"] = ((k.shift(1) < 20) & (k >= 20)).astype("int8")
    # 三次超賣反彈但 close 仍創新低（簡化版）
    out["k_triple_weak"] = (
        (k.shift(2) < 20) & (k.shift(1) >= 20) & (close < close.shift(10))
    ).astype("int8")


# ── 5.6 RSI ──────────────────────────────────────────────────────────────────

def _rsi_features(df: pd.DataFrame, out: dict) -> None:
    close = df["close"]

    def _rsi(n: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean().replace(0, np.nan)
        return 100 - 100 / (1 + gain / loss)

    r7, r14, r21 = _rsi(7), _rsi(14), _rsi(21)
    out["rsi_7"] = r7.astype("float32")
    out["rsi_14"] = r14.astype("float32")
    out["rsi_21"] = r21.astype("float32")
    out["rsi_7_21_diff"] = (r7 - r21).astype("float32")
    out["rsi_slope_5d"] = ((r14 - r14.shift(5)) / 5).astype("float32")

    _add_streak(out, "rsi_overbought", r14 > 70)
    _add_streak(out, "rsi_oversold", r14 < 30)
    out["rsi_oversold_rebound"] = ((r14.shift(1) < 30) & (r14 >= 30)).astype("int8")


# ── 5.7 價格動能 ──────────────────────────────────────────────────────────────

def _momentum_features(df: pd.DataFrame, out: dict,
                        market_close: pd.Series | None = None) -> None:
    close = df["close"]

    for n in [1, 5, 20, 60]:
        out[f"return_{n}d"] = _clean_return(close.pct_change(n)).astype("float32")

    if market_close is not None:
        mkt = market_close.reindex(df["date"]).reset_index(drop=True)
        for n in [5, 20, 60]:
            out[f"rs_{n}d"] = (_clean_return(close.pct_change(n)) - mkt.pct_change(n)).astype("float32")
    else:
        for n in [5, 20, 60]:
            out[f"rs_{n}d"] = np.nan

    for n in [60, 240]:
        roll_high = close.rolling(n).max()
        roll_low = close.rolling(n).min()
        out[f"dist_high_{n}"] = ((close - roll_high) / roll_high).astype("float32")
        out[f"dist_low_{n}"] = ((close - roll_low) / roll_low).astype("float32")

    out["is_new_high_60"] = (close >= close.rolling(60).max()).astype("int8")
    out["momentum_accel"] = (
        close.pct_change(5) > close.pct_change(20) / 4
    ).astype("int8")
    out["momentum_align"] = (
        (close.pct_change(1) > 0) &
        (close.pct_change(5) > 0) &
        (close.pct_change(20) > 0)
    ).astype("int8")


# ── 5.8 波動度 ────────────────────────────────────────────────────────────────

# ATR 的標準週期（Wilder 1978）
ATR_WINDOW = 14


def _volatility_features(df: pd.DataFrame, out: dict) -> None:
    close, high, low = df["close"], df["high"], df["low"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # 2026-08-07 修正：ATR 是 Wilder (1978) 的具名指標，用的是 α=1/14 的
    # 指數平滑，不是 14 日簡單平均。TA-Lib 的 talib.ATR 也是 Wilder。
    # 改之前同一份表裡並存兩種慣例：atr_14 / natr_14（Wilder，來自 TA-Lib）
    # 與 atr_ratio（SMA，這裡）—— 同名指標算法不同，比較起來沒有意義。
    # Wilder 平滑等價於 ewm(alpha=1/14)；adjust=False 才是遞迴式定義。
    atr14 = tr.ewm(alpha=1 / ATR_WINDOW, adjust=False, min_periods=ATR_WINDOW).mean()

    # 2026-08-01 修正：日均真實區間不可能超過股價本身（台股漲跌幅 ±10%）。
    # 超過 1.0 代表 ATR 還留著崩跌前的記憶、而現價已極低，該值無意義。
    atr_r = (atr14 / close).where(lambda x: x <= 1.0).astype("float32")
    out["atr_ratio"] = atr_r
    out["atr_rank"] = atr_r.rolling(60).rank(pct=True).astype("float32")
    out["std_ratio"] = (close.rolling(20).std() / close).where(lambda x: x <= 1.0).astype("float32")

    daily_ret = close.pct_change()
    # 2026-08-07 修正：下跌側原本用 `<= 0`，把平盤日算進下跌側。
    # 這個指標的語意是「上漲日 vs 下跌日」，平盤日兩者皆非。台股平盤日佔比高，
    # 把一堆報酬為 0 的日子塞進下跌側會壓低下跌側標準差（分母），讓比值系統性
    # 偏高。上漲側用的是嚴格 `> 0`，下跌側也必須對稱地用嚴格 `< 0`。
    # v2（features_v2/price_flow.py）本來就是嚴格 `< 0`，這裡對齊。
    up_ret = daily_ret.where(daily_ret > 0)
    dn_ret = daily_ret.where(daily_ret < 0)
    up_std = up_ret.rolling(20, min_periods=5).std()
    # 2026-08-01 修正：只擋分母「恰好為 0」不夠，接近零時比率會爆到 1e7。
    # 日報酬標準差低於 1e-5（0.001%）視為無效。
    dn_std = dn_ret.rolling(20, min_periods=5).std()
    # 2026-08-01 追加：分母下限拉到 1e-4，且比值超過 100 視為無效（上下波動
    # 相差百倍以上代表其中一側幾乎沒有樣本）。
    dn_std = dn_std.where(dn_std >= 1e-4)
    _va = up_std / dn_std
    out["vol_asymmetry"] = _va.where(_va <= 100).astype("float32")


# ── 5.9 量價 ──────────────────────────────────────────────────────────────────

def _volume_features(df: pd.DataFrame, out: dict) -> None:
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    # 2026-08-01 修正：OBV 是會穿越零的累積量，pct_change 在零附近會爆炸
    # （實測 obv_pct_20d 出現 -1.9e6）。分母需達到自身量級的 1% 才計算。
    _obv_scale = obv.abs().rolling(60, min_periods=10).mean()
    for _n in (5, 20):
        _base = obv.shift(_n)
        _base = _base.where(_base.abs() >= (_obv_scale * 0.01).clip(lower=1e-8))
        out[f"obv_pct_{_n}d"] = ((obv - _base) / _base).astype("float32")

    vol_ma5 = vol.rolling(5).mean().replace(0, np.nan)
    out["vol_ratio"] = (vol / vol_ma5).astype("float32")

    daily_ret = close.pct_change()
    # 2026-08-07 修正：同 vol_asymmetry，下跌側改嚴格 `< 0`。
    # 平盤日不屬於上漲也不屬於下跌，塞進下跌側會汙染分母。
    up_vol = vol.where(daily_ret > 0)
    dn_vol = vol.where(daily_ret < 0)
    up_avg = up_vol.rolling(20, min_periods=5).mean()
    dn_avg = dn_vol.rolling(20, min_periods=5).mean().replace(0, np.nan)
    # 上漲日均量 / 下跌日均量，超過 100 倍代表其中一側樣本極少，無參考價值
    out["vol_confirm_rate"] = (up_avg / dn_avg).where(lambda x: x <= 100).astype("float32")

    hl_range = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl_range
    mfv = mfm * vol
    # 2026-07-26 修正（資料審核發現）：原本 rolling(20) 沒指定 min_periods，
    # pandas 預設要求整個20天窗口都不是NaN才有結果——台股常見漲跌停鎖死
    # （high==low）導致 hl_range 變 NaN，20天裡只要有1天鎖死整個窗口就全部
    # 變NaN，造成 cmf null率異常偏高(部分月份超過70%)。加 min_periods=10
    # 跟同檔案其他20日滾動指標(如 vol_confirm_rate)一致的容錯寬度。
    vol_sum = vol.rolling(20, min_periods=10).sum().replace(0, np.nan)
    out["cmf"] = (mfv.rolling(20, min_periods=10).sum() / vol_sum).astype("float32")

    out["vol_breakout"] = (
        (out["vol_ratio"] > 2) & (daily_ret > 0)
    ).astype("int8")

    vol_ma20 = vol.rolling(20).mean()
    vol_ma60 = vol.rolling(60).mean().replace(0, np.nan)
    out["avg_vol_ratio"] = (vol_ma20 / vol_ma60).astype("float32")
    # turnover_ratio 需要 market_cap，由 build_features.py merge 時補


# ── per-stock pipeline ────────────────────────────────────────────────────────

def _compute_stock(group: pd.DataFrame,
                   market_close: pd.Series | None = None) -> pd.DataFrame:
    group = group.sort_values("date").reset_index(drop=True)
    out: dict = {"date": group["date"], "stock_id": group["stock_id"]}

    _ma_features(group, out)
    _macd_features(group, out)
    _bb_features(group, out)
    _kd_features(group, out)
    _rsi_features(group, out)
    _momentum_features(group, out, market_close)
    _volatility_features(group, out)
    _volume_features(group, out)

    return pd.DataFrame(out)


# ── entry point ───────────────────────────────────────────────────────────────

def run(full: bool = False, dry_run: bool = False) -> pd.DataFrame:
    price = _read("price")
    if price.empty:
        raise RuntimeError("price.parquet 不存在，請先執行 fetch_price.py")

    price["date"] = pd.to_datetime(price["date"])

    # 找出缺少特徵的日期
    existing = _read("price_features")
    if full or existing.empty:
        new_dates = set(price["date"].dt.normalize().unique())
        logger.info("全量重算")
    else:
        existing["date"] = pd.to_datetime(existing["date"])
        done = set(existing["date"].dt.normalize().unique())
        new_dates = set(price["date"].dt.normalize().unique()) - done
        if not new_dates:
            logger.info("price_features.parquet 已是最新，無需更新")
            return pd.DataFrame()
        logger.info(f"增量補算 {len(new_dates)} 個交易日")

    # 大盤指數（若有 TWII，用於相對強弱）
    market_close: pd.Series | None = None
    if "TWII" in price["stock_id"].values:
        twii = price[price["stock_id"] == "TWII"].set_index("date")["close"]
        market_close = twii

    # 逐支計算
    results = []
    stocks = [s for s in price["stock_id"].unique() if s != "TWII"]
    total = len(stocks)

    for i, sid in enumerate(stocks):
        grp = price[price["stock_id"] == sid].copy()
        if len(grp) < WARMUP:
            continue  # 歷史不足，跳過

        feat = _compute_stock(grp, market_close)
        # 只保留新的日期
        feat = feat[feat["date"].dt.normalize().isin(new_dates)]
        if not feat.empty:
            results.append(feat)

        if (i + 1) % 200 == 0:
            logger.info(f"  {i+1}/{total} 支完成")

    if not results:
        logger.warning("無新特徵產出")
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)
    logger.info(f"產出 {len(df)} 筆特徵（{df['stock_id'].nunique()} 支）")

    if dry_run:
        logger.info("[dry-run] 不寫入")
    else:
        _upsert("price_features", df, keys=["date", "stock_id"])

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="全量重算（忽略現有快取）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = run(full=args.full, dry_run=args.dry_run)
    if args.dry_run and not df.empty:
        print(f"欄位數：{len(df.columns)}")
        print(df.dtypes.to_string())
