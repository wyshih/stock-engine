"""
每日資料驗證：價格合理性、缺漏欄位、假日偵測。
用法：python validate_data.py --date 2026-06-27
"""
import logging
import argparse
from datetime import date, timedelta

import pandas as pd

from engine.data_source.utils import parse_date, read_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 台股漲跌幅上限（普通股 ±10%）
MAX_DAILY_CHANGE = 0.11

# OHLC 不變式容忍誤差（浮點運算用，不是業務邏輯門檻）
OHLC_EPSILON = 1e-9

# 當日 OHLC 缺值率上限。零星停牌／新股上市首日會有少量空值屬正常，
# 但整批抓取失敗時會一口氣衝到接近 100%（2026-07-31 是 99.8%）。
MAX_OHLC_MISSING_RATE = 0.05

# price 當日列數相對 chip 當日列數的下限。兩者都涵蓋全市場，正常時比值接近 1；
# 2025-08-01 是 price 15 列 vs chip 1,953 列（比值 0.008）。
MIN_PRICE_CHIP_ROW_RATIO = 0.5

# 全市場當日最少列數；低於此值視為假日或整批異常
MIN_MARKET_ROWS = 50

# 當日檔數相對前一交易日的下限。2026-08-20 抓到 186 檔停在 08-13 卻無人察覺：
# 當時 price/chip 比值 0.85 遠高於 MIN_PRICE_CHIP_ROW_RATIO，整批漏抓輕鬆過關。
# 正常情況下市場檔數只會因新上市／下市小幅變動，一天掉 5% 必定是抓取出問題。
MIN_ROW_RATIO_VS_PREV = 0.95

# 零成交量比例上限。2026-07-10 台股休市，yfinance 卻回了一整天 forward-fill 的
# 假 K 棒（1,956/1,961 = 99.7% 零成交、OHLC 四價相同），當時只印了一行 warning。
MAX_ZERO_VOLUME_RATE = 0.5

# 跨日漲跌幅上限。台股漲跌幅限制是相對**前一交易日收盤價**，超過即為結構性錯誤
# （分割、減資、除權息），不是市場現象。原本的檢查算的是 (close-open)/open ——
# 那是「當日振幅」，看的是同一列裡的兩個欄位，而接縫存在於相鄰兩列之間，
# 因此在設計上就抓不到：5314 在 2026-07-28 當日振幅僅 −4.7%，跨日卻是 −77.5%。
MAX_OVERNIGHT_CHANGE = 0.11

OHLC_COLS = ["open", "high", "low", "close"]

# 已知的 Yahoo Finance 上游資料缺口（2026-08-07 查證）。
#
# 這三天 chip 分別有 1,681 / 1,749 / 1,953 列，是貨真價實的交易日；但 Yahoo
# 就是沒有這幾天的個股報價 —— 拉 2330.TW 整段 2019-09-09~2026-08-01 的單一
# request，這三天不在 index 裡；窄區間單獨重抓 2317/2454/1101.TW 也一樣直接
# 從 07-31 跳到 08-04。全市場 22 個批次重抓後列數毫無變化（51/40/15）。
# 不是 rate limit，也沒有其他價格來源可補（repo 內 price 只有 yfinance）。
#
# 決策：接受缺口，不補值、不刪列（見 doc/BACKTEST_LOG.md）。
# 這裡把 row-count 類檢查對這三天降級成 WARNING —— 只降級這幾個「已經查證過」
# 的日期，規則本身照常運作，才攔得住「新出現」的缺口（2025-08-01 當初就是被
# 這條規則抓出來的）。OHLC 不變式等資料正確性檢查不降級，照常 FAIL。
KNOWN_UPSTREAM_GAPS: frozenset[date] = frozenset({
    date(2019, 9, 9),
    date(2021, 4, 6),
    date(2025, 8, 1),
})

KNOWN_GAP_NOTE = "Yahoo Finance 上游無資料，已知且接受，見 doc/BACKTEST_LOG.md"


def _check_ohlc_missing_rate(day_df: pd.DataFrame, target_date: date) -> bool:
    """
    當日 OHLC 缺值率超過 MAX_OHLC_MISSING_RATE 即視為整批抓取失敗。

    這條是 2026-07-31 漏網的直接補丁：那天 1,943 列裡只有 4 列 close 非 NaN，
    卻通過了全部既有檢查。根因是 pandas 的 `NaN > NaN` 回傳 False，
    不變式那組布林運算對 NaN 列一律判定為「沒有違反」；列數檢查又只看 len()，
    NaN 列照樣被算成一列。缺值本身必須單獨查，不能靠其他檢查順帶攔。
    """
    if not set(OHLC_COLS).issubset(day_df.columns):
        return True

    n_missing = int(day_df[OHLC_COLS].isna().any(axis=1).sum())
    missing_rate = n_missing / len(day_df)
    if missing_rate > MAX_OHLC_MISSING_RATE:
        logger.error(
            f"{target_date} OHLC 缺值率 {missing_rate:.1%}（{n_missing}/{len(day_df)} 列）"
            f"超過上限 {MAX_OHLC_MISSING_RATE:.0%}，判定為整批抓取失敗，請重抓"
        )
        return False
    if n_missing:
        logger.warning(f"{target_date} OHLC 缺值 {n_missing} 列（{missing_rate:.2%}），在容忍範圍內")
    return True


def _check_price_chip_row_ratio(day_df: pd.DataFrame, target_date: date,
                                allow_gap: bool = False) -> bool:
    """
    price 與 chip 都涵蓋全市場，正常日兩者列數應該相當。

    這條補的是 2025-08-01：price 只有 15 列、chip 有 1,953 列，但兩個檢查
    各自獨立回報「通過」，沒有任何一步比對過兩者，矛盾就這樣溜過去。
    chip 當天沒資料時無從比對，交叉檢查跳過（由 validate_chip 各自負責）。
    """
    chip = read_parquet("chip")
    if chip.empty or "date" not in chip.columns:
        return True

    chip_rows = int((chip["date"].dt.date == target_date).sum())
    if chip_rows == 0:
        return True

    ratio = len(day_df) / chip_rows
    if ratio >= MIN_PRICE_CHIP_ROW_RATIO:
        return True

    detail = (
        f"{target_date} price 只有 {len(day_df)} 列、chip 有 {chip_rows} 列"
        f"（比值 {ratio:.3f} < {MIN_PRICE_CHIP_ROW_RATIO}）"
    )
    if allow_gap:
        logger.warning(f"{detail}：{KNOWN_GAP_NOTE}")
        return True
    logger.error(f"{detail}，代表 price 整批抓取失敗，請重抓")
    return False


def _previous_trading_day(df: pd.DataFrame, target_date: date):
    """資料裡 target_date 之前最近的一個有資料的日期。"""
    earlier = df.loc[df["date"].dt.date < target_date, "date"]
    return None if earlier.empty else earlier.max()


def _exright_ids(target_date: date) -> set[str]:
    """當日除權息／分割的股票 —— 這些跨日跳空是合法的，不算異常。

    事件表還沒建立時回空集合：寧可多報幾檔要人確認，也不要放行真正的接縫。
    """
    events = read_parquet("exright")
    if events.empty or "date" not in events.columns:
        return set()
    same_day = events[pd.to_datetime(events["date"]).dt.date == target_date]
    return set(same_day["stock_id"].astype(str))


def _check_row_count_vs_prev(df: pd.DataFrame, day_df: pd.DataFrame,
                             target_date: date) -> bool:
    previous = _previous_trading_day(df, target_date)
    if previous is None:
        return True
    previous_total = int((df["date"] == previous).sum())
    if previous_total < MIN_MARKET_ROWS:
        return True
    ratio = len(day_df) / previous_total
    if ratio >= MIN_ROW_RATIO_VS_PREV:
        return True
    logger.error(
        f"{target_date} 檔數 {len(day_df)} 相對前一交易日 {previous.date()} 的 "
        f"{previous_total} 檔只有 {ratio:.1%}（下限 {MIN_ROW_RATIO_VS_PREV:.0%}），"
        f"代表有股票整批漏抓")
    return False


def _check_zero_volume(day_df: pd.DataFrame, target_date: date) -> bool:
    if "volume" not in day_df.columns:
        return True
    rate = float((day_df["volume"] == 0).mean())
    if rate <= MAX_ZERO_VOLUME_RATE:
        return True
    logger.error(
        f"{target_date} 零成交量佔 {rate:.1%}（上限 {MAX_ZERO_VOLUME_RATE:.0%}），"
        f"多半是把休市日抓成了一整天 forward-fill 的假 K 棒")
    return False


def _check_overnight_change(df: pd.DataFrame, day_df: pd.DataFrame,
                            target_date: date) -> bool:
    """跨日漲跌幅。除權息當日的跳空是合法的，用事件表當白名單排除。"""
    previous = _previous_trading_day(df, target_date)
    if previous is None:
        return True
    prev_close = (df.loc[df["date"] == previous, ["stock_id", "close"]]
                  .set_index("stock_id")["close"])
    merged = day_df.set_index("stock_id")["close"].to_frame("close")
    merged["prev"] = prev_close
    merged = merged.dropna()
    merged = merged[merged["prev"] > 0]
    if merged.empty:
        return True

    change = merged["close"] / merged["prev"] - 1
    breached = change[change.abs() > MAX_OVERNIGHT_CHANGE]
    breached = breached[~breached.index.astype(str).isin(_exright_ids(target_date))]
    if breached.empty:
        return True

    worst = breached.reindex(breached.abs().sort_values(ascending=False).index)
    detail = ", ".join(f"{sid} {value:+.1%}" for sid, value in worst.head(10).items())
    logger.error(
        f"{target_date} 跨日漲跌幅超過 ±{MAX_OVERNIGHT_CHANGE:.0%} 共 {len(breached)} 檔，"
        f"且不在除權息事件表內：{detail}"
        + ("..." if len(breached) > 10 else ""))
    return False


def validate_price(target_date: date) -> bool:
    df = read_parquet("price")
    if df.empty:
        logger.error("price.parquet 不存在")
        return False

    day_df = df[df["date"].dt.date == target_date]
    if day_df.empty:
        logger.warning(f"{target_date} 無價格資料")
        return False

    total = len(day_df)
    is_known_gap = target_date in KNOWN_UPSTREAM_GAPS

    # 假日偵測：全市場 >95% 資料為空視為假日
    if total < MIN_MARKET_ROWS:
        if not is_known_gap:
            logger.warning(f"{target_date} 僅 {total} 筆，可能為假日或異常")
            return False
        logger.warning(f"{target_date} 僅 {total} 筆：{KNOWN_GAP_NOTE}")

    if not _check_ohlc_missing_rate(day_df, target_date):
        return False

    if not _check_price_chip_row_ratio(day_df, target_date, allow_gap=is_known_gap):
        return False

    # OHLC 不變式：low <= min(open, close) 且 high >= max(open, close) 且 high >= low。
    # 這不是統計上的異常，是資料結構性錯誤——任何一列違反都代表 open/close
    # 落在當日高低區間之外，下游 KD、CMF 等以區間為分母的指標會直接跑出定義域
    # 且遞迴污染後續數十個交易日（見 doc/BACKTEST_LOG.md 2026-08-04、
    # repair_ohlc.py 的說明）。發現即視為驗證失敗。
    if {"open", "high", "low", "close"}.issubset(day_df.columns):
        invariant_violation = (
            (day_df["close"] > day_df["high"] + OHLC_EPSILON)
            | (day_df["open"] > day_df["high"] + OHLC_EPSILON)
            | (day_df["close"] < day_df["low"] - OHLC_EPSILON)
            | (day_df["open"] < day_df["low"] - OHLC_EPSILON)
            | (day_df["high"] < day_df["low"] - OHLC_EPSILON)
        )
        n_violation = int(invariant_violation.sum())
        if n_violation > 0:
            bad_ids = day_df.loc[invariant_violation, "stock_id"].tolist()
            logger.error(
                f"OHLC 不變式違反：{n_violation} 筆（close/open 落在 [low, high] 之外，"
                f"或 high < low），股票：{bad_ids[:20]}"
                + ("..." if len(bad_ids) > 20 else "")
            )
            return False

    # 跨日漲跌幅 / 檔數倒退 / 零成交量 —— 三個都是 error，會中止 pipeline
    if not _check_overnight_change(df, day_df, target_date):
        return False

    if not _check_row_count_vs_prev(df, day_df, target_date):
        return False

    if not _check_zero_volume(day_df, target_date):
        return False

    # 當日振幅：保留為輔助資訊。它抓的是同一列內 open→close 的變動，與上面的
    # 跨日檢查是不同的東西，不能互相取代（見 MAX_OVERNIGHT_CHANGE 的說明）。
    if "open" in day_df.columns and "close" in day_df.columns:
        intraday = ((day_df["close"] - day_df["open"]).abs()
                    / day_df["open"].replace(0, float("nan")))
        anomaly = int((intraday > MAX_DAILY_CHANGE).sum())
        if anomaly > 0:
            logger.warning(f"當日振幅 > {MAX_DAILY_CHANGE*100:.0f}%：{anomaly} 筆")

    logger.info(f"{target_date} 價格驗證通過：{total} 筆")
    return True


def validate_chip(target_date: date) -> bool:
    df = read_parquet("chip")
    if df.empty:
        logger.warning("chip.parquet 不存在，略過驗證")
        return True

    day_df = df[df["date"].dt.date == target_date]
    if day_df.empty:
        logger.warning(f"{target_date} 無籌碼資料")
        return False

    logger.info(f"{target_date} 籌碼驗證通過：{len(day_df)} 筆")
    return True


def run(target_date: date) -> bool:
    price_ok = validate_price(target_date)
    chip_ok = validate_chip(target_date)
    ok = price_ok and chip_ok
    if ok:
        logger.info(f"{target_date} 全部驗證通過")
    else:
        logger.error(f"{target_date} 驗證失敗，請檢查資料")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()

    target = parse_date(args.date) if args.date else date.today() - timedelta(days=1)
    ok = run(target)
    exit(0 if ok else 1)
