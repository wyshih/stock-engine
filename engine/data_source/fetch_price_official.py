"""從 TWSE／TPEx 官方端點抓每日行情 —— 真實成交價，不會被追溯改寫。

為什麼要換掉 yfinance（2026-08-20 查證）：
yfinance 的 `Close` **永遠會做分割調整**（`auto_adjust` 只控制配息還原）。某檔股票
一旦發生分割，它回傳的整段歷史就會改成新基準。而我們是增量抓取、只重寫尾端視窗，
舊的列永遠停在舊基準 —— 於是每次分割都在存檔裡留下一道永久的假跳空。

實例：5314 在 2026-08-14 做 1 股拆 4.157 股。
    我們檔案     7/27 收 56.70 → 7/28 收 12.75   （−77.5%，假的）
    TPEx 官方    7/27 收 56.70 → 7/28 收 53.00   （−6.5%，真的）
全市場有 905 / 2,028 檔中招，共 3,599 道接縫，訓練期最重。

官方端點給的是當日真實成交價，任何公司行動都不會回頭改寫，所以增量抓取永遠安全。
除權息的還原留給 `fetch_exright.py` 抓事件表，需要報酬率時再自己算。

端點（2026-08-20 實測可用）：
  上市 https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX  type=ALLBUT0999
  上櫃 https://www.tpex.org.tw/www/zh-tw/afterTrading/otc     type=AL
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests


from engine.data_source.utils import retry, upsert_parquet  # noqa: E402

logger = logging.getLogger(__name__)

TWSE_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/otc"
OUT_NAME = "price_official"

TWSE_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.twse.com.tw/"}
TPEX_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.tpex.org.tw/"}

# 天與天之間的間隔。沿用 fetch_chip.py 的 3 秒 —— 兩個站台同一個來源 IP，
# 抓太快會被擋，而全量回補要跑 1600 多天，被擋一次就得從頭確認缺哪幾天。
THROTTLE_SECONDS = 3.0
# 只留 4 碼純數字的證券代號：排除權證、受益憑證、可轉債等。
# TPEx 那支端點一天回 10,212 列，真正的上櫃股票只有一千出頭。
STOCK_ID_LENGTH = 4
COLUMNS = ["date", "stock_id", "open", "high", "low", "close", "volume", "amount",
           "ex_flag"]
# 每抓幾天才寫一次檔。`upsert_parquet` 每次都會把整份讀出來重寫，逐日寫入在全量
# 回補（約 2000 天）時是 O(n²)：跑到後期每天要重寫四百萬列，等於跑不完。
# 批次累積在記憶體裡，一批約 12 萬列，很小。
BATCH_DAYS = 60


def _to_float(value) -> float:
    """官方端點的數字是含逗號的字串；停牌那天會給 '--' 或空字串。"""
    text = str(value).replace(",", "").strip()
    if text in ("", "--", "---", "N/A", "無"):
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _ex_flag(row: list, index: dict, alias: dict) -> str | None:
    """漲跌欄位非數字時，它寫的就是公司行動的種類。"""
    field = alias.get("change")
    if field is None or field not in index:
        return None
    text = str(row[index[field]]).replace(",", "").strip()
    if not text or text in ("+", "-", "X"):
        return None
    try:
        float(text)
    except ValueError:
        return text
    return None


def _is_stock(code: str) -> bool:
    code = str(code).strip()
    return len(code) == STOCK_ID_LENGTH and code.isdigit()


# 兩個站台的 stat 大小寫不同：TWSE 回 "OK"、TPEx 回 "ok"。第一版寫死比對 "OK"，
# 結果上櫃整個被判成無資料卻不報錯 —— 靜默失敗正是這次要修掉的那類 bug。
OK_STATUSES = {"ok"}
# 休市日兩個站台都會回這類訊息，那是正常的「今天沒有資料」，不是失敗。
NO_DATA_HINTS = ("無資料", "沒有符合", "查無")


@retry(max_attempts=3, base_delay=5.0)
def _get_json(url: str, params: dict, headers: dict) -> dict:
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def _has_data(payload: dict, market: str, day: pd.Timestamp) -> bool:
    """狀態判讀。認不得的狀態一律拋錯，不能安靜地當成休市。"""
    status = str(payload.get("stat", "")).strip()
    if status.lower() in OK_STATUSES:
        return True
    if any(hint in status for hint in NO_DATA_HINTS):
        return False
    raise RuntimeError(f"{market} {day.date()} 回傳未知狀態：{status!r}")


def _extract(payload: dict, day: pd.Timestamp, alias: dict[str, str],
             strip_fields: bool) -> pd.DataFrame:
    """從回傳的多張 table 裡挑出行情那張，轉成統一 schema。

    `alias` 是「統一欄名 → 該站台欄名」的對照；兩個站台欄位名稱不同
    （上市『收盤價』、上櫃『收盤 』），但結構一致，共用這段。
    """
    rows = []
    for table in payload.get("tables", []):
        fields = [str(name).strip() if strip_fields else str(name)
                  for name in table.get("fields", [])]
        if alias["close"] not in fields or alias["stock_id"] not in fields:
            continue
        index = {name: position for position, name in enumerate(fields)}
        for row in table.get("data", []):
            code = str(row[index[alias["stock_id"]]]).strip()
            if not _is_stock(code):
                continue
            rows.append({
                "date": day, "stock_id": code,
                "open": _to_float(row[index[alias["open"]]]),
                "high": _to_float(row[index[alias["high"]]]),
                "low": _to_float(row[index[alias["low"]]]),
                "close": _to_float(row[index[alias["close"]]]),
                "volume": _to_float(row[index[alias["volume"]]]),
                "amount": _to_float(row[index[alias["amount"]]]),
                # 除權息旗標：這兩個站台在除權息日不填漲跌數字，改寫「除權」／
                # 「除息」／「除權息」。TPEx 沒有公司行動的歷史查詢端點，這是
                # 唯一拿得到上櫃除權息日期的來源，別把它丟了。
                "ex_flag": _ex_flag(row, index, alias),
            })
        break
    return pd.DataFrame(rows, columns=COLUMNS)


TWSE_ALIAS = {"stock_id": "證券代號", "open": "開盤價", "high": "最高價",
              "low": "最低價", "close": "收盤價", "volume": "成交股數",
              "amount": "成交金額", "change": "漲跌價差"}
TPEX_ALIAS = {"stock_id": "代號", "open": "開盤", "high": "最高",
              "low": "最低", "close": "收盤", "volume": "成交股數",
              "amount": "成交金額(元)", "change": "漲跌"}


def fetch_twse_day(day: pd.Timestamp) -> pd.DataFrame:
    payload = _get_json(TWSE_URL,
                        {"date": day.strftime("%Y%m%d"), "type": "ALLBUT0999",
                         "response": "json"},
                        TWSE_HEADERS)
    if not _has_data(payload, "上市", day):
        return pd.DataFrame(columns=COLUMNS)
    return _extract(payload, day, TWSE_ALIAS, strip_fields=False)


def fetch_tpex_day(day: pd.Timestamp) -> pd.DataFrame:
    """上櫃的欄位名稱帶了不對稱的空白（'收盤 '、' 成交金額(元)'），要先 strip。"""
    payload = _get_json(TPEX_URL,
                        {"date": day.strftime("%Y/%m/%d"), "type": "AL",
                         "response": "json"},
                        TPEX_HEADERS)
    if not _has_data(payload, "上櫃", day):
        return pd.DataFrame(columns=COLUMNS)
    return _extract(payload, day, TPEX_ALIAS, strip_fields=True)


FETCHERS = {"twse": ("上市", fetch_twse_day), "tpex": ("上櫃", fetch_tpex_day)}


def fetch_day(day: pd.Timestamp, markets: tuple[str, ...] = ("twse", "tpex")) -> pd.DataFrame:
    """單日行情。休市日回空表。

    兩個市場預設一起抓，但 TWSE 與 TPEx 是**不同主機、速率限制各自獨立**，
    排隊輪流跑等於白等一倍的節流時間。全量回補時用 `--market` 拆成兩個行程並行，
    各自寫各自的檔（同時 upsert 同一個檔會互相覆蓋）。
    """
    frames, counts = [], {}
    for name, fetcher in (FETCHERS[m] for m in markets):
        try:
            frame = fetcher(day)
        except Exception as error:                      # noqa: BLE001
            # 單一市場失敗不能靜默變成「休市」—— 那正是這次要修的那類 bug
            logger.error(f"{day.date()} {name} 抓取失敗：{error!r}")
            raise
        frames.append(frame)
        counts[name] = len(frame)
        time.sleep(THROTTLE_SECONDS)

    # 一邊有資料、另一邊卻是 0，只可能是端點改版或被擋，不會是市場現象
    if len(counts) > 1 and any(counts.values()) and not all(counts.values()):
        raise RuntimeError(f"{day.date()} 兩市場列數不一致：{counts}，疑似端點改版")
    return pd.concat(frames, ignore_index=True)


# ── 大盤指數 ──────────────────────────────────────────────────────────────
# `features_v2/common.py` 的 MARKET_ID = "TWII"，整組大盤特徵都靠它。個股行情的
# 端點只回個股，4 碼過濾也會把指數擋掉，所以要另外抓。兩支都是「一次一個月」：
#   TAIEX/MI_5MINS_HIST  日開高低收指數
#   afterTrading/FMTQIK  日成交股數與金額
TAIEX_URL = "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST"
FMTQIK_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK"
MARKET_ID = "TWII"


def _roc_to_date(text: str) -> pd.Timestamp:
    """民國日期 '113/07/01' → 2024-07-01。"""
    parts = str(text).strip().split("/")
    if len(parts) != 3:
        return pd.NaT
    year, month, day = (int(part) for part in parts)
    return pd.Timestamp(year + 1911, month, day)


def fetch_market_month(month_start: pd.Timestamp) -> pd.DataFrame:
    """一個月份的大盤 OHLC + 量額，組成一列 stock_id='TWII' 的行情。"""
    stamp = month_start.strftime("%Y%m01")
    ohlc = _get_json(TAIEX_URL, {"date": stamp, "response": "json"}, TWSE_HEADERS)
    time.sleep(THROTTLE_SECONDS)
    turnover = _get_json(FMTQIK_URL, {"date": stamp, "response": "json"}, TWSE_HEADERS)
    time.sleep(THROTTLE_SECONDS)

    frame = pd.DataFrame([
        {"date": _roc_to_date(row[0]), "open": _to_float(row[1]),
         "high": _to_float(row[2]), "low": _to_float(row[3]),
         "close": _to_float(row[4])}
        for row in (ohlc.get("data") or [])
    ])
    volume = pd.DataFrame([
        {"date": _roc_to_date(row[0]), "volume": _to_float(row[1]),
         "amount": _to_float(row[2])}
        for row in (turnover.get("data") or [])
    ])
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    merged = frame.merge(volume, on="date", how="left").dropna(subset=["date"])
    merged["stock_id"] = MARKET_ID
    merged["ex_flag"] = None
    return merged[COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--dates-file",
                        help="只抓這個檔案裡列的日期（每行一個 YYYY-MM-DD）。"
                             "補抓特定幾天時用，不必重掃整段區間")
    parser.add_argument("--out", default=OUT_NAME,
                        help="輸出的 parquet 名稱（預設 price_official，不動現有 price）")
    parser.add_argument("--batch-days", type=int, default=BATCH_DAYS,
                        help="每幾天寫一次檔（預設 60；設 1 等於逐日寫，很慢）")
    parser.add_argument("--market", choices=["twse", "tpex", "both", "index"], default="both",
                        help="只抓單一市場。兩個站台主機不同、限流獨立，"
                             "全量回補時拆兩個行程並行可省一半時間")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.market == "index":
        months = pd.date_range(args.start, args.end, freq="MS")
        frames = []
        for position, month in enumerate(months, start=1):
            frame = fetch_market_month(month)
            frames.append(frame)
            logger.info(f"  [{position}/{len(months)}] {month.strftime('%Y-%m')} "
                        f"{len(frame)} 個交易日")
        combined = pd.concat(frames, ignore_index=True)
        upsert_parquet(args.out, combined, keys=["date", "stock_id"])
        logger.info(f"完成：大盤 {len(combined):,} 列寫入 {args.out}.parquet")
        return

    markets = ("twse", "tpex") if args.market == "both" else (args.market,)
    if args.dates_file:
        listed = Path(args.dates_file).read_text().split()
        days = pd.DatetimeIndex(sorted(pd.to_datetime(listed).unique()))
        logger.info(f"依清單抓取 {len(days)} 個指定日期")
    elif args.start and args.end:
        days = pd.bdate_range(args.start, args.end)
        logger.info(f"抓取 {args.start} ~ {args.end}，共 {len(days)} 個工作日")
    else:
        parser.error("要給 --start/--end 或 --dates-file")

    def flush(batch: list[pd.DataFrame]) -> None:
        if batch:
            upsert_parquet(args.out, pd.concat(batch, ignore_index=True),
                           keys=["date", "stock_id"])

    empty_days, total_rows = 0, 0
    batch: list[pd.DataFrame] = []
    try:
        for position, day in enumerate(days, start=1):
            frame = fetch_day(day, markets)
            if frame.empty:
                empty_days += 1
                logger.info(f"  [{position}/{len(days)}] {day.date()} 休市或無資料")
                continue
            batch.append(frame)
            total_rows += len(frame)
            logger.info(f"  [{position}/{len(days)}] {day.date()} {len(frame)} 檔"
                        f"（累計 {total_rows:,} 列）")
            if len(batch) >= args.batch_days:
                flush(batch)
                batch = []
                logger.info(f"  ── 已存檔，累計 {total_rows:,} 列")
    finally:
        # 抓取中斷（連線被切、Ctrl-C）時把記憶體裡的批次寫下去。少了這段，
        # 一次崩潰會丟掉最多 BATCH_DAYS 天的成果 —— 2026-08-20 上櫃在
        # 2022-07-18 斷線，已抓到 07-15 卻只存到 06-20，19 天白跑。
        if batch:
            flush(batch)
            logger.info(f"  ── 中斷前存檔，累計 {total_rows:,} 列")

    logger.info(f"完成：{total_rows:,} 列寫入 {args.out}.parquet，{empty_days} 天無資料")


if __name__ == "__main__":
    main()
