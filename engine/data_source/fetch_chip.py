"""
從 TWSE 官方 API 抓取三大法人買賣超與融資融券。
免費、無須 API key、歷史資料到 2015 年。
資料來源：
  三大法人 https://www.twse.com.tw/exchangeReport/T86
  融資融券 https://www.twse.com.tw/exchangeReport/MI_MARGN
用法：
  python fetch_chip.py --date 2026-06-27
  python fetch_chip.py --start 2026-07-16 --end 2026-07-17   # 區間：TWSE API
      本身不支援一次查詢一個區間，這裡只是幫忙自動迴圈跑區間內每個交易日，
      天與天之間仍會各自發一次 request（有間隔避免 rate limit）。
"""
import logging
import argparse
from datetime import date, timedelta

import pandas as pd
import requests

from engine.data_source.utils import retry, upsert_parquet, parse_date, twse_date_str

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TWSE_T86_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"
TWSE_MARGN_URL = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.twse.com.tw/",
}


# ---------------------------------------------------------------------------
# 三大法人（T86）
# ---------------------------------------------------------------------------

@retry(max_attempts=3, base_delay=5.0)
def fetch_institutional(target_date: date) -> dict:
    params = {"date": twse_date_str(target_date), "selectType": "ALL", "response": "json"}
    resp = requests.get(TWSE_T86_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def process_institutional(raw: dict, target_date: date) -> pd.DataFrame:
    if raw.get("stat") != "OK" or not raw.get("data"):
        return pd.DataFrame()

    rows = raw.get("data", [])
    df = pd.DataFrame(rows)

    # 欄位位置（TWSE T86 新格式）：
    # 0=代號, 2=外資買進, 3=外資賣出, 4=外資買賣超,
    # 8=投信買進, 9=投信賣出, 10=投信買賣超, 11=自營商買賣超
    col_idx = {
        "stock_id": 0,
        "foreign_buy": 2,
        "foreign_sell": 3,
        "foreign_net": 4,
        "trust_buy": 8,
        "trust_sell": 9,
        "trust_net": 10,
        "dealer_net": 11,
    }
    result = {}
    for name, idx in col_idx.items():
        if idx < df.shape[1]:
            result[name] = df.iloc[:, idx]
    df = pd.DataFrame(result)

    for col in df.columns:
        if col != "stock_id":
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""), errors="coerce"
            )

    df = df[df["stock_id"].str.match(r"^\d{4,5}$")].copy()
    df.insert(0, "date", pd.Timestamp(target_date))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 融資融券（MI_MARGN）
# ---------------------------------------------------------------------------

@retry(max_attempts=3, base_delay=5.0)
def fetch_margin(target_date: date) -> dict:
    params = {"date": twse_date_str(target_date), "selectType": "ALL", "response": "json"}
    resp = requests.get(TWSE_MARGN_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def process_margin(raw: dict, target_date: date) -> pd.DataFrame:
    if raw.get("stat") != "OK":
        return pd.DataFrame()

    # 新版 MI_MARGN 回傳 tables 結構，tables[1] 為個股明細
    tables = raw.get("tables", [])
    if len(tables) < 2 or not tables[1].get("data"):
        return pd.DataFrame()

    rows = tables[1]["data"]
    df = pd.DataFrame(rows)

    # 欄位位置（TWSE MI_MARGN 新格式）：
    # 0=代號, 1=名稱, 2=融資買進, 3=融資賣出, 6=融資今日餘額,
    # 8=融券買進, 9=融券賣出, 12=融券今日餘額
    col_idx = {
        "stock_id": 0,
        "margin_buy": 2,
        "margin_sell": 3,
        "margin_balance": 6,
        "short_buy": 8,
        "short_sell": 9,
        "short_balance": 12,
    }
    result = {}
    for name, idx in col_idx.items():
        if idx < df.shape[1]:
            result[name] = df.iloc[:, idx]
    df = pd.DataFrame(result)

    for col in df.columns:
        if col != "stock_id":
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""), errors="coerce"
            )

    df = df[df["stock_id"].str.match(r"^\d{4,5}$")].copy()
    df.insert(0, "date", pd.Timestamp(target_date))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 合併寫入
# ---------------------------------------------------------------------------

def run(target_date: date, dry_run: bool = False) -> pd.DataFrame:
    logger.info(f"開始抓取 {target_date} 籌碼資料...")

    inst_raw = fetch_institutional(target_date)
    inst_df = process_institutional(inst_raw, target_date)

    margin_raw = fetch_margin(target_date)
    margin_df = process_margin(margin_raw, target_date)

    if inst_df.empty and margin_df.empty:
        logger.warning(f"{target_date} 無籌碼資料（可能為假日）")
        return pd.DataFrame()

    if inst_df.empty or margin_df.empty:
        df = inst_df if not inst_df.empty else margin_df
    else:
        df = inst_df.merge(margin_df, on=["date", "stock_id"], how="outer")

    if dry_run:
        logger.info(f"[dry-run] 不寫入，共 {len(df)} 筆")
    else:
        upsert_parquet("chip", df, keys=["date", "stock_id"])
        logger.info(f"籌碼完成：{len(df)} 筆")
    return df


def run_range(start_date: date, end_date: date, dry_run: bool = False) -> pd.DataFrame:
    """依序跑區間內每個交易日（TWSE API 本身不支援區間查詢，這裡只是方便的迴圈包裝）。"""
    import time
    frames = []
    d = start_date
    first = True
    while d <= end_date:
        if not first:
            time.sleep(3)  # 天與天之間間隔，避免 rate limit
        first = False
        df = run(d, dry_run=dry_run)
        if not df.empty:
            frames.append(df)
        d += timedelta(days=1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--start", type=str, default=None, help="區間模式起始日（需搭配 --end）")
    parser.add_argument("--end", type=str, default=None, help="區間模式結束日，含當天（需搭配 --start）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.start or args.end:
        if not (args.start and args.end):
            parser.error("--start 和 --end 必須一起指定")
        df = run_range(parse_date(args.start), parse_date(args.end), dry_run=args.dry_run)
    else:
        target = parse_date(args.date) if args.date else date.today() - timedelta(days=1)
        df = run(target, dry_run=args.dry_run)
    if args.dry_run:
        print(df.head(10).to_string())
