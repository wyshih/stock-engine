"""
從 TWSE 抓取每日本益比、殖利率、本淨比（全市場一次拉取）。
僅涵蓋上市（TWSE）股票；上櫃（TPEX）暫無免費 API，fundamental 欄位為 NaN。
資料來源：TWSE BWIBBU_d https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d
欄位位置（T86 同源版本）：
  0=代號, 2=本益比, 3=殖利率(%), 6=本淨比
用法：
  python fetch_fundamental.py --date 2025-06-24
  python fetch_fundamental.py --start 2026-07-16 --end 2026-07-17   # 區間：TWSE API
      不支援一次查詢一個區間，這裡只是幫忙自動迴圈跑區間內每個交易日。
"""
import logging
import argparse
from datetime import date, timedelta

import pandas as pd
import requests

from engine.data_source.utils import retry, upsert_parquet, parse_date, twse_date_str

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TWSE_BWIBBU_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.twse.com.tw/",
}


@retry(max_attempts=3, base_delay=5.0)
def fetch_raw(target_date: date) -> dict:
    params = {
        "date": twse_date_str(target_date),
        "selectType": "ALL",
        "response": "json",
    }
    resp = requests.get(TWSE_BWIBBU_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def process(raw: dict, target_date: date) -> pd.DataFrame:
    if raw.get("stat") != "OK" or not raw.get("data"):
        return pd.DataFrame()

    rows = raw["data"]
    df = pd.DataFrame(rows)

    # 欄位位置（TWSE BWIBBU_d）：
    # 0=代號, 2=本益比, 3=殖利率(%), 6=本淨比
    col_idx = {
        "stock_id": 0,
        "per": 2,
        "dividend_yield": 3,
        "pbr": 6,
    }
    result = {}
    for name, idx in col_idx.items():
        if idx < df.shape[1]:
            result[name] = df.iloc[:, idx]
    df = pd.DataFrame(result)

    for col in ["per", "dividend_yield", "pbr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""), errors="coerce"
            )

    df = df[df["stock_id"].str.match(r"^[1-9]\d{3}$")].copy()
    df.insert(0, "date", pd.Timestamp(target_date))
    return df.reset_index(drop=True)


def run(target_date: date, dry_run: bool = False) -> pd.DataFrame:
    logger.info(f"開始抓取 {target_date} 基本面...")
    raw = fetch_raw(target_date)
    df = process(raw, target_date)

    if df.empty:
        logger.warning(f"{target_date} 無基本面資料（可能為假日或非交易日）")
        return df

    if dry_run:
        logger.info(f"[dry-run] 不寫入，共 {len(df)} 筆")
    else:
        upsert_parquet("fundamental", df, keys=["date", "stock_id"])
        logger.info(f"基本面完成：{len(df)} 筆（僅上市股）")
    return df


def run_range(start_date: date, end_date: date, dry_run: bool = False) -> pd.DataFrame:
    """依序跑區間內每個交易日（TWSE API 本身不支援區間查詢，這裡只是方便的迴圈包裝）。"""
    import time
    frames = []
    d = start_date
    first = True
    while d <= end_date:
        if not first:
            time.sleep(3)
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
        if args.dry_run:
            print(df.head(10).to_string())
        import sys; sys.exit(0)

    target = parse_date(args.date) if args.date else date.today() - timedelta(days=1)
    df = run(target, dry_run=args.dry_run)
    if args.dry_run and not df.empty:
        print(df.head(10).to_string())
