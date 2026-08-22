"""
從 TPEX（證券櫃檯買賣中心）官方 API 抓取上櫃股票的三大法人買賣超與融資融券。
免費、無須 API key。
資料來源：
  三大法人 https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php
      （上櫃股票三大法人買賣明細，逐日查詢，日期為民國年 YYY/MM/DD）
  融資融券 https://www.tpex.org.tw/www/zh-tw/margin/balance
      （上櫃股票融資融券餘額，逐日查詢，日期為民國年 YYY/MM/DD）
輸出欄位跟 fetch_chip.py（TWSE 版）完全一致，可直接 upsert 進同一份 chip.parquet：
  date, stock_id, foreign_buy/sell/net, trust_buy/sell/net, dealer_net,
  margin_buy/sell/balance, short_buy/sell/balance

用法：
  python fetch_chip_tpex.py --date 2026-06-27
  python fetch_chip_tpex.py --start 2022-01-03 --end 2026-07-24   # 區間：TPEX API
      本身不支援一次查詢一段區間，這裡只是幫忙自動迴圈跑區間內每個交易日，
      天與天之間仍會各自發一次 request（有間隔避免 rate limit）。
"""
import logging
import argparse
from datetime import date, timedelta

import pandas as pd
import requests

from engine.data_source.utils import retry, upsert_parquet, parse_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TPEX_3INSTI_URL = "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
TPEX_MARGIN_URL = "https://www.tpex.org.tw/www/zh-tw/margin/balance"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.tpex.org.tw/",
}

STOCK_ID_RE = r"^\d{4,5}$"


def tpex_date_str(d: date) -> str:
    """TPEX API 使用的日期格式：民國年 YYY/MM/DD（西元年 - 1911）。"""
    return f"{d.year - 1911}/{d.month:02d}/{d.day:02d}"


# ---------------------------------------------------------------------------
# 三大法人（上櫃股票三大法人買賣明細表）
# ---------------------------------------------------------------------------

@retry(max_attempts=3, base_delay=5.0)
def fetch_institutional_tpex(target_date: date) -> dict:
    params = {
        "l": "zh-tw",
        "se": "EW",  # 上櫃普通股（含全部）
        "t": "D",
        "d": tpex_date_str(target_date),
        "s": "0,asc,0",
    }
    resp = requests.get(TPEX_3INSTI_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def process_institutional_tpex(raw: dict, target_date: date) -> pd.DataFrame:
    tables = raw.get("tables", [])
    if not tables or not tables[0].get("data"):
        return pd.DataFrame()

    rows = tables[0]["data"]
    df = pd.DataFrame(rows)

    # 欄位位置（TPEX 三大法人買賣明細表，25 欄，0-indexed）：
    # 0=代號, 1=名稱,
    # 2-4  =外資及陸資(不含外資自營商) 買/賣/買賣超
    # 5-7  =外資自營商 買/賣/買賣超
    # 8-10 =外資及陸資合計 買/賣/買賣超   <- 對應 foreign_buy/sell/net
    # 11-13=投信 買/賣/買賣超            <- 對應 trust_buy/sell/net
    # 14-16=自營商(自行買賣) 買/賣/買賣超
    # 17-19=自營商(避險) 買/賣/買賣超
    # 20-22=自營商合計 買/賣/買賣超       <- 22 對應 dealer_net
    # 23   =三大法人買賣超合計
    col_idx = {
        "stock_id": 0,
        "foreign_buy": 8,
        "foreign_sell": 9,
        "foreign_net": 10,
        "trust_buy": 11,
        "trust_sell": 12,
        "trust_net": 13,
        "dealer_net": 22,
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

    df = df[df["stock_id"].astype(str).str.match(STOCK_ID_RE)].copy()
    df.insert(0, "date", pd.Timestamp(target_date))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 融資融券（上櫃股票融資融券餘額）
# ---------------------------------------------------------------------------

@retry(max_attempts=3, base_delay=5.0)
def fetch_margin_tpex(target_date: date) -> dict:
    params = {"date": tpex_date_str(target_date)}
    resp = requests.get(TPEX_MARGIN_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def process_margin_tpex(raw: dict, target_date: date) -> pd.DataFrame:
    tables = raw.get("tables", [])
    if not tables or not tables[0].get("data"):
        return pd.DataFrame()

    rows = tables[0]["data"]
    df = pd.DataFrame(rows)

    # 欄位位置（TPEX 上櫃股票融資融券餘額表，0-indexed）：
    # 0=代號, 1=名稱, 2=前資餘額, 3=資買, 4=資賣, 5=現償, 6=資餘額,
    # 7=資屬證金, 8=資使用率, 9=資限額,
    # 10=前券餘額, 11=券賣, 12=券買, 13=券償, 14=券餘額,
    # 15=券屬證金, 16=券使用率, 17=券限額, 18=資券相抵, 19=備註
    # 注意：融券欄位順序是「券賣」在前、「券買（回補）」在後，
    # 跟 TWSE MI_MARGN（融資融券買進在前）的欄位順序不同，對應到同一組
    # short_buy/short_sell 命名時要對調索引。
    col_idx = {
        "stock_id": 0,
        "margin_buy": 3,
        "margin_sell": 4,
        "margin_balance": 6,
        "short_sell": 11,
        "short_buy": 12,
        "short_balance": 14,
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

    df = df[df["stock_id"].astype(str).str.match(STOCK_ID_RE)].copy()
    df.insert(0, "date", pd.Timestamp(target_date))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 合併寫入
# ---------------------------------------------------------------------------

def run(target_date: date, dry_run: bool = False) -> pd.DataFrame:
    logger.info(f"開始抓取 {target_date} TPEX 籌碼資料...")

    inst_raw = fetch_institutional_tpex(target_date)
    inst_df = process_institutional_tpex(inst_raw, target_date)

    margin_raw = fetch_margin_tpex(target_date)
    margin_df = process_margin_tpex(margin_raw, target_date)

    if inst_df.empty and margin_df.empty:
        logger.warning(f"{target_date} 無 TPEX 籌碼資料（可能為假日）")
        return pd.DataFrame()

    if inst_df.empty or margin_df.empty:
        df = inst_df if not inst_df.empty else margin_df
    else:
        df = inst_df.merge(margin_df, on=["date", "stock_id"], how="outer")

    if dry_run:
        logger.info(f"[dry-run] 不寫入，共 {len(df)} 筆")
    else:
        upsert_parquet("chip", df, keys=["date", "stock_id"])
        logger.info(f"TPEX 籌碼完成：{len(df)} 筆")
    return df


def run_range(start_date: date, end_date: date, dry_run: bool = False) -> pd.DataFrame:
    """依序跑區間內每個交易日（TPEX API 本身不支援區間查詢，這裡只是方便的迴圈包裝）。"""
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
