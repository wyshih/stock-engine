"""
從 FinMind 取得普通股清單（上市+上櫃），排除 ETF/權證/特別股。
資料來源：FinMind TaiwanStockInfo
用法：python fetch_stock_list.py
"""
import os
import logging
import argparse
import re

import pandas as pd
import requests
from dotenv import load_dotenv

from engine.data_source.utils import retry, upsert_parquet

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


@retry(max_attempts=3, base_delay=5.0)
def fetch_raw() -> list[dict]:
    params = {
        "dataset": "TaiwanStockInfo",
        "token": FINMIND_TOKEN,
    }
    resp = requests.get(FINMIND_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != 200:
        raise ValueError(f"FinMind 回傳錯誤：{data.get('msg')}")
    return data["data"]


VALID_MARKETS = {"twse", "tpex"}  # FinMind 回傳值，排除 emerging（興櫃）

def is_common_stock(row: dict) -> bool:
    """判斷是否為普通股（排除 ETF / 權證 / 特別股 / 興櫃）。"""
    sid = str(row.get("stock_id", ""))
    name = str(row.get("stock_name", ""))
    market_type = str(row.get("type", "")).lower()

    # 只接受上市(twse)和上櫃(otc)，排除興櫃(emerging)
    if market_type not in VALID_MARKETS:
        return False

    # 台股普通股代號：4 位純數字，且第一碼為 1~9
    # 以 0 開頭的（0050, 006xx 等）都是 ETF 或特殊商品
    if not re.match(r"^[1-9]\d{3}$", sid):
        return False

    # ETF / 基金名稱關鍵字
    etf_keywords = ["ETF", "基金", "反1", "正2", "貨幣", "債券"]
    if any(k in name for k in etf_keywords):
        return False

    # 特別股
    if name.endswith("特") or "特別股" in name:
        return False

    return True


def process(raw: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    if df.empty:
        return df

    # 先過濾（使用原始欄位名稱 type）
    mask = df.apply(is_common_stock, axis=1)
    df = df[mask].copy()

    # 再 rename
    df = df.rename(columns={
        "stock_id": "stock_id",
        "stock_name": "stock_name",
        "type": "market",
        "industry_category": "industry",
    })

    # market 對應
    df["market"] = df["market"].str.lower().map({"twse": "TWSE", "tpex": "TPEX"}).fillna(df["market"])

    # 預設欄位
    for col in ["is_full_cash", "is_disposed", "is_warning", "is_active"]:
        if col not in df.columns:
            df[col] = False
    df["is_active"] = True

    keep = ["stock_id", "stock_name", "market", "industry",
            "is_active", "is_full_cash", "is_disposed", "is_warning"]
    existing_cols = [c for c in keep if c in df.columns]
    df = df[existing_cols].drop_duplicates(subset=["stock_id"], keep="last")
    return df.reset_index(drop=True)


def run() -> pd.DataFrame:
    logger.info("開始抓取股票清單...")
    raw = fetch_raw()
    df = process(raw)
    if df.empty:
        logger.warning("股票清單為空，略過寫入")
        return df
    upsert_parquet("stock_list", df, keys=["stock_id"])
    logger.info(f"股票清單完成，共 {len(df)} 支普通股")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只顯示不寫入")
    args = parser.parse_args()

    df = run()
    if args.dry_run:
        print(df.head(20).to_string())
