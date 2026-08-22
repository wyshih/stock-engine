"""
大盤（加權指數 TWII）特徵，供 M1（大盤方向子模型）使用（PLAN.md 5.7/5.2 概念延伸）。
資料來源：price.parquet 裡 stock_id="TWII" 的列（fetch_price.py 已順便抓 ^TWII）。
輸出：data/market_features.parquet（只有 date + mkt_ 開頭欄位，無 stock_id，
      在 build_features.py 用 date 對齊 broadcast 到每一支股票）。

用法：python build_market_features.py [--full]
"""
import argparse
import sys
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 專案路徑一律走 code/paths.py（唯一來源），不要各檔自行推導
from engine.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402


def build(twii: pd.DataFrame) -> pd.DataFrame:
    twii = twii.sort_values("date").reset_index(drop=True)
    close = twii["close"]

    ma5  = close.rolling(5,  min_periods=3).mean()
    ma10 = close.rolling(10, min_periods=5).mean()
    ma20 = close.rolling(20, min_periods=10).mean()
    ma60 = close.rolling(60, min_periods=30).mean()

    out = pd.DataFrame({"date": twii["date"]})
    out["mkt_close_ma5"]  = close / ma5  - 1
    out["mkt_close_ma10"] = close / ma10 - 1
    out["mkt_close_ma20"] = close / ma20 - 1
    out["mkt_close_ma60"] = close / ma60 - 1
    out["mkt_ma5_ma20"]   = ma5 / ma20 - 1
    out["mkt_ma_bull"]    = ((ma5 > ma10) & (ma10 > ma20)).astype("int8")
    out["mkt_above_ma20"] = (close > ma20).astype("int8")
    out["mkt_above_ma60"] = (close > ma60).astype("int8")
    out["mkt_return_5d"]  = close.pct_change(5)
    out["mkt_return_20d"] = close.pct_change(20)
    out["mkt_return_60d"] = close.pct_change(60)
    out["mkt_vol_20d"]    = close.pct_change().rolling(20, min_periods=10).std()

    return out


def run(full: bool = False) -> pd.DataFrame:
    price = pd.read_parquet(DATA_DIR / "price.parquet")
    price["date"] = pd.to_datetime(price["date"])
    twii = price[price["stock_id"] == "TWII"][["date", "close"]]
    if twii.empty:
        raise RuntimeError("price.parquet 找不到 TWII（大盤指數），請確認 fetch_price.py 有抓 ^TWII")

    out = build(twii)
    out_path = DATA_DIR / "market_features.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"market_features.parquet 寫入：{len(out)} 筆 x {len(out.columns)} 欄")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run(full=args.full)
