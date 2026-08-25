"""找出「無法用公司行動解釋」的跨日跳空，記錄下來供下游遮蔽。

## 為什麼需要這支

上櫃的**分割與減資沒有任何官方歷史來源**。2026-08-25 查證過：

- `exright.parquet`（TWSE `exRight/TWT49U` + `reducation/TWTAUU` + `change/TWTB8U`）
  只涵蓋上市 —— TWSE 1,063/1,218 檔有事件，TPEX 只有 1/917
- `price_official.parquet` 的 `ex_flag` 補得到上櫃的**除權息**（5,024 筆），
  但補不到分割與減資
- TPEx OpenAPI 只有 `/tpex_exright_daily`（上櫃股票除權除息計算結果表），
  **實測只回當日快照 2 天、不能回溯**，且同樣不含分割減資
- TPEx 的 `www/zh-tw/afterTrading/exRight` 等路徑實測全是 404
  （⚠️ TPEx 對不存在的路徑回 HTTP 200 + HTML 404 頁，不要被騙）

所以只能偵測，不能查表。

## 判準

一筆跨日跳空要被列為「無法解釋」，必須四個條件同時成立：

1. `|報酬| > THRESHOLD`
2. 前後是**相鄰交易日**（跨停牌的價格落差不是當日報酬，本來就不該當報酬看）
3. 該股**上市已滿 NEW_LISTING_BARS 個交易日**（新股前五日無漲跌幅限制，
   +46% 那種是真實報酬，不是錯誤）
4. `exright` 與 `ex_flag` 兩份白名單都沒有當日事件

實測全歷史 874 筆 `|報酬|>11%` 的分類：
    exright 事件         418
    ex_flag（多為上櫃）   248
    新上市 5 日內          77
    跨停牌                123
    ★ 無法解釋              8   ← 只有這些會被記錄
其中 `|報酬|>30%` 只有 1 筆：3293 鈊象 2024-07-26（1465→786，1:2 分割）。

## 為什麼不直接放寬 `_clean_return` 的門檻

`build_price_features._clean_return` 是用 ±100% 濾掉公司行動階梯。把門檻壓到
±30% 會**誤殺新上市股的真實報酬**（實測最大 +46%）。判準必須包含「有沒有事件
可以解釋」，那是單看報酬值做不到的。

## 輸出

`data/suspect_jumps.csv`，欄位 `date,stock_id,prev_close,close,ret`。
**不刪任何資料** —— 這份清單是可查的記錄，由 `build_price_features` 讀取後
把對應的報酬設為 NaN（而不是留著假值）。
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from engine.paths import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 判定門檻。台股漲跌幅限制 ±10%，留一點餘裕避免邊界誤判
THRESHOLD = 0.11
# 上市未滿這麼多個交易日就不判定（前五日無漲跌幅限制，實測有 +46% 的真實報酬）
NEW_LISTING_BARS = 5
OUT_NAME = "suspect_jumps"


def find_unexplained(price: pd.DataFrame, exright: pd.DataFrame,
                     official: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """回傳無法用公司行動解釋的跨日跳空。"""
    px = price[["date", "stock_id", "close"]].sort_values(["stock_id", "date"]).copy()
    px["prev_close"] = px.groupby("stock_id")["close"].shift(1)
    px["prev_date"] = px.groupby("stock_id")["date"].shift(1)
    px["ret"] = px["close"] / px["prev_close"] - 1

    # 交易日序號：用來判斷「相鄰」與「上市幾天」，不能用日曆天（週末假日會誤判）
    calendar = pd.DatetimeIndex(sorted(px["date"].unique()))
    pos = pd.Series(range(len(calendar)), index=calendar)
    px["gap"] = px["date"].map(pos) - px["prev_date"].map(pos)
    first_day = px.groupby("stock_id")["date"].transform("min")
    px["bars_listed"] = px["date"].map(pos) - first_day.map(pos)

    big = px[px["ret"].abs() > threshold].copy()
    big = big[(big["gap"] == 1) & (big["bars_listed"] > NEW_LISTING_BARS)]

    explained = set()
    if not exright.empty and {"date", "stock_id"} <= set(exright.columns):
        explained |= set(zip(pd.to_datetime(exright["date"]), exright["stock_id"].astype(str)))
    if not official.empty and "ex_flag" in official.columns:
        flagged = official[official["ex_flag"].notna() & (official["ex_flag"] != "---")]
        explained |= set(zip(pd.to_datetime(flagged["date"]), flagged["stock_id"].astype(str)))

    cols = ["date", "stock_id", "prev_close", "close", "ret"]
    if big.empty:
        # 空 DataFrame 的布林索引會把欄位一起丟掉 —— 直接回傳有正確欄位的空表。
        # （2026-08-25 踩到：真實資料永遠有 8 筆，只有測試的空集合才會觸發）
        return pd.DataFrame(columns=cols)

    keys = list(zip(big["date"], big["stock_id"].astype(str)))
    big = big[[k not in explained for k in keys]]
    if big.empty:
        return pd.DataFrame(columns=cols)
    return big[cols].reset_index(drop=True)


def load_suspects() -> set[tuple[pd.Timestamp, str]]:
    """給 `build_price_features` 用：讀清單回傳 (date, stock_id) 集合。

    檔案不存在就回空集合 —— 這份清單是選用的防護，沒有它流程照樣能跑。
    """
    path = DATA_DIR / f"{OUT_NAME}.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, parse_dates=["date"])
    return set(zip(df["date"], df["stock_id"].astype(str)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--dry-run", action="store_true", help="只印，不寫檔")
    args = parser.parse_args()

    price = pd.read_parquet(DATA_DIR / "price.parquet", columns=["date", "stock_id", "close"])
    exright = pd.read_parquet(DATA_DIR / "exright.parquet")
    official = pd.read_parquet(DATA_DIR / "price_official.parquet",
                               columns=["date", "stock_id", "ex_flag"])

    found = find_unexplained(price, exright, official, args.threshold)
    logger.info(f"無法用公司行動解釋的跨日跳空：{len(found)} 筆"
                f"（門檻 ±{args.threshold:.0%}）")
    if not found.empty:
        for _, row in found.sort_values("ret", key=abs, ascending=False).head(10).iterrows():
            logger.info(f"  {row['date'].date()} {row['stock_id']} "
                        f"{row['prev_close']} → {row['close']} ({row['ret']:+.1%})")
    if args.dry_run:
        logger.info("--dry-run，不寫檔")
        return
    path = DATA_DIR / f"{OUT_NAME}.csv"
    found.to_csv(path, index=False)
    logger.info(f"已寫入 {path}")


if __name__ == "__main__":
    main()
