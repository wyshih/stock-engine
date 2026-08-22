"""`price_official.parquet` → `price.parquet`（下游唯一讀的那一份）。

為什麼需要這一支（2026-08-22 新寫）：
換資料源之後 `fetch_price_official.py` 產出的是 `price_official.parquet`，
但所有下游（特徵、label、回測、前端、conditional_stats）讀的都是
`price.parquet`。這一步當初是**手動**做的，repo 裡沒有任何程式 ——
換句話說整條流程有一個斷點，重跑重建不出來。這支把它補上。

規則（從兩份現有檔案反推並逐列驗證，誤差 0）：

    price.parquet = price_official.parquet
                    篩掉 ETF（代號以 "00" 開頭）
                    丟掉 ex_flag 欄
                    欄位順序 date, stock_id, open, high, low, close, volume, amount

2026-08-22 實測：official 2,081 檔、篩完 2,070 檔，差的 11 檔剛好就是
0050/0051/…/0061 這批 ETF；共同股票的列數差 0，TWII 兩邊都 1,854 列，
逐列數值全等。

⚠️ 為什麼判準是代號而不是 `stock_list.parquet`：
`stock_list` 來自 FinMind 的 TaiwanStockInfo，只有**現存**的股票。實測有 26 檔
（1333 / 1566 / 2883 / 2941 / …）還在 `price.parquet` 裡卻已經不在 `stock_list`
裡 —— 它們是真的個股，只是下市或被 FinMind 漏掉了。用 stock_list 當篩子會把這
26 檔的歷史整段刪掉，而下游特徵檔是 upsert 寫入、舊列會殘留，兩邊就對不上
（CLAUDE.md 規則 5 踩過的那種不一致）。代號規則沒有這個問題：官方端點的個股
代號全是 4 碼，ETF 一律 "00" 開頭，個股沒有任何一檔是。

`--prune` 才會**額外**套用 stock_list 篩選（確定要清掉不在名單上的代號時用）。
不管有沒有 --prune，`TWII` 一律保留（`build_market_features.py` 要它）。

冪等：每次都從 `price_official.parquet` 全量重算後覆寫，跑幾次結果都一樣。

用法：
  python -m engine.data_source.promote_price
  python -m engine.data_source.promote_price --prune     # 額外套用 stock_list 篩選
  python -m engine.data_source.promote_price --dry-run   # 只印統計，不寫檔
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from engine.paths import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_NAME = "price_official"
TARGET_NAME = "price"
STOCK_LIST_NAME = "stock_list"

# 下游（build_price_features / build_labels / backtest / 前端）依賴的欄位與順序
OUTPUT_COLUMNS = ["date", "stock_id", "open", "high", "low", "close", "volume", "amount"]
# 大盤指數不在 stock_list 裡，但 build_market_features.py 要它
INDEX_IDS = ("TWII",)
# ETF 的代號前綴。官方端點的個股代號全是 4 碼且沒有任何一檔以 "00" 開頭。
ETF_PREFIX = "00"
# 只在 price_official 裡出現、不屬於個股的欄位
DROP_COLUMNS = ("ex_flag",)


def is_etf(stock_id: str) -> bool:
    return str(stock_id).startswith(ETF_PREFIX)


def allowed_ids(official: pd.DataFrame, stock_list: pd.DataFrame, prune: bool) -> set[str]:
    """可以留在 price.parquet 的代號集合。

    預設只剔除 ETF。`prune=True` 才另外要求代號在 `stock_list` 裡
    （會連帶刪掉已下市的個股，見檔頭說明）。
    """
    keep = {sid for sid in official["stock_id"].astype(str) if not is_etf(sid)}
    if prune:
        keep &= set(stock_list["stock_id"].astype(str))
    return keep | set(INDEX_IDS)


def promote(official: pd.DataFrame, keep: set[str]) -> pd.DataFrame:
    """套用篩選與欄位整理。"""
    missing = [c for c in OUTPUT_COLUMNS if c not in official.columns]
    if missing:
        raise ValueError(f"{SOURCE_NAME}.parquet 缺欄位 {missing}，無法產生 {TARGET_NAME}.parquet")

    out = official[official["stock_id"].astype(str).isin(keep)].copy()
    out = out.drop(columns=[c for c in DROP_COLUMNS if c in out.columns])
    out = out[OUTPUT_COLUMNS]
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["date", "stock_id"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prune", action="store_true",
                        help="額外套用 stock_list 篩選（會刪掉已下市的個股，見檔頭說明）")
    parser.add_argument("--dry-run", action="store_true", help="只印統計，不寫檔")
    args = parser.parse_args()

    source = DATA_DIR / f"{SOURCE_NAME}.parquet"
    target = DATA_DIR / f"{TARGET_NAME}.parquet"
    stock_list_path = DATA_DIR / f"{STOCK_LIST_NAME}.parquet"
    for path in (source, stock_list_path):
        if not path.exists():
            raise SystemExit(f"{path} 不存在，請先執行 `make bootstrap` 或 `make update`")

    official = pd.read_parquet(source)
    stock_list = pd.read_parquet(stock_list_path, columns=["stock_id"])

    keep = allowed_ids(official, stock_list, args.prune)
    out = promote(official, keep)

    dropped = sorted(set(official["stock_id"].astype(str)) - set(out["stock_id"].astype(str)))
    unlisted = sorted(set(out["stock_id"].astype(str))
                      - set(stock_list["stock_id"].astype(str)) - set(INDEX_IDS))
    logger.info(f"{SOURCE_NAME}：{len(official):,} 列 / {official['stock_id'].nunique():,} 檔")
    logger.info(f"{TARGET_NAME}  ：{len(out):,} 列 / {out['stock_id'].nunique():,} 檔"
                f"（{out['date'].min().date()} ~ {out['date'].max().date()}）")
    logger.info(f"篩掉 {len(dropped)} 檔 ETF：{', '.join(dropped) if dropped else '（無）'}")
    if unlisted:
        logger.info(f"保留但不在 stock_list 的 {len(unlisted)} 檔（多半是已下市）："
                    f"{', '.join(unlisted[:15])}{' …' if len(unlisted) > 15 else ''}")

    if args.dry_run:
        logger.info("--dry-run，不寫檔")
        return

    tmp = target.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False, engine="pyarrow")
    tmp.replace(target)
    logger.info(f"已寫入 {target}")


if __name__ == "__main__":
    main()
