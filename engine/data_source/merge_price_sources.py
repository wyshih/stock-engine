"""把分開抓的行情檔合併成 `price_official.parquet`。

## 為什麼要分開抓

TWSE 與 TPEx 是兩台不同的主機，本來就該並行 —— 但 `fetch_price_official.py`
的預設 `--market both` 是在**同一個行程裡**先打 TWSE、sleep、再打 TPEx、sleep，
等於讓兩台互不相干的主機排隊，每天要 8 秒。

舊 repo 的做法（`logs/backfill_twse.log` 與 `backfill_tpex.log` 的第一行時間戳
同一秒可以證實）是兩個行程並行、各自寫到獨立檔案：

    fetch_price_official --market twse --out price_official_twse   1h55m（3.0s/天）
    fetch_price_official --market tpex --out price_official_tpex   2h27m（4.0s/天）

並行後總耗時是 2h27m，而不是 4h22m。本專案 2026-08-23 的第一次回補用了預設的
`--market both`，花了約 4.4 小時 —— 慢一倍，就是這個原因。

## 為什麼需要這支程式

分開抓就會有三個檔案，而下游（`promote_price.py`）只讀 `price_official.parquet`。
舊 repo 的合併是**手動做的、沒有留下程式**（跟 `promote_price` 當初的情況一樣），
所以整條流程無法從零重現。這支把那個步驟補成程式。

驗證（舊 repo 的實際數字）：
    price_official_twse 1,844,981 + price_official_tpex 1,496,899 + TWII 1,854
    = 3,343,734 = price_official.parquet　✔ 分毫不差

冪等：每次都從來源檔全量重算後覆寫。
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from engine.paths import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 合併順序不影響結果（最後會依 date, stock_id 排序），列出來是為了讓缺檔訊息好讀
SOURCES = ("price_official_twse", "price_official_tpex", "price_official_index")
TARGET = "price_official"
KEYS = ["date", "stock_id"]


def merge(sources: tuple[str, ...] = SOURCES) -> pd.DataFrame:
    frames, missing = [], []
    for name in sources:
        path = DATA_DIR / f"{name}.parquet"
        if not path.exists():
            missing.append(name)
            continue
        frame = pd.read_parquet(path)
        logger.info(f"  {name}: {len(frame):,} 列")
        frames.append(frame)

    if not frames:
        raise SystemExit(
            f"找不到任何來源檔（找過 {', '.join(sources)}）——"
            " 請先跑 make bootstrap，或確認 --out 名稱是否一致")
    if missing:
        # 不當成錯誤：只抓上市或只補指數的情況都合理，但要講出來，
        # 免得少一個來源卻沒人發現（例如上櫃那個 lane 其實失敗了）
        logger.warning(f"  ⚠️ 缺少來源：{', '.join(missing)}（若非刻意，資料會不完整）")

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    # 同一個 (date, stock_id) 只留一筆：來源之間理論上不重疊，重疊時以後者為準
    before = len(out)
    out = out.drop_duplicates(subset=KEYS, keep="last").sort_values(KEYS).reset_index(drop=True)
    if before != len(out):
        logger.warning(f"  ⚠️ 來源間有 {before - len(out):,} 列重複鍵，已去重")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="只印統計，不寫檔")
    args = parser.parse_args()

    out = merge()
    logger.info(f"合併結果：{len(out):,} 列 / {out['stock_id'].nunique()} 檔 / "
                f"{out['date'].min().date()} ~ {out['date'].max().date()}")
    if args.dry_run:
        logger.info("--dry-run，不寫檔")
        return
    path = DATA_DIR / f"{TARGET}.parquet"
    out.to_parquet(path, index=False)
    logger.info(f"已寫入 {path}")


if __name__ == "__main__":
    main()
