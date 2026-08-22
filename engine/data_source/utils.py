"""
data_collection 內部共用工具，不依賴 code/ 其他模組。
"""
import time
import logging
import sys
import functools
from pathlib import Path
from datetime import date, datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# 專案路徑一律走 code/paths.py（唯一來源），不要各檔自行推導
from engine.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------

def retry(max_attempts: int = 3, base_delay: float = 5.0):
    """指數退避重試 decorator。"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"{func.__name__} 第 {attempt} 次失敗：{e}，{delay:.0f}s 後重試"
                    )
                    time.sleep(delay)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# parquet 讀寫
# ---------------------------------------------------------------------------

def read_parquet(name: str) -> pd.DataFrame:
    """讀取 data/{name}.parquet，不存在則回傳空 DataFrame。"""
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def upsert_parquet(name: str, new_df: pd.DataFrame, keys: list[str]) -> None:
    """
    將 new_df 合併進 data/{name}.parquet。
    以 keys 為唯一鍵，重複列保留新的。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = read_parquet(name)
    if existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=keys, keep="last")

    combined = combined.sort_values(keys).reset_index(drop=True)
    path = DATA_DIR / f"{name}.parquet"
    combined.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"已寫入 {path}（{len(combined)} 筆）")


# ---------------------------------------------------------------------------
# 日期工具
# ---------------------------------------------------------------------------

def parse_date(d) -> date:
    """將字串、datetime 或 date 統一轉成 date。"""
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(str(d), "%Y-%m-%d").date()


def twse_date_str(d: date) -> str:
    """TWSE API 使用的日期格式 YYYYMMDD。"""
    return d.strftime("%Y%m%d")
