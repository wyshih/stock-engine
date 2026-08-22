"""抓公司行動事件表（除權息、減資、分割）→ data/exright.parquet

用途有兩個：
1. `validate_data.py` 的跨日漲跌幅檢查拿它當白名單 —— 除權息當天的跳空是合法的。
2. 之後算報酬率時用 `ratio` 把價格序列還原成可連續比較的形式。

為什麼需要三個端點（2026-08-21 實測）：
    exRight/TWT49U     除權除息（配股配息）      8,559 筆 / 2019~2026
    reducation/TWTAUU  減資恢復買賣              例：2371 大同 40.15 → 41.73
    change/TWTB8U      面額變更、股票分割        例：4763 材料-KY 885.00 → 88.50

只用 TWT49U 的話，官方版 530 筆上市跳空只解釋得了 333 筆（62.8%）—— 剩下的
197 筆全是分割與減資，那不是「除權息」，在 TWT49U 裡查不到。

⚠️ 這三支都只涵蓋**上市**。上櫃的公司行動 TPEx 沒有對應的歷史查詢端點（openapi
的兩支只有最近兩週），但上櫃的日行情把「漲跌」欄位寫成「除權」／「除息」，旗標
可以從行情檔本身取得。

`ratio` 的意義：恢復買賣參考價 ÷ 停止買賣前收盤價。要把事件日之前的舊價格換算成
事件後的基準，乘上 ratio 即可。
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests


from engine.data_source.utils import retry, upsert_parquet  # noqa: E402

logger = logging.getLogger(__name__)

BASE = "https://www.twse.com.tw/rwd/zh"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.twse.com.tw/"}
OUT_NAME = "exright"
THROTTLE_SECONDS = 3.0
FIRST_YEAR = 2019

# (端點, 事件類型, 前收盤價欄名, 參考價欄名)。三張表的欄位名稱不同但語意相同。
SOURCES = (
    ("exRight/TWT49U", "除權息", "除權息前收盤價", "除權息參考價"),
    ("reducation/TWTAUU", "減資", "停止買賣前收盤價格", "恢復買賣參考價"),
    ("change/TWTB8U", "分割", "停止買賣前收盤價格", "恢復買賣參考價"),
)

_ROC_PATTERNS = (
    re.compile(r"(\d+)年(\d+)月(\d+)日"),      # 115年07月01日
    re.compile(r"(\d+)/(\d+)/(\d+)"),          # 114/06/30
)


def _parse_roc_date(text: str) -> pd.Timestamp:
    """民國年轉西元。兩張表用不同格式，所以兩種都試。"""
    text = str(text).strip()
    for pattern in _ROC_PATTERNS:
        matched = pattern.match(text)
        if matched:
            year, month, day = (int(group) for group in matched.groups())
            return pd.Timestamp(year + 1911, month, day)
    return pd.NaT


def _to_float(value) -> float:
    text = str(value).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return float("nan")


@retry(max_attempts=3, base_delay=5.0)
def _get_json(path: str, params: dict) -> dict:
    response = requests.get(f"{BASE}/{path}", params=params, headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_source(path: str, kind: str, prev_field: str, ref_field: str,
                 year: int) -> pd.DataFrame:
    """一個端點一年份。這三張表都吃 startDate/endDate，一次一整年沒問題。"""
    payload = _get_json(path, {"startDate": f"{year}0101", "endDate": f"{year}1231",
                               "response": "json"})
    status = str(payload.get("stat", "")).strip()
    if status.upper() != "OK":
        raise RuntimeError(f"{path} {year} 回傳狀態 {status!r}")

    fields = [str(name).strip() for name in payload.get("fields", [])]
    rows = payload.get("data") or []
    if not rows:
        return pd.DataFrame()
    missing = {prev_field, ref_field, "股票代號"} - set(fields)
    if missing:
        raise RuntimeError(f"{path} {year} 缺少欄位 {missing}，端點可能改版")

    index = {name: position for position, name in enumerate(fields)}
    date_field = next(name for name in fields if "日期" in name)
    frame = pd.DataFrame({
        "date": [_parse_roc_date(row[index[date_field]]) for row in rows],
        "stock_id": [str(row[index["股票代號"]]).strip() for row in rows],
        "prev_close": [_to_float(row[index[prev_field]]) for row in rows],
        "ref_price": [_to_float(row[index[ref_field]]) for row in rows],
    })
    frame["kind"] = kind
    return frame.dropna(subset=["date"])


def build(first_year: int, last_year: int) -> pd.DataFrame:
    frames = []
    for path, kind, prev_field, ref_field in SOURCES:
        for year in range(first_year, last_year + 1):
            frame = fetch_source(path, kind, prev_field, ref_field, year)
            logger.info(f"  {kind:5s} {year}: {len(frame)} 筆")
            frames.append(frame)
            time.sleep(THROTTLE_SECONDS)

    events = pd.concat(frames, ignore_index=True)
    # ratio = 事件後參考價 ÷ 事件前收盤價。把事件日之前的舊價乘上它即可換算成
    # 事件後的基準；連乘多個事件就能把整段歷史還原到今天的基準。
    events["ratio"] = events["ref_price"] / events["prev_close"].replace(0, float("nan"))
    events = events.dropna(subset=["ratio"]).sort_values(["date", "stock_id", "kind"])
    return events.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-year", type=int, default=FIRST_YEAR)
    parser.add_argument("--last-year", type=int, default=pd.Timestamp.today().year)
    parser.add_argument("--out", default=OUT_NAME)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    events = build(args.first_year, args.last_year)
    upsert_parquet(args.out, events, keys=["date", "stock_id", "kind"])
    logger.info(f"完成：{len(events):,} 筆事件"
                f"（{events['date'].min().date()} ~ {events['date'].max().date()}）")
    logger.info(f"各類型：{events['kind'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
