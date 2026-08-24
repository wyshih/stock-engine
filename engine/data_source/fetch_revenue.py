"""
從 MOPS 公開資訊觀測站抓取月營收（IFRS 格式），每月 10 日後執行。
免費、無 API key，逐支股票 POST 請求。
資料來源：https://mopsov.twse.com.tw/mops/web/ajax_t05st10_ifrs
欄位說明（table[1] row index）：
  0=當月營收, 1=上月營收, 2=增減金額, 3=增減%, 4=當月累積, 5=去年累積
用法：
  python fetch_revenue.py --date 2026-06-12                       # 每日模式（逐支股票）
  python fetch_revenue.py --bulk-start 2019-01 --bulk-end 2021-12 # 批次模式（整月一次取回）

批次模式改用 MOPS 的整月彙總頁 t21sc03，一個月份 4 次請求（上市/上櫃 × 國內/國外公司）
就能拿到全部公司，補歷史資料時比逐支股票快約 500 倍。

已知限制：MOPS 會用「現行公開發行公司名單」重新產生歷史彙總頁，已下市/合併消滅的
公司會被從歷史檔案中移除（實測：2888 新光金在 2021-03 的檔案裡已不存在，同業 2891
中信金則在）。因此批次模式只補得到現仍存在的公司，已下市個股的歷史營收需靠每日模式
在當時抓下來的存量。
"""
import io
import logging
import argparse
import time
from datetime import date, timedelta

import pandas as pd
import requests

from engine.data_source.utils import retry, upsert_parquet, parse_date, read_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MOPS_URL = "https://mopsov.twse.com.tw/mops/web/ajax_t05st10_ifrs"
# 整月彙總頁：{market} 為 sii(上市) / otc(上櫃)，日期為民國年_月，結尾 {suffix} 為公司別。
# 資料來源：https://mopsov.twse.com.tw/nas/t21/{sii,otc}/t21sc03_<民國年>_<月>_<0|1>.html
MOPS_BULK_URL = (
    "https://mopsov.twse.com.tw/nas/t21/{market}/t21sc03_{roc_year}_{month}_{suffix}.html"
)
# MOPS 把同一個月的營收統計表拆成兩份檔案（CSV 下載鈕的說明：「檔案內容包含國內及國外公司」）：
#   _0 = 國內公司、_1 = 國外公司（第一上市/上櫃的 -KY 公司與 -DR 存託憑證）。
# 舊版只抓 _0，導致 121 檔仍在市交易的外國公司（120 檔 -KY + 9105 泰金寶-DR）
# 完全沒有月營收。實測 2019-01~2026-07 兩個市場的 _1 檔皆存在（_2 為 404）。
MOPS_BULK_SUFFIXES = ("0", "1")
MOPS_INDEX = "https://mopsov.twse.com.tw/mops/web/index"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": MOPS_INDEX,
}

_session: requests.Session | None = None


def _new_session() -> requests.Session:
    """建立新 session 並訪問首頁取得 cookie。"""
    global _session
    _session = requests.Session()
    _session.headers.update(HEADERS)
    try:
        _session.get(MOPS_INDEX, timeout=10)
    except Exception:
        pass
    return _session


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _new_session()
    return _session


@retry(max_attempts=3, base_delay=3.0)
def fetch_one(stock_id: str, typek: str, roc_year: int, month: int) -> float | None:
    """
    抓取單支股票月營收（千元）。
    typek: "sii" 上市 / "otc" 上櫃
    roc_year: 民國年（西元年 - 1911）
    """
    r = _get_session().post(
        MOPS_URL,
        data={
            "encodeURIComponent": "1",
            "step": "1",
            "firstin": "1",
            "off": "1",
            "isQuery": "Y",
            "TYPEK": typek,
            "year": str(roc_year),
            "month": f"{month:02d}",
            "co_id": stock_id,
        },
        timeout=20,
    )
    r.raise_for_status()
    r.encoding = "big5"

    try:
        tables = pd.read_html(io.StringIO(r.text))
    except ValueError:
        return None  # 無 table（公司未揭露或查無資料）

    # 找 2 欄且列數 >= 4 的表：即月營收摘要表
    data_t = next((t for t in tables if t.shape[1] == 2 and t.shape[0] >= 4), None)
    if data_t is None:
        return None

    try:
        return float(data_t.iloc[0, 1])
    except (ValueError, TypeError):
        return None


def _already_fetched(rev_year: int, month: int) -> bool:
    existing = read_parquet("revenue")
    if existing.empty or "revenue_year" not in existing.columns:
        return False
    return bool(
        ((existing["revenue_year"] == rev_year) & (existing["revenue_month"] == month)).any()
    )


def run(target_date: date, dry_run: bool = False) -> pd.DataFrame:
    if target_date.day < 10:
        logger.info(f"{target_date} 未到月營收公告期（每月 10 日後），略過")
        return pd.DataFrame()

    # 上個月的年/月（本月 10 日後公告的是上月營收）
    prev = target_date.replace(day=1) - timedelta(days=1)
    roc_year = prev.year - 1911
    month = prev.month
    rev_year = prev.year

    if not dry_run and _already_fetched(rev_year, month):
        logger.info(f"{rev_year}-{month:02d} 月營收已存在，略過")
        return pd.DataFrame()

    stock_list = read_parquet("stock_list")
    if stock_list.empty:
        raise RuntimeError("stock_list.parquet 不存在，請先執行 fetch_stock_list.py")

    logger.info(f"開始抓取 {rev_year}-{month:02d} 月營收，共 {len(stock_list)} 支...")
    logger.info("  策略：每批 50 筆，批次間隔 60s，每筆間隔 1.5s（MOPS 速率限制）")

    BATCH = 50        # MOPS 約 70 筆後強制斷線，保守取 50
    BATCH_PAUSE = 60  # 批次間讓 server 冷卻

    records = []
    total = len(stock_list)
    rows = list(stock_list.iterrows())

    for i, (_, row) in enumerate(rows):
        # 每批開始前重建 session
        if i % BATCH == 0:
            if i > 0:
                logger.info(f"  批次暫停 {BATCH_PAUSE}s...")
                time.sleep(BATCH_PAUSE)
            _new_session()

        sid = str(row["stock_id"])
        typek = "sii" if str(row.get("market", "TWSE")) == "TWSE" else "otc"
        try:
            revenue = fetch_one(sid, typek, roc_year, month)
            if revenue is not None:
                records.append({
                    "announce_date": pd.Timestamp(target_date),
                    "stock_id": sid,
                    "revenue": revenue,
                    "revenue_month": month,
                    "revenue_year": rev_year,
                })
        except Exception as e:
            logger.warning(f"{sid} 月營收失敗：{e}")

        if (i + 1) % 100 == 0:
            logger.info(f"  進度 {i+1}/{total}，已取得 {len(records)} 筆")
        time.sleep(1.5)  # MOPS 速率限制：1.5s/筆

    if not records:
        logger.warning("無月營收資料")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if dry_run:
        logger.info(f"[dry-run] 不寫入，共 {len(df)} 筆")
    else:
        upsert_parquet("revenue", df, keys=["announce_date", "stock_id"])
        logger.info(f"月營收完成：{len(df)} 筆（{rev_year}-{month:02d}）")
    return df


# ---------------------------------------------------------------------------
# 批次模式：整月彙總頁
# ---------------------------------------------------------------------------

BULK_INTERVAL = 3.0  # 秒；批次模式請求極少，仍保持禮貌間隔


def bulk_url(market: str, roc_year: int, month: int, suffix: str = "0") -> str:
    """組出整月彙總頁網址。抽成函式是為了讓 URL 契約可以被單元測試釘住。"""
    return MOPS_BULK_URL.format(
        market=market, roc_year=roc_year, month=month, suffix=suffix
    )


@retry(max_attempts=3, base_delay=10.0)
def fetch_bulk_month(market: str, roc_year: int, month: int,
                     suffix: str = "0") -> pd.DataFrame:
    """取回某市場某月某公司別的全部公司營收。回傳 stock_id / revenue 兩欄。"""
    url = bulk_url(market, roc_year, month, suffix)
    resp = requests.get(url, headers=HEADERS, timeout=60)
    # 404 代表該月該公司別沒有檔案（例如更早年份沒有外國公司），視為空結果，
    # 不要讓 retry 白白重試三次。
    if resp.status_code == 404:
        return pd.DataFrame(columns=["stock_id", "revenue"])
    resp.raise_for_status()
    resp.encoding = "big5"

    tables = pd.read_html(io.StringIO(resp.text))
    frames = []
    for t in tables:
        if t.shape[1] < 10 or len(t) < 2:
            continue  # 版面用的小表
        t = t.copy()
        t.columns = [
            " ".join(str(c) for c in col if "Unnamed" not in str(c)).strip()
            if isinstance(col, tuple) else str(col)
            for col in t.columns
        ]
        id_col = next((c for c in t.columns if "代號" in c), None)
        rev_col = next((c for c in t.columns if "當月營收" in c and "累計" not in c), None)
        if id_col is None or rev_col is None:
            continue
        sub = pd.DataFrame({
            "stock_id": t[id_col].astype(str).str.strip(),
            "revenue": pd.to_numeric(t[rev_col], errors="coerce"),
        })
        # 濾掉合計列與非個股代號
        sub = sub[sub["stock_id"].str.match(r"^\d{4,6}$") & sub["revenue"].notna()]
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["stock_id", "revenue"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["stock_id"])


def run_bulk(start_ym: str, end_ym: str, dry_run: bool = False) -> pd.DataFrame:
    """start_ym / end_ym 格式 YYYY-MM，含頭尾。"""
    months = pd.period_range(start_ym, end_ym, freq="M")
    logger.info(f"批次模式：{start_ym} ~ {end_ym}，共 {len(months)} 個月")

    all_rows = []
    for i, pm in enumerate(months, 1):
        roc_year, month = pm.year - 1911, pm.month
        got = 0
        for market in ("sii", "otc"):
            for suffix in MOPS_BULK_SUFFIXES:
                try:
                    df = fetch_bulk_month(market, roc_year, month, suffix)
                except Exception as e:
                    logger.warning(f"  {pm} {market}_{suffix} 失敗：{e}")
                    continue
                time.sleep(BULK_INTERVAL)
                if df.empty:
                    continue
                # 營收於次月 10 日公告
                announce = (pm + 1).to_timestamp() + pd.Timedelta(days=9)
                df["announce_date"] = announce
                df["revenue_month"] = month
                df["revenue_year"] = pm.year
                all_rows.append(df)
                got += len(df)
        logger.info(f"  [{i}/{len(months)}] {pm}（民國{roc_year}年{month}月）：{got} 筆")

    if not all_rows:
        logger.warning("批次模式無資料")
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)[
        ["announce_date", "stock_id", "revenue", "revenue_month", "revenue_year"]
    ]
    if dry_run:
        logger.info(f"[dry-run] 共 {len(out)} 筆，{out['stock_id'].nunique()} 檔")
    else:
        upsert_parquet("revenue", out, keys=["announce_date", "stock_id"])
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--bulk-start", type=str, default=None,
                        help="批次模式起始月份 YYYY-MM（需搭配 --bulk-end）")
    parser.add_argument("--bulk-end", type=str, default=None,
                        help="批次模式結束月份 YYYY-MM，含當月")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.bulk_start or args.bulk_end:
        if not (args.bulk_start and args.bulk_end):
            parser.error("--bulk-start 和 --bulk-end 必須一起指定")
        df = run_bulk(args.bulk_start, args.bulk_end, dry_run=args.dry_run)
        if args.dry_run and not df.empty:
            print(df.head(10).to_string())
        import sys; sys.exit(0)

    target = parse_date(args.date) if args.date else date.today()
    df = run(target, dry_run=args.dry_run)
    if args.dry_run and not df.empty:
        print(df.head(10).to_string())
