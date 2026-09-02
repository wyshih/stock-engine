"""衍生檔的相依圖與過期偵測。

為什麼需要這支：每一層的增量判斷都只看「這個日期在不在」，不看「上游有沒有
變過」。2026-08-29 修籌碼特徵時踩到 —— chip_features 修好了，但 build_features
看到那幾天日期已存在就跳過，修正沒有往下傳，而且**沒有任何錯誤訊息**。當時得
手動去每一層刪掉那幾天，逼它重算。

做法：每次建完衍生檔，把上游檔案的指紋（大小＋修改時間）記在 sidecar；下次要
建之前先比對，對不上就中止並告訴你該下什麼指令。刻意不自動重算 —— 重算範圍可能
很大，而這個專案的經驗是「大聲擋下來」比「安靜做事」可靠。

用法：
  python -m engine.features.pipeline_graph --check       # 列出過期的衍生檔
  python -m engine.features.pipeline_graph --stamp <名稱> # 建完後蓋章
  python -m engine.features.pipeline_graph --invalidate 2026-08-24
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from engine.paths import DATA_DIR

logger = logging.getLogger(__name__)

STAMP_FILE = DATA_DIR / ".pipeline_stamps.json"

# 衍生檔 → 它直接依賴的上游檔案。只列直接上游，遞移關係由圖自己走。
DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "price_features":       ("price",),
    "chip_features":        ("chip", "price", "stock_list"),
    "fundamental_features": ("fundamental", "price"),
    "revenue_features":     ("revenue", "price"),
    "market_features":      ("price",),
    "relative_features":    ("price",),
    "trendline_features":   ("price",),
    "features":             ("price_features", "chip_features", "fundamental_features",
                             "revenue_features", "market_features", "relative_features",
                             "trendline_features"),
    "labels":               ("price",),
    "labels_mdd10":         ("labels", "price"),
    # steady20 不吃 labels.parquet —— 它不是在 label_up20 之上加條件，
    # 而是完全獨立的定義（報酬 vs 自身波動 + 站上 20 日線）。
    "labels_steady20":      ("price",),
}


def _path(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


def fingerprint(name: str) -> str | None:
    """檔案指紋。用大小＋修改時間而不是內容雜湊 —— 這些檔動輒幾百 MB，
    每次建檔前都全檔雜湊會比省下來的重算還慢。"""
    p = _path(name)
    if not p.exists():
        return None
    st = p.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def _stamps() -> dict:
    if not STAMP_FILE.exists():
        return {}
    try:
        return json.loads(STAMP_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning(f"{STAMP_FILE.name} 讀不起來，視為沒有蓋過章")
        return {}


def stamp(name: str) -> None:
    """記下這個衍生檔建立當下，它每個上游的指紋。"""
    data = _stamps()
    data[name] = {up: fingerprint(up) for up in DEPENDENCIES.get(name, ())}
    STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STAMP_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def stale(name: str) -> list[str]:
    """回傳這個衍生檔有哪些上游自從上次建檔後變過。沒蓋過章就回空的
    （第一次跑不該全部報過期）。"""
    recorded = _stamps().get(name)
    if recorded is None or not _path(name).exists():
        return []
    return [up for up, fp in recorded.items() if fingerprint(up) != fp]


def check_all() -> dict[str, list[str]]:
    return {n: s for n in DEPENDENCIES if (s := stale(n))}


def require_fresh(name: str) -> None:
    """建檔前呼叫。上游變過就中止，並告訴使用者怎麼修。"""
    changed = stale(name)
    if not changed:
        return
    raise SystemExit(
        f"{name} 的上游變過了：{', '.join(changed)}。\n"
        f"增量只看日期在不在，不會發現上游被修正過 —— 直接跑會留下舊值。\n"
        f"請先讓 {name} 重算受影響的日期：\n"
        f"  python -m engine.features.pipeline_graph --invalidate <最早受影響的日期>")


def invalidate_from(date: str, names: tuple[str, ...] | None = None) -> dict[str, int]:
    """把指定日期起的列從衍生檔刪掉，逼下次建檔重算那一段。"""
    cutoff = pd.Timestamp(date)
    removed = {}
    for name in (names or tuple(DEPENDENCIES)):
        p = _path(name)
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        keep = df["date"] < cutoff
        n = int((~keep).sum())
        if n:
            df[keep].to_parquet(p, index=False)
            removed[name] = n
            logger.info(f"  {name}：刪掉 {n:,} 列")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--stamp", metavar="NAME")
    parser.add_argument("--invalidate", metavar="DATE")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.stamp:
        stamp(args.stamp)
        logger.info(f"{args.stamp} 已蓋章")
    elif args.invalidate:
        removed = invalidate_from(args.invalidate)
        logger.info(f"共清掉 {len(removed)} 個檔的 {sum(removed.values()):,} 列，"
                    "接著重跑 `make features` 等步驟")
    else:
        bad = check_all()
        if not bad:
            logger.info("所有衍生檔都是新的 ✓")
        else:
            for name, ups in bad.items():
                logger.warning(f"{name} 過期：上游 {', '.join(ups)} 變過")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
