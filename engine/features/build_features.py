"""
合併所有特徵，輸出 features.parquet（PLAN.md 5.16 + 最終 merge）。
執行順序：
  1. build_price_features.py      → price_features.parquet
  2. build_chip_features.py       → chip_features.parquet
  3. build_fundamental_features.py → fundamental_features.parquet + revenue_features.parquet
  4. build_features.py（本腳本）  → features.parquet

Point-in-time 規則：
  - price / chip：直接 join（date, stock_id）
  - fundamental（PER/PBR）：forward-fill（含 NaN 的交易日用最近一筆補）
  - revenue：asof merge on announce_date（只用已公告的最新一筆）

用法：python build_features.py [--full] [--dry-run]
"""
import logging
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 專案路徑一律走 code/paths.py（唯一來源），不要各檔自行推導
from engine.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402


def _read(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _upsert(name: str, df: pd.DataFrame, keys: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read(name)
    combined = pd.concat([existing, df], ignore_index=True) if not existing.empty else df
    combined = (combined
                .drop_duplicates(subset=keys, keep="last")
                .sort_values(keys)
                .reset_index(drop=True))
    path = DATA_DIR / f"{name}.parquet"
    combined.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"寫入 {path}（{len(combined)} 筆）")


# ── merge helpers ─────────────────────────────────────────────────────────────

def _asof_revenue(price_dates: pd.DataFrame, rev_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Point-in-time revenue merge：每個交易日配對最近一筆已公告的月營收特徵。
    price_dates: (date, stock_id)
    rev_feat:    (announce_date, stock_id, revenue_yoy, ...)
    """
    rev_cols = [c for c in rev_feat.columns
                if c not in ("announce_date",)]

    results = []
    for sid, grp in price_dates.groupby("stock_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        rev_sid = rev_feat[rev_feat["stock_id"] == sid].sort_values("announce_date")
        if rev_sid.empty:
            empty = pd.DataFrame(index=grp.index, columns=rev_cols)
            empty["stock_id"] = sid
            empty["date"] = grp["date"]
            results.append(empty)
            continue

        merged = pd.merge_asof(
            grp[["date"]],
            rev_sid.rename(columns={"announce_date": "date"}),
            on="date",
            direction="backward",
        )
        merged["stock_id"] = sid
        results.append(merged)

    return pd.concat(results, ignore_index=True)


def _forward_fill_fundamental(price_dates: pd.DataFrame,
                               fund_feat: pd.DataFrame) -> pd.DataFrame:
    """
    逐支 forward-fill fundamental 特徵到每個交易日。
    """
    fund_cols = [c for c in fund_feat.columns if c not in ("date", "stock_id")]
    results = []

    all_dates = pd.Series(sorted(price_dates["date"].unique()), name="date")

    for sid, grp in price_dates.groupby("stock_id"):
        dates_s = grp[["date"]].sort_values("date")
        fund_sid = fund_feat[fund_feat["stock_id"] == sid].sort_values("date")
        if fund_sid.empty:
            row = pd.DataFrame({"date": dates_s["date"], "stock_id": sid})
            for c in fund_cols:
                row[c] = np.nan
            results.append(row)
            continue

        merged = dates_s.merge(fund_sid, on="date", how="left")
        merged["stock_id"] = sid
        merged[fund_cols] = merged[fund_cols].ffill()
        results.append(merged)

    return pd.concat(results, ignore_index=True)


# ── 5.16 異常狀態 / 流動性特徵 ────────────────────────────────────────────────

def _add_status_features(df: pd.DataFrame, sl: pd.DataFrame,
                          price_feat: pd.DataFrame) -> pd.DataFrame:
    """
    加入 stock_list 靜態旗標與流動性特徵。
    """
    # 2026-08-01 移除：is_full_cash / is_disposed / is_warning 三個旗標整欄全 0
    # ——— stock_list.parquet 的資料來源（FinMind TaiwanStockInfo）根本沒有這些
    # 欄位，從未被填值。全 0 的欄位對模型零貢獻（RF 重要性並列最後），
    # 只是佔位。要有值必須另外去 TWSE 抓每日處置股/警示股/全額交割股名單。

    # limit_hit_rate：20 日漲跌停天數比率（close = high 或 close = low 且 close 異常）
    # 需要 open, close → 從 price 取
    if not price_feat.empty and "return_1d" in price_feat.columns:
        # 借用 return_1d 近似：±9.5% 以上視為觸停板
        ret = price_feat.set_index(["date", "stock_id"])["return_1d"]
        limit_hit = ret.abs() >= 0.095
        limit_rate = limit_hit.groupby(level="stock_id").transform(
            lambda s: s.rolling(20, min_periods=5).mean()
        )
        limit_df = limit_rate.reset_index().rename(columns={"return_1d": "limit_hit_rate"})
        df = df.merge(limit_df, on=["date", "stock_id"], how="left")

    # is_low_liquidity：20日均成交金額 < 5000萬
    if not price_feat.empty and "avg_vol_ratio" in price_feat.columns:
        # 若有 price 的 volume 和 close，可以算成交金額；暫用 vol_ratio 代理
        pass  # 留待有 amount 欄位時補充

    return df


# ── 補算衍生特徵（需跨 parquet 資料） ─────────────────────────────────────────

def _add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """需要同時有 price + chip 才能算的特徵。"""
    # margin_slope_5d = margin_slope_5d_raw / close（chip 存的是絕對斜率）
    if "margin_slope_5d_raw" in df.columns and "close_ma20_ratio" in df.columns:
        # 用 close_ma20_ratio × MA20 估 close 不現實；直接保留 raw 版本並改名
        df = df.rename(columns={"margin_slope_5d_raw": "margin_slope_5d"})

    # turnover_ratio（若有市值）
    # 暫缺 market_cap，留空

    return df


# ── 跨股票排名特徵（同一天，這支股票在全市場的相對位置）───────────────────────
# rs_Nd 是跟大盤指數比的超額報酬；這裡是跟「當天所有其他股票」比的百分位排名，
# 兩者訊號不同：rs_Nd 抓的是「有沒有跑贏大盤」，rank 抓的是「這波動能在全市場算強還是弱」
RANK_SOURCE_COLS = ["return_5d", "return_20d", "return_60d", "rs_20d", "rs_60d", "avg_vol_ratio"]


def _add_cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    for col in RANK_SOURCE_COLS:
        if col not in df.columns:
            continue
        df[f"rank_{col}"] = (
            df.groupby("date")[col].rank(pct=True).astype("float32")
        )
    return df


# ── 找出需要補算的日期 ─────────────────────────────────────────────────────────

def _new_dates(price_feat: pd.DataFrame, full: bool) -> set:
    existing = _read("features")
    if full or existing.empty:
        return set(price_feat["date"].dt.normalize().unique())
    existing["date"] = pd.to_datetime(existing["date"])
    done = set(existing["date"].dt.normalize().unique())
    return set(price_feat["date"].dt.normalize().unique()) - done


# ── entry point ───────────────────────────────────────────────────────────────

def run(full: bool = False, dry_run: bool = False) -> pd.DataFrame:
    price_feat = _read("price_features")
    if price_feat.empty:
        raise RuntimeError("price_features.parquet 不存在，請先執行 build_price_features.py")

    price_feat["date"] = pd.to_datetime(price_feat["date"])
    new_dates = _new_dates(price_feat, full)
    if not new_dates:
        logger.info("features.parquet 已是最新")
        return pd.DataFrame()

    logger.info(f"合併 {len(new_dates)} 個交易日...")
    base = price_feat[price_feat["date"].dt.normalize().isin(new_dates)].copy()

    # ── join chip features ────────────────────────────────────────────────────
    chip_feat = _read("chip_features")
    if not chip_feat.empty:
        chip_feat["date"] = pd.to_datetime(chip_feat["date"])
        base = base.merge(chip_feat, on=["date", "stock_id"], how="left")
        logger.info("chip_features 合併完成")

    # ── forward-fill fundamental (PER/PBR) ────────────────────────────────────
    fund_feat = _read("fundamental_features")
    if not fund_feat.empty:
        fund_feat["date"] = pd.to_datetime(fund_feat["date"])
        fund_new = base[["date", "stock_id"]].drop_duplicates()
        fund_merged = _forward_fill_fundamental(fund_new, fund_feat)
        fund_cols = [c for c in fund_merged.columns if c not in ("date", "stock_id")]
        base = base.merge(fund_merged[["date", "stock_id"] + fund_cols],
                          on=["date", "stock_id"], how="left")
        logger.info("fundamental_features 合併完成")

    # ── asof merge revenue (point-in-time) ────────────────────────────────────
    rev_feat = _read("revenue_features")
    if not rev_feat.empty:
        rev_feat["announce_date"] = pd.to_datetime(rev_feat["announce_date"])
        price_dates = base[["date", "stock_id"]].drop_duplicates()
        rev_merged = _asof_revenue(price_dates, rev_feat)
        rev_cols = [c for c in rev_merged.columns
                    if c not in ("date", "stock_id", "announce_date")]
        base = base.merge(rev_merged[["date", "stock_id"] + rev_cols],
                          on=["date", "stock_id"], how="left")
        logger.info("revenue_features asof 合併完成")

    # ── ta-lib 全量指標（5.13）────────────────────────────────────────────────
    talib_feat = _read("talib_features")
    if not talib_feat.empty:
        talib_feat["date"] = pd.to_datetime(talib_feat["date"])
        base = base.merge(talib_feat, on=["date", "stock_id"], how="left")
        logger.info("talib_features 合併完成")

    # ── swing / trendline 特徵（5.10~5.12）────────────────────────────────────
    swing_feat = _read("swing_features")
    if not swing_feat.empty:
        swing_feat["date"] = pd.to_datetime(swing_feat["date"])
        base = base.merge(swing_feat, on=["date", "stock_id"], how="left")
        logger.info("swing_features 合併完成")

    # ── 大盤（加權指數）特徵（2026-07-19 新增，供 M1 用）──────────────────────
    # 只有 date 維度（無 stock_id），broadcast 到當天全部股票
    mkt_feat = _read("market_features")
    if not mkt_feat.empty:
        mkt_feat["date"] = pd.to_datetime(mkt_feat["date"])
        base = base.merge(mkt_feat, on="date", how="left")
        logger.info("market_features 合併完成")

    # ── shape 特徵已移除（2026-08-05）────────────────────────────────────────
    # build_shape_features.py 的 KMeans 群心用 FIT_CUTOFF="2023-12-31" 以前的
    # 資料 fit，而新切分的測試期在 2022/2023 —— 那 14 欄是分布層級的洩漏來源。
    # 腳本本身保留供未來參考，但不再併進 features.parquet。

    # ── 自我正規化特徵（2026-07-29 新增，見 build_relative_features.py）───────
    # 「相對這檔股票自己的歷史分布」這個軸，既有特徵幾乎沒有涵蓋
    # （現有的是「跨股票同日排名」與「相對自己的價格水位」兩軸）。
    rel_feat = _read("relative_features")
    if not rel_feat.empty:
        rel_feat["date"] = pd.to_datetime(rel_feat["date"])
        base = base.merge(rel_feat, on=["date", "stock_id"], how="left")
        logger.info("relative_features 合併完成")

    # ── 新定義趨勢線（2026-07-29 新增，見 build_trendline_features.py）───────
    # 修正舊 support_*/resist_* 的定義問題並補上突破事件；與舊版並存供對照。
    tl_feat = _read("trendline_features")
    if not tl_feat.empty:
        tl_feat["date"] = pd.to_datetime(tl_feat["date"])
        base = base.merge(tl_feat, on=["date", "stock_id"], how="left")
        logger.info("trendline_features 合併完成")

    # ── 5.16 status features ─────────────────────────────────────────────────
    sl = _read("stock_list")
    base = _add_status_features(base, sl, price_feat)

    # ── 跨 parquet 衍生特徵 ───────────────────────────────────────────────────
    base = _add_cross_features(base)

    # ── 跨股票排名特徵（2026-07-13 新增）───────────────────────────────────────
    base = _add_cross_sectional_rank(base)

    logger.info(f"features.parquet：{len(base)} 筆 × {len(base.columns)} 欄")

    if dry_run:
        logger.info("[dry-run] 不寫入")
    else:
        _upsert("features", base, keys=["date", "stock_id"])

    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    df = run(full=args.full, dry_run=args.dry_run)
    if args.dry_run and not df.empty:
        print("cols=%d, rows=%d, stocks=%d" % (
            len(df.columns), len(df), df["stock_id"].nunique()))
        print(df.columns.tolist())
