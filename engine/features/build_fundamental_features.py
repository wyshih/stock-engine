"""
計算 fundamental_features.parquet（PLAN.md 5.15 節）。
依賴：fundamental.parquet（PER/PBR/殖利率）, revenue.parquet（月營收）
用法：python build_fundamental_features.py [--full] [--dry-run]
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


# ── 5.15 基本面特徵（PER/PBR 部分）────────────────────────────────────────────

# 估值指標的合理上界；超過視為無意義（多半是微利或虧損轉盈的公司）
VALUATION_CAP = {"per": 200.0, "pbr": 50.0, "dividend_yield": 50.0}

# 60 月 rolling percentile 的窗口（以交易日估算，約 60×21=1260 天）
VALUATION_RANK_WINDOW = 1260

# 窗口暖機門檻。原本是 20，等於「只要有 20 天就給一個號稱五年百分位的數字」。
# 但資料只從 2019-01 開始，實測訓練期（2020-01~2022-11）的實際視窗中位數只有
# 362~841 個交易日，測試期（2024+）才滿 1260 —— 同一個欄位在 train 與 test
# 量的是不同長度的分布。35.3% 的測試列百分位位移超過 20 個百分點
# （train/test 對照的 Spearman 只有 0.8229）。
# 視窗未滿就給 NaN，不要給一個語意不同的數字讓模型學到假的分布。
# 420 ≈ 兩年交易日，是「還算得上跨景氣循環」的最低要求。
MIN_VALUATION_PERIODS = 420


def _build_valuation_features(fund: pd.DataFrame) -> pd.DataFrame:
    """
    逐支計算 PER/PBR/殖利率的 60 月 rolling percentile。
    fund 欄位：date, stock_id, per, pbr, dividend_yield
    """
    fund = fund.copy()
    fund["date"] = pd.to_datetime(fund["date"])
    fund = fund.sort_values(["stock_id", "date"]).reset_index(drop=True)

    results = []
    for sid, grp in fund.groupby("stock_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        out = pd.DataFrame({"date": grp["date"], "stock_id": grp["stock_id"]})

        for col, rank_col in [("per", "per_rank"), ("pbr", "pbr_rank"),
                               ("dividend_yield", "yield_rank")]:
            if col in grp.columns:
                s = pd.to_numeric(grp[col], errors="coerce")
                # 2026-08-01 修正：TWSE 對微利公司會給出極端本益比（實測最高
                # 10,795）。超過 VALUATION_CAP 的估值在投資判斷上等同「無意義」，
                # 保留原值只會讓整欄的尺度被少數幾筆拉走。設 NaN。
                s = s.where((s > 0) & (s <= VALUATION_CAP[col]))
                out[col] = s.astype("float32")
                out[rank_col] = s.rolling(
                    VALUATION_RANK_WINDOW, min_periods=MIN_VALUATION_PERIODS
                ).rank(pct=True).astype("float32")
            else:
                out[col] = np.nan
                out[rank_col] = np.nan

        results.append(out)

    return pd.concat(results, ignore_index=True)


# ── 5.15 月營收特徵 ────────────────────────────────────────────────────────────

def _streak(condition: pd.Series) -> pd.Series:
    g = (~condition).cumsum()
    return condition.astype(int).groupby(g).cumsum()


def _add_streak(out: dict, prefix: str, condition: pd.Series) -> None:
    s = _streak(condition)
    out[f"{prefix}_1d"] = condition.astype("int8")
    out[f"{prefix}_3d"] = (s >= 3).astype("int8")
    out[f"{prefix}_5d"] = (s >= 5).astype("int8")
    out[f"{prefix}_10d"] = (s >= 10).astype("int8")
    out[f"{prefix}_log"] = np.log1p(s).astype("float32")


def _build_revenue_features(rev: pd.DataFrame) -> pd.DataFrame:
    """
    月營收特徵，以 announce_date 為基準（point-in-time）。
    rev 欄位：announce_date, stock_id, revenue, revenue_month, revenue_year
    輸出仍以 announce_date 為 key，後續在 build_features.py 用 asof merge 對齊交易日。
    """
    if rev.empty:
        return pd.DataFrame()

    rev = rev.copy()
    rev["announce_date"] = pd.to_datetime(rev["announce_date"])
    rev["revenue"] = pd.to_numeric(rev["revenue"], errors="coerce")
    rev = rev.sort_values(["stock_id", "announce_date"]).reset_index(drop=True)

    results = []
    for sid, grp in rev.groupby("stock_id"):
        grp = grp.sort_values("announce_date").reset_index(drop=True)
        r = grp["revenue"]

        out: dict = {
            "announce_date": grp["announce_date"],
            "stock_id": grp["stock_id"],
            "revenue_month": grp["revenue_month"],
            "revenue_year": grp["revenue_year"],
        }

        # YoY（同月去年比較）：shift 12 個月
        # 2026-08-01 修正：只擋「恰好為 0」不夠。基期營收極小的公司會讓年增率
        # 爆到 2.5e7%（實測）。營收單位為千元，基期低於 1000 千元（100 萬元）
        # 的月份視為無意義，設 NaN。
        _MIN_REV = 1000.0
        r_yoy_base = r.shift(12).where(lambda x: x.abs() >= _MIN_REV)
        out["revenue_yoy"] = ((r - r_yoy_base) / r_yoy_base * 100).astype("float32")

        # MoM（上月比較）
        r_mom_base = r.shift(1).where(lambda x: x.abs() >= _MIN_REV)
        out["revenue_mom"] = ((r - r_mom_base) / r_mom_base * 100).astype("float32")

        # 2026-08-01 追加：即使加了基期下限，仍有 19 萬% 這種值（基期小公司暴衝）。
        # 成長率超過 ±1000%（10 倍）視為離群，對模型只是噪音，設 NaN。
        for _c in ("revenue_yoy", "revenue_mom"):
            _s = pd.Series(out[_c], dtype="float32")
            out[_c] = _s.where(_s.abs() <= 1000).astype("float32")

        # YoY 加速度（本月 YoY - 上月 YoY）
        yoy = pd.Series(out["revenue_yoy"], dtype="float32")
        out["revenue_accel"] = (yoy - yoy.shift(1)).astype("float32")

        # 連續 YoY 正成長月數（streak）
        _add_streak(out, "revenue_pos_months", yoy > 0)

        results.append(pd.DataFrame(out))

    return pd.concat(results, ignore_index=True)


# ── entry point ───────────────────────────────────────────────────────────────

def run(full: bool = False, dry_run: bool = False) -> pd.DataFrame:
    fund = _read("fundamental")
    rev = _read("revenue")

    if fund.empty and rev.empty:
        logger.warning("fundamental.parquet 與 revenue.parquet 皆不存在，略過")
        return pd.DataFrame()

    results = []

    if not fund.empty:
        fund["date"] = pd.to_datetime(fund["date"])

        # 增量模式只決定「輸出哪些日期」，不能拿來篩 rolling 的輸入。
        # 舊版先把 fund 篩成只剩新日期才丟進 _build_valuation_features()，
        # 增量跑一天等於只有一列進 rolling(1260)，min_periods 永遠不滿，
        # per_rank / pbr_rank / yield_rank 會整片變 NaN。
        # rolling 一律吃完整歷史，只在最後輸出時取新日期。
        new_dates: set | None = None
        existing = _read("fundamental_features")
        if not full and not existing.empty:
            existing["date"] = pd.to_datetime(existing["date"])
            done = set(existing["date"].dt.normalize().unique())
            new_dates = set(fund["date"].dt.normalize().unique()) - done
            if new_dates:
                logger.info(f"增量補算 fundamental {len(new_dates)} 天（rolling 仍讀完整歷史）")
            else:
                logger.info("fundamental_features 已是最新")

        if new_dates is None or new_dates:
            val_df = _build_valuation_features(fund)
            if new_dates is not None:
                val_df = val_df[val_df["date"].dt.normalize().isin(new_dates)]
            results.append(val_df)

    if not rev.empty:
        logger.info("計算月營收特徵...")
        rev_feat = _build_revenue_features(rev)
        logger.info(f"月營收特徵：{len(rev_feat)} 筆")
        # 存到獨立的 revenue_features.parquet，供 build_features.py asof merge
        if not dry_run:
            _upsert("revenue_features", rev_feat, keys=["announce_date", "stock_id"])

    if not results:
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)
    logger.info(f"產出 fundamental_features：{len(df)} 筆")

    if dry_run:
        logger.info("[dry-run] 不寫入")
    else:
        _upsert("fundamental_features", df, keys=["date", "stock_id"])

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    df = run(full=args.full, dry_run=args.dry_run)
    if args.dry_run and not df.empty:
        print(df.dtypes.to_string())
