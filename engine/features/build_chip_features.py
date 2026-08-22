"""
計算 chip_features.parquet（PLAN.md 5.14 節）。
依賴：chip.parquet, price.parquet（volume 換算張數）, stock_list.parquet（shares_outstanding）
用法：python build_chip_features.py [--full] [--dry-run]
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


# ── I/O ──────────────────────────────────────────────────────────────────────

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


# ── helpers ───────────────────────────────────────────────────────────────────

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


# 成交量低於此值（張）的交易日不計算任何「除以成交量」的比率，避免除以趨近零。
# 實測 price.parquet 最小成交量為 0.001 張（1 股），不設下限會產生 1e9 級離群值。
MIN_VOL_ZHANG = 1.0

# 融券餘額低於此值（張）時，融資融券比無意義
MIN_SHORT_ZHANG = 10.0


# ── 5.14 籌碼特徵 ─────────────────────────────────────────────────────────────

def _compute_chip_stock(chip: pd.DataFrame, vol_zhang: pd.Series,
                         shares: float | None, close: pd.Series) -> dict:
    """
    chip: 單支股票 chip 資料（已 sort by date）
    vol_zhang: 對齊後的成交量（張，yfinance shares ÷ 1000）
    shares: 流通股數（股），若無則為 None
    close: 對齊後的收盤價
    """
    out: dict = {}

    # 2026-08-01 修正（兩個 bug，見下）：
    #   1. 單位不一致：TWSE T86 的法人買賣超單位是「股」，vol_zhang 是「張」，
    #      原式 net / vol_zhang 讓比率膨脹 1000 倍。改成除以股數。
    #   2. 除以趨近零：原本只 replace(0, nan)，擋不掉極小值。實際最小成交量是
    #      0.001 張（1 股），5 萬股的買賣超除下去會變成 50 億，這種離群值會把
    #      整欄的標準差撐爆，導致下游標準化把正常值全部壓到 0 附近。
    #      改成低於 MIN_VOL_ZHANG 張的日子一律不計算比率（設 NaN）。
    vol_zhang_safe = vol_zhang.where(vol_zhang >= MIN_VOL_ZHANG)
    vol_shares_safe = vol_zhang_safe * 1000

    # 法人買賣超 / 當日成交量（%）
    # 買賣超淨額不可能超過當日總成交量，故 |ratio| <= 100 是物理上界。超過代表
    # 籌碼資料與價格資料對不上（來源不同、或當日 volume 異常），設 NaN 而非保留。
    for col, name in [("foreign_net", "foreign"), ("trust_net", "trust"), ("dealer_net", "dealer")]:
        ratio = chip[col] / vol_shares_safe * 100
        ratio = ratio.where(ratio.abs() <= 100)
        out[f"{name}_net_ratio"] = ratio.astype("float32")

    # 三大法人合計
    inst_net = (chip["foreign_net"].fillna(0)
                + chip["trust_net"].fillna(0)
                + chip["dealer_net"].fillna(0))
    # 同樣換算成股，並套用最小量門檻
    vol_ma10 = (vol_zhang_safe.rolling(10).mean() * 1000)
    inst_10d = inst_net.rolling(10).sum()
    inst_5d = inst_net.rolling(5).sum()
    # 法人 10 日買賣超佔均量比例，超過 1000%（10 倍均量）代表分母仍過小，設 NaN
    out["institutional_10d"] = ((inst_10d / vol_ma10 * 100)
                                .where(lambda x: x.abs() <= 1000).astype("float32"))
    # 2026-08-01 修正：原本是「10日總和 − 5日總和」的原始股數差值，沒有正規化，
    # 實測範圍 ±5.8e8、72% 的列超過 1e4。大型股的絕對股數天生就比小型股大好幾個
    # 數量級，模型實際學到的是「這是不是大公司」而不是籌碼動能。
    # 改成與 institutional_10d 相同的分母（10日均量，股），單位變成 % 後可跨股比較。
    out["institutional_10_5d"] = (((inst_10d - inst_5d) / vol_ma10 * 100)
                                  .where(lambda x: x.abs() <= 1000).astype("float32"))

    # 外資連續買超（streak）
    _add_streak(out, "foreign_consec_buy", chip["foreign_net"] > 0)

    # 外資 + 投信同向
    _add_streak(out, "foreign_trust_align",
                (chip["foreign_net"] > 0) & (chip["trust_net"] > 0))

    # 吸籌但滯漲：法人持續買超、但股價沒什麼反應（起漲點常見的「盤整吸籌」形態）
    inst_buy_10d = inst_net.rolling(10).apply(lambda x: (x > 0).mean(), raw=True)
    price_chg_10d = close.pct_change(10)
    out["absorb_buy_ratio_10d"] = inst_buy_10d.astype("float32")
    out["absorb_price_chg_10d"] = price_chg_10d.astype("float32")
    out["absorb_stall_10d"] = ((inst_buy_10d >= 0.6) & (price_chg_10d.abs() < 0.03)).astype("int8")

    # 融資
    mb = chip.get("margin_balance", pd.Series(dtype=float))
    ms = chip.get("short_balance", pd.Series(dtype=float))

    # 2026-07-26 修正（資料審核發現）：原本這裡想用流通股數算比例，但
    # stock_list.parquet 從沒有 shares_outstanding 欄位，`shares` 永遠是 None，
    # 上面那個公式是從沒真正執行過的死代碼，已移除，只保留一直在跑的
    # 60日均量替代公式（相對變化仍有意義）。`shares` 參數保留供未來若真的補上
    # 流通股數資料時使用。
    # 融資融券餘額單位是「張」，與 vol_zhang 一致，不需換算；但同樣要擋極小分母
    vol_ma60 = vol_zhang_safe.rolling(60).mean()
    out["margin_ratio"] = (mb / vol_ma60).astype("float32")
    out["short_ratio"] = (ms / vol_ma60).astype("float32")

    # 融資 5 日斜率（需搭配 close，此處先存絕對斜率，build_features merge 後再除 close）
    # 2026-08-01 修正：原本是「張/日」的絕對斜率（實測 ±4.6e4），且 build_features
    # 的註解自承「直接保留 raw 版本並改名」＝ 從未正規化。改除以 60 日均量，
    # 語意為「融資餘額每日變動佔均量的比例」，可跨股比較。
    out["margin_slope_5d_raw"] = (((mb - mb.shift(5)) / 5) / vol_zhang_safe.rolling(60).mean()).astype("float32")

    # 融資融券比
    # 2026-08-01 修正：原本只擋「融券餘額為 0」，但餘額只有 1~2 張時比值會衝到
    # 幾十萬（實測 max 4.9e5，中位數才 78）。融券餘額低於 MIN_SHORT_ZHANG 張時
    # 這個比值沒有意義，設 NaN。
    short_nz = ms.where(ms >= MIN_SHORT_ZHANG)
    # 融資融券比中位數約 42，超過 1000 代表融券餘額仍過小，設 NaN
    out["margin_short_ratio"] = (mb / short_nz).where(lambda x: x <= 1000).astype("float32")

    return out


def _compute_chip(chip: pd.DataFrame, price: pd.DataFrame,
                   stock_shares: dict[str, float]) -> pd.DataFrame:
    price["date"] = pd.to_datetime(price["date"])
    chip["date"] = pd.to_datetime(chip["date"])

    # yfinance volume 單位是股，÷1000 換算張
    price_vol = (price.set_index(["date", "stock_id"])["volume"] / 1000)
    price_close = price.set_index(["date", "stock_id"])["close"]

    results = []
    for sid, grp in chip.groupby("stock_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        # 對齊 price volume / close
        vol = price_vol.xs(sid, level="stock_id") if sid in price_vol.index.get_level_values("stock_id") else None
        if vol is not None:
            vol = vol.reindex(grp["date"]).values
            vol_s = pd.Series(vol, index=grp.index)
        else:
            vol_s = pd.Series(np.nan, index=grp.index)

        close = price_close.xs(sid, level="stock_id") if sid in price_close.index.get_level_values("stock_id") else None
        if close is not None:
            close = close.reindex(grp["date"]).values
            close_s = pd.Series(close, index=grp.index)
        else:
            close_s = pd.Series(np.nan, index=grp.index)

        shares = stock_shares.get(sid)
        feat = _compute_chip_stock(grp, vol_s, shares, close_s)
        feat_df = pd.DataFrame(feat, index=grp.index)
        feat_df.insert(0, "date", grp["date"])
        feat_df.insert(1, "stock_id", grp["stock_id"])
        results.append(feat_df)

    return pd.concat(results, ignore_index=True)


# ── entry point ───────────────────────────────────────────────────────────────

def run(full: bool = False, dry_run: bool = False) -> pd.DataFrame:
    chip = _read("chip")
    if chip.empty:
        raise RuntimeError("chip.parquet 不存在")

    existing = _read("chip_features")
    if full or existing.empty:
        new_dates = set(pd.to_datetime(chip["date"]).dt.normalize().unique())
        logger.info("全量重算")
    else:
        existing["date"] = pd.to_datetime(existing["date"])
        done = set(existing["date"].dt.normalize().unique())
        chip["date"] = pd.to_datetime(chip["date"])
        new_dates = set(chip["date"].dt.normalize().unique()) - done
        if not new_dates:
            logger.info("chip_features.parquet 已是最新")
            return pd.DataFrame()
        logger.info(f"增量補算 {len(new_dates)} 個交易日")

    chip = chip[pd.to_datetime(chip["date"]).dt.normalize().isin(new_dates)].copy()

    price = _read("price")
    sl = _read("stock_list")
    stock_shares: dict[str, float] = {}
    if not sl.empty and "shares_outstanding" in sl.columns:
        stock_shares = sl.set_index("stock_id")["shares_outstanding"].dropna().to_dict()

    df = _compute_chip(chip, price, stock_shares)
    logger.info(f"產出 {len(df)} 筆（{df['stock_id'].nunique()} 支）")

    if dry_run:
        logger.info("[dry-run] 不寫入")
    else:
        _upsert("chip_features", df, keys=["date", "stock_id"])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    df = run(full=args.full, dry_run=args.dry_run)
    if args.dry_run and not df.empty:
        print(df.dtypes.to_string())
