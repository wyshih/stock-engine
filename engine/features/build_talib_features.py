"""
ta-lib 全量技術指標（PLAN.md 5.13）。
涵蓋：
  - Momentum Indicators（30 種動量指標，相對值）
  - Pattern Recognition（61 種 CDL K 線型態，±100/0）
  - Volatility / Volume / Statistic 精選

輸出：data/talib_features.parquet（date, stock_id, 各指標欄位）
用法：python build_talib_features.py [--full]
"""
import logging
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import talib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 專案路徑一律走 code/paths.py（唯一來源），不要各檔自行推導
from engine.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402


# ── 指標清單 ──────────────────────────────────────────────────────────────────

# 動量類（已是相對值或 0-100 範圍，直接保留）
MOMENTUM_FUNCS = [
    "ADX", "ADXR",               # 趨勢強度
    "APO",                        # Absolute Price Oscillator
    "AROON", "AROONOSC",         # Aroon（Up/Down/Osc）
    "CCI",                        # Commodity Channel Index
    "CMO",                        # Chande Momentum Oscillator
    "DX",                         # Directional Movement Index
    "MFI",                        # Money Flow Index
    "MOM",                        # Momentum（n=10）
    "PPO",                        # Percentage Price Oscillator
    "ROC", "ROCP",               # Rate of Change / Percent
    "RSI",                        # RSI（補充 14 以外的期數）
    "STOCH", "STOCHF", "STOCHRSI", # KD 變體
    "TRIX",                       # 三重 EMA 變化率
    "ULTOSC",                     # Ultimate Oscillator
    "WILLR",                      # Williams %R
]

# K 線型態（全部 61 種）
CDL_FUNCS = [f for f in talib.get_functions() if f.startswith("CDL")]

# 精選：波動度 + 量價
EXTRA_FUNCS = [
    "ATR", "NATR",               # True Range（相對值 NATR）
    "OBV",                        # On Balance Volume（累積，需標準化）
    "AD",                         # Accumulation/Distribution
    "BETA",                       # 相對大盤 beta（需 high/low）
]


# ── 計算核心 ──────────────────────────────────────────────────────────────────

# 2026-08-01 新增：有界指標的定義域。超出範圍代表計算無效（多半是 high==low
# 導致 TA-Lib 內部除以零），一律設 NaN 而非保留錯誤值。
BOUNDED_RANGES = {
    "stoch_k": (0, 100), "stoch_d": (0, 100),
    "stochf_k": (0, 100), "stochf_d": (0, 100),
    "stochrsi_k": (0, 100), "stochrsi_d": (0, 100),
    "ultosc": (0, 100), "mfi_14": (0, 100), "natr_14": (0, 100),
    "rsi_9": (0, 100), "rsi_28": (0, 100),
    "adx_14": (0, 100), "adxr_14": (0, 100), "dx_14": (0, 100),
    "aroon_up_14": (0, 100), "aroon_dn_14": (0, 100),
    "aroonosc_14": (-100, 100), "willr_14": (-100, 0),
}

# 分母至少要達到序列自身量級的這個比例，否則比率視為無效
_MIN_DEN_FRAC = 0.01


def _safe_ratio(num, den) -> "np.ndarray":
    """比率計算，分母接近零時回 NaN（避免 1e6 級離群值撐爆標準差）。"""
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    scale = pd.Series(np.abs(num)).rolling(60, min_periods=10).mean().values
    floor = np.maximum(scale * _MIN_DEN_FRAC, 1e-8)
    ok = np.isfinite(den) & (np.abs(den) >= floor)
    return np.where(ok, num / np.where(ok, den, 1.0), np.nan)


def _compute_stock(grp: pd.DataFrame) -> pd.DataFrame:
    """對單支股票計算全部 ta-lib 特徵。"""
    grp = grp.sort_values("date").reset_index(drop=True)

    op = grp["open"].values.astype(float)
    hi = grp["high"].values.astype(float)
    lo = grp["low"].values.astype(float)
    cl = grp["close"].values.astype(float)
    vol = grp["volume"].values.astype(float)

    out: dict[str, np.ndarray] = {}

    # ── 動量指標 ─────────────────────────────────────────────────────────
    for fname in MOMENTUM_FUNCS:
        try:
            fn = getattr(talib, fname)
            if fname in ("AROON",):
                dn, up = fn(hi, lo, timeperiod=14)
                out["aroon_dn_14"] = dn
                out["aroon_up_14"] = up
            elif fname == "STOCH":
                k, d = fn(hi, lo, cl)
                out["stoch_k"] = k
                out["stoch_d"] = d
            elif fname == "STOCHF":
                k, d = fn(hi, lo, cl)
                out["stochf_k"] = k
                out["stochf_d"] = d
            elif fname == "STOCHRSI":
                k, d = fn(cl)
                out["stochrsi_k"] = k
                out["stochrsi_d"] = d
            elif fname in ("ADX", "ADXR", "DX"):
                out[fname.lower() + "_14"] = fn(hi, lo, cl, timeperiod=14)
            elif fname == "AROONOSC":
                out["aroonosc_14"] = fn(hi, lo, timeperiod=14)
            elif fname == "MFI":
                out["mfi_14"] = fn(hi, lo, cl, vol, timeperiod=14)
            elif fname == "ULTOSC":
                out["ultosc"] = fn(hi, lo, cl)
            elif fname == "WILLR":
                out["willr_14"] = fn(hi, lo, cl, timeperiod=14)
            elif fname == "CCI":
                out["cci_14"] = fn(hi, lo, cl, timeperiod=14)
            elif fname in ("MOM", "ROC", "ROCP", "CMO"):
                out[fname.lower() + "_10"] = fn(cl, timeperiod=10)
            elif fname in ("APO", "PPO", "TRIX"):
                out[fname.lower()] = fn(cl)
            elif fname == "RSI":
                out["rsi_9"]  = fn(cl, timeperiod=9)
                out["rsi_28"] = fn(cl, timeperiod=28)
            else:
                out[fname.lower()] = fn(cl)
        except Exception:
            pass

    # ── K 線型態（全部 61 種）────────────────────────────────────────────
    for fname in CDL_FUNCS:
        try:
            result = getattr(talib, fname)(op, hi, lo, cl)
            out[fname.lower()] = result.astype("int8")
        except Exception:
            pass

    # ── 波動度 / 量價 ─────────────────────────────────────────────────────
    try:
        out["atr_14"]  = talib.ATR(hi, lo, cl, timeperiod=14)
        out["natr_14"] = talib.NATR(hi, lo, cl, timeperiod=14)
        # OBV 標準化（相對 60日均值）
        obv = talib.OBV(cl, vol)
        # OBV / AD 會在零附近來回穿越，只擋「恰好為 0」的分母不夠：分母接近零
        # 時比率會爆到 1e6 級。改成要求分母至少達到該序列自身量級的 1%。
        out["obv_ratio"] = _safe_ratio(obv, pd.Series(obv).rolling(60, min_periods=10).mean().values)
        # AD 標準化
        ad = talib.AD(hi, lo, cl, vol)
        out["ad_ratio"] = _safe_ratio(ad, pd.Series(ad).rolling(60, min_periods=10).mean().values)
    except Exception:
        pass

    # 2026-08-01 修正：有界振盪指標超出定義域一律設 NaN。
    # 漲停/跌停等 high==low 的日子會讓 TA-Lib 內部除以零，實測 stoch_k 出現
    # 3.3e7、ultosc 出現 6.6e9，這種值會把整欄標準差撐爆，使下游標準化把正常
    # 值全部壓到 0 附近。超界代表計算無效，設 NaN 讓補值機制處理才正確。
    # 2026-08-01 修正：APO / MOM 是「元」為單位的絕對量（實測 ±1.7e5、±3.9e5），
    # 高價股天生數值大。除以股價變成 %，才能跨股比較。
    # 2026-08-01 二次修正：上一版用 cl > 0 當保護不夠 —— 股價 0.13 元的個股會讓
    # apo/mom 除下去變成 -4.8e5（實測 max 才 164，極度不對稱）。台股最低價約
    # 0.13 元，要求至少 1 元；換算後再限制 ±500%，超過視為無效。
    _cl_safe = np.where(np.isfinite(cl) & (cl >= 1.0), cl, np.nan)
    for _c in ("apo", "mom_10"):
        if _c in out:
            _v = np.asarray(out[_c], dtype="float64") / _cl_safe * 100
            out[_c] = np.where(np.abs(_v) <= 500, _v, np.nan)

    # 2026-08-07 修正：atr_14 沒除以股價，同段的 apo / mom_10 都有除。
    # 後果是它變成「股價高低」的代理變數而不是波動度 —— 中位數從 2020 年的
    # 0.735 元一路漲到 2024+ 的 1.191 元（+62%），漲的是台股均價不是波動。
    # 訓練期與測試期的分布因此系統性偏移，模型學到的門檻搬不過去。
    # 除以收盤價轉成 %，與 natr_14 同一尺度（真實區間不可能超過股價本身 → 上限 100%）。
    #
    # ⚠️ 注意：TA-Lib 的 NATR 定義就是 100 * ATR / close，所以這樣normalize之後
    # atr_14 與 natr_14 在數值上完全相同（實測 330 萬列 corr=1.0，最大差
    # 4.1e-6 只是 float32 精度）。兩欄同時進特徵集是純粹的重複，見 Phase 2 回報。
    if "atr_14" in out:
        _a = np.asarray(out["atr_14"], dtype="float64") / _cl_safe * 100
        out["atr_14"] = np.where(np.isfinite(_a) & (_a >= 0) & (_a <= 100), _a, np.nan)

    for _c, (_lo, _hi) in BOUNDED_RANGES.items():
        if _c in out:
            _v = np.asarray(out[_c], dtype="float64")
            out[_c] = np.where(np.isfinite(_v) & (_v >= _lo) & (_v <= _hi), _v, np.nan)

    result_df = pd.DataFrame(out, index=grp.index)
    result_df.insert(0, "date",     grp["date"])
    result_df.insert(1, "stock_id", grp["stock_id"])

    # float64 → float32 節省記憶體（CDL 已是 int8）
    for c in result_df.columns:
        if result_df[c].dtype == np.float64:
            result_df[c] = result_df[c].astype("float32")

    return result_df


# ── I/O ───────────────────────────────────────────────────────────────────────

def _read(name: str) -> pd.DataFrame:
    p = DATA_DIR / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _upsert(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read("talib_features")
    combined = pd.concat([existing, df], ignore_index=True) if not existing.empty else df
    combined = (combined
                .drop_duplicates(subset=["date", "stock_id"], keep="last")
                .sort_values(["date", "stock_id"])
                .reset_index(drop=True))
    path = DATA_DIR / "talib_features.parquet"
    combined.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"talib_features.parquet：{len(combined)} 筆 × {len(combined.columns)} 欄")


# ── entry point ───────────────────────────────────────────────────────────────

def run(full: bool = False) -> pd.DataFrame:
    price = _read("price")
    if price.empty:
        raise RuntimeError("price.parquet 不存在")

    price["date"] = pd.to_datetime(price["date"])
    price = price.sort_values(["stock_id", "date"]).reset_index(drop=True)

    existing = _read("talib_features")
    if full or existing.empty:
        logger.info("全量計算 ta-lib 特徵...")
        target_price = price
    else:
        existing["date"] = pd.to_datetime(existing["date"])
        done = set(existing["date"].dt.normalize().unique())
        new_dates = set(price["date"].dt.normalize().unique()) - done
        if not new_dates:
            logger.info("talib_features.parquet 已是最新")
            return pd.DataFrame()
        # 需要 WARMUP 才能算 rolling 指標，拉回 300 天
        min_new = min(new_dates)
        warmup_start = min_new - pd.Timedelta(days=420)
        target_price = price[price["date"] >= warmup_start]
        logger.info(f"增量計算 {len(new_dates)} 天（含 warmup）")

    stocks = target_price["stock_id"].unique()
    logger.info(f"共 {len(stocks)} 支股票...")

    results = []
    for i, sid in enumerate(stocks):
        grp = target_price[target_price["stock_id"] == sid]
        try:
            feat = _compute_stock(grp)
            # 增量模式：只留新日期
            if not full and not existing.empty:
                feat = feat[feat["date"].dt.normalize().isin(new_dates)]
            results.append(feat)
        except Exception as e:
            logger.warning(f"{sid}: {e}")

        if (i + 1) % 200 == 0:
            logger.info(f"  {i+1}/{len(stocks)} 完成")

    if not results:
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)
    _upsert(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run(full=args.full)
