"""產出 public dashboard 用的資料包（2026-08-22 新寫）。

範圍刻意只有**模型測試期的全市場**（Round 4 的 test + test2＝2025-02-01 ~
2026-07-31）。理由：public repo 是展示與檢驗用，不是即時預測服務。訓練期資料、
完整特徵檔、模型 pkl 一律不出去。

產出（預設寫到 ../dashboard/public_data/）：

    price_test.parquet     該期間全市場 OHLCV（前端的圖表與指標都由這份現算）
    scores_test.parquet    10 個模型 × 該期間每日每股的分數（long format）
    pattern_hits.parquet   148 條說法的命中矩陣，int8，該期間全市場
                           —— 前端因此不需要 96 欄特徵值，也不需要 TA-Lib
    pattern_stats.json     148 條說法的**全市場全歷史**條件統計（含對照組 baseline）
    sigcurve_m*.csv ×10    各模型 val_sel 門檻曲線（前端滑桿旁的數字讀這個）
    backtest_summary.csv   絕對門檻版 + 訊號數對齊版（每日前 1.5%）兩張表
    stock_list.parquet     代號 / 名稱 / 市場 / 產業
    manifest.json          期間、模型與門檻、產生時間、資料口徑、免責聲明

⚠️ 體積護欄：任一檔 > MAX_FILE_MB 或總量 > MAX_TOTAL_MB 就中止並報告。
推不上 GitHub 的資料包產出來只是浪費時間。

⚠️ 統計算法一律呼叫 `engine.app.frontend.conditional_stats` 的同一套函式，
不在這裡另寫一份 —— 這個專案吃過方法漂移的虧（CLAUDE.md 規則 8）。

用法：
  python -m engine.export.build_public_bundle
  python -m engine.export.build_public_bundle --out /path/to/dashboard/public_data
  python -m engine.export.build_public_bundle --skip-backtest    # 只更新資料，不重跑回測
  python -m engine.export.build_public_bundle --backtest-only    # 只跑回測表（make backtest）
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from engine.backtest import summary as summary_mod
from engine.backtest.summary import MATCHED_TOP_PCT, MODEL_KEYS
from engine.models.bundle import CHOSEN_THRESHOLDS, score_path, sigcurve_path
from engine.paths import DATA_DIR, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 期間：Round 4 的 test（2025-02~2025-12）+ test2（2026-01~2026-07）─────────
TEST_START = "2025-02-01"
TEST_END = "2026-07-31"
SPLITS = ("test", "test2")

DEFAULT_OUT = PROJECT_ROOT.parent / "dashboard" / "public_data"

# GitHub 對單檔 > 100MB 直接拒收，>50MB 會警告。留一點餘裕。
MAX_FILE_MB = 90
MAX_TOTAL_MB = 200


DISCLAIMER = (
    "本站顯示的是模型測試期（2025-02 ~ 2026-07）的回溯結果，不是即時預測，"
    "不構成任何投資建議。歷史績效不代表未來表現。"
)
CAVEATS = [
    "資料源為 TWSE / TPEx 官方端點，不含已下市股票 —— 全部統計都帶生存偏差，數字偏樂觀。",
    "測試期（2025-02~2026-07）不在訓練期內，但門檻是在 2024 下半年的 val_sel 上由人挑的。",
    "10 個模型全部是 Round 4 切分：train 2020-01~2023-11、val 2024、test 2025-02~2026-07，兩個交界各留一個月 embargo。",
    "回測口徑 dedup=False（每筆超過門檻的訊號獨立進場），與挑門檻時看的曲線同一把尺。",
    "出場規則：獲利 15% 後啟動移動停利、從最高收盤回落 10% 出場、固定停損 20%。",
    "148 條技術說法的統計是全市場全歷史，不是個股自己的統計，也沒有納入產業與籌碼結構。",
]


# ── 各項產出 ──────────────────────────────────────────────────────────────────

def build_price(out_dir: Path) -> pd.DataFrame:
    price = pd.read_parquet(DATA_DIR / "price.parquet")
    price["date"] = pd.to_datetime(price["date"])
    price = price[(price["date"] >= TEST_START) & (price["date"] <= TEST_END)]
    price = price.sort_values(["stock_id", "date"]).reset_index(drop=True)
    price.to_parquet(out_dir / "price_test.parquet", index=False)
    logger.info(f"price_test：{len(price):,} 列 / {price['stock_id'].nunique():,} 檔 / "
                f"{price['date'].nunique()} 個交易日")
    return price


def build_scores(out_dir: Path) -> pd.DataFrame:
    frames = []
    for key in MODEL_KEYS:
        for split in SPLITS:
            path = score_path(key, split)
            if not path.exists():
                raise SystemExit(
                    f"找不到 {path}。10 個模型的分數檔要先有才能出資料包 —— 請先 `make train`。")
            part = pd.read_parquet(path)
            part["date"] = pd.to_datetime(part["date"])
            part = part[(part["date"] >= TEST_START) & (part["date"] <= TEST_END)]
            part["model"] = key
            frames.append(part[["date", "stock_id", "model", "score"]])
    scores = pd.concat(frames, ignore_index=True)
    scores = scores.drop_duplicates(subset=["date", "stock_id", "model"])
    scores["model"] = scores["model"].astype("category")
    scores["score"] = scores["score"].astype("float32")
    scores = scores.sort_values(["date", "model", "stock_id"]).reset_index(drop=True)
    scores.to_parquet(out_dir / "scores_test.parquet", index=False)
    logger.info(f"scores_test：{len(scores):,} 列 / {scores['model'].nunique()} 個模型")
    return scores


def build_pattern_hits(out_dir: Path) -> pd.DataFrame:
    """148 條說法在測試期全市場的命中矩陣（int8）。

    前端因此完全不需要特徵值，也不需要 TA-Lib —— 判定在這裡一次算完。
    """
    from engine.app.frontend import conditional_stats as cs
    from engine.app.frontend.patterns import PATTERNS

    columns = tuple(dict.fromkeys(c for p in PATTERNS for c in p.columns))
    features = cs._load_columns(columns)
    if features.empty:
        raise SystemExit("讀不到任何特徵欄位，無法產生 pattern_hits（特徵檔還沒建？）")
    features["date"] = pd.to_datetime(features["date"])
    features = features[(features["date"] >= TEST_START) & (features["date"] <= TEST_END)]

    hits = features[["date", "stock_id"]].copy()
    skipped = []
    for pattern in PATTERNS:
        try:
            hits[pattern.key] = pattern.predicate(features).fillna(False).astype("int8")
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append((pattern.key, str(exc)))
    if skipped:
        logger.warning(f"{len(skipped)} 條說法缺欄位、已跳過：{[k for k, _ in skipped]}")
    hits = hits.sort_values(["date", "stock_id"]).reset_index(drop=True)
    hits.to_parquet(out_dir / "pattern_hits.parquet", index=False, compression="zstd")
    logger.info(f"pattern_hits：{len(hits):,} 列 × {len(hits.columns) - 2} 條說法")
    return hits


def build_pattern_stats(out_dir: Path) -> dict:
    """148 條說法的全市場全歷史條件統計 + 對照組。使用者要求「全放」。"""
    from engine.app.frontend import conditional_stats as cs
    from engine.app.frontend.patterns import PATTERNS

    payload = {
        "baseline": cs.baseline(),
        "min_samples": cs.MIN_SAMPLES,
        "hold_days": list(cs.HOLD_DAYS),
        "edge_tolerance": cs.EDGE_TOLERANCE,
        "patterns": {},
    }
    for i, pattern in enumerate(PATTERNS, 1):
        stats = cs.pattern_stats(pattern)
        payload["patterns"][pattern.key] = {
            "name": pattern.name, "tone": pattern.tone, "group": pattern.group,
            "text": pattern.text, "stats": stats,
        }
        if i % 25 == 0:
            logger.info(f"  pattern_stats 進度 {i}/{len(PATTERNS)}")
    (out_dir / "pattern_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    logger.info(f"pattern_stats：{len(payload['patterns'])} 條說法")
    return payload


def copy_sigcurves(out_dir: Path) -> list[str]:
    copied = []
    for key in MODEL_KEYS:
        src = sigcurve_path(key)
        if not src.exists():
            raise SystemExit(f"找不到 {src}，請先 `make curve`。")
        dst = out_dir / f"sigcurve_{key}.csv"
        dst.write_bytes(src.read_bytes())
        copied.append(dst.name)
    logger.info(f"sigcurve：複製 {len(copied)} 個檔")
    return copied


def copy_stock_list(out_dir: Path) -> None:
    cols = ["stock_id", "stock_name", "market", "industry"]
    sl = pd.read_parquet(DATA_DIR / "stock_list.parquet")
    sl[[c for c in cols if c in sl.columns]].to_parquet(
        out_dir / "stock_list.parquet", index=False)
    logger.info(f"stock_list：{len(sl):,} 檔")


# ── 回測 ──────────────────────────────────────────────────────────────────────
# 實作在 engine/backtest/summary.py（唯一一份，`make backtest` 也走那裡）。
# 這裡只是把 public 展示窗口的區間傳進去 —— 內部驗證與 public 展示共用同一套
# 回測，區間不同而已。回測邏輯不得在這裡出現第二份（CLAUDE.md 規則 8）。

def build_backtest_summary(out_dir: Path, tmp_dir: Path) -> pd.DataFrame:
    """public 資料包用的回測表：固定在展示窗口 TEST_START ~ TEST_END。"""
    return summary_mod.build_backtest_summary(
        out_dir, tmp_dir, splits=SPLITS, start=TEST_START, end=TEST_END,
        keys=MODEL_KEYS)


# ── 體積護欄 ──────────────────────────────────────────────────────────────────

def check_size(out_dir: Path) -> None:
    files = sorted(p for p in out_dir.iterdir() if p.is_file())
    total = 0.0
    over = []
    print("\n產出檔案：")
    for path in files:
        mb = path.stat().st_size / 1024 / 1024
        total += mb
        flag = ""
        if mb > MAX_FILE_MB:
            flag = f"   ← 超過單檔上限 {MAX_FILE_MB}MB"
            over.append(path.name)
        print(f"  {path.name:28s} {mb:8.2f} MB{flag}")
    print(f"  {'總計':28s} {total:8.2f} MB\n")

    problems = []
    if over:
        problems.append(f"單檔超過 {MAX_FILE_MB}MB：{', '.join(over)}")
    if total > MAX_TOTAL_MB:
        problems.append(f"總量 {total:.1f}MB 超過上限 {MAX_TOTAL_MB}MB")
    if problems:
        raise SystemExit("❌ 體積護欄擋下：\n  - " + "\n  - ".join(problems)
                         + "\n推不上 GitHub 的資料包不要產。請縮小期間或欄位後重跑。")


def write_manifest(out_dir: Path, price: pd.DataFrame, scores: pd.DataFrame) -> None:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "period": {"start": TEST_START, "end": TEST_END,
                   "trading_days": int(price["date"].nunique())},
        "coverage": {"stocks": int(price["stock_id"].nunique()),
                     "price_rows": int(len(price)), "score_rows": int(len(scores))},
        "split": "Round 4（train 2020-01~2023-11 / val_es 2024H1 / val_sel 2024H2 / "
                 "test 2025-02~2025-12 / test2 2026-01~2026-07）",
        "models": [{"key": k, "threshold": CHOSEN_THRESHOLDS[k]} for k in MODEL_KEYS],
        "data_source": "TWSE MI_INDEX / TPEx otc 官方端點（不使用 yfinance）",
        "backtest": {
            "implementation": "engine.backtest.backtest.simulate()/performance()",
            "dedup": False,
            "exit_rules": {"trail_trigger": 0.15, "trail_pct": 0.10, "stop_loss": 0.20},
            "matched_top_pct": MATCHED_TOP_PCT,
        },
        "caveats": CAVEATS,
        "disclaimer": DISCLAIMER,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("manifest.json 已寫入")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-stats", action="store_true",
                        help="跳過 pattern_stats（全歷史統計，很慢）")
    parser.add_argument("--backtest-only", action="store_true",
                        help="只跑回測表（`make backtest` 走這條），不產資料包")
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    if args.backtest_only:
        build_backtest_summary(out_dir, tmp_dir)
        for path in tmp_dir.iterdir():
            path.unlink()
        tmp_dir.rmdir()
        print(f"\n回測表 → {out_dir / 'backtest_summary.csv'}")
        return

    price = build_price(out_dir)
    scores = build_scores(out_dir)
    build_pattern_hits(out_dir)
    if not args.skip_stats:
        build_pattern_stats(out_dir)
    copy_sigcurves(out_dir)
    copy_stock_list(out_dir)
    if not args.skip_backtest:
        build_backtest_summary(out_dir, tmp_dir)
    write_manifest(out_dir, price, scores)

    for path in tmp_dir.iterdir():
        path.unlink()
    tmp_dir.rmdir()
    check_size(out_dir)
    print(f"✅ 資料包完成 → {out_dir}")
    print("   接著跑 `make publish-public` 看要下哪些 git 指令。")


if __name__ == "__main__":
    main()
