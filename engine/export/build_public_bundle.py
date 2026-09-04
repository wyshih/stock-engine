"""產出 public dashboard 用的資料包（2026-08-22 新寫）。

範圍刻意只有**模型測試期的全市場**（Round 4 的 test + test2＝2025-02-01 ~
2026-07-31）。理由：public repo 是展示與檢驗用，不是即時預測服務。訓練期資料、
完整特徵檔、模型 pkl 一律不出去。

產出（預設寫到 ../dashboard/public_data/）：

    price_test.parquet     該期間全市場 OHLCV，另多帶 20 個交易日供驗證訊號結果
    scores_test_m*.parquet 各模型該期間每日每股的分數（一個模型一個檔）
    pattern_hits.parquet   148 條說法的命中矩陣，int8，該期間全市場
                           —— 前端因此不需要 96 欄特徵值，也不需要 TA-Lib
    pattern_stats.json     148 條說法的**全市場全歷史**條件統計（含對照組 baseline）
    sigcurve_m*.csv.gz     各模型 val_sel（2024H2）門檻曲線，整條 gzip、不抽樣
                           —— 統計母體與 price/scores 的測試期不重疊，見 manifest
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
import gzip
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from engine.backtest import summary as summary_mod
from engine.backtest.summary import MATCHED_TOP_PCT, MODEL_KEYS
from engine.models.bundle import CHOSEN_THRESHOLDS, model_label, sigcurve_path
from engine.models.score_source import combined_scores
from engine.paths import DATA_DIR, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 期間：Round 4 的 test（2025-02~2025-12）+ test2（2026-01~2026-07）─────────
TEST_START = "2025-02-01"
TEST_END = "2026-07-31"
SPLITS = ("test", "test2")

# 訊號止於 TEST_END，但價格多帶這麼多個交易日 —— 標的看的是「未來 20 個交易日」，
# 價格跟訊號同時截斷的話，期間最後那批訊號在站上只看得到中途的回檔，
# 會被誤讀成模型失準（4739 那次就是）。
LOOKAHEAD_TRADING_DAYS = 20

DEFAULT_OUT = PROJECT_ROOT.parent / "dashboard" / "public_data"

# 前端滑桿的刻度：0.50 ~ 1.00，每 1% 一格。曲線只需要抽樣到這些點上
# —— 前端唯一會查的就是這 51 個值，比這更細的解析度沒有人看得到。
SLIDER_MIN = 0.50
SLIDER_MAX = 1.00
SLIDER_STEP = 0.01
SLIDER_GRID = [round(SLIDER_MIN + i * SLIDER_STEP, 2)
               for i in range(int(round((SLIDER_MAX - SLIDER_MIN) / SLIDER_STEP)) + 1)]

# GitHub 對單檔 > 100MB 直接拒收，>50MB 會警告。曲線整條匯出（不抽樣），
# 目前最大單檔約 18MB、總量約 106MB。上限貼著實際用量抓餘裕，
# 多出來的東西要能被問到，不是隨便放寬。
MAX_FILE_MB = 30
MAX_TOTAL_MB = 140


DISCLAIMER = (
    "本站顯示的是模型測試期（2025-02 ~ 2026-07）的回溯結果，不是即時預測，"
    "不構成任何投資建議。歷史績效不代表未來表現。"
)
CAVEATS = [
    "資料源為 TWSE / TPEx 官方端點，不含已下市股票 —— 全部統計都帶生存偏差，數字偏樂觀。",
    "測試期（2025-02~2026-07）不在訓練期內，但門檻是在 2024 下半年的 val_sel 上由人挑的。",
    "2 個模型都是 Round 4 切分：train 2020-01~2023-11、val 2024、test 2025-02~2026-07，兩個交界各留一個月 embargo。",
    "兩者用同一份特徵集，只差標的：m1_base_up20 只看未來 20 日的上漲天數；m1_mdd10 再要求期間最低收盤不跌破 −10%。",
    "回測口徑 dedup=False（每筆超過門檻的訊號獨立進場），與挑門檻時看的曲線同一把尺。",
    "出場規則：獲利 15% 後啟動移動停利、從最高收盤回落 10% 出場、固定停損 20%。",
    "148 條技術說法的統計是全市場全歷史，不是個股自己的統計，也沒有納入產業與籌碼結構。",
]


# 契約改過之後被取代的產物 —— 不清掉的話會殘留在 out_dir，被算進體積、
# 也會讓讀資料包的人以為那還是有效的檔案（2026-08-27 稽核 MEDIUM-3）。
#
# 2026-09-02 起也含**被移除模型**的每模型產物（m2/m3/m6/m8）：那些檔是上一次
# 匯出留在 out_dir 的，MODEL_KEYS 縮短之後不會再被覆寫，只會安靜地留著，
# 讓 public 站以為那幾個模型還在。刻意寫死代號而不是掃 glob —— glob 會連
# 「這次還沒寫出來的」也一起刪掉，順序一錯就把有效產物清了。
RETIRED_MODEL_KEYS = ("m2_nomkt_up20", "m3_v3_up20", "m6_base_nobear", "m8_v3_nobear")
SUPERSEDED_OUTPUTS = (
    ("scores_test.parquet",)
    # 舊契約的未壓縮曲線（現行是 sigcurve_{k}.csv.gz）
    + tuple(f"sigcurve_{k}.csv" for k in MODEL_KEYS + RETIRED_MODEL_KEYS)
    # 被移除模型的每模型產物
    + tuple(f"scores_test_{k}.parquet" for k in RETIRED_MODEL_KEYS)
    + tuple(f"sigcurve_{k}.csv.gz" for k in RETIRED_MODEL_KEYS))


# ── 各項產出 ──────────────────────────────────────────────────────────────────

def price_end_date(all_dates: pd.Series) -> pd.Timestamp:
    """價格要帶到 TEST_END 之後第 LOOKAHEAD_TRADING_DAYS 個交易日（不足就給到底）。"""
    after = sorted(d for d in all_dates.unique() if d > pd.Timestamp(TEST_END))
    if not after:
        return pd.Timestamp(TEST_END)
    return after[min(LOOKAHEAD_TRADING_DAYS, len(after)) - 1]


def lookahead_days(dates: pd.Series) -> int:
    """實際帶到 TEST_END 之後的交易日數 —— 不足 LOOKAHEAD_TRADING_DAYS 時會少於它。"""
    return int(dates[dates > pd.Timestamp(TEST_END)].nunique())


def build_price(out_dir: Path) -> pd.DataFrame:
    price = pd.read_parquet(DATA_DIR / "price.parquet")
    price["date"] = pd.to_datetime(price["date"])
    end = price_end_date(price["date"])
    price = price[(price["date"] >= TEST_START) & (price["date"] <= end)]
    price = price.sort_values(["stock_id", "date"]).reset_index(drop=True)
    price.to_parquet(out_dir / "price_test.parquet", index=False)
    extra = lookahead_days(price["date"])
    logger.info(f"price_test：{len(price):,} 列 / {price['stock_id'].nunique():,} 檔 / "
                f"{price['date'].nunique()} 個交易日"
                f"（含 TEST_END 之後 {extra} 天供驗證訊號結果）")
    return price


def build_scores(out_dir: Path) -> pd.DataFrame:
    """各模型在測試期的分數，一個模型一個檔。

    分數來源一律走 `score_source.combined_scores()` —— 回測也走同一支，
    兩邊才不會像 2026-08-27 那次一樣各自組出不同的訊號宇宙。
    """
    frames = []
    for key in MODEL_KEYS:
        part = combined_scores(key, SPLITS, TEST_START, TEST_END)
        if part.empty:
            raise SystemExit(
                f"{key} 在測試期沒有任何分數 —— 請先 `make train`。")
        part["model"] = key
        frames.append(part[["date", "stock_id", "model", "score"]])
    scores = pd.concat(frames, ignore_index=True)
    scores["model"] = scores["model"].astype("category")
    scores["score"] = scores["score"].astype("float32")
    scores = scores.sort_values(["date", "model", "stock_id"]).reset_index(drop=True)

    # 分數必須蓋滿宣告的期間。少了最後幾天不會報錯，只會讓最新的訊號悄悄消失
    # —— 4739 那次就是這樣被誤讀的，所以在這裡擋下來。
    price_days = pd.to_datetime(
        pd.read_parquet(out_dir / "price_test.parquet", columns=["date"])["date"])
    # 價格會多帶 TEST_END 之後的日子，分數只需要蓋到 TEST_END 為止。
    in_period = sorted(set(price_days[price_days <= pd.Timestamp(TEST_END)]))
    if not in_period:
        raise SystemExit("price_test.parquet 在宣告期間內沒有任何交易日。")
    for key in MODEL_KEYS:
        model_days = set(scores[scores["model"] == key]["date"])
        missing = [d for d in in_period if d not in model_days]
        if missing:
            raise SystemExit(
                f"{key} 的分數少了 {len(missing)} 個交易日"
                f"（最早 {missing[0].date()}、最晚 {missing[-1].date()}）。"
                "請先執行 `python -m engine.models.score_recent` 補分數。")

    for key in MODEL_KEYS:
        part = scores[scores["model"] == key].drop(columns=["model"])
        part.reset_index(drop=True).to_parquet(
            out_dir / f"scores_test_{key}.parquet", index=False)
        logger.info(f"scores_test_{key}：{len(part):,} 列")
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


def build_fpm_rule_hits(out_dir: Path) -> pd.DataFrame:
    """fpm 專案挖出的平盤起漲點規則，測試期以來的歷史命中買點（int8 起漲旗標）。

    下限沿用 TEST_START（2025-02 之前的任何一筆都不能進 public repo，CLAUDE.md
    硬性規定），但**上限不卡 TEST_END**——TEST_END 是模型 Round 4 官方測試期
    （test2 到 2026-07-31）的定義，不是我們自己能決定的數字；fpm 資料本來就
    每天在累積（見 validation_level 分兩級），使用者要求「全部都要看得到」，
    所以這裡自己不設上限，出到來源檔案有的最新一天為止。
    來源 `data/fpm_rules/rules_hitlist_oos.csv` 本身已經是 walk-forward
    樣本外命中明細，這裡只是再篩一次期間下限、換成 public 資料包的檔名。
    """
    src = DATA_DIR / "fpm_rules" / "rules_hitlist_oos.csv"
    if not src.exists():
        logger.warning("找不到 data/fpm_rules/rules_hitlist_oos.csv，跳過 fpm_rule_hits")
        return pd.DataFrame()

    hits = pd.read_csv(src, parse_dates=["date"])
    hits = hits[hits["date"] >= TEST_START]
    hits["stock_id"] = hits["stock_id"].astype(str)
    hits["label"] = hits["label"].fillna(0).astype("int8")
    if "validation_level" not in hits.columns:
        hits["validation_level"] = "window顯著性驗證過"  # 舊格式相容：沒有這欄的一律視為驗證過
    hits = hits[["date", "stock_id", "rule_id", "r_end", "mdd", "label", "validation_level"]] \
        .sort_values(["date", "stock_id"]).reset_index(drop=True)
    hits.to_parquet(out_dir / "fpm_rule_hits.parquet", index=False, compression="zstd")
    logger.info(f"fpm_rule_hits：{len(hits):,} 列（{TEST_START}~{hits['date'].max().date()}）")
    return hits


def build_fpm_rule_stats(out_dir: Path) -> dict:
    """fpm 規則清單 + 統計（勝率/lift/樣本外期望報酬）。統計本身是全歷史 walk-forward
    彙總數字，跟 pattern_stats 一樣屬於「全市場全歷史統計」，不受 2025-02 這條線
    限制——受限的是逐筆帶日期的原始紀錄（fpm_rule_hits），不是彙總後的統計量。
    """
    src = DATA_DIR / "fpm_rules" / "rules_registry.yaml"
    if not src.exists():
        logger.warning("找不到 data/fpm_rules/rules_registry.yaml，跳過 fpm_rule_stats")
        return {}
    import yaml
    registry = yaml.safe_load(src.read_text(encoding="utf-8")) or []
    payload = {
        "source": "fpm（獨立的型態探勘專案，見該專案 README/PLAN.md）",
        "caveats": [
            "permutation test 從計畫原訂 500 次降到 20 次（跑不完全量，已在 fpm 專案記錄）",
            "只驗證了一種「平盤」定義（trend_r2 最低五分位），另一種（價格區間窄）驗證中",
            "資料不含下市股票，存在存活者偏差，結果為上界估計",
            "單筆命中勝率（嚴格定義）只有 5~7%，不是穩贏訊號，是統計期望值為正的邊際優勢",
        ],
        "rules": registry,
    }
    (out_dir / "fpm_rule_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    logger.info(f"fpm_rule_stats：{len(registry)} 條規則")
    return payload


def export_sigcurves(out_dir: Path) -> list[str]:
    """匯出門檻曲線 —— 整條原樣 gzip，不抽樣。

    使用者要求資料包保留完整曲線（每個訊號一列），不做任何取樣。這裡壓縮的是
    **原始位元組**，解開後與來源檔逐位元組相同 —— 壓縮不是取樣，一列都沒少。

    用 gzip 是因為這幾個檔佔資料包 70%（六條共 94MB），而且每次重出內容會整個
    改寫；不壓的話 public repo 每更新一次就永久多長 ~100MB 的 git 歷史。
    壓完約剩 32%。`pandas.read_csv` 認得 .gz，前端不用改任何一行。
    """
    written = []
    for key in MODEL_KEYS:
        src = sigcurve_path(key)
        if not src.exists():
            raise SystemExit(f"找不到 {src}，請先 `make curve`。")
        # 挑定的門檻一定要落在滑桿刻度上，否則前端根本拉不到它。
        chosen = CHOSEN_THRESHOLDS[key]
        if chosen not in SLIDER_GRID:
            raise SystemExit(
                f"{key} 挑定的門檻 {chosen} 不在滑桿刻度上"
                f"（{SLIDER_MIN}~{SLIDER_MAX} step {SLIDER_STEP}）。"
                "請把 CHOSEN_THRESHOLDS 改成刻度上的值。")
        dst = out_dir / f"sigcurve_{key}.csv.gz"
        raw = src.read_bytes()
        dst.write_bytes(gzip.compress(raw, compresslevel=6))
        # 立刻驗回來 —— 壓縮壞掉的資料包比沒有更糟，而且要到前端才會發現。
        if gzip.decompress(dst.read_bytes()) != raw:
            raise SystemExit(f"{dst.name} 壓縮後解不回原檔，中止。")
        written.append(dst.name)
        logger.info(f"  {key}：{len(raw)/1048576:.1f}MB → "
                    f"{dst.stat().st_size/1048576:.1f}MB")
    logger.info(f"sigcurve：{len(written)} 個檔，整條 gzip（不抽樣）")
    return written


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


def _lookahead_note(price: pd.DataFrame) -> str:
    """如實描述多帶了幾天 —— 不足時不可以照抄 LOOKAHEAD_TRADING_DAYS。

    2026-08-27 稽核抓到：note 無條件寫「多帶 20 個交易日」，實際只有 15 天，
    對外宣告與資料不符。
    """
    actual = lookahead_days(price["date"])
    base = (f"價格多帶 {actual} 個交易日（目標 {LOOKAHEAD_TRADING_DAYS} 天），"
            "讓期間最後那批訊號也看得到後續走勢；那段沒有分數。")
    if actual < LOOKAHEAD_TRADING_DAYS:
        short = LOOKAHEAD_TRADING_DAYS - actual
        base += (f" ⚠️ 還差 {short} 個交易日才滿 {LOOKAHEAD_TRADING_DAYS} 天，"
                 "期間末尾的訊號結果尚未定案，前端會顯示為空白而非虧損。")
    return base


def write_manifest(out_dir: Path, price: pd.DataFrame, scores: pd.DataFrame) -> None:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "period": {"start": TEST_START, "end": TEST_END,
                   "trading_days": int(price[price["date"] <= TEST_END]["date"].nunique())},
        "price_period": {
            "start": TEST_START,
            "end": price["date"].max().date().isoformat(),
            "lookahead_trading_days": lookahead_days(price["date"]),
            "lookahead_target": LOOKAHEAD_TRADING_DAYS,
            "note": _lookahead_note(price)},
        "coverage": {"stocks": int(price["stock_id"].nunique()),
                     "price_rows": int(len(price)), "score_rows": int(len(scores))},
        "split": "Round 4（train 2020-01~2023-11 / val_es 2024H1 / val_sel 2024H2 / "
                 "test 2025-02~2025-12 / test2 2026-01~2026-07）",
        # name 讓公開站的選單顯示「全特徵·漲勢」而不是 m1_base_up20；
        # key 仍是唯一識別，檔名與 backtest_summary 都用它。
        "models": [{"key": k, "name": model_label(k), "threshold": CHOSEN_THRESHOLDS[k]}
                   for k in MODEL_KEYS],
        # 曲線的統計母體是 val_sel，與上面的 period 完全不重疊。檔名不帶 split，
        # 直接讀資料包的人分辨不出來，所以在這裡明講（2026-08-27 稽核 MEDIUM-1）。
        "sigcurve": {
            "split": "val_sel",
            "start": "2024-07-01",
            "end": "2024-12-31",
            "note": "sigcurve_*.csv 是**驗證期（2024 下半年）**的門檻曲線，"
                    "不是測試期統計。門檻就是在這條曲線上由人挑的。"},
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


def clear_superseded(out_dir: Path) -> list[str]:
    removed = []
    for name in SUPERSEDED_OUTPUTS:
        path = out_dir / name
        if path.exists():
            path.unlink()
            removed.append(name)
    if removed:
        logger.info(f"清掉被取代的舊產物：{', '.join(removed)}")
    return removed


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
    clear_superseded(out_dir)
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
    build_fpm_rule_hits(out_dir)
    build_fpm_rule_stats(out_dir)
    export_sigcurves(out_dir)
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
