"""模型回測彙總表 —— **唯一實作**（CLAUDE.md 規則 8 / 9）。

`make backtest`（內部驗證）與 `make export-public`（public 展示）都呼叫這裡，
不是各寫一份。這兩條路徑之所以必須共用，是因為本專案已經吃過一次虧：
scratchpad 版與正式版的回測混用，數字對不上，人被搞混（CLAUDE.md 規則 8）。

差別只有**區間**：內部驗證預設 test + test2 全段，可用參數改；public 匯出
固定在展示窗口（`build_public_bundle.TEST_START/TEST_END`）。區間由呼叫端傳入，
不寫死在回測邏輯裡。

固定不變的部分（不接受參數覆寫，這是使用者指定的方法）：
  - `simulate()` / `performance()`，出場一律取 `CURRENT_EXIT_RULES`
    （trail_trigger 0.15 / trail_pct 0.10 / stop_loss 0.20）
  - `dedup=False` —— 訊號層級口徑，與挑門檻時看的曲線同一把尺
  - 每個模型都跑兩列：`absolute`（人挑定的門檻）與 `matched_top`（每日前 1.5%）
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd

from engine.models.bundle import CHOSEN_THRESHOLDS
from engine.models.score_source import combined_scores

logger = logging.getLogger(__name__)

# 訊號數對齊的比例。不對齊的模型比較沒有意義 —— 本系統「訊號越少報酬越高」，
# 固定門檻的比較會退化成門檻鬆緊的比較（doc/BACKTEST_LOG.md #28）。
MATCHED_TOP_PCT = 0.015

# 持有期上限，兩種都產（2026-08-25 使用者指定）。
#   None  不限 —— 既有行為，與 norf 的歷史記錄同口徑，但 2024H2 的訊號平均抱
#         108 根 bar、一路抱到 2025/4 崩盤，跨 split 的數字不可比
#   20    與 label_up20 的 horizon 一致（未來 20 個交易日），模型預測什麼就賺什麼
# 並列才看得出「績效有多少來自持有期拉長、有多少來自選股本身」。
HOLD_VARIANTS = (None, 20)

# 2026-09-02 使用者要求只留這兩個。砍掉的 m2/m3/m6/m8 是「去大盤」「v3 特徵集」
# 「去空頭 label」三種變體，連同 v3 特徵管線與 `label_nobear` 一併移除。
# 代號中間有空號是刻意的 —— 沿用原編號，才對得上 BACKTEST_LOG 裡的 ①。
# 兩者同特徵集、同搜尋空間，**只差標的**：m1 是 label_up20，m1_mdd10 再要求
# 「20 日內最低收盤不跌破 −10%」。差異只能來自標的，不會混進調參的運氣。
MODEL_KEYS = ("m1_base_up20", "m1_mdd10")

# 內部驗證的預設區間＝Round 4 的樣本外全段
DEFAULT_SPLITS = ("test", "test2")


def combined_score_file(key: str, tmp_dir: Path, splits: tuple[str, ...],
                        start: str | None = None, end: str | None = None) -> Path:
    """把多個切分的分數接成一份暫存檔。

    `simulate()` 一次只讀一個分數檔，但 Round 4 的 test 只有 11 個月、test2 只有
    7 個月，分開看區間太短。
    """
    # 分數來源與公開資料包共用 `score_source.combined_scores()`，
    # 兩邊各組一份的話會像 2026-08-27 那次一樣，同一模型同一門檻給出不同訊號數。
    out = combined_scores(key, tuple(splits), start, end)
    path = tmp_dir / f"_bt_{key}.parquet"
    out.to_parquet(path, index=False)
    return path


def top_pct_score_file(score_file: Path, tmp_dir: Path, key: str, pct: float) -> Path:
    """每日只留分數最高的前 pct 比例 —— 訊號數對齊版的輸入。"""
    src = pd.read_parquet(score_file)
    # 用 rank 而不是 groupby().apply(head)：後者會把 date 收進 index，
    # 寫出去的 parquet 就沒有 date 欄，simulate() 直接拒收。
    rank = src.groupby("date")["score"].rank(method="first", ascending=False)
    limit = src.groupby("date")["score"].transform("size").mul(pct).round().clip(lower=1)
    keep = src[rank <= limit].sort_values(["date", "stock_id"]).reset_index(drop=True)
    path = tmp_dir / f"_bt_{key}_top.parquet"
    keep.to_parquet(path, index=False)
    return path


def run_one(key: str, score_file: Path, threshold: float, mode: str,
            max_hold_bars: int | None = None) -> dict:
    """單一模型單一口徑的回測。出場規則一律取 CURRENT_EXIT_RULES，不接受覆寫。

    `max_hold_bars`：持有期上限（交易日），None＝不限。兩種都要產 ——
    不限上限時，2024H2 的訊號平均抱 108 根 bar、一路抱到 2025/4 的崩盤，
    跨 split 的數字在方法論上不可比；限 20 日則與 `label_up20` 的 horizon 一致
    （模型預測什麼就賺什麼）。並列才看得出「績效有多少來自持有期而非選股」。
    """
    from engine.backtest.backtest import CURRENT_EXIT_RULES as R, performance, simulate

    trades, price = simulate(
        "test", score_path=score_file, threshold=threshold, dedup=False,
        max_hold_bars=max_hold_bars,
        take_profit=R["take_profit"], stop_ma=R["stop_ma"],
        trail_trigger=R["trail_trigger"], trail_pct=R["trail_pct"],
        stop_loss=R["stop_loss"],
    )
    row = {"model": key, "mode": mode,
           "max_hold": "無上限" if max_hold_bars is None else f"{max_hold_bars}日",
           "threshold": threshold, "trades": 0}
    if trades.empty:
        return row
    perf = performance(trades, price)
    row.update({
        "trades": perf["trades"], "win_rate": perf["win_rate"],
        "avg_return": perf["avg_return"], "total_return": perf["total_return"],
        "sharpe": perf["sharpe"], "max_drawdown": perf["max_drawdown"],
    })
    return row


def build_backtest_summary(out_dir: Path, tmp_dir: Path,
                           splits: tuple[str, ...] = DEFAULT_SPLITS,
                           start: str | None = None, end: str | None = None,
                           keys: tuple[str, ...] = MODEL_KEYS) -> pd.DataFrame:
    """兩張表：絕對門檻版 + 訊號數對齊版（每日前 1.5%）。

    ⚠️ 兩張都要 —— 只看固定門檻的比較會退化成「門檻鬆緊」的比較
    （doc/BACKTEST_LOG.md #28）。不要為了省時間只跑其中一張。
    """
    rows = []
    for key in keys:
        thr = CHOSEN_THRESHOLDS[key]
        combined = combined_score_file(key, tmp_dir, splits, start, end)
        top = top_pct_score_file(combined, tmp_dir, key, MATCHED_TOP_PCT)
        # 2×2：絕對門檻 / 訊號數對齊　×　無持有上限 / 20 日上限
        for hold in HOLD_VARIANTS:
            label = "無上限" if hold is None else f"{hold} 日"
            logger.info(f"回測 {key}：絕對門檻 {thr}　持有{label}")
            rows.append(run_one(key, combined, thr, "absolute", hold))
            logger.info(f"回測 {key}：訊號數對齊（每日前 {MATCHED_TOP_PCT:.1%}）　持有{label}")
            rows.append(run_one(key, top, 0.0, "matched_top", hold))
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "backtest_summary.csv", index=False)
    logger.info(f"backtest_summary：{len(df)} 列 → {out_dir / 'backtest_summary.csv'}")
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="回測彙總（絕對門檻 + 訊號數對齊）")
    parser.add_argument("--out", type=Path, default=Path("data/backtest"))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS),
                        help="逗號分隔，預設 test,test2（Round 4 的樣本外全段）")
    parser.add_argument("--start", help="額外的起日篩選，預設不限（用整個 split）")
    parser.add_argument("--end", help="額外的迄日篩選，預設不限")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        df = build_backtest_summary(args.out, Path(tmp),
                                    splits=tuple(args.splits.split(",")),
                                    start=args.start, end=args.end)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
