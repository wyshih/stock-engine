"""
交易門檻調校（2026-07-29 新增）：在 meta_val(2025) 上用**實際交易模擬報酬**
搜出生產用門檻，寫進 `models/threshold_trading.pkl`。

為什麼需要這個檔（見 doc/AUDIT_20260728.md §C-4）：
  `train_meta.py` Step 3 用 F-beta(β=0.5) 在 label_meta 上搜出的
  `threshold.pkl`，是 **Meta 分類任務**的最佳門檻，不是交易門檻。#13 改了
  label_meta 定義之後，F-beta 最佳點從 0.78 掉到 0.62，而 `predict.py` 讀的
  正是這個檔 —— 導致每日推薦名單走一個從未被交易回測驗證過的門檻
  （實測 0.62 在 2026：3226 筆 / 平均 2.97% / 勝率 33.9%）。

  現在兩個門檻分開存：
    models/threshold_fbeta.pkl   ← train_meta.py Step 3 寫，純分類診斷用
    models/threshold_trading.pkl ← 本腳本寫，predict.py / backtest.py / 前端用

紀律（CLAUDE.md 回測規則）：一律用 `backtest.py` 的 simulate()/performance()
搭配 `models/` 底下的正式模型，不另寫模擬器、不用臨時訓練的模型。
調校只在 meta_val(2025) 上做，meta_test(2026) 不參與挑選。

用法：
  python tune_trading_threshold.py                    # 預設網格
  python tune_trading_threshold.py --lo 0.70 --hi 0.90 --step 0.01
  python tune_trading_threshold.py --dry-run          # 只印表格，不寫檔
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 專案路徑一律走 code/paths.py（唯一來源），不要各檔自行推導
from engine.paths import MODEL_DIR, PROJECT_ROOT  # noqa: E402



from engine.backtest.backtest import performance, simulate  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# 調校一律在 meta_val(2025)，meta_test(2026) 保留當驗證期，不參與挑選
TUNE_SPLIT = "meta_val"

# 樣本量下限：BACKTEST_LOG #5/#9/#10/#12 反覆踩到「在幾十筆的池子裡雕出樣本外
# 會消失的假訊號」。低於這個筆數的門檻一律不列入候選，不管報酬數字多好看。
MIN_TRADES = 40


def sweep(lo: float, hi: float, step: float, split: str = TUNE_SPLIT) -> pd.DataFrame:
    """對門檻網格逐一跑正式回測，回傳每個門檻的績效表。"""
    rows = []
    for thr in np.arange(lo, hi + 1e-9, step):
        thr = round(float(thr), 4)
        trades, price = simulate(split=split, threshold=thr)
        if trades.empty:
            rows.append({"threshold": thr, "trades": 0})
            print(f"  門檻 {thr:.3f}：無交易")
            continue
        perf = performance(trades, price)
        rows.append({"threshold": thr, **perf})
        print(f"  門檻 {thr:.3f}：{perf['trades']:4d} 筆　"
              f"平均 {perf['avg_return']:+.2%}　中位 {trades['return'].median():+.2%}　"
              f"勝率 {perf['win_rate']:.1%}　Sharpe {perf['sharpe']:.3f}")
    return pd.DataFrame(rows)


# 平台寬度：取前後各這麼多個網格點做移動平均，用來找「穩定區」而非「尖點」
PLATEAU_HALFWIDTH = 2


def pick_best(table: pd.DataFrame, min_trades: int = MIN_TRADES) -> dict:
    """
    從網格表挑生產門檻 —— 取「平台」而非「單點最高」。

    2026-08-02 修改（原本是直接取 avg_return 最高的單點）：
    單點最高幾乎必然落在極端值上。實測發現舊做法挑出的門檻由**單一異常月份**
    決定 —— 2025-04 關稅崩跌後的反彈讓那個月的預測機率遠高於其他月份
    （該月最大機率 0.828，其他月份只有 0.53~0.70），報酬峰值全部落在那批樣本，
    於是門檻被訂在 0.81。套到沒有同等級急跌的期間，一筆交易都篩不到。

    改法：對 avg_return 取 ±PLATEAU_HALFWIDTH 個網格點的移動平均，選平滑後最高者。
    平台代表「對門檻不敏感」，比尖點穩健得多。仍保留 MIN_TRADES 下限。

    注意：這裡刻意不做「取鄰近門檻的平均值當作門檻」—— #13 踩過那個坑
    （會算出超出機率範圍、0 筆交易的門檻）。移動平均只用來評分，
    回傳的門檻一定是網格上真實存在、且已驗證有交易的點。
    """
    ok = table[table.get("trades", 0) >= min_trades].copy()
    if ok.empty:
        raise RuntimeError(
            f"沒有任何門檻的交易數 >= {min_trades} 筆，無法挑出可信的生產門檻。"
            f"請放寬網格範圍或檢查模型/機率矩陣是否正常。"
        )
    ok = ok.sort_values("threshold").reset_index(drop=True)
    win = 2 * PLATEAU_HALFWIDTH + 1
    ok["plateau_score"] = (
        ok["avg_return"].rolling(win, center=True, min_periods=1).mean()
    )
    best = ok.loc[ok["plateau_score"].idxmax()]
    return {
        "threshold":     round(float(best["threshold"]), 4),
        "tuned_on":      TUNE_SPLIT,
        "trades":        int(best["trades"]),
        "avg_return":    float(best["avg_return"]),
        "plateau_score": float(best["plateau_score"]),
        "win_rate":      float(best["win_rate"]),
        "sharpe":        float(best["sharpe"]),
        "min_trades":    min_trades,
        "selection":     f"plateau(+-{PLATEAU_HALFWIDTH})",
        "source":        "tune_trading_threshold.py",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lo",   type=float, default=0.70)
    parser.add_argument("--hi",   type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--min-trades", type=int, default=MIN_TRADES)
    parser.add_argument("--dry-run", action="store_true", help="只印表格，不寫檔")
    parser.add_argument("--threshold", type=float, default=None,
                        help="人工指定門檻（看過完整曲線後自行決定），跳過自動挑選")
    args = parser.parse_args()

    print(f"在 {TUNE_SPLIT} 上掃描門檻 {args.lo}~{args.hi}（step {args.step}）…\n")
    table = sweep(args.lo, args.hi, args.step)

    if args.threshold is not None:
        row = table.iloc[(table["threshold"] - args.threshold).abs().idxmin()]
        best = {"threshold": round(float(row["threshold"]), 4), "tuned_on": TUNE_SPLIT,
                "trades": int(row.get("trades", 0)), "avg_return": float(row.get("avg_return", 0)),
                "win_rate": float(row.get("win_rate", 0)), "sharpe": float(row.get("sharpe", 0)),
                "min_trades": args.min_trades, "selection": "manual",
                "source": "tune_trading_threshold.py"}
    else:
        best = pick_best(table, args.min_trades)
    print(f"\n最佳交易門檻：{best['threshold']:.1%}"
          f"（{best['trades']} 筆，平均 {best['avg_return']:+.2%}，"
          f"勝率 {best['win_rate']:.1%}，Sharpe {best['sharpe']:.3f}）")
    print(f"※ 樣本量下限 {args.min_trades} 筆以下的門檻已被排除")

    if args.dry_run:
        print("\n--dry-run：未寫檔")
        return

    out = MODEL_DIR / "threshold_trading.pkl"
    with open(out, "wb") as f:
        pickle.dump(best, f)
    print(f"\n已寫入 {out}")

    table_path = MODEL_DIR / "threshold_trading_sweep.csv"
    table.to_csv(table_path, index=False)
    print(f"完整掃描表：{table_path}")


if __name__ == "__main__":
    main()
