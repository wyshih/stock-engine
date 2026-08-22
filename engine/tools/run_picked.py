"""用使用者挑定的門檻，對每個模型在完整測試期跑回測對照。

一律呼叫 `backtest.simulate()` / `performance()`，出場參數用 `CURRENT_EXIT_RULES`，
口徑 `dedup=False`（每筆超過門檻的訊號獨立進場＝使用者的實際用法，也與挑門檻時
看的曲線同一把尺）。不另外算報酬 —— 這個專案吃過方法漂移的虧。

測試期把該模型所有 test* 切分接起來（Round 4 的 test 只有 11 個月、test2 只有
7 個月，分開看區間太短）。

用法：
  python code/backtest/run_picked.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd


from engine.models import bundle as B  # noqa: E402
from engine.backtest.backtest import CURRENT_EXIT_RULES as R, performance, simulate  # noqa: E402
from engine.paths import DATA_DIR  # noqa: E402

# 使用者 2026-08-15 從 val_sel 曲線上挑定
PICKED = {
    "r4_rf": 0.72,
    "sigma_r4": 0.87,
    "labelA_r4": 0.0,      # 低於該模型最低分 0.0732 → 等於全買（零技巧基準）
    "exit7_r4": 0.75,
    "labelB_r4": 0.47,
    "labelC_r4": 0.53,
}


def combined_scores(key: str) -> Path | None:
    frames = []
    for sp in ("test", "test2", "test3"):
        p = B.score_path(key, sp)
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(subset=["date", "stock_id"]).sort_values(["date", "stock_id"])
    path = Path(tempfile.gettempdir()) / f"picked_{key}.parquet"
    out.to_parquet(path, index=False)
    return path


def run(key: str, thr: float) -> dict | None:
    path = combined_scores(key)
    if path is None:
        return None
    span = pd.read_parquet(path, columns=["date"])["date"]
    trades, price = simulate(
        "test", score_path=path, threshold=thr, dedup=False,
        take_profit=R["take_profit"], stop_ma=R["stop_ma"],
        trail_trigger=R["trail_trigger"], trail_pct=R["trail_pct"],
        stop_loss=R["stop_loss"],
    )
    if trades.empty:
        return {"model": key, "threshold": thr, "trades": 0}
    perf = performance(trades, price)
    return {
        "model": key, "threshold": thr,
        "span": f"{span.min().date()}~{span.max().date()}",
        "trades": perf["trades"], "win_rate": perf["win_rate"],
        "avg_return": perf["avg_return"], "total_return": perf["total_return"],
        "sharpe": perf["sharpe"], "max_drawdown": perf["max_drawdown"],
        "trail_stop_pct": perf["trail_stop_pct"],
    }


def main() -> None:
    rows = []
    for key, thr in PICKED.items():
        print(f"\n=== {key} @ {thr} ===", flush=True)
        r = run(key, thr)
        if r:
            rows.append(r)
            print(f"  {r.get('trades', 0):,} 筆　勝率 {r.get('win_rate', 0):.1%}　"
                  f"平均報酬 {r.get('avg_return', 0):+.2%}", flush=True)

    df = pd.DataFrame(rows)
    out = DATA_DIR / "backtest_picked_20260815.csv"
    df.to_csv(out, index=False)
    print("\n" + "=" * 70)
    print(df.to_string(index=False))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
