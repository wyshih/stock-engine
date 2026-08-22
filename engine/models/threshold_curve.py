"""門檻挑選曲線：驗證期的**實際回測交易**由高分到低分排序後，逐點的勝率與報酬。

用法：
  python threshold_curve.py --tag final_lightgbm --split val_sel

做法（使用者指定）：把驗證期的預測機率**由高到低排序**，逐一以每個預測機率當門檻，
算出累積的勝率與平均／中位報酬，畫成曲線，用眼睛看「取到哪裡開始掉」再決定門檻。

⚠️ **驗證與測試一律用同一套方法**（CLAUDE.md 回測規則）：本檔的每一筆交易都來自
`backtest.py` 的 `simulate()` —— 隔日開盤進場、MA 停損／移動停利出場、同一支不重複
進場，跟正式回測逐字相同。**不得在這裡自行定義報酬**（2026-08-06 修正：舊版自己算
「固定持有 20 日收盤報酬」，與回測是兩把尺，導致驗證期看到 +6.62%、測試期只有
+0.47%，落差 84% 來自這個不一致）。

為什麼不用自動規則挑：單點最高幾乎必然落在極端值上（BACKTEST_LOG #13 踩過 ——
門檻被單一異常月份決定，套到其他期間一筆都篩不到）。曲線讓人同時看到「平台在哪」
與「樣本數掉到多少」，這是自動規則給不了的資訊。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 專案路徑一律走 code/paths.py（唯一來源），不要各檔自行推導
from engine.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402
from engine.backtest.backtest import simulate  # noqa: E402


MIN_SAMPLES = 40  # 低於這個筆數的深度不畫，避免在雜訊上挑點

INK = "#2c2c2c"
MUTED = "#8a8a8a"
SERIES = "#3b6ea5"
BASELINE = "#b0b0b0"


# 出場規則的預設值。**全部可由 CLI 覆寫** —— 出場規則還在調整中，
# 挑門檻時必須能跟著當下要測的規則走，寫死會逼人改程式碼。
# 不論怎麼調，門檻曲線與正式回測一定用同一組值（驗證與測試同一把尺）。
EXIT_DEFAULTS = dict(take_profit=0.20, trail_trigger=0.25, trail_pct=0.10,
                     stop_ma=20, stop_loss=None)


def backtest_trades(tag: str, split: str, floor: float, exit_kw: dict) -> pd.DataFrame:
    """跑一次 `simulate()`，取得該切分在 floor 以上的全部實際交易。

    只跑一次：門檻由高往低掃時，提高門檻只會**移除**訊號，所以高門檻的交易集合
    是低門檻集合的子集（同股去重可能讓極少數較晚的訊號補上，屬可忽略的近似）。
    """
    score_path = DATA_DIR / f"score_{tag}_{split}.parquet"
    trades, _ = simulate(split=split, score_path=score_path,
                         threshold=floor, **exit_kw)
    return trades


def cumulative_by_threshold(trades: pd.DataFrame) -> pd.DataFrame:
    """把**每一筆交易的進場分數**當門檻，逐點算「分數 >= 該門檻」的勝率與報酬。

    x 軸是門檻分數值（不是「取前幾 %」）—— 門檻是模型輸出的實際數值。
    報酬一律取自 `simulate()` 的 `return` 欄，不在這裡另行計算。
    """
    ranked = trades.sort_values("score", ascending=False).reset_index(drop=True)
    r = ranked["return"].to_numpy()
    n = np.arange(1, len(ranked) + 1)

    out = pd.DataFrame({
        "threshold": ranked["score"].to_numpy(),
        "n": n,
        "win_rate": np.cumsum(r > 0) / n,
        "avg_return": np.cumsum(r) / n,
        "med_return": [float(np.median(r[:i])) for i in n],
    })
    return out[out["n"] >= MIN_SAMPLES].reset_index(drop=True)


def plot_curve(curve: pd.DataFrame, baseline: dict, tag: str, split: str,
               out_path: Path) -> None:
    """上下兩張小圖：勝率、平均報酬。兩者尺度不同，不共用 y 軸。"""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    x = curve["threshold"]

    panels = [
        (axes[0], curve["win_rate"] * 100, baseline["win_rate"] * 100,
         "win rate (%) — realised, simulate() exit rules"),
        (axes[1], curve["avg_return"] * 100, baseline["avg_return"] * 100,
         "avg return (%) — realised per trade"),
    ]
    for ax, y, base, title in panels:
        ax.plot(x, y, color=SERIES, linewidth=2)
        ax.axhline(base, color=BASELINE, linewidth=1.5, linestyle="--")
        ax.annotate(f"all {base:.2f}", xy=(x.iloc[-1], base), xytext=(4, 4),
                    textcoords="offset points", ha="left", fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    # 標幾個關鍵門檻的樣本數 —— 高門檻端筆數很少，是判讀時最需要看到的資訊
    for target in (100, 500, 2000, 10000):
        idx = (curve["n"] - target).abs().idxmin()
        row = curve.loc[idx]
        axes[1].annotate(
            f"n={int(row['n']):,}",
            xy=(row["threshold"], row["avg_return"] * 100),
            xytext=(0, -16), textcoords="offset points",
            ha="center", fontsize=8, color=MUTED,
        )

    axes[1].invert_xaxis()  # 門檻由高到低，與「往下走」的直覺一致
    axes[1].set_xlabel("threshold (predicted probability, high → low)",
                       fontsize=10, color=INK)
    fig.suptitle(f"{tag} threshold curve ({split}, sorted by score desc)",
                 fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--split", default="val_sel")
    parser.add_argument("--floor", type=float, default=0.0,
                        help="掃描下限；低於此分數的訊號不進場（純粹為了控制模擬量）")
    # ── 出場規則：與正式回測共用同一組參數，這裡調什麼回測就要調什麼 ──────
    parser.add_argument("--stop-loss", type=float, default=None,
                        help="固定百分比停損，例如 0.20；設定時不使用 MA 停損")
    parser.add_argument("--trail-trigger", type=float, default=0.25,
                        help="移動停利啟動點；設 0 或負數表示關閉，改用固定停利")
    parser.add_argument("--trail-pct", type=float, default=0.10, help="移動停利回落幅度")
    parser.add_argument("--take-profit", type=float, default=0.20,
                        help="固定停利（僅在 trail-trigger 關閉時生效）")
    parser.add_argument("--stop-ma", type=int, default=20, choices=[10, 20])
    args = parser.parse_args()

    exit_kw = dict(EXIT_DEFAULTS)
    exit_kw.update(
        take_profit=args.take_profit, trail_pct=args.trail_pct,
        stop_ma=args.stop_ma, stop_loss=args.stop_loss,
        trail_trigger=args.trail_trigger if args.trail_trigger and args.trail_trigger > 0 else None,
    )
    print(f"出場規則：{exit_kw}")

    trades = backtest_trades(args.tag, args.split, args.floor, exit_kw)
    if trades.empty:
        raise SystemExit(f"{args.tag}/{args.split} 在 floor={args.floor} 下沒有任何交易")

    r = trades["return"]
    # 基準 = 同一套出場規則下、該切分的全部交易（等於門檻放到最寬）
    baseline = {
        "win_rate": float((r > 0).mean()),
        "avg_return": float(r.mean()),
        # 中位數比平均重要：平均容易被少數大漲個股撐高（CLAUDE.md 驗證規則）
        "med_return": float(r.median()),
    }
    curve = cumulative_by_threshold(trades)

    csv_path = DATA_DIR / f"threshold_curve_{args.tag}_{args.split}.csv"
    png_path = DATA_DIR / f"threshold_curve_{args.tag}_{args.split}.png"
    curve.to_csv(csv_path, index=False)
    plot_curve(curve, baseline, args.tag, args.split, png_path)

    print(f"\n全體（門檻放到最寬，{len(trades):,} 筆實際交易）："
          f"勝率 {baseline['win_rate']:.1%}，平均報酬 {baseline['avg_return']:+.2%}，"
          f"中位 {baseline['med_return']:+.2%}\n")

    # 完整曲線每筆交易一點，終端印代表深度方便判讀，CSV 存全部
    print(f"{'門檻':>8}  {'n':>7}  {'勝率':>7}  {'平均報酬':>9}  {'中位報酬':>9}")
    for target in (40, 60, 100, 150, 250, 400, 600, 1000, 1600, 2500, 4000, 6000, 10000):
        if target > curve["n"].max():
            break
        row = curve.iloc[(curve["n"] - target).abs().idxmin()]
        print(f"{row['threshold']:8.3f}  {int(row['n']):7,}  {row['win_rate']*100:6.1f}%  "
              f"{row['avg_return']*100:+8.2f}%  {row['med_return']*100:+8.2f}%")
    print(f"\n{png_path}\n{csv_path}")


if __name__ == "__main__":
    main()
