"""在『相同交易筆數』下比較（修正固定門檻比較的不公平，見 BACKTEST_LOG #19）。

手動分析工具，不在自動流程上。用法：python -m engine.backtest.matched_n <標籤>

⚠️ 兩件事要知道（2026-08-22 搬進本 repo 時確認）：

1. **這支是舊時代的腳本**：切分寫死 `meta_val` / `meta_test`，那是已拆除的
   委員會系統的切分名稱，m1~m10 用的是 Round 4 的 `val_sel` / `test` / `test2`。
   直接跑會找不到分數檔。保留是因為 #19 的比較方法有參考價值。
2. **現行的訊號數對齊實作不在這裡**，在 `engine/backtest/summary.py` 的
   `matched_top`（每日前 1.5%），`make backtest` 走那條。要做 #28 那種比較用
   那支，不要用這支。

頂層直接讀 `sys.argv[1]`（原樣保留，舊 repo 也是這樣），所以不能被 import，
只能當腳本跑。
"""
import sys

from engine.backtest.backtest import simulate, performance
from engine.backtest.benchmark import attach_peer_benchmark, alpha_summary
import numpy as np, pandas as pd

def at_n(split, target_n):
    rows=[]
    for thr in np.arange(0.60, 0.95, 0.005):
        thr=round(float(thr),4)
        t,p=simulate(split=split, threshold=thr)
        if t.empty: break
        q=performance(t,p); rows.append((thr,q['trades']))
        if q['trades']<20: break
    c=pd.DataFrame(rows,columns=['thr','n'])
    thr=float(c.loc[(c.n-target_n).abs().idxmin(),'thr'])
    t,p=simulate(split=split, threshold=thr); q=performance(t,p)
    t=attach_peer_benchmark(t,p,stop_ma=20); a=alpha_summary(t)
    return dict(thr=thr,n=q['trades'],avg=q['avg_return'],med=float(t['return'].median()),
        win=q['win_rate'],sharpe=q['sharpe'],peer=a['peer_mean'],
        alpha_med=a['alpha_median'],win_peer=a['win_vs_peer'],drop5=a.get('alpha_mean_drop5'))

label=sys.argv[1]
for split,tn in (('meta_val',93),('meta_test',89)):
    r=at_n(split,tn)
    print(f"[{label}] {split} 匹配N={tn}: thr={r['thr']:.3f} 實得{r['n']}筆 "
          f"avg{r['avg']:+.2%} med{r['med']:+.2%} | peer{r['peer']:+.2%} "
          f"alphaMed{r['alpha_med']:+.2%} winPeer{r['win_peer']:.1%} drop5{r['drop5']:+.2%}", flush=True)
