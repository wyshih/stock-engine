"""`label_steady20`：盤整緩漲的標的 —— 漲幅要跑贏個股自己的波動，且期間站得住 20 日線。

## 這個模型的角色（2026-09-04 使用者定案，第三輪）

前兩輪（本檔第一版、`label_xsrank20`）都以「取代 / 打敗 m1_base_up20」為標準，
兩次都失敗：免訓練驗證證實 m1 選中的崩跌反彈**是真正賺錢的訊號，不是雜訊**
（見 doc/BACKTEST_LOG.md #33）——剔除崩跌股只會讓 m1 的 Sharpe 從 0.777 掉到
0.727~0.755，不會變好。也就是說「盤整緩漲」在目前的特徵集下，可預測性本來就
比「崩跌反彈」弱很多（AUC 天花板約 0.55），這不是 label 設計的問題，是特徵集
在這個子問題上訊號較弱。

**所以這個模型的定位改變了**：不是 m1 的競爭者或替代品，是**覆蓋率用途的
獨立第二選單**——目的是在崩跌反彈之外，額外給一條「盤整緩漲」的候選名單，
用它自己的絕對標準判斷（勝率是否過半、平均報酬是否為正、Sharpe 是否為正），
不要求它超越或接近 m1 的績效。第一輪的訓練結果剛好符合這個較低標準：
勝率 53.2%、平均報酬 +4.32%、Sharpe 0.445（無上限持有），訊號組成裡盤整緩漲
佔 85.2%、崩跌只佔 7.3%——方向對、經濟意義為正，只是量級遠不如 m1。

⚠️ 上一輪這個模型是**借用 m1_base_up20 的組態看成效**（bundle 的
`config_source_key="m1_base_up20"`），不是依規定各自調參。這一輪決定正式出貨，
所以要重新走一次完整的調參（`train_all.sh` 的 sweep 步驟），拿掉 `--config-key`。

## 為什麼要這個標的（2026-09-02，最初的診斷，仍然成立）

`label_up20` 數的是**上漲天數的頻率**，而頻率是波動度的代理 —— 剛崩跌的股票
日線方向來回擺盪，湊滿 20 天裡 10 天上漲容易得多。實測（全歷史）：

    進場前 20 日報酬 < -20% 的樣本，label_up20 正例率 63.2%
    進場前 20 日報酬 -5~0% 的樣本，          正例率 35.4%   → 偏差 1.78 倍

模型忠實地最佳化了這個目標，於是 m1 在門檻 0.77 之上的 14,867 筆訊號裡，
**97.8% 是「過去 20 日跌超過 10%」的股票**（全市場只有 13.5%），86.6% 距 60 日
高點跌超過 20%。更直接的證據：`label_up20` 正例裡「盤整緩漲」只佔 53.3%，
**比全市場基準 56.7% 還低** —— 它不是沒偏好緩漲，是實際上在排斥。

## 試過但無效的方向（都實測過，別再走一次）

    候選                                    崩跌/持平  正例中緩漲%  正例報酬中位
    label_up20（現行）                          1.78     53.3%      +5.2%
    label_mdd10（加最大回檔限制）                 1.62        —          —
    絕對報酬門檻（漲>8% 且回檔<5%）                3.09        —          —
    趨勢掃描 t 值（de Prado）                    1.78        —          —
    「上漲」改成收盤 > 當日開盤                    1.57        —          —
    站上 5 日線 >= 10 天                        1.41     56.6%      +4.5%
    橫斷面風險調整排名前 20%（label_xsrank20）      1.12     ——        ——（Sharpe 0.264，比本標的差，未復活）

  * `_trend_scanning_label`（`build_labels.py` 裡留著沒用的那支，註解說「下一階段
    label 重新設計會用到」）**解決不了這個問題** —— V 型反彈套 OLS 一樣是漂亮的
    上升直線。別把時間花在那裡。
  * 絕對報酬門檻**更糟**：崩跌後本來就更容易漲 8%。
  * 「站上 5 日線」是中性的（56.6% ≈ 全市場基準 56.7%），而且**門檻越收緊越糟**
    —— 從 >=10 天拉到 >=16 天，偏差從 1.74 惡化到 2.79。MA5 太快，V 型反彈整段
    都在 5 日線上，這條線分不出反彈與緩漲。
  * 橫斷面排名（`label_xsrank20`）雖然把正例率釘死在 20%，且分數上限碰到了
    滑桿下限，但績效比本標的更差（Sharpe 0.264、勝率 50.1%=擲硬幣），已從
    程式碼移除（見 doc/BACKTEST_LOG.md #32，git 歷史可查）。

## 規則（與 `build_labels.py` 同為收盤制）

    sigma  = 進場前 20 個交易日的日報酬標準差 × sqrt(20)
    target = max(K_SIGMA × sigma, MIN_RETURN)

    label_steady20 = 1  ⟺  未來 20 日報酬 > target
                       且  未來 20 日中至少 MIN_DAYS_ABOVE 天收盤站上當日 20 日線

兩個條件各自負責一件事，缺一不可：

  * **報酬條件**負責「賺得到錢」。用個股自己的波動當分母是關鍵 —— 緩漲股波動小、
    要求的漲幅自動變低；剛崩跌的股票波動大、要求自動變高。這一條就把偏差從
    1.78 壓到 1.13。
  * **均線條件**負責「路徑是持續的而非一次跳空」。單獨用它偏差是 1.05 但正例報酬
    只有 +4.2%（純粹的動能持續性，賺不到錢）；跟報酬條件疊起來才補上最後那點偏差。

實測（全歷史，K=1.5 / 下限 5% / 20 日線 >= 10 天）：

    基準率 10.0%　崩跌/持平 0.98　正例中緩漲 63.3%（基準 56.7%）
    正例中崩跌 8.2%（基準 11.2%）　正例報酬中位 +18.6%　正例期間最深回檔中位 -0.3%

偏差打平，正例組成往緩漲偏，報酬是 `label_up20` 的 3.6 倍，且持有期間幾乎不套牢。

⚠️ **分母沒有洩漏**：`sigma` 與 20 日線都只用進場**當下已知**的資料算出基準；
未來的部分只有「未來 20 日的收盤」與「未來 20 日當天的 20 日線」，後者是標籤
視窗內的量，與 `build_labels.py` 的前瞻視窗同一個性質。

⚠️ **基準率 10.0%，遠低於 `label_up20` 的 39.7%**。三個後果：
  1. 類別更不平衡，學習難度變高
  2. AUC 與現行模型**不可直接比較**
  3. 門檻必須重新挑（CLAUDE.md 規則 7），舊模型的絕對值搬不過來

⚠️ 以上都是 **label 側**的證據，證明的是「目標不再偏袒崩跌反彈」。

## 2026-09-03 訓練結果：偏差修好了，但模型賺不到錢（見 doc/BACKTEST_LOG.md #31）

**label 側成功**：訊號中崩跌股 18.9%→7.3%（低於全市場基準 11.1%）、
緩漲 32.8%→85.2%（基準 57.2%）。它準確地找到了要的那種標的。

**但績效每一項都輸 m1**（訊號數對齊口徑，test+test2）：

    模型            持有    勝率     平均報酬   Sharpe
    m1_base_up20   無上限   65.8%   +9.00%   0.777
    m1_steady20    無上限   53.2%   +4.32%   0.445
    m1_base_up20   20日    55.3%   +3.27%   0.638
    m1_steady20    20日    44.9%   +0.99%   0.364   ← 低於擲硬幣

**病根不是調參，是這個標的的正例率隨市場環境擺盪**：train 10.6% / val_es 14.5% /
val_sel 5.3% / test 7.9% / test2 10.1%（對照 `label_up20` 各期穩定在 35~39%）。
同時要求「漲贏 1.5σ」與「站上 20 日線」，下跌或震盪年份幾乎沒有樣本達標，於是
val_sel 挑的門檻搬到 test2 就不成立 —— 比規則 7 講的更嚴重，是結構性的。
val_sel 的 AUC 也只有 0.5541，接近隨機。

**所以：不要因為它「修好了偏差」就出貨。** 找對型態與賺得到錢是兩回事 ——
盤整緩漲這個型態本身在測試期沒有超額報酬。`m1_steady20` 留在
`bundle.EXPERIMENTAL_KEYS`，沒有門檻、不進回測表與公開資料包。

**還有一道牆**：基準率 10% 讓 RF 的機率輸出整片壓低，m1_steady20 在
val_sel+test+test2 的**分數最大值只有 0.443**，碰不到前端滑桿的下限 0.50
（m1 是 0.971、m1_mdd10 是 0.859）。即使績效過關，這個模型在現行前端上一個訊號
都選不出來，`build_public_bundle` 的刻度檢查也會擋掉。滑桿的 0.50 下限是配著
`label_up20` 約 40% 基準率長出來的預設 —— **任何低基準率的標的都會撞到它**。

**下一步（未執行）**：改成**橫斷面相對**的標的（例如「當日全市場 20 日報酬排名
前 20%」），正例率被定義釘死，不隨大盤起伏。那是新實驗，不是調參能解決的。
⚠️ 但要先決定怎麼解上面那道牆：把分數改成當日橫斷面百分位（讓刻度與基準率脫鉤），
或讓滑桿範圍隨模型而定。兩者都要動前端契約與匯出格式。

用法：
  python -m engine.models.build_labels_steady
  python -m engine.models.build_labels_steady --k-sigma 1.0 --min-days-above 14
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from engine.paths import DATA_DIR

logger = logging.getLogger(__name__)

OUTPUT_LABEL = "label_steady20"
HOLD_BARS = 20
LOOKBACK_BARS = 20
MA_WINDOW = 20
# 1.5：偏差 1.13、基準率 10.6%。1.0 只降到 1.36；2.0 過頭到 0.76（反過來偏袒
# 牛皮股）且基準率掉到 7.3%。三個都實測過，取中間那個。
DEFAULT_K_SIGMA = 1.5
# 絕對下限。只除以波動的話，偏差會從「偏袒崩跌股」翻成「偏袒牛皮股」——
# 依進場波動分五組，正例率 16.4%（最低波動）遞減到 7.6%（最高波動），而且最低
# 波動組正例的實際漲幅中位只有 8.1%、最高波動組是 41.5%，同樣叫「達標」但賺的
# 錢差五倍。加 5% 下限後最低波動組壓到 13.7%，基準率只從 11.1% 掉到 10.6%。
DEFAULT_MIN_RETURN = 0.05
# 20 天裡至少 10 天站上 20 日線＝「一半以上的時間維持在趨勢之上」。
# 拉到 14 天會過頭（偏差 0.63），基準率掉到 8.6% 而改善很小。
DEFAULT_MIN_DAYS_ABOVE = 10


def _forward(series: pd.Series, group: pd.Series, k: int) -> pd.Series:
    """每支股票獨立的 forward shift（未來第 k 期）。"""
    return series.groupby(group).shift(-k)


def trailing_sigma(close: pd.Series, group: pd.Series,
                   bars: int = LOOKBACK_BARS) -> pd.Series:
    """進場前 `bars` 個交易日的日報酬標準差 × sqrt(bars)＝該期間的「一個標準差」。

    ⚠️ 只用進場當下已知的資料。`min_periods=bars` 是刻意的：暖機期不足時回 NaN，
    讓那些列被剔除，而不是拿半截樣本算出一個偏小的 sigma —— 偏小的 sigma 會讓
    門檻變低、把那些列灌成假正例。
    """
    daily = close.groupby(group).pct_change(fill_method=None)
    return (daily.groupby(group)
                 .rolling(bars, min_periods=bars).std()
                 .reset_index(level=0, drop=True) * np.sqrt(bars))


def days_above_ma(close: pd.Series, group: pd.Series, window: int = MA_WINDOW,
                  bars: int = HOLD_BARS) -> tuple[pd.Series, pd.Series]:
    """未來 `bars` 天裡，收盤站上「當天那條 `window` 日線」的天數。

    回傳 (天數, 這一列算不算得出來)。均線本身 `min_periods=window`，暖機期不足
    的日子是 NaN；只要視窗內有任何一天算不出來就整列作廢，不用半截視窗充數。
    """
    ma = (close.groupby(group)
               .rolling(window, min_periods=window).mean()
               .reset_index(level=0, drop=True))
    fwd_close = [_forward(close, group, k) for k in range(1, bars + 1)]
    fwd_ma = [_forward(ma, group, k) for k in range(1, bars + 1)]
    days = sum((c > m).astype("int8") for c, m in zip(fwd_close, fwd_ma))
    ok = (pd.concat(fwd_close, axis=1).notna().all(axis=1)
          & pd.concat(fwd_ma, axis=1).notna().all(axis=1))
    return days, ok


def build(price: pd.DataFrame,
          k_sigma: float = DEFAULT_K_SIGMA,
          min_return: float = DEFAULT_MIN_RETURN,
          min_days_above: int = DEFAULT_MIN_DAYS_ABOVE) -> pd.DataFrame:
    frame = price[["date", "stock_id", "close"]].sort_values(
        ["stock_id", "date"]).reset_index(drop=True)
    close, sid = frame["close"], frame["stock_id"]

    fwd_ret = _forward(close, sid, HOLD_BARS) / close - 1
    target = pd.Series(np.maximum(k_sigma * trailing_sigma(close, sid), min_return),
                       index=frame.index)
    above, ma_ok = days_above_ma(close, sid)

    frame[OUTPUT_LABEL] = ((fwd_ret > target) & (above >= min_days_above)).astype("int8")

    # 未來資料不足（尾端）或暖機期不足（sigma / 均線算不出來）一律剔除，不可當
    # 負例 —— 「未定的未來視窗補 0」是本專案修過的 bug（doc/AUDIT_20260728.md §A-1）。
    keep = fwd_ret.notna() & target.notna() & ma_ok
    out = frame.loc[keep, ["date", "stock_id", OUTPUT_LABEL]].reset_index(drop=True)

    logger.info(
        f"{len(out):,} 列：正例率 {out[OUTPUT_LABEL].mean():.1%}"
        f"（k={k_sigma}、下限 {min_return:.0%}、站上 {MA_WINDOW} 日線 >= "
        f"{min_days_above} 天；要求漲幅中位 {target[keep].median():.1%}）")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price", type=Path, default=DATA_DIR / "price.parquet")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "labels_steady20.parquet")
    parser.add_argument("--k-sigma", type=float, default=DEFAULT_K_SIGMA)
    parser.add_argument("--min-return", type=float, default=DEFAULT_MIN_RETURN)
    parser.add_argument("--min-days-above", type=int, default=DEFAULT_MIN_DAYS_ABOVE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    price = pd.read_parquet(args.price, columns=["date", "stock_id", "close"])
    price["date"] = pd.to_datetime(price["date"])

    out = build(price, args.k_sigma, args.min_return, args.min_days_above)
    out.to_parquet(args.out, index=False)
    logger.info(f"{len(out):,} 列 → {args.out}")


if __name__ == "__main__":
    main()
