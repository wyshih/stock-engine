# 特徵計算修正（2026-08-02）

## ⚠️ 重要：既有模型已失效

`models/*.pkl` 訓練於 **2026-07-30**，用的是修正前的 `features.parquet`。
本次修正已於 **2026-08-02 16:59** 重建 `features.parquet`。

**`predict.py` 會讀 `features.parquet`，所以現在是「新特徵餵給舊模型」的
train/serve 不一致狀態，推薦結果不可信，而且不會報錯。**

處置：**重訓全部子模型與 Meta**（`train_submodels.py` → `train_meta.py` →
`tune_trading_threshold.py`），或在重訓完成前停用 `predict.py` 的輸出。

---

## 修正的內容

被自動掃描標記為數值異常的欄位：**48 → 6**。剩下 6 個中 3 個是該特徵本身的
分布形狀（`institutional_10_5d` / `revenue_accel` / `revenue_mom`，都在 ±1000 內），
3 個是全空欄位（見下）。

### A. 邏輯錯誤：黃金交叉特徵整欄全 0

`macd_golden_3d/5d/10d`、`kd_golden_3d/5d/10d` **原本整欄都是 0**。

原因：交叉是瞬間事件（`dif > sig 且 昨天 dif <= sig`），今天剛穿越、明天
`shift(1)` 的比較就反向，**在定義上不可能連續兩天成立**，卻被拿去算「連續天數」
（`_add_streak`）。要求連續 3/5/10 天的欄位必然恆為 0。

修法：新增 `_add_event()`，欄位名稱不變但語意改為「過去 N 天內曾交叉」，
`_log` 欄改為 `log(1 + 距上次交叉幾天)`。修正後非零比例：
`macd_golden_1d` 3.93% / `_3d` 15.51% / `_5d` 22.87% / `_10d` 39.78%。

其他用 `_add_streak` 的條件（均線多頭排列、柱狀體擴張、突破布林上軌）都是
可持續的**狀態**，不受影響。

### B. 單位不一致

`build_chip_features.py`：TWSE T86 的法人買賣超單位是「股」，`vol_zhang` 是
「張」（＝股/1000），原式 `net / vol_zhang` 讓比率膨脹 1000 倍。已改為除以股數。

### C. 除以趨近零（同一種錯誤重複十餘處）

原本多處只寫 `.replace(0, np.nan)`，擋不掉極小值。實測最小成交量 0.001 張
（1 股），5 萬股的買賣超除下去變成 50 億。這種離群值會把整欄標準差撐爆，
使下游標準化把正常值全部壓到 0 附近 —— **對 LR/MLP/LSTM 等尺度敏感的模型
等同於廢掉該欄**（RF 對尺度免疫，所以先前沒被發現）。

已加分母下限的欄位：法人/融資券各比率、`institutional_10d`、`obv_ratio`、
`ad_ratio`、`obv_pct_5d/20d`、`dif_pct_change`、`dif_slope_5d`、`vol_asymmetry`、
`hist_ratio`、`atr_ratio`、`std_ratio`、`ma_squeeze`、`vol_confirm_rate`、
`tl_channel_width`、`revenue_yoy/mom`。

### D. 超出定義域

`stoch_k/d`、`stochf_k/d`、`ultosc`、`mfi_14`、`rsi`、`adx`、`aroon`、`willr`
等有界指標實測出現 3.3e7、6.6e9 等值（定義上限是 100）。原因是漲停/跌停日
`high == low`，TA-Lib 內部除以 (high−low) 為零。TA-Lib 本身沒問題，是**餵進去
的資料沒做邊界檢查**。已加 `BOUNDED_RANGES` 表，超界一律設 NaN。

### E. 未正規化的絕對量

`apo`、`mom_10`、`close_ols_slope`、`ma5_ols_slope`（單位：元）、
`margin_slope_5d`（單位：張）原本未正規化，高價股天生數值大，模型學到的是
「股價高低」而非型態。已除以股價或均量。

註：`build_features.py` 原本的註解自承 margin_slope「直接保留 raw 版本並改名」，
即從未正規化。

### F. 前波高低點距離的分母

`high1~3_dist`、`low1~3_dist`、`vh1~3_dist`、`vl1~3_dist`、`support_dist`、
`resist_dist`、`tl_support_dist`、`tl_resist_dist` 原本除以**現價**，股價從高點
崩跌時分母趨近零，實測 `high1_dist` 最低到 −13,378（上界卻只有 0.99，嚴重不對稱）。

已改為除以**基準價**（前波高低點的價位 / 趨勢線價位），語意變成「自該點以來的
報酬率」，下界自然有界於 −1。同時把基準從轉折日的收盤價改為最高/最低價
（既然叫前波「高點」，壓力位看的是最高價）。

### G. 移除全空欄位

`is_full_cash` / `is_disposed` / `is_warning` 整欄全 0 —— `stock_list.parquet`
的資料來源（FinMind TaiwanStockInfo）沒有這些欄位，從未被填值。已移除 merge。
（欄位仍以 NaN 形式存在於 parquet，對模型無作用。）

---

## 價格資料清理（新增 `code/data_collection/clean_price_outliers.py`）

yfinance 的兩類錯誤，都會污染所有以價格為基礎的特徵：

1. **單日跳點**：整根 K 棒被放大 N 倍、隔日恢復。實測 1752 在 2025-01-10
   收盤 4030.99（前後兩天都是 40 上下，剛好 100 倍）。命中 2 根。
2. **整段水準異常**：3666 在 2019-01 ~ 2022-05 的價格是 11 萬 ~ 88 萬元
   （台股史上最高約 6,000 元），之後跌到 16.6 元 —— 還原股價把某次公司行動
   算錯，整段放大約 1 萬倍。命中 816 列。

判定條件刻意設計成不會誤刪真實的減資階梯（減資後價格永久改變，前後日不會
互相接近）。全市場約 2,000 檔零誤判。命中列的 OHLC 設為 NaN，保留列與日期
以維持交易日對齊。

另在 `build_price_features.py` 加報酬率防護：單日/N 日報酬超過 ±100% 設 NaN
（減資、合併造成的價格階梯是真實的價格變動，但不是可交易的報酬）。

---

## 月營收抓取修正（`fetch_revenue.py`）

原本的逐筆模式（每檔一次 POST 到 `ajax_t05st10_ifrs`）**大量靜默失敗** ——
`except` 只寫 warning 就跳過，流程看起來成功，實際每月 2,128 檔只抓到約 583 檔。
這是 `AUDIT_20260728.md` 記錄的「基本面/營收特徵 62.4% NaN」的真正原因。

已新增 `--bulk-start YYYY-MM --bulk-end YYYY-MM` 批次模式，改用 MOPS 整月彙總頁
`https://mopsov.twse.com.tw/nas/t21/{sii|otc}/t21sc03_{民國年}_{月}_0.html`，
一個月只要 2 次請求拿到全部公司。

| | 逐筆 | 批次 |
|---|---|---|
| 每月請求數 | 2,128 | 2 |
| 36 個月耗時 | ~57 小時 | ~2 分鐘 |
| 每月涵蓋 | ~583 檔 | ~1,775 檔 |

正確性驗證：2023-05 重疊的 451 檔數值 100% 相同。
重抓 2019-01 ~ 2026-06 後 `revenue.parquet` 從 34,969 筆變 166,134 筆。

**每日模式未更動**，`--date` 的行為完全不變。

---

## 資料補齊

全部用 repo 原有腳本補到 2019-01-02（原本從 2022-01 起）：

| 資料 | 結果 |
|---|---|
| price | 340 萬筆 |
| chip（TWSE + TPEX）| 2019-2021 補齊，各年 242~245 個交易日 |
| fundamental | 2019-2021 補齊（仍僅上市）|
| revenue | 2019-2026 全量重抓 |

`features.parquet`：2,088,487 列 → **3,411,592 列**，NaN 率 **4.3% → 3.0%**
（基本面/營收群從 62.4% 降到 15.3%）。

---

## 修正對模型表現的影響

**幾乎沒有。** 同一組切分下 RF 20 日的 test AUC：修正前 0.6392 → 修正後 0.6371。
四個模型的排序（RF > MLP > LR > LSTM）也完全不變。

修正的價值在**資料正確性與可信度**，不在績效提升。但既有模型仍必須重訓，
因為它們學到的分裂點對應的是舊的數值分布。

---

## 備份

所有被修改的檔案都留有備份：
- `code/features/build_*.py.bak` / `.bak2`
- `code/data_collection/fetch_revenue.py.bak`
- `data/*.parquet.bak_before_2019_backfill`
- `data/price.parquet.bak_before_spike_clean_20260801_185106`

## 驗收方式

`~/stock_committee_experiments/updays/full_scan.py` —— 全欄位掃描，用五個不依賴
名稱的判準標記可疑欄位（有 inf / 極端偏斜 / 尾端脫節 / 值過大 / 近常數）。
任何特徵改動後應重跑此腳本確認。
