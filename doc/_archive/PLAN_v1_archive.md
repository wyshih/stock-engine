# 台股智能預測系統 PLAN v1.0

> 本文件為唯一需求來源，依據 REVIEW.md 全部決議改寫自 DESIGN.md。
> 實作前先讀此文件，每完成一項在對應 [ ] 標記 [x]。

---

## 目錄

1. [系統概覽](#1-系統概覽)
2. [整體架構](#2-整體架構)
3. [資料搜集系統](#3-資料搜集系統)
4. [資料儲存設計](#4-資料儲存設計)
5. [特徵工程](#5-特徵工程)
6. [預測模型系統](#6-預測模型系統)
7. [每日自動化流程](#7-每日自動化流程)
8. [回測系統](#8-回測系統)
9. [開發介面](#9-開發介面)
10. [開發路線圖](#10-開發路線圖)

---

## 1. 系統概覽

### 1.1 系統定位
台股輔助決策工具，每天盤後產生推薦買入名單，供使用者參考。採寧缺勿濫原則，沒有足夠信心的標的不推薦。最終買賣決策由使用者自行判斷。

### 1.2 股票範圍
- 台股上市（TWSE）+ 上櫃（TPEX）**普通股**，約 1700 支
- **排除**：ETF、權證、特別股、REITs、TDR

### 1.3 使用者
- Phase 1：單一使用者，本機使用
- 無帳號系統，無 web service

### 1.4 開發機器
| 規格 | 說明 |
|------|------|
| 型號 | MacBook Air M5 |
| 記憶體 | 24GB 統一記憶體 |
| 核心數 | 10 核心 CPU |
| 作業系統 | macOS |

---

## 2. 整體架構

```
Oracle Cloud Free VM（每日 16:00 台灣時間）
  ├── 抓取當日資料（yfinance + TWSE API + FinMind）
  ├── 驗證 + 寫入 Parquet
  └── git push → private GitHub repo
                        │
                        ▼
                Mac M5（每日盤後）
                  ├── git pull
                  ├── 特徵工程
                  ├── 子模型預測（22 個子模型，只用 RF，見 §6.1）
                  ├── Meta 模型預測
                  └── Streamlit 介面顯示結果
```

### 2.1 技術選型
| 類別 | 選擇 |
|------|------|
| 語言 | Python 3.11+ |
| 資料格式 | Parquet（pyarrow） |
| 特徵計算 | pandas + ta-lib |
| ML | scikit-learn（LightGBM 已於 2026-07-19 停用，見 §6.7）|
| DL（可選） | PyTorch |
| 介面 | Streamlit（Phase 1） |
| 排程 | Oracle VM cron job |
| 版控 | Git + private GitHub repo |
| 測試 | pytest + hypothesis |

---

## 3. 資料搜集系統

### 3.1 資料來源

| 資料類型 | 來源 | 更新頻率 | 歷史範圍 |
|---------|------|---------|---------|
| 日線價格量（還原權息） | yfinance（auto_adjust=True） | 每日 | 2022~至今 |
| 三大法人買賣超 | TWSE 官方 API（T86） | 每日 | 2015~至今 |
| 融資融券 | TWSE 官方 API（MI_MARGN） | 每日 | 2015~至今 |
| 月營收 | FinMind（TaiwanStockMonthRevenue） | 每月 | 2022~至今 |
| 基本面（PER/PBR/殖利率） | FinMind（TaiwanStockPER） | 每日 | 2022~至今 |
| 股票清單 | FinMind（TaiwanStockInfo） | 每季手動更新 |  |
| 大盤加權指數（^TWII） | yfinance | 每日 | 2022~至今 |
| 美股 S&P500（^GSPC） | yfinance | 每日 | 2022~至今 |
| VIX（^VIX） | yfinance | 每日 | 2022~至今 |
| 台幣匯率（TWD=X） | yfinance | 每日 | 2022~至今 |

> TWSE API 端點（免費無限制，已驗證可取得歷史資料）：
> - `https://www.twse.com.tw/exchangeReport/T86?date=YYYYMMDD&selectType=ALL`
> - `https://www.twse.com.tw/exchangeReport/MI_MARGN?date=YYYYMMDD&selectType=ALL`
>
> TPEX 上櫃 API 待 Oracle VM 設好後實機測試確認（WebFetch 會 403）。
>
> 大戶散戶比（TDCC）：只有 1 年保存期，無歷史 API，Phase 1 略過，預留模組位置。

### 3.2 歷史資料初始化

```
code/data_collection/init/
  init_price.py        # yfinance 批次拉 2022~至今
  init_chip.py         # TWSE API 逐日拉 2022~至今
  init_revenue.py      # FinMind 月營收 2022~至今
  init_fundamental.py  # FinMind 基本面 2022~至今
  init_progress.json   # checkpoint（.gitignore）
```

- FinMind 免費版 1,500 次/小時，初始化約 4.5 小時
- 支援中斷續跑（讀 init_progress.json checkpoint）
- 一次性作業

### 3.3 每日更新腳本

```
code/data_collection/
  fetch_price.py       # yfinance 當日全市場
  fetch_chip.py        # TWSE API 當日三大法人+融資券
  fetch_revenue.py     # FinMind 月初才執行（每月 10 日後）
  fetch_fundamental.py # FinMind 每日 PER/PBR/殖利率
  fetch_stock_list.py  # 每季手動觸發
  validate_data.py     # 資料驗證
```

所有腳本支援 `--date YYYY-MM-DD` 參數供手動補跑。
錯誤處理：網路逾時/rate limit → 指數退避重試 3 次；解析錯誤 → log 後跳過。

### 3.4 假日判斷
- 使用 `exchange_calendars`（XTAI）為主，年初自動同步 TWSE 官方休市公告
- 執行時若全市場 >95% 資料為空 → 自動判定突發休市並記錄
- 次日自動補跑前日缺口，無需人工介入

### 3.5 冪等性
- Append 前 `drop_duplicates(subset=['date', 'stock_id'])`
- 重跑同一天不產生重複列

### 3.6 資料時間對齊原則
- **籌碼**：t 日籌碼只用於預測 t+1 日
- **月營收**：以公告日為準 forward-fill（次月 10 日後才使用）
- **季報**：以公告日為準（季末 + 45 天後才使用）
- **前波高低點**：只用過去已確認的點，不使用未來資料

---

## 4. 資料儲存設計

### 4.1 格式
- 全部使用 **Parquet**（pyarrow），每種資料一個檔案
- 複合 key：`date`（datetime）+ `stock_id`（string）
- GitHub repo 儲存上限估算：每日約 200KB，10 年約 500MB，遠低於 5GB 建議上限

### 4.2 目錄結構

```
/（root）
├── code/
│   ├── data_collection/
│   │   ├── fetch_price.py
│   │   ├── fetch_chip.py
│   │   ├── fetch_revenue.py
│   │   ├── fetch_fundamental.py
│   │   ├── fetch_stock_list.py
│   │   ├── validate_data.py
│   │   └── init/
│   │       ├── init_price.py
│   │       ├── init_chip.py
│   │       ├── init_revenue.py
│   │       ├── init_fundamental.py
│   │       └── init_progress.json   ← .gitignore
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_submodels.py
│   │   ├── train_meta.py
│   │   └── predict.py
│   └── backtest/
│       └── backtest.py
├── data/
│   ├── price.parquet
│   ├── chip.parquet
│   ├── revenue.parquet
│   ├── fundamental.parquet
│   └── stock_list.parquet
├── models/                          ← .gitignore（不進 git）
├── tests/
├── notebooks/
│   ├── 01_data_preview.ipynb
│   ├── 02_feature_inspection.ipynb
│   ├── 03_train_test_split.ipynb
│   └── 04_model_output.ipynb
├── .env                             ← .gitignore
├── .env.example
├── .gitignore
├── requirements.txt                 ← 所有套件鎖版本號
└── streamlit_app.py
```

### 4.3 Parquet 欄位定義

#### price.parquet
| 欄位 | 型別 | 說明 |
|------|------|------|
| date | datetime | 交易日期 |
| stock_id | string | 股票代號 |
| open | float | 還原開盤價 |
| high | float | 還原最高價 |
| low | float | 還原最低價 |
| close | float | 還原收盤價 |
| volume | float | 還原成交量（股） |
| amount | float | 成交金額（元） |

#### chip.parquet
| 欄位 | 型別 | 說明 |
|------|------|------|
| date | datetime | 交易日期 |
| stock_id | string | 股票代號 |
| foreign_buy | float | 外資買進張數 |
| foreign_sell | float | 外資賣出張數 |
| foreign_net | float | 外資買賣超張數 |
| trust_buy | float | 投信買進張數 |
| trust_sell | float | 投信賣出張數 |
| trust_net | float | 投信買賣超張數 |
| dealer_net | float | 自營商買賣超張數 |
| margin_balance | float | 融資餘額（張） |
| margin_buy | float | 融資買進張數 |
| margin_sell | float | 融資賣出張數 |
| short_balance | float | 融券餘額（張） |
| short_buy | float | 融券買進張數 |
| short_sell | float | 融券賣出張數 |

#### revenue.parquet
| 欄位 | 型別 | 說明 |
|------|------|------|
| announce_date | datetime | 實際公告日期 |
| stock_id | string | 股票代號 |
| revenue | float | 當月營收（元） |
| revenue_month | int | 營收所屬月份 |
| revenue_year | int | 營收所屬年份 |

#### fundamental.parquet
| 欄位 | 型別 | 說明 |
|------|------|------|
| date | datetime | 日期 |
| stock_id | string | 股票代號 |
| per | float | 本益比 |
| pbr | float | 股價淨值比 |
| dividend_yield | float | 殖利率（%） |
| eps_ttm | float | 近四季 EPS |
| gross_margin | float | 毛利率（%） |

#### stock_list.parquet
| 欄位 | 型別 | 說明 |
|------|------|------|
| stock_id | string | 股票代號 |
| stock_name | string | 股票名稱 |
| market | string | TWSE / TPEX |
| industry | string | 產業別 |
| is_active | bool | 是否仍在交易 |
| is_full_cash | bool | 全額交割股 |
| is_disposed | bool | 處置中 |
| is_warning | bool | 警示股 |
| shares_outstanding | float | 流通股數（股） |
| market_cap | float | 流通市值（元，每日更新） |

---

## 5. 特徵工程

### 5.1 標準化原則
- **絕對價格/金額** → 除以 close 或用比率，確保跨股票可比
- **天數特徵** → 同時產生 `streak_1d`（≥1）、`streak_3d`（≥3）、`streak_5d`（≥5）、`streak_10d`（≥10）四個布林值，加上 `log(days+1)` 連續值，共 5 個維度
- **已是 0~100 / % / 比率** → 直接保留
- **K 線型態** → +100（看多）/ 0（無）/ -100（看空），直接保留

### 5.2 均線特徵

| 特徵 | 計算方式 |
|------|---------|
| close/MA_n | close 除以 MA（n=5,10,20,60,120,240） |
| MA_short/MA_long | MA5/MA20、MA10/MA60、MA20/MA120 等比率 |
| close_zscore | close 的 60 日 rolling z-score |
| above_ma_n | close > MA_n（布林，n=5,10,20,60,120,240） |
| ma_slope_n | (MA_n_t - MA_n_{t-5}) / (5 × close)（n=20,60） |
| ma_squeeze | max(MA5,MA10,MA20) / min(MA5,MA10,MA20) - 1 |
| is_3ma_bull / bear | MA5>MA10>MA20（布林）+ streak 5 維度 |
| is_4ma_bull / bear | MA5>MA10>MA20>MA60（布林）+ streak 5 維度 |
| is_5ma_bull / bear | MA5>MA10>MA20>MA60>MA120（布林）+ streak 5 維度 |

### 5.3 MACD 特徵

| 特徵 | 計算方式 |
|------|---------|
| dif_ratio | DIF / close |
| macd_ratio | MACD / close |
| hist_ratio | Histogram / close |
| dif_positive | DIF > 0（布林）|
| dif_pct_change | DIF 的單日變化率 |
| dif_slope_5d | (DIF_t - DIF_{t-5}) / (5 × close) |
| macd_golden | 黃金交叉（布林）+ streak 5 維度 |
| hist_expand | Histogram 擴大（布林）+ streak 5 維度 |
| hist_3bar_same | Histogram 連續 3 根同向（布林）|
| macd_top_div | price 創 N 日新高但 DIF 不創新高（頂背離，布林）|
| macd_bot_div | price 創 N 日新低但 DIF 不創新低（底背離，布林）|

### 5.4 布林通道特徵

| 特徵 | 計算方式 |
|------|---------|
| bb_pct_b | (close - lower) / (upper - lower)（0~1）|
| bb_pct_b_change | bb_pct_b 的 5 日變化 |
| bb_width_pct | (upper - lower) / middle |
| bb_width_rank | bb_width_pct 的 60 日 rolling percentile |
| bb_upper_break | close > upper（布林）+ streak 5 維度 |
| bb_lower_break | close < lower（布林）+ streak 5 維度 |
| bb_expand | 通道擴大（布林）+ streak 5 維度 |

### 5.5 KD 特徵

| 特徵 | 計算方式 |
|------|---------|
| k_value | KD 的 K 值（0~100）|
| d_value | KD 的 D 值（0~100）|
| kd_diff | (K - D) / 100 |
| k_change_3d | K_t - K_{t-3} |
| kd_golden | 黃金交叉（布林）+ streak 5 維度 |
| k_overbought | K > 80（布林）|
| k_oversold | K < 20（布林）|
| k_oversold_rebound | 前日 K<20 且今日 K>20（布林）|
| k_triple_weak | 連續 3 次從超賣反彈但價格新低（布林）|

### 5.6 RSI 特徵

| 特徵 | 計算方式 |
|------|---------|
| rsi_14 | 14 日 RSI（0~100）|
| rsi_7 | 7 日 RSI |
| rsi_21 | 21 日 RSI |
| rsi_7_21_diff | RSI7 - RSI21 |
| rsi_slope_5d | (RSI_t - RSI_{t-5}) / 5 |
| rsi_overbought | RSI > 70（布林）+ streak 5 維度 |
| rsi_oversold | RSI < 30（布林）+ streak 5 維度 |
| rsi_oversold_rebound | 前日 RSI<30 且今日 RSI>30（布林）|

### 5.7 價格動能特徵

| 特徵 | 計算方式 |
|------|---------|
| return_1d / 5d / 20d / 60d | N 日報酬率（%）|
| rs_5d / 20d / 60d | N 日報酬 - 同期大盤（加權指數）報酬（相對強弱）|
| dist_high_60 | (close - high_60) / high_60 |
| dist_low_60 | (close - low_60) / low_60 |
| dist_high_240 | (close - high_240) / high_240 |
| dist_low_240 | (close - low_240) / low_240 |
| is_new_high_60 | 是否創 60 日新高（布林）|
| momentum_accel | 動能是否加速（return_5d > return_20d/4）（布林）|
| momentum_align | 1/5/20 日報酬均為正（布林）|

### 5.8 波動度特徵

| 特徵 | 計算方式 |
|------|---------|
| atr_ratio | ATR14 / close |
| atr_rank | atr_ratio 的 60 日 rolling percentile |
| std_ratio | 20 日 rolling std / close |
| vol_asymmetry | 漲日 std / 跌日 std（20 日窗口）|

### 5.9 量價特徵

| 特徵 | 計算方式 |
|------|---------|
| obv_pct_5d / 20d | OBV 的 5/20 日 pct_change |
| vol_ratio | 今日量 / 5 日均量 |
| vol_confirm_rate | 漲日均量 / 跌日均量（20 日窗口）|
| cmf | Chaikin Money Flow（20 日）|
| vol_breakout | 量比 > 2 且上漲（布林）|
| avg_vol_ratio | 20 日均量 / 60 日均量 |
| turnover_ratio | 20 日均成交金額 / market_cap |

### 5.10 前波高低點（Swing High/Low）

| 特徵 | 計算方式 |
|------|---------|
| swing_high_1/2/3_dist | 前三個前波高點距離%（close 為基準）|
| swing_high_1/2/3_days | 前三個前波高點距今天數（log + streak）|
| swing_high_vol_ratio | 前波高點形成時成交量 / 同期 60 日均量 |
| swing_high_cluster | 三個前波高點價格 std / close（密集程度）|
| swing_low_1/2/3_dist | 前三個前波低點距離%（類同上）|
| swing_low_1/2/3_days | 前三個前波低點距今天數（log + streak）|
| is_higher_highs | 高點越來越高（布林）|
| is_higher_lows | 低點越來越高（布林）|
| is_lower_highs | 高點越來越低（布林）|
| is_lower_lows | 低點越來越低（布林）|

> 確認方式：前後各 N 天都比它低/高才確認（不使用未來資料）

### 5.11 大量高低點

| 特徵 | 計算方式 |
|------|---------|
| vol_high_1/2/3_dist | 前三個大量高點距離%（大量定義：量比 > 2）|
| vol_high_1/2/3_days | 距今天數（log + streak）|
| vol_high_strength | 大量高點當日量 / 60 日均量 |
| vol_low_1/2/3_dist | 前三個大量低點距離%（類同上）|
| is_above_vol_high1 | close > 最近大量高點（布林）+ streak 5 維度 |
| is_below_vol_low1 | close < 最近大量低點（布林）+ streak 5 維度 |

### 5.12 趨勢線

| 特徵 | 計算方式 |
|------|---------|
| support_dist | 距支撐線%（close 為基準）|
| resist_dist | 距壓力線%（close 為基準）|
| support_slope | 支撐線斜率 / close |
| resist_slope | 壓力線斜率 / close |
| support_r2 | 支撐線 R²（0~1）|
| resist_r2 | 壓力線 R²（0~1）|
| trendline_touches | 趨勢線觸及次數（連續值）|
| trendline_days | 趨勢線持續天數（log）|
| is_triangle | 支撐線與壓力線收斂（布林）|
| slope_align | 支撐線與壓力線斜率同向（布林）|

### 5.13 ta-lib 全量技術指標

使用 ta-lib 提供的所有指標，運算快速者全部納入，讓模型自動學習特徵重要性：

**動量類（已是相對值，直接保留）**
- CCI（Commodity Channel Index）
- MFI（Money Flow Index，0~100）
- Williams %R（-100~0）
- AROON Up / Down / Oscillator（0~100）
- TRIX（三重 EMA 變化率，已是%）
- Ultimate Oscillator（0~100）
- StochRSI K / D（0~1）
- PPO（Percentage Price Oscillator，已是%）
- CMO（Chande Momentum Oscillator）
- MOM / ROC（N 日動量/變化率）

**K 線型態（ta-lib CDL* 系列，全部 61 種）**
- 輸出 +100（看多）/ 0（無型態）/ -100（看空）
- 直接保留，不需標準化

### 5.14 籌碼特徵

| 特徵 | 計算方式 |
|------|---------|
| foreign_net_ratio | 外資買賣超 / 當日總成交量（%）|
| trust_net_ratio | 投信買賣超 / 當日總成交量（%）|
| dealer_net_ratio | 自營商買賣超 / 當日總成交量（%）|
| institutional_10d | 三大法人 10 日累計買賣超 / 均量（%）|
| institutional_10_5d | 10 日累計 - 5 日累計（分離近遠期）|
| foreign_consec_buy | 外資連續買超天數（log + streak）|
| foreign_trust_align | 外資與投信同時淨買超（布林）+ streak 5 維度 |
| margin_ratio | 融資餘額 / 流通股數（%）|
| margin_slope_5d | 融資餘額 5 日斜率 / close |
| short_ratio | 融券餘額 / 流通股數（%）|
| margin_short_ratio | 融資餘額 / 融券餘額（槓桿對比）|
| sec_lending_ratio | 借券賣出量 / 流通股數（%，如有資料）|

### 5.15 基本面／營收特徵

| 特徵 | 計算方式 |
|------|---------|
| per | 本益比（直接保留）|
| pbr | 股價淨值比（直接保留）|
| dividend_yield | 殖利率%（直接保留）|
| per_rank | PER 的 60 月 rolling percentile |
| pbr_rank | PBR 的 60 月 rolling percentile |
| yield_rank | 殖利率的 60 月 rolling percentile |
| revenue_yoy | 月營收 YoY%（以公告日對齊）|
| revenue_mom | 月營收 MoM% |
| revenue_cum_yoy | 累計營收 YoY% |
| revenue_accel | 本月 YoY - 上月 YoY（加速度）|
| revenue_cum_diff | 累計 YoY - 單月 YoY |
| revenue_pos_months | 連續 YoY 為正月數（log + streak）|
| eps_yoy | 最新季 EPS YoY%（以公告日+45天對齊）|
| gross_margin_slope | 近三期毛利率趨勢斜率（標準化）|

### 5.16 異常狀態／流動性特徵

| 特徵 | 計算方式 |
|------|---------|
| is_full_cash | 全額交割（布林）|
| is_disposed | 處置中（布林）|
| is_warning | 警示股（布林）|
| limit_hit_rate | 觸及漲跌停天數 / 20 日 |
| is_low_liquidity | 20 日均成交金額 < 5000 萬元（布林）|
| avg_vol_20_60 | 20 日均量 / 60 日均量（流動性趨勢）|

---

## 6. 預測模型系統

### 6.1 整體架構

> **2026-07-12 重新打底**：原本的 T1~T6/T1B~T4B/T1C 系統已整套移除（commit
> `a5809eb`）。原因：舊系統預設了「起漲點長什麼樣子」（動能/反彈/反轉的人工定義），
> 但資料驗證後發現多數 label 實際上在回答「漲勢會不會延續」而不是「這是不是正確
> 進場點」。新系統改為「先無條件掃描出真正賺錢的買點，再讓模型從正樣本本身的
> 共同特徵去學」，不從假設出發。
>
> **2026-07-18 Ground Truth 設計實驗 + 第二輪清單整理**：針對「怎麼定義買點」做了
> 系統性實驗，比較四種做法——(1) fixed-horizon 終點快照（原始 BUY5/10/20）、
> (2) triple-barrier 路徑判定、(3) 局部低點 NPMM、(4) trend-scanning——並對表現較好
> 的做法做參數調整（停利門檻、停損用 MA10 或 MA20）與 López de Prado 式兩階段
> meta-labeling 架構（primary 篩候選、secondary 判斷要不要出手），在 2026-06
> out-of-sample 資料上以每日 Top-N 推薦名單驗證。結論：
> - **Trend-scanning 證實在 5/10 天版皆無效**（ROC-AUC 貼近 0.5，10天版 unified
>   precision 只有 1.4%、平均報酬轉負），BUY5/10/20_TREND 全部未納入正式清單。
> - **Triple-barrier 的 MA20 停損版全面優於原本的 MA10 版**（5天版平均報酬 1.86%+
>   vs 1.30%，10/20天版同樣驗證更優），`BUY5/10/20_TB_MA20` 取代原本的
>   `BUY5/10/20`（fixed-horizon 基準版）與 `BUY5/10_TB`（舊MA10版）成為正式版本，
>   **舊版已於 2026-07-19 從正式清單移除**（不再保留對照）。
> - **局部低點（NPMM）單獨用不夠準，但當兩階段架構的 primary、TB_MA20 當
>   secondary 時效果最好**（5天版平均報酬 2.46%，全場最高）——`BUY5_LOCAL`/
>   `BUY5_LOCAL_P5` 保留作為這個兩階段組合的 primary 輸入；10/20天版的
>   LOCAL/TREND 驗證後判定較弱，已不再訓練。
> - **EARLY_RALLY、T123_MERGED、M2、M3 已移除**：前兩者不再需要；M2/M3 於
>   2026-07-19 驗證 lift < 1（比隨機還差，因為特徵仍是個股價格特徵、不是真正的
>   大盤指數特徵），屬於負貢獻，直接移除而非保留當弱訊號。
> - **LR、GBT、LightGBM 演算法已全面停用**，只用 RF（原因：LR/GBT 在既有比較中
>   沒有優勢；LightGBM 因 lightgbm/sklearn 版本不相容報錯，直接拔除而非修版本；
>   §6.7 已同步更新）。
> - **正式清單最終定案為 10 個子模型**（見下方），詳細實驗數據與各參數變體的
>   precision/報酬比較見專案 memory（ground truth 設計實驗記錄）。

```
第一層（子模型）× 10 個，只用 RF（LR/GBT/LightGBM 已全面停用，見上方）
  BUY5_TB_MA20   買進後 5 天，triple-barrier（7%停利+MA20停損）
  BUY10_TB_MA20  買進後 10 天，triple-barrier（12%停利+MA20停損）
  BUY20_TB_MA20  買進後 20 天，triple-barrier（21%停利+MA20停損）
  BUY5_LOCAL     局部低點 NPMM（前後5天最低點+5天漲幅≥7%）
  BUY5_LOCAL_P5  局部低點 NPMM 參數版（同上，效果最好的版本）
  C1 籌碼強度（三大法人 10 天累計）
  C2 三大法人動向（外資 5 天累計）
  C3 外資持股趨勢（外資 20 天累計近似）
  F1 營收動能
  M1 大盤短線方向（5 天，特徵仍是個股價格特徵，非真正大盤指數特徵，訊號偏弱）

第二層（Meta 模型，對話/文件中稱「完整版」，跟下方兩階段實驗架構區分）
  輸入：10 個子模型 RF 輸出機率（純 stacking，不混入原始特徵——2026-07-19 實測
    加原始特徵沒有幫助，PR-AUC/lift 幾乎不變，故維持純機率輸入的簡單設計）
  Ground Truth：label_meta（triple-barrier，10%停利+MA20停損，5天horizon，
    見 §6.3）；為避免球員兼裁判，任何跟這個定義相同的子模型不得餵入 Meta 特徵
  模型：RF（LightGBM 已停用，見 §6.7）
  模型檔案：models/meta_model.pkl + models/threshold.pkl
```

- **「完整版」現況（2026-07-19，val PR-AUC=0.1156，lift≈2.75x，門檻75%）**：
  用跟子模型一致的方法（2026-06 out-of-sample，每日 Top-N，統一 5天≥7%報酬檢查）
  驗證，**完整版目前平均報酬明顯偏低**（Top-10 僅 0.57%、Top-20 僅 0.89%），precision
  跟單一子模型差不多（約28%），但平均報酬遠低於單獨用 `BUY5_TB_MA20`（1.86-2.28%）
  或兩階段架構（2.46-2.85%）。**原因待查**——目前只是把子模型機率餵給 RF 做
  stacking，尚未確認是否選到「勉強達標但漲不多」的訊號組合，是後續優化重點。
- 特徵集：BUY 系列（TB_MA20/LOCAL 全部變體）統一使用完整長中短價格特徵 + Swing +
  籌碼原始特徵 + 狀態特徵 + 大盤原始特徵（`build_market_features.py`／
  `MARKET_PREFIXES`，2026-07-19 完成並接入）；
  C1~C3/F1/M1 各自用對應領域的精簡特徵集。詳細對應見
  `code/models/submodel_config.py: feature_cols()`。
- **兩階段 meta-labeling 架構**（`BUY5_LOCAL`/`BUY5_LOCAL_P5` 當 primary、
  `BUY5_TB_MA20` 當 secondary，模型檔 `models/META_BUY5_LOCAL*_MA20_RF.pkl`）目前
  只有實驗腳本（scratchpad 的 `train_meta_labeling.py`/`validate_meta.py`），
  **尚未整合進「完整版」的正式 pipeline**，是待辦事項——樣本外平均報酬比完整版
  現況好上數倍，值得優先評估是否要正式納入。

### 6.2 子模型清單

> 以下為 2026-07-19 最終定案的 10 個正式子模型，2026-07-25 新增第 11 個
> `BUY10_PERSIST7`（見下方）。fixed-horizon 基準版（BUY5/10/20）、
> triple-barrier 舊 MA10 版（BUY5_TB/10_TB/20_TB）、
> EARLY_RALLY、T123_MERGED、M2、M3、trend-scanning 全系列、以及這次 ground
> truth 實驗中沒選中的 LOCAL/TB 參數變體，皆已於同日從 `ALL_MODEL_IDS` 移除
> （驗證結果不如新版，或 lift < 1）。歷史比較數據見專案 memory。

| 模型ID | 預測目標 | Ground Truth（`build_labels.py` 實際定義） |
|--------|---------|-------------|
| BUY5_TB_MA20 / 10_TB_MA20 / 20_TB_MA20 | 買進後 N 天，triple-barrier 路徑判定版買點（★正式主力，門檻7%/12%/21%） | `label_buyN_tb_ma20`：triple-barrier，停損用 MA20 |
| BUY5_LOCAL / BUY5_LOCAL_P5 | 局部低點版買點（NPMM，5天版） | `label_buy5_local[_p5]`：t 為前後5天收盤最低點 + 5天漲幅≥7% |
| BUY10_PERSIST7 | 「持續漲」版買點（2026-07-25新增，10天7%門檻+過半天數上漲，見下方說明） | `label_buy10_persist7` |
| C1 | 未來 10 天三大法人累計買超為正 | `label_C1`：10 天後 (外資+投信+自營) 淨買賣超累計 > 0 |
| C2 | 未來 5 天外資淨買超為正 | `label_C2`：5 天後外資淨買賣超累計 > 0 |
| C3 | 未來 20 天外資買賣超動能為正（近似持股比例上升） | `label_C3`：20 天後外資淨買賣超累計 > 0 |
| F1 | 下期月營收 YoY 高於本期 YoY | `label_F1`：以公告日 asof 對齊，next_yoy > yoy |
| M1 | 大盤未來 5 天看多（訊號偏弱，lift≈1.08x，留著但非優先優化對象） | `label_M1`：全市場等權中位數報酬 5 天累計 > 0 |

**持續漲定義（`label_buy10_persist7`，2026-07-25 新增）**：買進後 10 天報酬
`ret_10 >= 7%`，且 10 天持有期中超過一半（>5天）是上漲日（收盤價逐日比較，
共10段漲跌，上漲天數 > 5 才算）。用意是排除「單日跳空達標、後續走平或拉回」
的假訊號，比 `BUY10_TB_MA20` 門檻更低（7% vs 12%）但額外要求持續性。scratchpad
實驗（feature_cols 沿用跟 BUY10_TB_MA20 相同的完整特徵集，RF + RandomizedSearchCV
調參）樣本外回測（2026-01~今，每日 Top-10）：10天報酬平均 +5.02%、中位數
+1.22%、7%達標率 37.3%，優於既有子模型與「完整版」Meta（詳見專案 memory：
ground truth 設計實驗）。另測試過「起漲點」（MA5>MA10>MA20多頭排列+斜率轉正，
未來3天確認）與兩者交集版本，皆驗證效果較差（起漲點單獨版中位數為負、交集版
正樣本率過低只剩0.18%），未採用。

**Triple-barrier 定義**（`_triple_barrier()`，de Prado 式，門檻與對應 BUY5/10/20 完全相同）：
從 t+2 起逐日檢查收盤價 —— 先觸及停利門檻（收盤 ≥ entry×(1+profit)）記 1；
停損採「連續兩天收盤都跌破 MA20」才觸發記 0（單日正常拉回不算，避免誤殺真訊號）；
horizon 天內都沒觸及任一 barrier（vertical barrier）記 0。停利不需連續確認，
停損需要連續確認 —— 兩者防呆不對稱是刻意設計。

**局部低點 NPMM 定義**（`_local_extrema_label()`）：t 當天收盤是前後 window 天內
的最低點（結構轉折條件），且買進後 window 天漲幅達門檻，兩者同時成立才算正樣本；
用意是把「這是不是正確進場點」跟「買了會不會賺」分開驗證。

### 6.3 Meta Ground Truth

> **2026-07-18 改版**：原本的「T123 AND 盤整 AND 不空頭 AND MA多頭排列」組合條件
> 已換成 ground truth 設計實驗驗證出樣本外表現最好的 triple-barrier 版本，跟
> `BUY5_TB_P10_MA20` 定義完全相同（停利門檻拉高到 10%，其餘同 §6.2 的
> triple-barrier 定義）。**這代表整個系統的持有策略從「買進抱 21 天」正式轉向
> 「5 天短打 + MA20 動態停損」**，§6.4 買賣價定義與 §8.1 回測規則已同步更新。
> `BUY5_TB_P10_MA20` 本身**不**作為子模型存在（跟 Meta target 完全相同會球員兼
> 裁判，已從子模型清單移除，只保留在 `label_meta` 這個定義裡）。

```
label_meta = 1  當且僅當（triple-barrier，horizon=5天）：
  進場後 5 天內收盤先觸及「收盤 ≥ entry×1.10」（停利 10%）→ 記 1
  若先觸及「連續兩天收盤跌破 MA20」（動態停損）→ 記 0
  5 天內兩者都沒觸及（vertical barrier）→ 記 0

label_meta = 0  其餘情況
```

- 正負比約 4.3%（`label_meta` 實測 pos_rate，取代舊版的 1.5%；因為定義完全改變，
  兩者不可直接比較，門檻搜索、precision 目標都要重新校準，不可沿用舊版數字）。
- **待辦**：專家審查（machine-learning-ops:data-scientist）指出這個「子模型層級
  贏家直接升級為 Meta 層級 target」的決定尚未在 Meta 層級驗證過（子模型輸入是
  原始特徵，Meta 輸入是子模型機率，是不同任務）；建議重新 fit Meta 頭 + 門檻
  （用現成的 `prob_meta_train/val/test.parquet`，不需重練子模型），比較新舊
  `label_meta` 的 **lift**（PR-AUC / 正樣本率，不可比較原始 PR-AUC，因為正樣本率
  變了）再決定是否正式採用。

### 6.4 買入價定義

> **2026-07-18 策略轉向**：原本「固定持有 21 天」的中線策略，已改為跟 Meta
> ground truth（§6.3）一致的「5 天短打 + MA20 動態停損」。這是刻意的產品定位
> 決定，不是純技術調整——系統從「中線持有」變成「短線波段、有紀律停損」。

```
買入價 = t+1 收盤價（entry）
出場條件（逐日檢查，先觸及哪個就出場，最長持有 5 個交易日）：
  1. 停利：收盤 ≥ 買入價 × 1.10 → 立即出場（單日觸及即算，不需連續確認）
  2. 停損：連續兩天收盤都跌破 MA20 → 出場（需連續確認，避免正常拉回被誤殺）
  3. 兩者都沒觸及，第 5 個交易日收盤強制出場（vertical barrier）
報酬率 = (出場價 - 買入價) / 買入價
```

- 舊版「t+1 日均價進場、t+21 日收盤出場」的邏輯已停用，`backtest.py` 尚未同步
  更新（見 §8.1 待辦）。

### 6.5 時間切割

| 資料集 | 時間範圍 | 用途 |
|--------|---------|------|
| 子模型 Train | 2022-01 ~ 2023-12 | 訓練各子模型（含 2022 空頭週期）|
| — | Gap：20 交易日 | — |
| Meta Train | 2024-01 ~ 2024-12 | 子模型對 2024 做預測（out-of-sample）→ Meta 訓練資料 |
| — | Gap：20 交易日 | — |
| Meta Val | 2025-01 ~ 2025-12 | Meta 驗證、機率門檻搜索 |
| — | Gap：20 交易日 | — |
| Meta Test | 2026-01 ~ 至今 | 最終評估（不用於任何調參）|

### 6.6 訓練流程

```
Step 1：子模型訓練
  - 在 2022~2023 訓練所有子模型（見 §6.1 清單）
  - 每個子模型：只用 RF（2026-07-18 起停用 LR/GBT），
    RandomizedSearchCV(n_iter=20) + TimeSeriesSplit(n_splits=3, gap=依horizon動態調整)
  - 輸出機率用 Isotonic Regression 校正（在 Meta Train 2024 上校正）

Step 2：DL 子模型（可選，暫緩）
  - PLAN 原規劃在 2022~2023 訓練 LSTM/CNN，比較 Meta Train 2024 PR-AUC 決定要不要納入
  - 2026-07-18 決定本輪不做，程式碼中目前也不存在（曾在某次清理中被移除，見 memory）

Step 3：Meta 訓練資料製備
  - 子模型對 2024 全年做預測（天然 out-of-sample）
  - 機率矩陣 shape：(2024年交易日 × 1700支) × 子模型數（只有 RF，不再乘上演算法數）

Step 4：Meta 模型訓練
  - 只用 RF（LightGBM 已停用，見 §6.7）
  - class_weight='balanced'

Step 5：機率門檻搜索
  - 在 Val 2025 上搜索最佳機率門檻（PR-AUC + F-beta β=0.5 引導）
  - 門檻決定後固定，Test 2026 只做最終評估
```

### 6.7 傳統 ML 演算法設定

> **2026-07-18 起只用 RandomForest**。LR 在既有比較中沒有展現優勢、GBT 從未
> 實際訓練出任何 `.pkl`，兩者已從 `train_submodels.py --algos` 預設值移除
> （`GBT_SPACE`/`LR_SPACE` 仍留在 `submodel_config.py` 供未來需要時手動指定）。

| 演算法 | 設定 |
|--------|------|
| RandomForest（唯一使用） | `class_weight='balanced'`, `n_jobs=-1` |
| ~~GradientBoosting~~ | 已停用，設定保留供未來參考：`sample_weight` 處理不平衡 |
| ~~LogisticRegression~~ | 已停用，設定保留供未來參考：`class_weight='balanced'`, saga solver, StandardScaler |
| 超參數搜索 | RandomizedSearchCV(n_iter=20) + TimeSeriesSplit(n_splits=3, gap=依horizon動態調整) |
| 機率校正 | Isotonic Regression（CalibratedClassifierCV(FrozenEstimator(clf))，sklearn≥1.6 寫法）|

### 6.8 DL 模型（暫緩）

> 2026-07-18 決定本輪不實作，以下規劃保留供之後參考。

**LSTM**（原規劃給短/中/長線動能子模型用，該套子模型已於 2026-07-12 移除）
```
Input (N, F) → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Linear(1) → Sigmoid
N=20（短線），N=60（中線），N=120（長線）
```

**1D-CNN**（原規劃給反彈/反轉子模型用，該套子模型已於 2026-07-12 移除）
```
Input (N, F) → Conv1d(64, k=3) → ReLU → Conv1d(128, k=3) → ReLU → AdaptiveMaxPool → Linear(1) → Sigmoid
N=20
```

訓練設定：BCEWithLogitsLoss(pos_weight)、Adam lr=1e-3、ReduceLROnPlateau、Early stopping（10 epoch）

### 6.9 評估指標

| 指標 | 說明 |
|------|------|
| PR-AUC | 主要指標，對不平衡資料最準確 |
| F-beta (β=0.5) | 精確率優先，寧少推不推錯 |
| 回測夏普比率 | 訊號是否有實際交易意義 |
| 最大回撤 | 風控角度 |

### 6.10 推薦名單產生

- 機率門檻在 Val 2025 搜索決定
- 超過門檻的股票納入推薦名單
- 依機率分數由高到低排序
- 資金分配由使用者自行決定

### 6.11 生存偏差
- Phase 1 不納入已下市股票
- Streamlit 介面標注「未納入已下市股票，回測結果存在生存偏差」

---

## 7. 每日自動化流程

### 7.1 Oracle VM Cron Job

```bash
# crontab（台灣時間 16:00 = UTC 08:00）
0 8 * * 1-5 /home/ubuntu/scripts/daily_update.sh
```

```bash
# daily_update.sh
cd /path/to/repo
python code/data_collection/fetch_price.py
python code/data_collection/fetch_chip.py
python code/data_collection/validate_data.py
git add data/price.parquet data/chip.parquet data/revenue.parquet data/fundamental.parquet data/stock_list.parquet
git commit -m "data: daily update $(date +%Y-%m-%d)"
git push origin main

# 月初才跑（每月 11 日以後）
if [ $(date +%d) -ge 11 ]; then
  python code/data_collection/fetch_revenue.py
fi
```

### 7.2 安全性

- `.gitignore` 排除：`.env`、`models/`、`init_progress.json`、`*.pyc`、`__pycache__/`
- Cron 腳本只 `git add` 指定的 parquet 檔，不使用 `git add -A`
- 敏感資訊（FINMIND_TOKEN 等）存 Oracle VM 的 `.env`，不進 git

### 7.3 Mac 端每日流程

```bash
# 盤後手動執行（或設定 launchd）
git pull origin main
python code/features/build_features.py
python code/models/predict.py
streamlit run streamlit_app.py
```

---

## 8. 回測系統

### 8.1 交易模擬規則

> **2026-07-18 改版，需與 §6.4 對齊**：策略從「固定持有 20 天」改為「5 天短打 +
> MA20 動態停損」，以下規則尚未在 `backtest.py` 程式碼中同步實作，是待辦事項。

- 買入價：t+1 收盤價（entry；舊版是 t+1 日均價，新策略改用收盤價跟 label 定義一致）
- 賣出條件（逐日檢查，最長 5 個交易日）：
  1. 停利：收盤 ≥ 買入價 × 1.10 → 單日觸及即出場
  2. 停損：連續兩天收盤跌破 MA20 → 出場
  3. 都沒觸及：第 5 天收盤強制出場
- 不納入交易成本（手續費/交易稅），在介面上標注
- 未納入已下市股票，介面上標注
- **待辦**：`backtest.py` 目前仍是舊版「固定 20 天 + 可配置停損門檻」邏輯，需要
  改成上述 triple-barrier 規則才能跟新版 `label_meta`/子模型的持有邏輯一致。

### 8.2 評估層次

**第一層：模型準確度**
- PR-AUC（主要）
- F-beta（β=0.5）
- Precision / Recall

**第二層：實際獲利**
- 每筆交易報酬率
- 勝率（賺錢次數 / 總次數）
- 平均報酬率
- 夏普比率
- 最大回撤
- 累計資產曲線

### 8.3 門檻搜索（只在 Val 2025 上）

> 停損門檻搜索已由固定百分比改為 §6.4/§8.1 的 MA20 動態停損機制，不再需要搜索
> 停損%數，只搜索機率門檻。

| 參數 | 範圍 | 顆粒度 |
|------|------|--------|
| 機率門檻 | 50% ~ 95% | 每隔 1% |

- 10 核心平行處理加速搜索
- 門檻固定後，Test 2026 只做最終驗證

---

## 9. 開發介面

### 9.1 Streamlit（Phase 1）

用於 debug 和結果呈現，包含：

| 頁面 | 功能 |
|------|------|
| 資料預覽 | 各 parquet 原始資料顯示（10~15 筆），支援日期/股票篩選 |
| 特徵預覽 | join 後的特徵矩陣，確認特徵值合理 |
| 模型訓練資料 | 各子模型的訓練/驗證集樣本與標籤分布 |
| 今日推薦 | 機率 > 門檻的股票清單，依分數排序 |
| 回測結果 | 累計資產曲線、每筆交易明細、各指標統計 |
| 模型成效 | 各子模型 PR-AUC、特徵重要性 Top 10 |

---

## 10. 開發路線圖

### Phase 1 實作順序

- [x] **Step 0**：建立目錄結構、.gitignore、requirements.txt（鎖版本）、.env.example
  - [x] 0-1：本機 Python 執行環境建立（Python 3.11.15 venv + `pip install -r requirements.txt`，含 Homebrew ta-lib/libomp）
- [ ] **Step 1**：pytest 框架建立，各模組先有測試檔
- [x] **Step 2**：歷史資料初始化
  - [x] 2-1：yfinance 日線價格（2022~至今）→ init_price.py
  - [x] 2-2：TWSE API 三大法人（2022~至今）→ init_chip.py（進行中）
  - [x] 2-3：TWSE API 融資融券（2022~至今）→ init_chip.py（進行中）
  - [x] 2-4：FinMind 月營收（2022~至今）→ init_revenue.py（進行中）
  - [x] 2-5：TWSE BWIBBU_d PER/PBR/殖利率（2022~至今）→ init_fundamental.py（進行中）
  - [x] 2-6：股票清單 → fetch_stock_list.py
  - [ ] 2-7：TPEX API 實機測試確認
- [ ] **Step 3**：Oracle VM 設定
  - [ ] 3-1：安裝環境、clone repo、設定 .env
  - [ ] 3-2：每日更新腳本（fetch_price/chip）
  - [ ] 3-3：cron job 設定
  - [ ] 3-4：驗證自動 commit/push
- [x] **Step 4**：特徵工程
  - [x] 4-1：均線、MACD、布林通道、KD、RSI 特徵 → build_price_features.py
  - [x] 4-2：價格動能、波動度、量價特徵 → build_price_features.py
  - [x] 4-3：ta-lib 全量指標 + 61 種 K 線型態 → build_talib_features.py
  - [x] 4-4：前波高低點、大量高低點、趨勢線 → build_swing_features.py
  - [x] 4-5：籌碼特徵（標準化）→ build_chip_features.py
  - [x] 4-6：基本面 / 營收特徵（Point-in-Time 對齊）→ build_fundamental_features.py
  - [x] 4-7：異常狀態 / 流動性特徵 → build_features.py
  - [x] 4-8：Streamlit 特徵預覽頁 → streamlit_app.py
- [x] **Step 5**：子模型訓練（10 個，見 §6.1/§6.2 最終清單，只用 RF）—— **全部訓練完成（2026-07-19）**：
  - [x] 5-1：建 Ground Truth label → build_labels.py（labels.parquet 已產出；探索用但未使用的欄位如 label_buy10/20_local、label_buyN_trend、label_T123 等仍留著供未來參考，docstring 已標註哪些【使用中】）
  - [x] 5-2：全部 10 個子模型已訓練（RF only）：C1、C2、C3、F1、M1、BUY5_LOCAL、BUY5_LOCAL_P5、BUY5_TB_MA20、BUY10_TB_MA20、BUY20_TB_MA20
  - [x] 5-3：Isotonic Regression 機率校正 → train_submodels.py（已隨 5-2 一併執行）
  - [ ] 5-4：DL 子模型 —— 本輪決定不做（2026-07-18），見 §6.6/6.8
  - [x] 5-5：C1/C2/C3 特徵前綴 bug 修正後已重新訓練，特徵集與目前 features.parquet 一致
  - [x] 5-6：LR、GBT、LightGBM 全面停用（2026-07-18/19 決定），`train_submodels.py`/`train_meta.py` 的演算法設定皆已改為只有 RF
  - [x] 5-7：EARLY_RALLY、T123_MERGED、M2、M3（lift<1，負貢獻）、BUY5/10/20 fixed-horizon 基準版、BUY5/10_TB 舊MA10版、以及這次 ground truth 實驗中沒選中的變體，全部已從 `ALL_MODEL_IDS` 移除並清除對應 `.pkl`
- [x] **Step 6**：Meta 模型（「完整版」）→ train_meta.py —— **已用最終子模型清單重跑（2026-07-19）**
  - [x] 6-1：子模型對 Meta Train 2024 做預測 → `data/prob_meta_{train,val,test}.parquet` 已用最終 10 個子模型重新產生
  - [x] 6-2：原規劃 RF vs LightGBM 比較，2026-07-19 因 lightgbm/sklearn 版本
    不相容（`force_all_finite` 參數已被 sklearn 移除）直接停用 LightGBM，Meta
    模型只用 RF
  - [x] 6-3：Val 2025 門檻搜索 → 最佳門檻 75%，precision=16.7%、recall=16.9%、F0.5=0.168
  - [ ] 6-4：Test 2026 最終評估 —— 尚未跑正式的 Test 2026 全年評估，只用 2026-06
    單月做過 Top-N 樣本外驗證（見 6-7）
  - [x] 6-5：純 stacking（只用子模型機率）vs 加原始特徵 的比較已做（2026-07-19），
    結果幾乎無差異（PR-AUC 0.1167 vs 0.1157），確認維持純 stacking 設計
  - [ ] 6-6：兩階段 meta-labeling 架構（BUY5_LOCAL 系列當 primary、BUY5_TB_MA20
    當 secondary）樣本外平均報酬（2.46-2.85%）遠優於「完整版」現況
    （0.57-0.89%，見 6-7），**優先待辦**：評估是否整合進正式 pipeline
  - [x] 6-7（新增）：用跟子模型一致的方法（2026-06 out-of-sample、每日 Top-N、
    統一5天≥7%門檻）驗證「完整版」，發現 **precision 尚可（約28%）但平均報酬明顯
    偏低**（Top-10僅0.57%、Top-20僅0.89%），比單獨用 BUY5_TB_MA20 或兩階段架構
    都差，原因待查（見 §6.1 紅字）——是目前最優先的問題
- [ ] **Step 7**：回測系統 + Streamlit 完整介面 → backtest.py + streamlit_app.py —— **程式碼邏輯尚未更新**：`backtest.py` 仍是「固定持有20天+可配置停損%」的舊邏輯，需要改成 §6.4/§8.1 的「5天+MA20動態停損」triple-barrier 規則，且依賴的 Step 6 產出也已過期，**需在 backtest.py 改版後才能視為有效結果**
- [ ] **Step 8**：整合每日自動流程，端對端測試

---

## 附錄：環境變數

```bash
# .env.example
FINMIND_TOKEN=your_token_here
```

---

*PLAN v1.0 | 依據 REVIEW.md 全部決議產出 | 2026-06-27*
