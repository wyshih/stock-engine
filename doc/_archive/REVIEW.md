# 設計審查記錄

> 五位專家 agent 審查 DESIGN.md 後產出的問題清單。  
> 每項討論後更新狀態與決議，最終依據決議改寫 DESIGN.md → 輸出 PLAN.md。
>
> **狀態**：⬜ 待討論 ／ ✅ 已決議 ／ ❌ 決定不改

---

## 問題總覽

### A. 核心架構

| ID | 嚴重 | 狀態 | 問題摘要 |
|----|------|------|---------|
| A1 | 高 | ✅ | 交易型資料（users/watchlist/prediction_log）不能存 CSV on GitHub |
| A2 | 高 | ✅ | Render 免費版即時跑 1800 檔預測 = 服務不可用 |
| A3 | 高 | ✅ | Render 512MB 載入幾十個模型 OOM（A2 解決後自動消失） |
| A4 | 中 | ✅ | 1800 個 CSV 效能差，建議改 Parquet |
| A5 | 中 | ✅ | 模型二進位進 Git 歷史，repo 無限膨脹 |

### D. 資料管線

| ID | 嚴重 | 狀態 | 問題摘要 |
|----|------|------|---------|
| D1 | 高 | ✅ | FinMind 初始化估 4 小時，實際需 13~20 小時 |
| D2 | 高 | ✅ | yfinance 在 Actions 共用 IP 易被 Yahoo Finance 封鎖 |
| D3 | 中 | ✅ | 資料撈取腳本模組化（使用者原始需求） |
| D4 | 中 | ✅ | GitHub Actions workflow 應拆成多個 yml |
| D5 | 中 | ✅ | CSV Append 非冪等（中途失敗重跑產生重複列） |
| D6 | 中 | ✅ | 假日判斷用「資料為空」太脆弱，需交易日曆 |

### M. ML 模型設計

| ID | 嚴重 | 狀態 | 問題摘要 |
|----|------|------|---------|
| M1 | 高 | ✅ | 週/月線 resample 包含未來資料（look-ahead bias） |
| M2 | 高 | ✅ | 前波高低點確認機制 train/inference 不一致 |
| M3 | 高 | ✅ | 基本面特徵未做 Point-in-Time（財報延遲未處理） |
| M4 | 高 | ✅ | Meta Ground Truth = T1∩T2∩T3，正樣本率只有 1~3% |
| M5 | 高 | ✅ | 訓練資料只有 2023~2024，沒有空頭週期 |
| M6 | 高 | ✅ | TimeSeriesSplit 缺少 Gap，label 洩漏到 Val |
| M7 | 高 | ✅ | Stacking 缺少 OOF 機制，Meta 訓練資料只有 6 個月 |
| M8 | 高 | ✅ | 評估指標應改用 PR-AUC + F-beta（而非 AUC-ROC） |
| M9 | 中 | ✅ | Meta 模型定義不一致（§6.1 RF vs §6.2.1 MLP） |
| M10 | 中 | ✅ | OBV 絕對值跨股票不可比，應改用變化率 |
| M11 | 中 | ✅ | 持股比例等跨股票可比性差，需自身歷史分位數標準化 |
| M12 | 中 | ✅ | DL 子模型（LSTM/CNN）缺少機率校正 |
| M13 | 中 | ✅ | LSTM DL 增益未驗證，建議先做 ablation study |
| M14 | 中 | ✅ | 買入價應改用 t+1 開盤價（而非收盤價） |
| M15 | 中 | ✅ | 訊號轉換邏輯缺失（每日推幾檔？資金分配？） |
| M16 | 中 | ✅ | 市場狀態（Market Regime）未顯式建模 |
| M17 | 低 | ✅ | 生存偏差：已下市股票不在訓練資料 |

### P. API 設計

| ID | 嚴重 | 狀態 | 問題摘要 |
|----|------|------|---------|
| P1 | 高 | ❌ | GET /recommendations/today 60 秒會 HTTP timeout |
| P2 | 高 | ❌ | POST /backtest/run 阻塞整個服務（Render 上估 30 分鐘） |
| P3 | 中 | ❌ | GET /stock/{id} 責任太多，應拆分為獨立端點 |
| P4 | 中 | ❌ | 缺少重要端點（/health、/auth/refresh、/backtest/status 等） |

### S. 資安

| ID | 嚴重 | 狀態 | 問題摘要 |
|----|------|------|---------|
| S1 | 高 | ❌ | /auth/register 完全公開，與「少數人共用」矛盾 |
| S2 | 高 | ❌ | JWT 永久登入缺 refresh token，洩漏後無法撤銷 |
| S3 | 高 | ✅ | joblib 反序列化 RCE 風險（模型檔被替換可執行任意程式） |
| S4 | 高 | ✅ | Actions 自動 commit 若用 git add -A 可能推送敏感檔 |
| S5 | 中 | ❌ | 缺少 rate limiting（登入暴力破解 + API 濫用） |
| S6 | 中 | ❌ | IDOR：watchlist 操作未驗證資源歸屬 |
| S7 | 中 | ❌ | CORS / 安全標頭未定義 |
| S8 | 中 | ❌ | stock_id 路徑未驗證，有路徑穿越風險 |
| S9 | 低 | ✅ | 依賴版本未鎖定，缺 pip-audit |

### F. 文件修正

| ID | 嚴重 | 狀態 | 問題摘要 |
|----|------|------|---------|
| F1 | 低 | ✅ | §2.2 寫「GitHub 無空間限制」是錯誤描述 |
| F2 | 低 | ✅ | 訓練資料時間範圍過時（現在 2026 年，切割需更新） |

---

## 問題詳細說明

### A1 — 交易型資料不能存 CSV on GitHub
**影響章節**：§4.1、§4.3、§11.1、§11.4

1. 安全：users.csv 含 email + bcrypt hash 進 git 歷史，私有 repo 也危險
2. 並發：兩個使用者同時寫 watchlist → push 衝突或覆蓋
3. 操作：prediction_log 需要 UPDATE（回填），CSV 整檔重寫越來越慢
4. 矛盾：§11.4 已提 Alembic，但 §4.1 說「所有資料存 CSV」

**建議**：users、watchlist、prediction_log 改用免費 Postgres（Supabase 或 Neon free tier）。

---

### A2 — Render 即時跑全市場預測 = 服務不可用
**影響章節**：§7.4、§11.1、§11.3

三層疊加：冷啟動 30-60s + 載入模型 OOM 風險 + 1800 檔重算數分鐘。

**建議**：
- 全市場預測移到每日 GitHub Actions 批次（runner 2-core/7GB）
- 結果持久化（寫 Postgres 或 commit predictions parquet）
- FastAPI 只讀，幾乎零延遲
- 盤中刷新單一股票保留即時計算

---

### D1 — FinMind 初始化時間低估
**影響章節**：§3.2

1800 檔 x 6 種資料 = 10,800+ 次請求，1,500 次/小時 → 最快 13-20 小時，不是 4 小時。

**建議**：加 checkpoint 機制（init_progress.json），可中斷續跑；優先用市場批次端點。

---

### D2 — yfinance Actions IP 被封
**影響章節**：§3.1、§3.3

**建議**：fallback 改用 TWSE OpenAPI（官方端點）；鎖定 yfinance 版本。

---

### D3 — 資料撈取腳本模組化
**影響章節**：§3.3、§4.2、§14.5

建議腳本結構：
```
code/data_collection/
  fetch_price.py       # yfinance + TWSE fallback
  fetch_chip.py        # 三大法人 + 融資融券
  fetch_fundamental.py # PER/PBR/殖利率
  fetch_revenue.py     # 月營收
  fetch_stock_status.py
  validate_data.py
  backfill_predictions.py
  init/
    init_price.py      # 歷史初始化（支援 checkpoint）
    init_progress.json # checkpoint（.gitignore）
```
統一 CLI 介面：python fetch_chip.py --date 2026-06-26 --dry-run
退出碼：0=成功，1=部分失敗，2=整體失敗
冪等性：寫入前查 date 是否存在

---

### D4 — 多個獨立 workflow yml
**影響章節**：§7.1、§7.2

```
.github/workflows/
  daily_data_fetch.yml   # 主排程 UTC 09:30，並行各 fetch job
  fetch_price.yml        # 可獨立手動觸發
  fetch_chip.yml
  fetch_fundamental.yml
  fetch_revenue.yml      # 月初才執行
  fetch_stock_status.yml
  daily_predict.yml      # 資料成功後觸發批次預測
  backfill_predictions.yml
  init_data.yml          # workflow_dispatch，歷史初始化
```
各 fetch job 加 continue-on-error: true，互不影響。

---

### D5 — CSV Append 冪等性
**影響章節**：§3.3、§7.3

Append 前先 drop_duplicates(subset=['date', 'stock_id'])。

---

### D6 — 假日判斷需交易日曆
**影響章節**：§7.1

維護 trading_calendar.csv，「該有資料卻抓到空」→ 告警 + 重試，不是跳過。

---

### M1 — 週/月線 resample look-ahead bias
**影響章節**：§5.1、§5.2

週三 resample 本週 → 週 K 收盤 = 週五（未來資料）。
建議：weekly = daily.resample('W-FRI').last().shift(1)

---

### M2 — 前波高低點確認機制不一致
**影響章節**：§5.4

訓練時用後 N 天確認，推論當下這 N 天還沒發生。
建議：「在 t-N 日之前，後面 N 天都未超越的點才算確認」，距今計算起點改為確認日。

---

### M3 — 基本面 Point-in-Time
**影響章節**：§5.8、§3.5

月營收次月 10 日公告，季報延遲 45 天，直接用「最新期」= look-ahead bias。
建議：as_of_date 機制，只使用公告日前資料。

---

### M4 — Meta Ground Truth 極端不平衡
**影響章節**：§6.3、§6.6

T1∩T2∩T3 正樣本率 1-3%，模型幾乎永遠預測負類。
選項：(1) T1 OR (T2 AND T3)；(2) Learning to Rank；(3) 維持交集 + SMOTE

---

### M5 — 訓練資料時間範圍不足
**影響章節**：§6.7

2023-2024 = 台股大多頭，沒有空頭週期。現在是 2026 年切割也需更新。
建議：Train: 2015-2023，Val: 2024，Test: 2025

---

### M6 — TimeSeriesSplit 缺少 Gap
**影響章節**：§6.2

T3 預測 60 天後，Train 最後 60 筆 label 與 Val 前 60 筆共享同段未來股價。
建議：TimeSeriesSplit(n_splits=5, gap=N)，T1: gap=5，T2: gap=20，T3: gap=60

---

### M7 — Stacking 需 OOF 機制
**影響章節**：§6.7

子模型對自身訓練資料機率偏高，Meta 學到假關係。Val 6 個月的 Meta 訓練資料也極少。
建議：Walk-forward OOF，把所有 OOF 預測拼接為 Meta 訓練資料。

---

### M8 — 評估指標
**影響章節**：§9.1、§6.2

建議：PR-AUC、F-beta（beta<1）、回測夏普比率、最大回撤。

---

### M9 — Meta 模型定義不一致
**影響章節**：§6.1、§6.2.1

§6.1 第 88 行寫 RandomForest，§6.2.1 表格寫 MLP。需統一。

---

### S1 — 關閉公開註冊
**影響章節**：§1.4、§11.3

建議：移除 /auth/register，管理者 seed script 建帳號（最簡單）。

---

### S2 — JWT 雙 token 模式
**影響章節**：§1.4、§11.2

Access token: 15-60 分鐘；Refresh token: 30 天，存 DB，存 HttpOnly cookie。
登出/改密碼 → 刪 DB 中 refresh token → 立即生效。

---

### S3 — joblib RCE 風險
**影響章節**：§4.1、§11.4

短期：記錄 SHA-256，載入前比對。長期：改用 ONNX 或 safetensors。

---

### S4 — Actions commit 白名單
**影響章節**：§7.2

只白名單 commit：git add data/price/ data/chip/ data/revenue/ data/fundamental/
絕不用 git add -A。

---

### P1 — /recommendations/today timeout
**影響章節**：§11.3

A2 解決後（預測移到 Actions）此問題自動消失。若維持即時計算，改非同步 + task_id 輪詢。

---

### P2 — /backtest/run 阻塞服務
**影響章節**：§11.3、§9.3

改為非同步：POST /backtest/run 立刻回傳 task_id，提供 GET /backtest/{task_id}/status 和 result。

---

### P3 — GET /stock/{id} 拆分
**影響章節**：§11.3

拆為 /prediction、/chart?period=D|W|M、/chip、/revenue。

---

### P4 — 缺少的重要端點
**影響章節**：§11.3

GET /health、POST /auth/refresh、POST /auth/logout、
GET /recommendations/status、GET /stock/{id}/chart 等（詳見問題總覽）。

---

### S5 — Rate limiting
**影響章節**：§11.3

登入失敗 N 次鎖定（slowapi）；/stock/{id}/refresh 每使用者 5 分鐘冷卻；backtest 同時只能一個任務。

---

### S6 — IDOR watchlist
**影響章節**：§11.3

所有 watchlist 操作用 token 中的 user_id 做 WHERE 條件。

---


### 特徵補充決議（2026-06-27）

| 日期 | 說明 |
|------|------|
| 2026-06-27 | ta-lib 全量指標：CCI、MFI、Williams %R、AROON Up/Down/OSC、TRIX、Ultimate Oscillator、StochRSI、PPO 等所有 ta-lib 動量/趨勢/成交量指標全部納入，已是相對值者直接保留，絕對值者除以 close 標準化。原則：運算夠快的指標全部加入，讓模型自動學習特徵重要性。 |
| 2026-06-27 | 持續天數編碼：所有 streak 特徵同時產生四個累積布林值（streak_1d= days>=1, streak_3d= days>=3, streak_5d= days>=5, streak_10d= days>=10）加上 log(days+1) 連續值，共 5 個維度。 |
| 2026-06-27 | K 線型態：使用 ta-lib 提供的全部 61 種 candlestick pattern（CDL* 系列），輸出 +100/0/-100，直接保留不需標準化。 |
| 2026-06-27 | 均線排列補充：新增四線多排/空排（MA5>MA10>MA20>MA60）、五線多排/空排（+MA120），及各持續天數 log(days+1)。 |
## 決議記錄

| 日期 | ID | 決議摘要 |
|------|----|---------|
| 2026-06-26 | A1 | Phase 1 只有一個使用者，不需要 Postgres。prediction_log 用本機 SQLite，users/watchlist 暫不需要 |
| 2026-06-26 | A2 | 預測全在本機 Mac 跑（M5 10 核），不需要 Render |
| 2026-06-26 | A3 | 自動消失（不用 Render） |
| 2026-06-26 | D4 | 改用 Oracle Cloud Free VM cron job，不用 GitHub Actions workflow |
| 2026-06-26 | P1-P4 | Phase 1 無 FastAPI/React，暫不適用。日後多人使用時再加 |
| 2026-06-26 | S1,S2,S5-S8 | Phase 1 無 web service，暫不適用 |
| 2026-06-26 | 整體架構 | Oracle VM 每天抓資料 → git push private GitHub repo → Mac git pull → 本機特徵工程 + 預測 → Streamlit 看結果 |
| 2026-06-26 | D1 | 股價用 yfinance（auto_adjust 自動處理除權息還原價）。三大法人／融資券改用 TWSE 官方 API（免費無限制，歷史到 2015 已驗證，一次請求拿全市場）。月營收／基本面季報用 FinMind（初始化約 4.5 小時可接受）。大戶散戶比 Phase 1 略過，預留模組位置。資料範圍限普通股，排除 ETF 與權證。 |
| 2026-06-26 | D2 | Oracle VM 固定 IP 每日增量拉取，yfinance 被封風險大幅降低。籌碼資料（三大法人／融資券）改用 TWSE 官方 API，完全脫離 yfinance／FinMind 依賴。TPEX 上櫃官方 API 待 Oracle VM 設好後實機測試確認。 |
| 2026-06-26 | D3 | 腳本分為 daily（fetch_price/chip/revenue/fundamental/stock_list）與 init（歷史初始化，含 checkpoint）兩層，預留 realtime/ 目錄給 Phase 2。錯誤處理：網路逾時/rate limit → 指數退避重試 3 次；資料為空或解析錯誤 → log 後跳過。每支腳本支援 --date 參數供手動補跑。 |
| 2026-06-27 | D5 | Append 前 drop_duplicates(subset=['date', 'stock_id'])，保證冪等。 |
| 2026-06-27 | D6 | 用 exchange_calendars（XTAI）為底，每年年初自動同步 TWSE 官方休市公告。執行時若全市場 >95% 資料為空則自動判定突發休市並記錄，次日自動補跑前日缺口，無需人工介入。 |
| 2026-06-27 | A4 | 改用 Parquet，每種資料一個檔案（price/chip/revenue/fundamental/stock_list.parquet），date+stock_id 為複合 key。 |
| 2026-06-27 | A5 | models/ 加入 .gitignore，不進 git，本機 Mac 保存。 |
| 2026-06-27 | M1 | 原設計已正確使用上週/上月完整 K 線，不適用。 |
| 2026-06-27 | M2 | 原設計已只往過去看確認高低點，不適用。 |
| 2026-06-27 | M3 | 基本面特徵以公告日為準 forward-fill（月營收用次月10日後，季報用季末+45天後），不使用尚未公告的數字。 |
| 2026-06-27 | M4 | Meta Ground Truth = MA5>MA10>MA20 條件下，20 交易日後報酬>15% 為正例。正負比約 1:12，訓練時加 class weight。N=20 與 15% 門檻為超參數，backtest 後可調整。 |
| 2026-06-27 | M5 | 子模型 Train 2022~2023（含 2022 空頭週期）。Meta Train 2024，Meta Val 2025，Meta Test 2026~至今。各期銜接留 20 交易日 Gap。 |
| 2026-06-27 | M6 | Train/Val 與 Val/Test 之間各留 20 交易日 Gap，避免 label（20 交易日後報酬）洩漏到下一段。 |
| 2026-06-27 | M7 | 子模型在 2022~2023 訓練後，對 2024/2025/2026 分別做預測，這些預測對子模型天然是 out-of-sample，直接作為 Meta 各期輸入，不需要 within-period OOF fold 機制。 |
| 2026-06-27 | M8 | 評估指標：PR-AUC（主指標）、F-beta β=0.5（精確率優先）、回測夏普比率、最大回撤，四項全部納入。 |
| 2026-06-27 | M9 | Meta 模型候選：RandomForest 與 LightGBM，在 Val 2025 以 PR-AUC 比較，擇優作為最終 Meta。MLP 移除。 |
| 2026-06-27 | M10 | OBV 改為 N 日 pct_change。所有絕對值特徵均需標準化（見 M11）。 |
| 2026-06-27 | F1 | 改為「GitHub repo 建議保持在 5GB 以下，parquet 資料量每天約 200KB，10 年也僅約 500MB，空間足夠」。 |
| 2026-06-27 | F2 | 時間切割已更新：子模型 Train 2022~2023，Meta Train 2024，Meta Val 2025，Meta Test 2026~至今。 |
| 2026-06-27 | S9 | requirements.txt 所有套件鎖定版本號，定期跑 pip-audit 檢查已知漏洞。 |
| 2026-06-27 | S4 | 雙重保護：.gitignore 排除 .env/models//init_progress.json 等，Oracle VM cron 的 commit 腳本只 git add 指定 parquet 檔（data/price.parquet data/chip.parquet data/revenue.parquet data/fundamental.parquet data/stock_list.parquet），不用 git add -A。 |
| 2026-06-27 | S3 | Phase 1 模型只存本機 Mac，無外部存取風險，暫不需要 SHA-256 驗證。 |
| 2026-06-27 | M17 | Phase 1 不納入已下市股票，接受生存偏差，在 Streamlit 介面標注說明。 |
| 2026-06-27 | M16 | 大盤模型（M1/M2/M3）輸出機率直接納入 Meta 輸入，Meta 共 13 個輸入（T1-T6, C1-C3, F1, M1-M3）。原警示邏輯移除，由 Meta 自動學習市場環境下的權重調整。 |
| 2026-06-27 | M15 | 機率門檻在 Val 2025 上搜索決定，超過門檻的股票納入推薦名單，依機率分數由高到低排序。資金分配由使用者自行決定，不在系統範圍內。 |
| 2026-06-27 | M14 | 買入價改用 t+1 日均價（成交金額 / 成交量），比收盤價或開盤價更貼近實際隨機時間買入的情境。報酬率 = (t+21日收盤 - t+1均價) / t+1均價。 |
| 2026-06-27 | M13 | LSTM/CNN 定位為可選額外成員模型。傳統 ML（RF/GBT/LR）為核心，DL 模型後續加入時比較 Val 2025 PR-AUC 是否提升，有提升才納入 Meta。 |
| 2026-06-27 | M12 | DL 子模型（LSTM/CNN）在 Val 2024 上做 Isotonic Regression 機率校正，與傳統 ML 一致。 |
| 2026-06-27 | M11 | 全特徵標準化規則：(1)價格相關→除以 close 或用比率；(2)天數→log(days+1)；(3)已是%/0-100→保留。各群決議如下：【均線】close/MA_n、MA間比率、rolling z-score、above_ma布林、多排布林、多排天數log、MA20斜率/close、均線糾結程度。【MACD】DIF/MACD/Histogram 各除以 close、DIF>0布林、DIF pct_change、DIF 5日斜率/close、交叉布林+天數log、Histogram擴縮布林+天數log、Histogram連續3根同向布林、頂/底背離布林。【布林通道】%B、%B 5日變化率、通道寬度歷史分位、突破布林、擴縮布林+天數log（ATR/close 與通道寬度高度共線，保留ATR移除通道寬度絕對值）。【KD】K/D值、(K-D)/100、K值3日變化速度、交叉布林+天數log、超買超賣布林、超賣反彈布林。【RSI】RSI14、RSI7-RSI21差值、RSI斜率5日、超買超賣布林、反彈布林、超買/超賣持續天數log。【價格動能】1/5/20/60日報酬率、距60/240日高低點%、創新高布林、動能一致性布林、個股相對強弱（N日報酬-大盤N日報酬，5/20/60日）。【波動度】ATR14/close、ATR/close相對60日均值比率、20日std/close、波動率非對稱性（漲日std/跌日std）。【量價】OBV N日pct_change、量比（今日/5日均量）、量能確認率（漲日均量/跌日均量20日）、CMF、放量突破布林。【前波高低點】距離%、天數log、高低點布林、轉折點成交量強度。【大量高低點】距離%、天數log、成交量強度、突破布林、突破後持續天數log。【趨勢線】距支撐/壓力%、斜率/close、R²、觸及次數、持續天數log、方向一致性布林。【籌碼】三大法人買賣超/總成交量%、5/10日累計（10日-5日分離信息）、持股比例%、融資融券餘額/流通股數%、融資5日斜率/close、外資+投信聯合買超布林、借券賣出/流通股數%、外資連續買超天數log。【基本面/營收】PER/PBR/殖利率、歷史分位數、月營收YoY/MoM、累計YoY、EPS YoY、毛利率趨勢斜率、營收加速度（本月YoY-上月YoY）、連續正成長月數log。【流動性】is_full_cash/disposed/warning、漲跌停頻率/20日、流動性不足標記布林、日均量→avg_vol_20/avg_vol_60、日均成交金額→avg_turnover_20/market_cap。 |

