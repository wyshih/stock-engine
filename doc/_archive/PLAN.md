# stock_committee — 單一 label 多模型架構（2026-08-04）

> 這份文件是唯一需求來源。委員會時代的脈絡備份在 `doc/PLAN_committee_archive.md`，
> 更早的在 `doc/PLAN_v1_archive.md`，兩份都不要動。
>
> 完整的實驗設計（切分理由、六個模型的 Optuna 搜尋空間逐項數值）在
> `~/.claude/plans/swirling-meandering-hearth.md`。本文件是它的執行狀態面板。

分支 `exp/drop-committee`（worktree `stock_committee_norf`，與主 repo
`stock_committee` 共用同一份 `data/`）。

---

## 1. 架構決策：拆掉委員會

**舊架構**：13 個子模型 + Meta stacking。

**拆掉的證據**：在相同 label、切分與出場規則下做頭對頭比較，委員會遠差於單一 RF，
也差於隨機對照。且把 RF / MLP / LR / LSTM 的預測做排序平均，AUC **單調變差**
（RF 單獨 0.6371 → 四個全上 0.6104）—— 模型間看似很低的重疊（top-1% 只有 5~17%）
反映的是弱成員在擬合雜訊，不是互補訊號。

**新架構**：單一 ground truth `label_up20`（未來 20 交易日上漲天數 ≥ 10，
持平不算上漲，base rate ≈ 37.6%）+ 多種模型類型，比較誰勝出。

三個實驗軸：**① 模型類型**（本輪在做）、② 出場規則、③ label 重新設計。
軸 2、3 待 Round 1 結果出來後另開討論。

---

## 2. 資料

| 檔案 | 來源 | 腳本 |
|---|---|---|
| `price.parquet` | yfinance（含大盤 TWII） | `fetch_price.py` |
| `chip.parquet` | TWSE 官方 API | `fetch_chip.py` |
| `fundamental.parquet` | TWSE BWIBBU_d（僅上市） | `fetch_fundamental.py` |
| `revenue.parquet` | MOPS 月營收 | `fetch_revenue.py` |
| `stock_list.parquet` | 股票清單 | `fetch_stock_list.py` |

可用區間 2019-01-02 ~ 今，3,411,592 列。**2019 只能當特徵暖機**
（NaN 19.63%，46 個欄位近乎全空），實際可用 2020-01 起。

### 2026-08-04 資料修復：OHLC 不變式

7,009 列（0.205%）的 `close`／`open` 落在當日 `[low, high]` 之外，偏離中位數 2.1%、
最大 21.9%。**根因在 Yahoo Finance 上游**：98.4% 集中在上櫃（違反率是上市的 75 倍），
單獨重抓仍重現，`auto_adjust=False` 也一樣。

- 修法：`code/data_collection/repair_ohlc.py` 擴張高低價把開收盤包進去
  （`close` 是最受信任、驅動報酬計算的欄位，不能動）
- 抓取端已內建同一個修復（`fetch_price.py`），`validate_data.py` 加了硬性檢查
- 影響：修復前 v1 的 `cmf` 有 6,998 列超出定義域 [-1,1]、`k_value` 1,644 列超出
  [0,100]（KD 是遞迴平滑，單一髒列污染後續數十個交易日）

---

## 3. 特徵

```
build_price_features → build_chip_features → build_fundamental_features
→ build_talib_features → build_swing_features → build_market_features
→ build_trendline_features → build_relative_features → build_features（合併，最後跑）
```

**381 欄**（原 395 − shape 14）。增量模式是預設，全量重算要加 `--full`。

**shape 已拔除**：`build_shape_features.py` 的 KMeans 群心用 `FIT_CUTOFF="2023-12-31"`
以前的資料 fit，而新切分的測試期在 2022/2023，那 14 欄是分布層級的洩漏來源。

特徵一律經 `submodel_config.feature_cols()` 取用，不可直接用 `features.parquet`
的全部欄位。

---

## 4. 切分（Part B）

Embargo 一律一個月（≥20 交易日 = 最長 horizon），避免 label 視窗跨界。

```
Round 1（篩選）  train  2020-01-01 ~ 2020-11-30    437k 列
                 val    2021-01-01 ~ 2021-11-30
                   val_es  2021-01-01 ~ 2021-08-31   ← early stopping + Optuna 目標
                   val_sel 2021-09-01 ~ 2021-11-30   ← 跨模型排名 + 挑門檻
                 test   2022（空頭年 −22%）
                 test2  2023（多頭年 +27%）

Round 2（決賽）  train  2020-01-01 ~ 2022-11-30    1.33M 列
                 val    2023-01-01 ~ 2023-11-30（同樣 2/3 + 1/3 切）
                 test   2024 / test2 2025 / test3 2026-01~07
```

**val 切兩段的理由**：LightGBM / LambdaRank / MLP / LSTM 都要用 val 做 early
stopping，會偷看 val 幾百次；RF / ExtraTrees 不用。不切的話 val 分數虛高，
六種模型站不到同一條線上（fold3 就發生過 val 排名與 test 幾乎顛倒）。

**Round 1 兩個獨立測試年**是這個設計的價值所在：一空一多，兩種制度下都排前面
才是真的好。

已知限制：Round 1 訓練期只有 2020 一年且只有 Round 2 的 1/3 大小；Round 2 的 val
（2023）就是 Round 1 的 test2，所以 Round 2 的 val 分數偏樂觀（但 test 2024~2026
完全乾淨，最終結論不受影響，報告時要寫明）。

---

## 5. 六個模型（Part C）

RF / ExtraTrees / LightGBM / LambdaRank / MLP / LSTM。
**已放棄**：Logistic Regression（使用者決定不跑）、等權重集成（實測 RF+MLP+LR+LSTM
排序平均讓 AUC 從 0.6371 單調掉到 0.6104；若 Round 2 出現兩個個別強度相近的模型，
再考慮加權組合，不做等權平均）。

### 搜尋空間（逐項與使用者確認過）

```python
# RF — 96 種，GridSampler 全掃
max_features     ∈ {"sqrt"(=19), 10, 30, 40}   # 絕對欄數；舊版固定 sqrt 從沒搜過
min_samples_leaf ∈ {50, 150, 200, 300}
max_depth        ∈ {10, 14, 20}
class_weight     ∈ {None, "balanced"}
n_estimators     = 300

# ExtraTrees — 同上 96 種，bootstrap=False。低成本對照組

# LightGBM — 432 種，TPESampler 80 trials
num_leaves        ∈ {15, 31, 63, 127}
min_child_samples ∈ {50, 150, 200, 300}        # 與 RF 同組，差異純來自演算法
learning_rate     ∈ {0.01, 0.03, 0.1}
feature_fraction  ∈ {0.1, 0.3, 0.6}
lambda_l2         ∈ {1, 10, 100}
bagging_fraction  = 0.7
n_estimators      = early stopping on val_es（上限 3000）

# LambdaRank — LightGBM 空間 × truncation，TPESampler 80 trials
objective                   = "lambdarank"
group                       = 每個交易日（全市場）
relevance                   = 二元 label_up20（與其他模型同一 ground truth）
lambdarank_truncation_level ∈ {10, 30, 100}
# 輸出是分數不是機率 → 門檻走每日分位數，不用絕對值

# MLP — 72 種，GridSampler 全掃
weight_decay ∈ {1e-4, 1e-3, 1e-2, 1e-1}        # 舊 grid 幾乎只有 1e-4，是主旋鈕
hidden       ∈ {(256,64), (128,)}
dropout      ∈ {0.1, 0.3, 0.5}
lr           ∈ {1e-5, 1e-4, 1e-3}
BATCH=8192, BatchNorm 保留
MAX_EPOCHS: 40 → 120   # lr=1e-5 配 40 epoch 只有約 7200 步，會被時間上限誤殺
PATIENCE=5 (on val_es)

# LSTM — 54 種，GridSampler 全掃
輸入      = 381 欄（沿用現有 build_seq_index 快取機制）
SEQ_LEN   ∈ {10, 20, 40}                       # 需新建 seq_index_10 / seq_index_40
hidden    ∈ {64, 128, 256}
layers    ∈ {1, 2}
dropout   ∈ {0.1, 0.2, 0.4}
```

- Optuna 目標 = **val_es AUC**（統一，樹模型也用同一段）
- 跨模型排名 = **val_sel AUC** + Round 1 兩個 test 年
- Round 1 跑完輸出完整表格（每個組態 × val_sel / test 2022 / test2 2023 的
  AUC 與 PR-AUC lift），**由使用者手動挑存活組態**，不設自動規則

前處理：訓練期統計量補值 + 標準化 + ±10σ winsorize。

---

## 6. Part D：特徵獨立雙實作（v2）

**動機**：上次人工審核抓到 26 個計算錯誤，且**都不會報錯**。獨立雙實作是抓這類
靜默錯誤的標準方法。

**做法**：從 v1 抽出最小規格（`doc/FEATURE_SPEC_V2.md`，只寫輸入／輸出／一句話
定義，不附公式），由另一個 agent 從零實作，**全程不得讀 `code/features/build_*.py`**。

**比對方式**（`code/features_v2/compare_v1_v2.py`）：

1. **逐欄診斷（主要產出）**：Spearman 排序相關 < 0.99、NaN 位置不一致、
   標準化後最大絕對差 —— 三項任一踩線就列為可疑，交人工裁決
2. **AUC + 回測（結案指標）**：兩版各跑相同組態的 RF，比 Round 1 的
   test 2022 / test2 2023，並各跑一次回測（配隨機對照）

**只比 AUC 不夠**：上次修 26 個確實存在的錯誤，test AUC 只從 0.6392 動到 0.6371。

v2 產出在 `data_v2/`（8 個特徵檔 + 合併的 `features.parquet` 371 欄），
解讀紀錄與裁決在 `doc/FEATURE_V2_DECISIONS.md`，分歧清單在 `doc/FEATURE_V2_DIFF.md`。

---

## 7. 執行狀態（2026-08-04）

### Part A 拆委員會

| 項目 | 狀態 |
|---|---|
| 刪 `train_meta` / `two_stage` / `train_two_stage` / `tune_primary_threshold` | ✅ |
| `backtest.py` 改讀單一模型分數、`meta_prob` → `score` | ✅ |
| v2 訓練表排除 shape | ✅ |
| `predict.py`（10 處 meta 耦合） | ❌ |
| `streamlit_app.py`（23 處） | ❌ |
| `submodel_config.py` 砍 13-key 對照表（保留 `feature_cols()` 與搜尋空間） | ❌ |
| `build_labels.py` 只留 `label_up20`（**5 個防洩漏原子工具全部保留**，軸 3 會用） | ✅ 338→292 行，5 個工具 diff 為零，pytest 7 passed；⏸ `labels.parquet` 待重算清掉舊欄位 |
| 抽出 `code/paths.py` | ❌ |
| v1 `build_features.py` 的 merge 移除 shape 14 欄 | ✅ 程式已改；⏸ `features.parquet` 待全量重建（394→380 欄） |

### Part D

| 項目 | 狀態 |
|---|---|
| v2 八個特徵檔 + 合併訓練表 | ✅ 全部符合規格、欄名與 v1 逐檔一致 |
| v2 單元測試 28 項（跨股票邊界／前視偏誤／數值穩定性） | ✅ |
| `labels.parquet` 全量重建（修好的資料上） | ✅ `label_up20` base rate 0.376，與計畫一致 |
| v1 特徵全量重建（修好的資料上） | ✅ `cmf`／`k_value` 越界歸零 |
| 逐欄診斷（Spearman／NaN 位置／標準化最大差） | ✅ 174/281 欄列為可疑，抽查主因是已裁決的定義差異 |
| **Part D 結案：兩版 AUC + 回測打平** | ✅ 見 `doc/BACKTEST_LOG.md` #22 |
| 訓練與評估 harness（`code/models/train_single.py`） | ✅ Round 1 切分 + 固定 RF 組態 + 分數檔輸出 |
| **v2 RF**（345 欄，`feature_cols("UP20")` 選出） | ✅ 見下表 |
| 逐欄診斷 | ⏸ 待 v1 |
| v1 RF：AUC + 兩版回測 | ⏸ 待 v1 |
| 分歧人工裁決 | ⏸ 待 AUC 與回測 |

**Round 1 RF 結果**（固定組態 `n_estimators=300, max_depth=14,
min_samples_leaf=150, max_features="sqrt", class_weight=None`）：

**兩版取欄位交集 344 欄**（v1 的 `features.parquet` 另有 6 個 `rank_*`、
`limit_hit_rate`、`margin_slope_5d` 不在 v2 規格內，不取交集就不只是比實作差異）：

| 版本 | val_es 2021H1 | val_sel 2021Q4 | test 2022（空頭） | test2 2023（多頭） |
|---|---|---|---|---|
| v1 | 0.5778 | 0.5914 | 0.6299 | 0.5870 |
| v2 | 0.5785 | 0.5955 | 0.6274 | 0.5877 |

差異全在 ±0.004 內。回測同樣打平（alpha 中位差 0.1pp），詳見 `BACKTEST_LOG.md` #22。

**Part D 的核心教訓**：兩版有數十欄定義不同，AUC 只動 0.004 ——
**特徵正確性必須靠逐欄比對，不能靠模型指標**。

### Part C Round 1 進度

| 模型 | 組態數 | 狀態 |
|---|---|---|
| RF | 96（GridSampler 全掃） | ✅ 單組平均 194 秒，總計 5.2 小時 |
| ExtraTrees | 96 | ✅ 單組 103 秒（2 trial 並行），**三段全面輸 RF** |
| LightGBM | 80（TPE） | ✅ **目前最佳** |
| LambdaRank | 80（TPE） | ✅ |
| MLP | 72 | ✅ 0.5990，但 `weight_decay` 完全沒作用（見下） |
| LSTM | 54 | ⏸ **中斷於 7/54**（2026-08-05 關機），最好 val_sel 0.5734 |

**MLP 的搜尋等於白搜**：前四名的 `weight_decay` 從 1e-4 到 1e-1 差一千倍，
AUC 卻完全相同到小數第四位（連 epochs 都同樣是 7）。AdamW 的解耦權重衰減每步
乘以 `(1 − lr × wd)`，在 lr=1e-4、只跑 7 個 epoch（約 336 步）下，wd=0.1 也只讓
權重縮 0.3%。計畫把它列為「主旋鈕」，但在這個 lr 與 early stopping 組合下轉不動。
要重搜得改成更大的 lr × wd 組合，或加大 patience。

**LSTM 重啟注意**：Optuna study 沒有持久化（沒設 storage），重跑會從 0 開始。
已完成的 7 組留在 `data/sweep_round1_lstm.csv`。單組平均 383 秒（比 MLP 慢 9 倍），
全部跑完約 5.7 小時。目前看起來會是最弱的一個（test2 2023 只有 0.5422）。

### 交易門檻（2026-08-05 補上，先前做錯）

**每次重新訓練都必須重調門檻**，方法是 `code/models/threshold_curve.py`：
驗證期預測由高到低排序，**逐一以每個預測機率當門檻**算累積勝率與報酬，畫成曲線，
**由人看圖挑點**。不是「取前幾 %」，也不是自動規則
（單點最高必然落在極端值上，BACKTEST_LOG #13 踩過）。

v1 在 val_sel 的曲線（`data/threshold_curve_v1_val_sel.png`）：

| 門檻 | n | 勝率 | 平均報酬 |
|---|---|---|---|
| 0.585 | 66 | 68.2% | 4.27% |
| 0.565 | 226 | 58.4% | 2.65% |
| 0.555 | 449 | 55.0% | 2.25% |
| 0.545 | 783 | 58.6% | 3.43% |
| 全體 | 112,596 | 54.9% | 2.80% |

**0.555~0.565 有一段凹陷**，不是單調衰減。0.585 以上很強但只有 66 筆撐不起結論；
0.535~0.545 有近千筆且勝率 58~59%，實用性較高。**門檻尚未拍板。**

⚠️ `BACKTEST_LOG.md` #22 的回測用的是寫死的每日前 0.1%，**不是**上述方法，
那組數字要重跑才算數。

**四個表格模型的最佳組態**（依 val_sel 挑，同一列是同一組態）：

| 模型 | 單組秒 | val_sel | test 2022 空頭 | test2 2023 多頭 |
|---|---|---|---|---|
| **LightGBM** | 27 | **0.6103** | **0.6353** | 0.6074 |
| RF | 194 | 0.6055 | 0.6308 | 0.5972 |
| LambdaRank | 25 | 0.6023 | 0.5960 | **0.6115** |
| ExtraTrees | 103 | 0.5735 | 0.6074 | 0.5578 |

三個發現：

1. **LightGBM 全面贏過 RF**（舊系統唯一用過的模型），而且**單組快 7 倍**。
2. **LambdaRank 有明顯的制度依賴**：多頭年最強（0.6115）、空頭年最弱（0.5960），
   與它直接優化「每日排序前 K 名」的目標一致 —— 空頭年前 K 名本身就沒東西可排。
3. **ExtraTrees 三段全輸**約 0.03，不是雜訊。極端隨機分裂在這種低訊噪比資料上
   過頭了；低成本對照組的任務達成 —— RF 的 bootstrap + 最佳分裂確實有價值。

**環境注意**：LightGBM 4.5.0 與 sklearn 1.9.0 不相容
（`check_X_y(force_all_finite=...)` 在 sklearn 1.6 改名、1.8 移除），
會在 `model.fit()` 當場 TypeError。已升級到 **LightGBM 4.7.0**。

**RF 最佳組態**（完整結果在 `data/sweep_round1_rf.csv`）：

| max_features | min_samples_leaf | max_depth | class_weight | val_sel | test 2022 | test2 2023 |
|---|---|---|---|---|---|---|
| **40** | **200** | **20** | None | **0.6055** | 0.6308 | **0.5972** |
| sqrt | 150 | 20 | None | 0.5988 | **0.6350** | 0.5926 |

兩個實質發現：

1. **`max_features` 舊版固定 `sqrt`（=19）從沒搜過，搜了發現 40 更好** ——
   40 在 val_sel 與 test2 都領先，只有 test 2022（空頭年）是 sqrt 勝出。
2. **`class_weight="balanced"` 幾乎全面落後**，base rate 37.6% 本來就不算失衡。

### 下一步

Part D 已結案（閘門通過）。**依使用者決定，照原順序推 Part C，出場規則不插隊。**

1. **Part C 六模型 sweep**（進行中）
2. 逐欄分歧的人工裁決（`doc/FEATURE_V2_DECISIONS.md` §5 已裁決 4 大類，
   §5.6 還有 6 項待定）
3. Part A 剩下 6 項（`predict.py`／`streamlit_app.py`／`submodel_config.py`／
   `build_labels.py`／`code/paths.py`／v1 移除 shape）
4. Part E 軸 2（出場規則）／軸 3（label 重新設計）

### ⚠️ 已知風險：出場規則可能讓模型比較失焦

Round 1 回測（`BACKTEST_LOG.md` #22）顯示 2023 多頭年是**負 alpha**
（中位 −2.6%，只有 37~40% 贏過同日漲停 peer），2022 空頭年才有 +1.2%。
目前 `simulate()` 的移動停利（+25% 啟動、回落 10%）配 MA20 停損，實測平均持有
遠長於 label 的 20 天 horizon，訊號沒機會兌現。

**在這個出場規則下比較六個模型，比到的可能是「誰挑的股票適合長抱」而不是
「誰最會挑 20 天內會漲的股票」。** 使用者已知悉並決定照原順序進行；
軸 2 會在 Round 1 結果出來後處理。

---

## 8. 驗證與回測規則

- **回測一律用 `code/backtest/backtest.py` 的 `simulate()`/`performance()`**，
  搭配當時的正式模型檔，不寫 scratchpad 版本、不用臨時模型、門檻讀檔不硬編
- **必須配隨機對照**（`benchmark.py` 的 `attach_peer_benchmark` / `alpha_summary`）
- **每次跑完立刻寫入 `doc/BACKTEST_LOG.md`**，含日期、改了什麼、完整數字、
  與上一筆的比較結論
- **中位數比平均重要**（平均易被少數大漲個股撐高）
- **分類指標與實際報酬不一定同向**，驗收以實際報酬為準
- **驗收用 Top-K 命中，不要用 PR-AUC**：策略只買最前面 0.04%，而 PR-AUC 是在
  全部 22 萬筆上算的，訊號被稀釋到看不見
- **不能用 val AUC 跨模型比較**（神經網路偷看 val 幾百次，分數虛高）
- **絕對機率門檻搬不動**，跨期一律用「每日前 q%」分位數門檻

---

## 9. 檔案索引

```
code/data_collection/  fetch_*.py, validate_data.py, repair_ohlc.py
code/features/         build_*_features.py, build_features.py（v1，381 欄）
code/features_v2/      common.py, swing_core.py, build_*.py, build_dataset.py,
                       compare_v1_v2.py（v2 獨立實作，371 欄）
code/models/           build_labels.py, submodel_config.py, train_submodels.py,
                       predict.py, tune_trading_threshold.py
code/backtest/         backtest.py, benchmark.py, matched_n.py
data/                  原始資料 + v1 特徵
data_v2/               v2 特徵
tests/                 test_models/, test_features_v2/, test_data_collection/
```
