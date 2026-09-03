# 台股預測系統 — engine（private）

這個 repo 是**全部**的資料抓取、特徵、訓練、回測與本機前端。
public 展示站是隔壁獨立的 `dashboard/` repo（不是 submodule），
只吃 `make export-public` 產出的資料包。

**最高原則：任何改動都不可以讓 m1_base_up20 與 m1_mdd10 這 2 個模型變得無法重建。**
判斷不確定的時候，選「重建得出來」的那條路。

## 2 個模型

全部 RandomForest、Round 4 切分（train 2020-01~2023-11 / val_es 2024H1 /
val_sel 2024H2 / test 2025-02~2025-12 / test2 2026-01~2026-07），
兩個交界各留一個月 embargo。

| 代號 | 特徵集 | 特徵數 | label | 門檻 |
|---|---|---|---|---|
| m1_base_up20 | base | 344 | label_up20 | 0.77 |
| m1_mdd10 | base | 344 | label_mdd10 | 0.68 |

**兩者用同一份特徵集、同一個搜尋空間，只差標的** —— 差異只能來自標的，
不會混進調參的運氣。`label_mdd10` 是 `label_up20` 再要求「未來 20 個交易日內
最低收盤不跌破 −10%」，原本標 1 但期間跌破的樣本改標 **0**（不是整列排除）。

⚠️ **`label_up20` 會系統性偏袒大跌反彈**（2026-09-02 實測）：進場前 20 日跌超過
20% 的樣本正例率 63.2%，持平的只有 35.4%（偏差 1.78 倍）。原因是它數的是「上漲
天數的頻率」，而頻率是波動度的代理。後果：m1 在門檻 0.77 之上的 14,867 筆訊號裡
**97.8% 是過去 20 日跌超過 10% 的股票**（全市場只有 13.5%）。

⚠️ **兩輪修這個偏差都失敗了，已移除（2026-09-03）**：
`label_steady20`（波動標準化+均線）與 `label_xsrank20`（橫斷面風險調整排名）
在 label 側都成功把偏差壓到接近 1.0，但**訓練出的模型績效遠比 label_up20 差**
（val_sel AUC 接近隨機、Sharpe 從 0.777 掉到 0.445 / 0.264）。**最貴的教訓**：
label 側全歷史統計算出的偏差，不代表模型真正選出的高分訊號（被 RF 揀選過的
極端 1.5%）也不偏頗 —— 兩者是不同的群體，兩輪都是訓練完才發現。完整記錄見
`doc/BACKTEST_LOG.md` #31 / #32；程式已從程式碼移除，在 git 歷史裡
（`git show 5c452f1:engine/models/build_labels_xsrank.py` 等）。
**下次要修這個偏差，先用小樣本或既有模型驗證高分訊號的組成，再決定值不值得
整輪訓練。**

**代號中間有空號是刻意的** —— 原本是 5 種特徵集 × 2 種 label = 10 個，
2026-08-22 收到 5 個（①②③⑥⑧），2026-09-02 使用者要求再收到這 2 個。
沿用原編號（①）才對得上 `doc/BACKTEST_LOG.md` 裡的實驗記錄。

2026-09-02 移除的東西，以及為什麼可以一起走：
- `m2_nomkt_up20`（去大盤）、`m3_v3_up20` / `m8_v3_nobear`（v3 特徵集）、
  `m6_base_nobear`（去空頭 label）
- **v3 特徵管線**（`engine/features/v3/`、`features_v3.parquet`、`feature_audit.csv`、
  `volproxy.csv`、`make features-v3`）—— 只有 m3/m8 在用
- **`label_nobear`**（`build_labels_nobear.py`、`labels_nobear.parquet`，以及
  `score_source.py` 裡對 nobear 家族的去空頭過濾）—— 只有 m6/m8 在用

更早砍掉的 m4/m5/m7/m9/m10 全是「去大盤」或「去波動度」變體：BACKTEST_LOG #28
證實它們在絕對門檻下的高報酬來自**門檻效應而非模型能力**。

門檻寫在 `engine/models/bundle.py` 的 `CHOSEN_THRESHOLDS`，由使用者看
val_sel 曲線挑定，不是自動算的。

⚠️ **每個模型各自調參，不得共用組態。** 每一個選定的模型都要用**自己的特徵集、
自己的 label** 跑一輪超參數搜尋，讀自己那份 `data/sweep_{key}_rf.csv`。即使兩個
模型的特徵集相同、只差 label，也各搜各的 —— label 換了，最佳組態就不保證一樣，
沿用等於拿別的問題調出來的參數。`train_label_variant.py` 不傳 `--config-round`
就會走這條路；`--config-round` 只保留給讀取既有歷史檔案用。新增的
`--config-key` 是給「先借別的模型組態看成效」的暫定路徑用，bundle 的
`config_source_key` 會留痕，成效不好就直接刪掉整個實驗，不必轉正。

⚠️ **`engine/models/config/drop_volatility.txt` 是版控產物**（人工挑定的 16 欄
波動度家族，程式推導不出來）。目前沒有模型在用（m5/m10 已砍），保留是為了之後
想復原那兩個時不必重挑。同層的 `sweep_round4_rf.csv` 是舊式共用組態的歷史檔案，
依上面的規定**不再用於訓練新模型**。

重建路徑：`make bootstrap` → `make rebuild-full` → `make train` → `make curve`
（人挑門檻）→ `make backtest` → 寫 `doc/BACKTEST_LOG.md`。

---

## 14 條踩坑規則

1. **不用 yfinance，只用 TWSE / TPEx 官方端點。** yfinance 的 `Close` 永遠會做
   分割調整，某檔股票一旦分割就回頭改寫整段歷史；而我們是增量抓取、只重寫尾端
   視窗，舊列停在舊基準 —— 每次分割都留下一道永久假跳空。全市場曾有 905/2028 檔
   中招、**3,599 道假接縫**，訓練期最重。官方端點給的是當日真實成交價。
2. **特徵一律經 `submodel_config.feature_cols()` 白名單取用**，不可以直接把
   `features.parquet` 的全部欄位餵給模型（會把 `label_*` / `hitrate_*` 一起餵進去）。
3. **shape 特徵永久排除。** KMeans 群心是用 2023-12-31 之前的資料 fit 的，
   是分布層級的洩漏。`build_shape_features.py` 不在本 repo，`train_single.py`
   另外再擋一次 `price_shape_` / `volume_shape_` 前綴。
4. **bundle 裡的訓練期 stats（median / mean / std）推論時一律沿用，不得重算。**
   重算等於把推論期的分布資訊倒灌回標準化。
5. **`--full` 重建前先刪舊衍生檔。** 那些 builder 是 upsert 寫檔，`--full` 只覆蓋
   算得出來的列，舊資料獨有的組合會殘留（2026-08-22 踩過：2026-07-10 那個假交易日
   的 1,948 列留在特徵裡）。用 `make rebuild-full`，它會先跑 `clean-derived`。
6. **兩個切分交界各留一個月 embargo。** `label_up20` 要看未來 20 個交易日，
   交界緊貼的話訓練期末端的答案會落在驗證期裡，等於偷看。
7. **門檻由人看 val_sel 曲線挑，不用自動規則。** 程式讀檔不硬編，
   **每次重訓都必須重挑**（同一組資料重訓出來的分數分布就會不一樣）。
8. **回測只用 `engine/backtest/backtest.py` 的 `simulate()` / `performance()`**，
   不得另寫 scratchpad 版、不得用臨時訓練的模型、不得手動硬編門檻。
   混用不同實作導致數字對不上，這個專案已經吃過一次虧。
9. **模型比較必須附訊號數對齊版（每日前 1.5%）。** 本系統「訊號越少報酬越高」，
   固定門檻的比較會退化成「門檻鬆緊」的比較，不是模型好壞的比較
   （`doc/BACKTEST_LOG.md` #28）。`make backtest` 兩張表一起出。
10. **不並行訓練。** 10 核機器上 RF 內層已經吃 8 核，外層再並行只會更慢。
    `train_all.sh` 是序列的，產出已存在就跳過，中斷後直接重跑即可續跑。
11. **`validate_data.py` 驗「資料裡最新那一天」，不是「今天」。** 開盤前跑的話
    今天還沒有資料，拿今天去驗必定失敗，整條 update 會中斷（2026-08-14 早上踩到）。
    用 `fetch_from.py --last` 取那一天。
12. **`fetch_stock_list` 必須最先跑，`build_features` 必須最後跑。**
    前者：抓價格是照 stock_list 對的，不先更新清單，新上市的股票永遠不會有資料。
    後者：它是合併步驟，前面九支都跑完才有東西可以合併。
13. **路徑只走 `engine/paths.py`。** 任何檔案都不得自行推導 `PROJECT_ROOT`。
14. **跑完回測立刻寫 `doc/BACKTEST_LOG.md`**，不要只留在對話裡 —— 包含日期、
    這次改了什麼、完整績效數字、跟上一筆的比較結論。

---

## 回測方法（唯一路徑）

`make backtest` 就是這個方法，不要另開一條：

- `engine.backtest.backtest.simulate()` / `performance()`
- 出場用 `CURRENT_EXIT_RULES`：`trail_trigger=0.15`、`trail_pct=0.10`、
  `stop_loss=0.20`（**`stop_loss` 有值時 MA20 停損不生效**）
- **`dedup=False`** —— 訊號層級口徑，每筆超過門檻的訊號都獨立進場。
  這是使用者的實際用法，也與挑門檻時看的曲線同一把尺，兩邊必須一致才比得起來
  （`doc/BACKTEST_LOG.md` #24 vs #25）
- 每次比較都附**訊號數對齊版**：每日取分數最高的前 1.5%
- 跑完立刻寫 `doc/BACKTEST_LOG.md`

實作只有一份：`engine/backtest/summary.py`。`make backtest`（內部驗證，預設
test + test2 全段）與 `make export-public`（public 展示，固定 2025-02~2026-07）
都呼叫它，差別只有區間。**不要在匯出流程裡另寫一份回測**。

## 額外規則

- 所有金融計算必須有 unit test 驗證邊界條件（尤其「未來資料不足時應為 NaN 而非 0」）。
- 不得 hardcode 任何 API key，一律用環境變數。
- 資料來源必須記錄在 code comment。
- `data/`（11GB）與 `models/`（1.6GB）不進 git；例外是 `engine/models/config/`
  底下的人工調參產物，那是原始碼的一部分。
- **`make export-public` 之後一定要跑 dashboard 的 pytest**（`make export-public`
  已經自動接了）。public repo 的安全檢查有幾條在 `public_data/` 是空的時候會
  skip —— 只有資料包產出後才真的驗得到「沒有 2025-02-01 之前的資料」。
- Commit 規範：`feat / fix / refactor / test / docs / chore`。

## 不要搬回來的東西

`code/features_v2/`、`build_shape_features.py`（洩漏）、yfinance 版的
`fetch_price.py` / `repair_ohlc.py` / `clean_price_outliers.py` /
`init/*.py`、`train_bundles.py` / `train_submodels.py` / `finalists.py`
（只服務已封存的 r1/r2/r4）、`run_queue.sh`、`data_v2/`。
它們全部留在舊 repo `stock_committee_norf`，不要重新引入。

2026-09-02 起也包含 **v3 特徵管線**（`engine/features/v3/`）與
**`build_labels_nobear.py`**：它們在本 repo 的 git 歷史裡，要復原是
`git show`，不是重寫一份。復原之前先確認真的有模型要用 —— 只有 m3/m8 用 v3、
只有 m6/m8 用 nobear，那四個模型已依使用者要求移除。
