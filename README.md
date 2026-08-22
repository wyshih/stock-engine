# 台股預測系統 — engine（private）

抓資料 → 建特徵 → 算 label → 訓練 → 回測 → 本機前端。
public 展示站在隔壁的 `../dashboard/`（獨立 repo，不是 submodule）。

規則與踩坑記錄看 [`CLAUDE.md`](CLAUDE.md)。**最高原則：任何改動都不可以讓
m1~m10 這 10 個模型變得無法重建。**

## 快速開始

```bash
make install     # venv + 檢查 TA-Lib C 函式庫 + 裝套件
make bootstrap   # 從 2019-01-01 起全量抓取（10~15 小時，只有第一次）
make rebuild-full  # 建特徵（含 v3）與 label
make train       # 序列訓練 m1~m10 + 產門檻曲線（數小時）
make curve       # 產曲線 → 由人看曲線挑門檻 → 寫進 bundle.py 的 CHOSEN_THRESHOLDS
make backtest    # 絕對門檻版 + 訊號數對齊版，跑完立刻寫 doc/BACKTEST_LOG.md
make app         # 本機前端 http://localhost:8501
```

日常只要 `make update`（抓資料 → promote → 特徵 → label → 補分數）。

## 目錄

```
engine/                 python package（全部程式）
  paths.py              路徑唯一來源
  data_source/          TWSE / TPEx 官方端點抓取 + promote_price
  features/             9 支 builder + v3/（520 欄的轉換版）
  models/               label / 訓練 / 門檻曲線 / bundle / 推論
  backtest/             simulate()/performance() 唯一實作 + benchmark + matched_n
  app/                  本機 Streamlit 前端（frontend/ 是技術面分析頁）
  tools/                手動工具：status / run_picked / tune_threshold / verify_vs_old
  export/               產出 public dashboard 的資料包
doc/                    BACKTEST_LOG / EXPERIMENT_STATUS（_archive/ 是已過期的）
tests/                  pytest
data/ models/ logs/     不進 git（.gitkeep 佔位）
```

## 資料

- 期間 2019-01-02 ~ 2026-08-21，資料源 TWSE `MI_INDEX` + TPEx `otc` 官方端點。
- `price_official.parquet` 是抓回來的原始檔；下游全部讀 `price.parquet`，
  兩者之間靠 `make promote`（`engine/data_source/promote_price.py`）。
- `features.parquet` 380 欄、`features_v3.parquet` 520 欄；
  模型實際用的是經 `submodel_config.feature_cols()` 白名單篩過的 344 / 509 欄。

## 驗證重構

`make verify-vs-old` 會拿本 repo 重建的結果對照舊 repo
`../../stock_committee_norf/data/`（唯讀）：原始資料逐列、特徵欄位集合與數值、
label 逐列全等；模型只比 AUC 與門檻曲線（RF 有隨機性，不比 pkl bytes）。
