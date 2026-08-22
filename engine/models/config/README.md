# 版控的訓練組態產物

這個資料夾放的是**人在迴圈裡挑出來、無法由程式重新推導**的產物。它們跟
`bundle.py` 的 `CHOSEN_THRESHOLDS` 是同一種東西：由人看結果決定，一旦遺失
就重建不出當初那 10 個模型。

因此**刻意放進版控**，不放 `data/`（`data/` 整個被 .gitignore 擋掉）。

| 檔案 | 誰在用 | 內容 |
|---|---|---|
| `drop_volatility.txt` | m5、m10（`--drop-file`） | 16 行波動度家族欄位名，人工挑定於 2026-08-16 |
| `sweep_round4_rf.csv` | m1、m2、m6、m7（`--config-round 4`） | Round 4 第一階段 27 組調參結果（n_estimators=150） |

## ⚠️ 這兩份是換官方資料源「之前」的產物

`sweep_round4_rf.csv` 是 2026-08-15 在 yfinance 資料上搜出來的。重訓時沿用是
**刻意的**，不是漏更新 —— `train_all.sh` 原本的註解就寫明「非 v3 的兩組沿用
Round 4 搜出的最佳參數（那次就是在這 344 欄上搜的）」。

要重新搜（27 組、約 15 小時）：`make sweep-base`。跑完會寫進 `data/`，
`best_config()` 會優先讀 `data/` 的新版本，本資料夾的版本自動退位。
