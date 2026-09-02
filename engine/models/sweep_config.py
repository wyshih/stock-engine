"""從 sweep CSV 讀出**該模型自己**的最佳超參數組。

## 規定：每個模型各自調參，不得共用組態

每一個選定的模型都必須用**自己的特徵集、自己的 label**跑一輪超參數搜尋，
讀自己那一份 `sweep_{key}_rf.csv`。不可以拿別的模型搜出來的組態套過來，
即使兩者的特徵集相同、只有 label 不同，也不行。

沒有 `--key` 的舊式呼叫（讀 `sweep_round{N}_rf.csv`）保留下來只為了讀取既有的
歷史檔案，**不可用於新模型**。

例外（2026-09-03 使用者指定）：`train_label_variant.py --config-key OTHER` 可以
明確借用另一個模型的組態。僅在**特徵集與搜尋空間都相同、只差標的**時才成立 ——
此時沿用同一組超參數反而讓「差異只能來自標的」更乾淨。借用會寫進 bundle 的
`config_source_key` 留痕。⚠️ 不要改用「複製一份 sweep_{key}_rf.csv」達成同樣效果：
那個檔裡的 val_sel AUC 是**別的 label** 算出來的，留著就是日後誤讀的地雷。

## 為什麼是新檔

`train_label_variant.py`（訓練入口）需要 `best_config()`，而這個函式原本住在
`code/models/finalists.py`。`finalists.py` 整支只服務已封存的 r1/r2/r4 委員會
流程，不搬；但這兩個函式是訓練路徑的必要相依。舊 repo 的 `pipeline/03_models/`
漏了這支，所以 `pipeline/train_all.sh` 其實跑不起來（ModuleNotFoundError）。

⚠️ 組態一律**從 sweep CSV 現讀**，不寫死 —— 寫死會拿到某次暫定值。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from engine.paths import DATA_DIR
from engine.models.train_single import current_round

logger = logging.getLogger(__name__)

# 版控的調參產物（人工挑定、程式推導不出來），見 config/README.md
CONFIG_DIR = Path(__file__).resolve().parent / "config"

# metrics 與計時欄位不是超參數，讀組態時要濾掉
NON_PARAM_COLS = {"model", "seconds", "epochs"}
# 字串型的超參數規格（例如 MLP 的 hidden "256-64"），不能被當成數字轉型
STR_PARAMS: set[tuple[str, str]] = set()


def best_config(family: str, config_round: int | None = None,
                key: str | None = None) -> dict:
    """從**該模型自己的** sweep CSV 取 val_sel AUC 最高的那一組超參數。

    跨模型排名一律看 val_sel，不看 val_es —— 神經網路拿 val_es 做 early
    stopping，會偷看幾百次，分數虛高（doc/PLAN.md §4）。

    `key`：模型代號，讀 `sweep_{key}_{family}.csv`。**新模型一律傳 key。**
    `config_round`：舊式共用組態的相容路徑，只用於讀既有歷史檔案。
    """
    if key:
        # 每個模型自己的搜尋結果。這是現行的唯一正路。
        name = f"sweep_{key}_{family}.csv"
    else:
        # 舊式：整個 round 共用一份。只用來讀既有歷史檔案，新模型不得使用。
        round_no = config_round or current_round()
        name = f"sweep_round{round_no}_{family}.csv"

    # 找檔順序：先 data/（重跑 sweep 產生的新結果優先），再退回版控的 config/。
    # round 4 的 CSV 只存在於 config/ —— 它是人工調參的既有成果，沒有任何程式
    # 會在日常流程中重新產生它（要重搜是 `make sweep-base`，約 15 小時）。
    # 這兩個檔一旦不在版控就會重蹈「scratchpad 產物隨 session 消失、模型
    # 再也重建不出來」的覆轍，見 config/README.md。
    path = DATA_DIR / name
    if not path.exists():
        path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {name}（已找過 {DATA_DIR} 與 {CONFIG_DIR}）。\n"
            f"round 5/6/7 由 `make train` 的階段一自動產生；"
            f"round 4 是版控產物，應在 {CONFIG_DIR}，請確認沒有被誤刪。"
        )
    logger.info(f"{family}：組態讀自 {path}")

    df = pd.read_csv(path)
    row = df.sort_values("val_sel_auc", ascending=False).iloc[0]
    # 印出 CSV 裡**實際存在**的 AUC 欄，不要用 eval_splits() 去猜 ——
    # 調參自 2026-08-23 起只評估 val_sel，CSV 沒有 test_auc / test2_auc 了。
    # 舊版寫死跑 eval_splits()，在這行純粹是給人看的 log 上 KeyError 中止，
    # 而選組態的邏輯（上一行的 sort_values）其實完全正常。
    auc_cols = [c for c in df.columns if c.endswith("_auc")]
    logger.info(
        f"{family}：{len(df)} 組中選 "
        + " ".join(f"{c[:-4]}={row[c]:.4f}" for c in auc_cols)
    )

    params = {}
    for key, value in row.items():
        if key in NON_PARAM_COLS or key.endswith(("_auc", "_lift")):
            continue
        params[key] = str(value) if (family, key) in STR_PARAMS else _coerce(value)
    return params


def _coerce(value):
    """CSV 讀回來的型別要還原成 sklearn / LightGBM 吃得下的原始型別。

    - class_weight 的 None 在 CSV 裡是空值 → NaN
    - max_features 同欄混了 "sqrt" 與數字 → 整欄變字串
    """
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
