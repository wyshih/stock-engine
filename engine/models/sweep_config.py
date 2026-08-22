"""從 sweep CSV 讀出某個模型家族的最佳超參數組。

為什麼是新檔（2026-08-22 搬進本 repo 時新寫）：
`train_label_variant.py`（訓練 5 個模型的唯一入口）需要 `best_config()`，
而這個函式原本住在 `code/models/finalists.py` 裡。`finalists.py` 整支只服務
已封存的 r1/r2/r4 委員會流程，不搬；但它裡面的 `best_config()` / `_coerce()`
是訓練路徑的必要相依。舊 repo 的 `pipeline/03_models/` 漏了這支，
所以 `pipeline/train_all.sh` 其實跑不起來（ModuleNotFoundError: finalists）。
這裡把這兩個函式原封搬出來，讓那幾個模型真的可以重建。

⚠️ 組態一律**從 sweep CSV 現讀**，不寫死 —— 寫死會拿到某次暫定值。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from engine.paths import DATA_DIR
from engine.models.train_single import current_round, eval_splits

logger = logging.getLogger(__name__)

# 版控的調參產物（人工挑定、程式推導不出來），見 config/README.md
CONFIG_DIR = Path(__file__).resolve().parent / "config"

# metrics 與計時欄位不是超參數，讀組態時要濾掉
NON_PARAM_COLS = {"model", "seconds", "epochs"}
# 字串型的超參數規格（例如 MLP 的 hidden "256-64"），不能被當成數字轉型
STR_PARAMS: set[tuple[str, str]] = set()


def best_config(family: str, config_round: int | None = None) -> dict:
    """從 sweep CSV 取 val_sel AUC 最高的那一組超參數。

    跨模型排名一律看 val_sel，不看 val_es —— 神經網路拿 val_es 做 early
    stopping，會偷看幾百次，分數虛高（doc/PLAN.md §4）。

    `config_round` 可與當前切分不同，例如「Round 1 的切分 + Round 2 選出的組態」。
    這是必要的：`sweep_round1_*.csv` 是在特徵重建**之前**跑的（含 revenue_year
    等已清除的欄位），選出的組態被污染，不該再拿來訓練。
    預設 None＝與當前 round 相同，維持原本行為。

    ── 以上為 finalists.py 的原始 docstring，原封保留 ──
    2026-08-22 搬進本 repo 時補充：m1/m2/m3/m6/m8 也走這條路。m3/m8 用
    `--config-round 5`（v3 重搜，寫在 sweep_round5_rf.csv）；
    m1/m2/m6 用 `--config-round 4`，而 round 4 的 CSV 是版控產物，見
    `engine/models/config/README.md`。
    """
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
    logger.info(
        f"{family}：{len(df)} 組中選 "
        + " ".join(f"{s}={row[f'{s}_auc']:.4f}" for s in eval_splits() if s != "val_es")
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
