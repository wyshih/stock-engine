"""專案路徑的唯一來源。

在此之前 `PROJECT_ROOT` / `DATA_DIR` / `MODEL_DIR` 在 22 個檔案裡各寫一遍，
而且寫法不一致：多數是 `Path(__file__).parent.parent.parent`（沒有 resolve），
少數是 `Path(__file__).resolve().parents[2]`。前者在 symlink 或相對路徑呼叫下
會指到別的地方，這裡統一用 resolve。

用法（本 repo 是正常的 package，直接 import）::

    from engine.paths import DATA_DIR

⚠️ 專案裡任何檔案都不得自行推導路徑（CLAUDE.md 規則 13）。舊 repo 用
`sys.path.insert` + `from paths import ...` 的 sibling-import 慣例已全部移除。
"""

from __future__ import annotations

from pathlib import Path

# engine/engine/paths.py → parents[1] 是 repo 根目錄
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_V2_DIR = PROJECT_ROOT / "data_v2"
MODEL_DIR = PROJECT_ROOT / "models"
DOC_DIR = PROJECT_ROOT / "doc"
