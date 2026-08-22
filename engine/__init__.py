"""台股預測系統的引擎：抓資料 → 建特徵 → 算 label → 訓練 → 回測 → 前端。

所有子模組一律用絕對 import（`from engine.xxx import ...`）。舊 repo 的
`sys.path.insert` sibling-import 慣例已全部移除。
"""
