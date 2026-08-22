"""
各子模型的特徵組與超參數搜索空間（PLAN.md 6.2 / 6.7）。
"""
from __future__ import annotations

# ── 特徵群組定義 ───────────────────────────────────────────────────────────────
# 每個前綴代表一組特徵（build_price_features / chip / fundamental 的欄位名稱）

SHORT_PRICE_PREFIXES = [
    "return_", "ma5", "ma10", "ma20",
    "close_ma5", "close_ma10", "close_ma20",
    "ma5_ma10", "ma10_ma20", "ma_squeeze",
    "bull_3ma_", "bear_3ma_", "bull_4ma_", "bear_4ma_",  # 原寫 is_3ma_/is_4ma_ 對不到實際欄位，2026-07-14 修正
    "dif_", "macd_", "hist_",
    "k_", "d_", "kd_",
    "rsi_",  # 原只列 rsi_7/rsi_14，改用前綴涵蓋 rsi_9/28/slope/overbought 等（2026-07-14）
    "bb_pct", "bb_width", "bb_upper_break_", "bb_lower_break_",
    "vol_ratio", "vol_price_",
    "limit_hit",
    "rank_",  # 跨股票排名特徵（2026-07-13 新增，同一天在全市場的相對位置）
    "above_ma5", "above_ma10", "above_ma20",  # 2026-07-14 補漏
    "close_ols_slope", "ma5_ols_slope", "close_slope_trend", "ma5_slope_trend",  # 2026-07-14 新增（最小平方法斜率）
    "rs_",  # 跟大盤指數比的超額報酬 rs_5d/20d/60d（2026-07-14 補漏）
    # ta-lib 全量指標（PLAN.md 5.13，2026-07-14 發現從未接進任何模型，補上）
    "adx_", "adxr_", "apo", "aroon", "cci_", "cmo_", "dx_", "mfi_", "mom_10",
    "ppo", "roc_", "rocp_", "stoch", "trix", "ultosc", "willr_", "atr_", "natr_",
    "obv_", "ad_ratio", "cmf", "vol_breakout", "vol_confirm",
    "cdl",  # K 線型態辨識，61 個（2026-07-14 補漏）
]

MED_PRICE_PREFIXES = [
    "ma60", "ma120",
    "close_ma60", "close_ma120",
    "ma20_ma120", "ma60_ma120",
    "bull_5ma_", "bear_5ma_",  # 原寫 is_5ma_ 對不到實際欄位，2026-07-14 修正
    "momentum_",  # 原只列 momentum_20d/60d，改前綴涵蓋 momentum_accel/align（2026-07-14）
    "vol_ma", "vol_asymm",
    "close_zscore",
    "bb_expand",
    "high_n_", "low_n_",
    "above_ma60", "above_ma120",  # 2026-07-14 補漏
    "dist_high_", "dist_low_", "is_new_high",  # 2026-07-14 補漏
    "std_ratio",  # 2026-07-14 補漏
]

LONG_PRICE_PREFIXES = [
    "ma240", "close_ma240",
    "per", "pbr", "dividend_yield", "per_rank", "pbr_rank", "yield_rank",
    "revenue_yoy", "revenue_mom", "revenue_accel", "revenue_pos_months",
    "above_ma240",  # 2026-07-14 補漏
]

SWING_PREFIXES = [
    "return_", "bb_pct_b", "bb_width",
    "rsi_14", "rsi_oversold",
    "k_oversold", "kd_golden",
    "vol_ratio", "vol_price_corr",
    "close_ma20", "close_ma60",
    "ma_slope_",
    # swing_features.parquet 實際欄位（前波/大量高低點、趨勢線，PLAN.md 5.10~5.12）
    "high1_", "high2_", "high3_",
    "low1_", "low2_", "low3_",
    "vh1_", "vh2_", "vh3_",
    "vl1_", "vl2_", "vl3_",
    "is_above_vh1", "is_below_vl1",
    "support_", "resist_",
    "is_triangle", "slope_align",
]

CHIP_PREFIXES = [
    "foreign_", "trust_", "dealer_",
    "institutional_", "margin_", "short_",
    "absorb_",  # 吸籌但滯漲：法人連續買超 + 股價沒反應（PLAN.md 之外，2026-07-12 新增）
]

REVENUE_PREFIXES = [
    "revenue_", "per", "pbr", "dividend_yield",
    "per_rank", "pbr_rank", "yield_rank",
]

STATUS_PREFIXES = [
    "is_full_cash", "is_disposed", "is_warning",
    "limit_hit_rate", "avg_vol_",
]

MARKET_PREFIXES = [
    "mkt_",  # 大盤(加權指數)特徵，2026-07-19 新增，見 build_market_features.py
]

TRENDLINE_PREFIXES: list[str] = [
    # 新定義趨勢線（2026-07-29 新增，見 code/features/build_trendline_features.py）：
    # 修正舊 support_*/resist_* 的五個定義問題（用 high/low 而非 close、回歸線平移到
    # 邊界而非穿過中間、至少 3 個樞紐點、touches 改數獨立回測、三角形要求真收斂），
    # 並補上舊版完全沒有的「突破事件」（tl_resist_break / tl_break_vol /
    # tl_false_breaks_20d）。舊的 SWING_PREFIXES 保留不動，兩者並存供對照。
    "tl_",
]

REL_PREFIXES: list[str] = [
    # ⚠️ 2026-07-29 驗證失敗，已從所有子模型移除（見 doc/BACKTEST_LOG.md #18）：
    # 同門檻對照 2026 平均報酬 10.05%→5.46%、alpha 中位數 -3.64%→-5.17%、
    # 贏過 peer 46.1%→39.4%，兩種門檻紀律、兩個年度、每個指標都變差。
    # 機制：重要度排 32~198/359、合計 5.68%（均勻應 7.24%）——模型確實在用，
    # 但沒有預測力，反而稀釋了 max_features="sqrt" 的抽樣，排擠掉有用的特徵。
    # 常數保留、欄位也留在 features.parquet，但不再接進任何子模型。
    # 自我正規化特徵（2026-07-29 新增，見 code/features/build_relative_features.py）：
    # 量能/風險調整動能/ATR 計價距離/指標自身百分位/籌碼異常倍數，全部是
    # 「相對這檔股票自己的歷史分布」而非絕對值或跨股票排名。
    # ⚠️ 本專案過去加特徵成功率為 0，本次驗收必須有「相同特徵集重訓」的對照組
    # （見 doc/BACKTEST_LOG.md 任務 #6），否則無法跟重訓本身的變異區分。
    "rel_",
]

SHAPE_PREFIXES: list[str] = [
    # 2026-07-26 新增過 price_shape_5d_/volume_shape_5d_（5天價量走勢
    # min-max normalize 後 KMeans 分群，one-hot 編碼），2026-07-27 驗證：
    # 兩種編碼（原始群編號、one-hot）在 BUY20_PERSIST10 上重要度都趨近 0
    # （排名 236~347/347），確認是特徵本身無訊號，不是編碼問題，故移除。
    # build_shape_features.py 保留供未來參考，但不再接進任何子模型。
]


# ── 前綴比對誤收、或與其他欄位完全等價的欄位，一律排除（2026-08-06）──────────
EXCLUDE_COLS: frozenset[str] = frozenset({
    # 日曆索引，不是特徵。`REVENUE_PREFIXES` 的 "revenue_" 前綴把營收資料的年月
    # 索引一起選了進來。revenue_year 的值就是當年年份，train(2020~2022) 與
    # test(2024~2026) 的分布**零重疊**（KS=1.0）——樹只要在這裡切一刀，
    # 全部測試列都會落進同一個分支，等於用少數訓練樣本服務所有預測。
    # 不是洩漏（當日確實已知），是結構性的泛化失效。
    "revenue_year", "revenue_month",

    # 與其他欄位代數等價，留著只會稀釋 max_features 的抽樣（Spearman ≈ 1.0000）：
    "rocp_10",    # ≡ roc_10 / 100
    "ppo",        # ≡ apo（apo 已除以股價）
    "stochf_d",   # ≡ stoch_k（慢速 K 的定義就是快速 D）
    "cmo_10",     # ≡ 2 * rsi_9 - 100

    # ATR 三胞胎：TA-Lib 的 NATR 定義就是 100 * ATR / close，而 atr_ratio 改用
    # Wilder 平滑後 = natr_14 / 100 —— 三欄是同一個指標的三種寫法（實測
    # natr_14/atr_ratio 中位數 101.99、corr 0.9665，改 Wilder 後會恰好等價）。
    # 留 natr_14（TA-Lib 標準、Wilder），另外兩欄排除，避免稀釋 max_features 抽樣。
    # 註：atr_rank 是 ATR 的滾動百分位，資訊不同，不在排除之列。
    "atr_14", "atr_ratio",
})


def _select(all_cols: list[str], prefixes: list[str]) -> list[str]:
    """從 all_cols 中選出以任一 prefix 開頭的欄位。"""
    result = []
    for c in all_cols:
        if any(c.startswith(p) for p in prefixes):
            result.append(c)
    return result


def feature_cols(model_id: str, all_feature_cols: list[str]) -> list[str]:
    """
    回傳指定子模型應使用的特徵欄位。

    2026-07-26：重建委員會架構（在修正過的乾淨資料上重跑，跟單一模型
    BUY20_PERSIST10 做公平對照，見 doc/PLAN.md）。
    """
    pm = {
        # ── 籌碼 / 大盤 / 營收：獨立診斷模型，各自預測自己的目標 ─────────────────
        "C1": CHIP_PREFIXES + SHORT_PRICE_PREFIXES,
        "C2": CHIP_PREFIXES + SHORT_PRICE_PREFIXES,
        "C3": CHIP_PREFIXES + MED_PRICE_PREFIXES,
        "F1": REVENUE_PREFIXES + LONG_PRICE_PREFIXES,
        "M1": MARKET_PREFIXES + SHORT_PRICE_PREFIXES,

        "BUY5_LOCAL": MARKET_PREFIXES + SHORT_PRICE_PREFIXES + MED_PRICE_PREFIXES + LONG_PRICE_PREFIXES + SWING_PREFIXES + CHIP_PREFIXES + STATUS_PREFIXES + SHAPE_PREFIXES + TRENDLINE_PREFIXES,
        "BUY5_LOCAL_P5":  MARKET_PREFIXES + SHORT_PRICE_PREFIXES + MED_PRICE_PREFIXES + LONG_PRICE_PREFIXES + SWING_PREFIXES + CHIP_PREFIXES + STATUS_PREFIXES + SHAPE_PREFIXES + TRENDLINE_PREFIXES,
        "BUY5_2STAGE": MARKET_PREFIXES + SHORT_PRICE_PREFIXES + MED_PRICE_PREFIXES + LONG_PRICE_PREFIXES + SWING_PREFIXES + CHIP_PREFIXES + STATUS_PREFIXES + SHAPE_PREFIXES + TRENDLINE_PREFIXES,
        "BUY5_TB_MA20":   MARKET_PREFIXES + SHORT_PRICE_PREFIXES + MED_PRICE_PREFIXES + LONG_PRICE_PREFIXES + SWING_PREFIXES + CHIP_PREFIXES + STATUS_PREFIXES + SHAPE_PREFIXES + TRENDLINE_PREFIXES,
        "BUY10_TB_MA20": MARKET_PREFIXES + SHORT_PRICE_PREFIXES + MED_PRICE_PREFIXES + LONG_PRICE_PREFIXES + SWING_PREFIXES + CHIP_PREFIXES + STATUS_PREFIXES + SHAPE_PREFIXES + TRENDLINE_PREFIXES,
        "BUY20_TB_MA20": MARKET_PREFIXES + SHORT_PRICE_PREFIXES + MED_PRICE_PREFIXES + LONG_PRICE_PREFIXES + SWING_PREFIXES + CHIP_PREFIXES + STATUS_PREFIXES + SHAPE_PREFIXES + TRENDLINE_PREFIXES,
        "BUY10_PERSIST7": MARKET_PREFIXES + SHORT_PRICE_PREFIXES + MED_PRICE_PREFIXES + LONG_PRICE_PREFIXES + SWING_PREFIXES + CHIP_PREFIXES + STATUS_PREFIXES + SHAPE_PREFIXES + TRENDLINE_PREFIXES,
        "BUY20_PERSIST10": MARKET_PREFIXES + SHORT_PRICE_PREFIXES + MED_PRICE_PREFIXES + LONG_PRICE_PREFIXES + SWING_PREFIXES + CHIP_PREFIXES + STATUS_PREFIXES + SHAPE_PREFIXES + TRENDLINE_PREFIXES,
        # 2026-08-02 新增。特徵集與其他委員一致（不含 rel_*）——
        # 消融實驗（同參數、同門檻 0.672、test 2025-01~2026-07）：
        #   含 rel_* (395欄)  AUC 0.6442  1664筆 勝率 78.1% 平均 +20.66%
        #   不含    (355欄)  AUC 0.6432  1604筆 勝率 79.6% 平均 +21.88%
        # AUC 幾乎相同但交易指標全面較佳，與 BACKTEST_LOG #18 在
        # BUY20_PERSIST10 上的結論一致：rel_* 無訊號，只稀釋 sqrt 抽樣。
        "UP20": MARKET_PREFIXES + SHORT_PRICE_PREFIXES + MED_PRICE_PREFIXES + LONG_PRICE_PREFIXES + SWING_PREFIXES + CHIP_PREFIXES + STATUS_PREFIXES + SHAPE_PREFIXES + TRENDLINE_PREFIXES + REVENUE_PREFIXES,
    }
    prefixes = pm.get(model_id, SHORT_PRICE_PREFIXES)
    cols = [c for c in _select(all_feature_cols, prefixes) if c not in EXCLUDE_COLS]
    # 保底：至少 20 欄。2026-07-30 修正——原本直接取 all_feature_cols[:50]，
    # 會把 label_* / hitrate_* 一起抓進特徵集（等於把答案餵給模型）。
    # production 每個模型都選到 200+ 欄所以從未觸發，但這是未爆彈，
    # 由 tests/test_models/test_build_labels.py::TestFeatureCols 釘住。
    if len(cols) < 20:
        cols = [c for c in all_feature_cols
                if c not in ("date", "stock_id")
                and not c.startswith(("label_", "hitrate_"))][:50]
    return cols


# ── 每個子模型 label 的前瞻天數（決定 train/meta split 之間需要多大 gap）────────
HORIZON_DAYS = {
    "C1": 10, "C2": 5,  "C3": 20,
    "F1": 21,
    "M1": 5,
    "BUY5_LOCAL": 5, "BUY5_LOCAL_P5": 5, "BUY5_TB_MA20": 5, "BUY5_2STAGE": 5,
    "BUY10_TB_MA20": 10,
    "BUY20_TB_MA20": 20,
    "BUY10_PERSIST7": 10,
    "BUY20_PERSIST10": 20,
    "UP20": 20,
}


def label_col(model_id: str) -> str:
    mapping = {
        "C1": "label_C1",
        "C2": "label_C2",
        "C3": "label_C3",
        "F1": "label_F1",
        "M1": "label_M1",
        "BUY5_LOCAL": "label_buy5_local",
        "BUY5_LOCAL_P5":  "label_buy5_local_p5",
        "BUY5_2STAGE": "label_buy5_tb_ma20",
        "BUY5_TB_MA20":   "label_buy5_tb_ma20",
        "BUY10_TB_MA20": "label_buy10_tb_ma20",
        "BUY20_TB_MA20": "label_buy20_tb_ma20",
        "BUY10_PERSIST7": "label_buy10_persist7",
        "BUY20_PERSIST10": "label_buy20_persist10",
        "UP20": "label_up20",
    }
    return mapping[model_id]


# ── 超參數搜索空間 ─────────────────────────────────────────────────────────────

RF_SPACE = {
    "n_estimators":      [50, 100, 150],
    "max_depth":         [3, 7, 10],
    "min_samples_leaf":  [10, 20, 50],
    "max_features":      ["sqrt", "log2"],
}

GBT_SPACE = {
    "n_estimators":      [50, 100],
    "max_depth":         [3, 5],
    "learning_rate":     [1e-2, 1e-1],
    "subsample":         [0.8, 1.0],
    "min_samples_leaf":  [20, 50],
}

LR_SPACE = {
    "C":                 [0.001, 0.01, 0.1, 1.0, 10.0],
    "max_iter":          [500],
}

ALL_MODEL_IDS = ["C1", "C2", "C3", "F1", "M1",
                 # 2026-07-30 移除 BUY5_2STAGE（doc/AUDIT_20260728.md §C-6）：
                 # 它跟 BUY5_TB_MA20 的特徵集與 label 完全相同，實測兩者機率
                 # max|diff| = 0.0、相關係數 1.000，13x13 相關矩陣因此有一個
                 # 特徵值恰為 0（矩陣奇異）。除了浪費算力，還會讓 RF 的
                 # feature_importances_ 在兩個完全相同的欄位間隨機分配
                 # （5.50% + 5.22%），使 streamlit「模型成效」頁的重要度排名失真。
                 # 下方 pm / HORIZON_DAYS / label_col 的條目保留不刪，
                 # 這樣舊的 models/BUY5_2STAGE_RF.pkl 若還在也能被讀取。
                 "BUY5_LOCAL", "BUY5_LOCAL_P5", "BUY5_TB_MA20",
                 "BUY10_TB_MA20",
                 "BUY20_TB_MA20",
                 "BUY10_PERSIST7",
                 "BUY20_PERSIST10",
                 # 2026-08-02 新增：獨立實驗驗證過的「未來20日上漲天數過半」模型，
                 # RF 在五個測試期 test AUC 穩定 0.60~0.64。
                 "UP20"]
