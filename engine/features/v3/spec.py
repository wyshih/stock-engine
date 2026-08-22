"""決定每個現行特徵在 v3 裡要做哪些轉換、原值留不留。

分類的依據全部是實測數字（不是看名字），來自兩份稽核檔：
* feature_audit.csv  ── 逐年（2020/2022/2024/2026）中位數與 P10~P90 寬度
* volproxy_old.csv   ── 與當日成交金額 amount、與 20 日已實現波動 rv20 的 Spearman

核心判斷（見 doc 回報）：模型分數與 amount 相關 0.60~0.70，但**沒有任何單一特徵
超過 0.50**（中位數只有 0.087）。真正的來源是一整個「波動度家族」
（natr_14 / bb_width_pct / std_ratio / ma_squeeze / tl_channel_width，
與 rv20 相關 0.64~0.90）被樹模型聚合起來。所以：

* 對這個家族做**橫斷面排名沒有用** —— 高波動股每天的排名都高，排名只是把
  「這是一檔高波動股」原封不動再編碼一次。
* 有用的是**跟自己歷史比**（方向 A）：「這檔股票現在的波動相對它自己算不算高」，
  這個轉換把個股之間的波動水位差整個消掉。
* 方向 B 仍然要有，但要作用在 A 的結果上（`_szx`）：「今天全市場裡，誰的波動
  相對自己最異常」—— 這才是不帶個股身分的市場比較。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ── 判定門檻 ──────────────────────────────────────────────────────────────────
# 與 rv20 或 amount 的 |Spearman| 超過這個值，就認定它的絕對水位主要在編碼
# 「這是哪一種股票」而不是「現在發生了什麼」。
IDENTITY_RV_THRESHOLD = 0.30
IDENTITY_AMOUNT_THRESHOLD = 0.30

RAW_ONLY_CATEGORIES = frozenset({"binary", "sparse_count", "market_level"})

# 後綴約定
SUFFIX_SELF_Z = "_sz"     # 方向 A：對自己過去 250 日的 z-score
SUFFIX_SELF_Z_XS = "_szx"  # 方向 B'：_sz 的當日橫斷面百分位
SUFFIX_XS = "_xs"         # 方向 B：原值的當日橫斷面百分位


@dataclass(frozen=True)
class FeaturePlan:
    """單一現行特徵在 v3 的處理方式。"""

    name: str
    category: str
    keep_raw: bool
    make_self_z: bool
    make_self_z_xs: bool
    make_xs: bool
    reason: str

    def output_columns(self) -> list[str]:
        cols = []
        if self.keep_raw:
            cols.append(self.name)
        if self.make_self_z:
            cols.append(self.name + SUFFIX_SELF_Z)
        if self.make_self_z_xs:
            cols.append(self.name + SUFFIX_SELF_Z_XS)
        if self.make_xs:
            cols.append(self.name + SUFFIX_XS)
        return cols


def _is_identity_carrier(row: pd.Series) -> bool:
    """絕對水位主要在編碼個股身分（波動度/流動性水位）而非當下狀態。"""
    rv = abs(row.get("sp_rv20", 0.0) or 0.0)
    amt = abs(row.get("sp_amount", 0.0) or 0.0)
    return rv > IDENTITY_RV_THRESHOLD or amt > IDENTITY_AMOUNT_THRESHOLD


def plan_for(name: str, row: pd.Series) -> FeaturePlan:
    category = row["category"]

    # 1) 二元旗標 / 稀疏型態計數 / 大盤層級：原樣保留，不排名。
    #    旗標本身 0/1 就跨年可比；大盤特徵排名會把「現在是什麼市況」抹掉，
    #    模型必須還知道大盤在哪裡（使用者明確要求）。
    if category in RAW_ONLY_CATEGORIES:
        return FeaturePlan(name, category, True, False, False, False,
                           "旗標/稀疏計數/大盤層級：原樣保留，不做任何排名")

    identity = _is_identity_carrier(row)
    is_drift = category == "drift"

    # 2) 身分編碼型（波動度/流動性/估值水位）與逐年漂移型：原值一律剔除。
    #    只留「相對自己歷史」(A) 與「A 的當日橫斷面」(B')。
    if identity or is_drift:
        why = []
        if identity:
            why.append(f"rv20={row['sp_rv20']:+.2f}/amount={row['sp_amount']:+.2f} 過高")
        if is_drift:
            why.append(f"逐年漂移 scale_ratio={row['scale_ratio']:.2f}")
        return FeaturePlan(name, category, False, True, True, False,
                           "剔除原值（" + "、".join(why) + "）→ 改用 _sz + _szx")

    # 3) 已經是「跟自己歷史比」的滾動百分位：原值就是方向 A，補上方向 B。
    if category == "self_relative":
        return FeaturePlan(name, category, True, False, False, True,
                           "本身即方向 A（滾動百分位），補當日橫斷面排名補齊方向 B")

    # 4) 其餘尺度無關的連續特徵：原值保留 + 當日橫斷面排名。
    #    不額外做 _sz —— 這批多是有界振盪指標（RSI/KD/威廉/CCI…），
    #    features.parquet 裡的 rel_pct_* 已經做過同一件事並實測無訊號
    #    （doc/BACKTEST_LOG.md #18），再加只會稀釋 max_features="sqrt" 抽樣。
    return FeaturePlan(name, category, True, False, False, True,
                       "尺度無關連續值：保留原值 + 當日橫斷面排名（方向 B）")


def build_plans(audit_csv: Path, volproxy_csv: Path) -> list[FeaturePlan]:
    audit = pd.read_csv(audit_csv, index_col="feature")
    vol = pd.read_csv(volproxy_csv, index_col="feature")
    merged = audit.join(vol[["sp_amount", "sp_rv20"]], how="left").fillna(
        {"sp_amount": 0.0, "sp_rv20": 0.0}
    )
    return [plan_for(name, row) for name, row in merged.iterrows()]


def plans_to_frame(plans: list[FeaturePlan]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": p.name,
                "category": p.category,
                "keep_raw": p.keep_raw,
                "make_self_z": p.make_self_z,
                "make_self_z_xs": p.make_self_z_xs,
                "make_xs": p.make_xs,
                "n_out": len(p.output_columns()),
                "reason": p.reason,
            }
            for p in plans
        ]
    ).set_index("feature")
