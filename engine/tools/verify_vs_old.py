"""把本 repo 重建出來的 data/ 對照舊 repo 的 data/，逐項比對（2026-08-22 新寫）。

存在的理由：這個 repo 是從 `stock_committee_norf` 重構過來的（改成真 package、
移掉 sys.path hack）。「程式跑得動」不等於「算出來的東西一樣」，必須有一支
能把兩邊的產出攤開來對的工具，否則沒有人敢說重構是安全的。

⚠️ 舊 repo 全程**唯讀**，這支只 read。

比對項目：

  原始資料   price / chip / fundamental / revenue / exright / stock_list
             → 列數、日期範圍、共同鍵的逐列數值
  特徵       9 個特徵檔 + features.parquet(380 欄) + features_v3.parquet(520 欄)
             → 欄位集合相同、共同列數值差 < TOLERANCE
  label      labels.parquet / labels_nobear.parquet → 逐列全等
  模型       bundle 的 AUC 與 val_sel 門檻曲線
             → **不比 pkl bytes**：RF 有隨機性，同樣的資料重訓也不會 byte 相同

用法：
  python -m engine.tools.verify_vs_old
  python -m engine.tools.verify_vs_old --old /path/to/stock_committee_norf
  python -m engine.tools.verify_vs_old --only features
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from engine.models.bundle import CHOSEN_THRESHOLDS
from engine.paths import DATA_DIR, MODEL_DIR

DEFAULT_OLD = Path("/Users/todd/Documents/projects/stock_committee_norf")

TOLERANCE = 1e-9
KEYS = ["date", "stock_id"]

RAW_FILES = ["price", "chip", "fundamental", "revenue", "exright", "stock_list"]
FEATURE_FILES = [
    "price_features", "chip_features", "fundamental_features", "talib_features",
    "swing_features", "market_features", "trendline_features", "relative_features",
    "revenue_features",
]
MERGED_FEATURES = {"features": 380, "features_v3": 520}
LABEL_FILES = {"labels": 3_327_632, "labels_nobear": 2_060_418}
MODEL_KEYS = tuple(CHOSEN_THRESHOLDS)

OK, FAIL, SKIP, NOTE = "✅", "❌", "⏭ ", "📌"


class Report:
    """一行一項的對照報告。任一項不符就整份標紅（exit code 1）。"""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, mark: str, name: str, detail: str) -> None:
        self.rows.append((mark, name, detail))
        print(f"  {mark} {name:32s} {detail}")

    def ok(self, name: str, detail: str = "") -> None:
        self.add(OK, name, detail)

    def fail(self, name: str, detail: str) -> None:
        self.add(FAIL, name, detail)

    def skip(self, name: str, detail: str) -> None:
        self.add(SKIP, name, detail)

    def note(self, name: str, detail: str) -> None:
        """有差異但不判失敗 —— 已知且已解釋的漂移，要讓人看見但不擋流程。"""
        self.add(NOTE, name, detail)

    @property
    def failures(self) -> list[tuple[str, str, str]]:
        return [r for r in self.rows if r[0] == FAIL]


def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=columns)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _span(df: pd.DataFrame) -> str:
    if "date" not in df.columns or df.empty:
        return "（無 date 欄）"
    return f"{df['date'].min().date()} ~ {df['date'].max().date()}"


def _compare_values(new: pd.DataFrame, old: pd.DataFrame, keys: list[str],
                    tol: float) -> tuple[int, str]:
    """共同鍵上的逐列比對。回傳 (不符欄位數, 說明)。"""
    merged = new.merge(old, on=keys, how="inner", suffixes=("_new", "_old"))
    shared = [c for c in new.columns if c in old.columns and c not in keys]
    bad = []
    for col in shared:
        a, b = merged[f"{col}_new"], merged[f"{col}_old"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            diff = (a.astype("float64") - b.astype("float64")).abs()
            mismatch = (diff > tol) & ~(a.isna() & b.isna())
        else:
            mismatch = (a.astype(str) != b.astype(str)) & ~(a.isna() & b.isna())
        n = int(mismatch.sum())
        if n:
            bad.append(f"{col}({n})")
    detail = f"共同 {len(merged):,} 列 × {len(shared)} 欄"
    if bad:
        detail += "；不符：" + ", ".join(bad[:8]) + (" …" if len(bad) > 8 else "")
    return len(bad), detail


def check_raw(report: Report, old_data: Path) -> None:
    print("\n原始資料")
    for name in RAW_FILES:
        new = _read(DATA_DIR / f"{name}.parquet")
        old = _read(old_data / f"{name}.parquet")
        if new is None or old is None:
            report.skip(name, f"新={'有' if new is not None else '無'} 舊={'有' if old is not None else '無'}")
            continue
        keys = KEYS if all(k in new.columns for k in KEYS) else ["stock_id"]
        problems = []
        if len(new) != len(old):
            problems.append(f"列數 {len(new):,} vs {len(old):,}")
        if "date" in new.columns and _span(new) != _span(old):
            problems.append(f"期間 {_span(new)} vs {_span(old)}")
        n_bad, detail = _compare_values(new, old, keys, TOLERANCE)
        if n_bad:
            problems.append(detail)
        (report.fail if problems else report.ok)(name, "；".join(problems) or f"{len(new):,} 列　{_span(new) if 'date' in new.columns else ''}")


def check_features(report: Report, old_data: Path) -> None:
    print("\n特徵檔")
    for name in FEATURE_FILES:
        new_path, old_path = DATA_DIR / f"{name}.parquet", old_data / f"{name}.parquet"
        if not new_path.exists() or not old_path.exists():
            report.skip(name, "檔案不存在")
            continue
        new_cols = set(pq.read_schema(new_path).names)
        old_cols = set(pq.read_schema(old_path).names)
        if new_cols != old_cols:
            report.fail(name, f"欄位集合不同：新多 {sorted(new_cols - old_cols)[:5]}，"
                              f"舊多 {sorted(old_cols - new_cols)[:5]}")
            continue
        n_bad, detail = _compare_values(_read(new_path), _read(old_path), KEYS, TOLERANCE)
        (report.fail if n_bad else report.ok)(name, detail)

    for name, expect_cols in MERGED_FEATURES.items():
        new_path, old_path = DATA_DIR / f"{name}.parquet", old_data / f"{name}.parquet"
        if not new_path.exists() or not old_path.exists():
            report.skip(name, "檔案不存在")
            continue
        new_cols = list(pq.read_schema(new_path).names)
        old_cols = list(pq.read_schema(old_path).names)
        problems = []
        if len(new_cols) != expect_cols:
            problems.append(f"欄數 {len(new_cols)}，應為 {expect_cols}")
        if set(new_cols) != set(old_cols):
            problems.append("欄位集合與舊 repo 不同")
        if problems:
            report.fail(name, "；".join(problems))
            continue
        # 380/520 欄 × 3.3M 列一次讀爆記憶體，分批比
        n_bad_total, checked = 0, 0
        value_cols = [c for c in new_cols if c not in KEYS]
        for i in range(0, len(value_cols), 40):
            batch = value_cols[i:i + 40]
            n_bad, _ = _compare_values(_read(new_path, KEYS + batch),
                                       _read(old_path, KEYS + batch), KEYS, TOLERANCE)
            n_bad_total += n_bad
            checked += len(batch)
        (report.fail if n_bad_total else report.ok)(
            name, f"{len(new_cols)} 欄　比對 {checked} 欄"
                  + (f"，{n_bad_total} 欄不符" if n_bad_total else ""))


# 封存 bundle 實際使用的特徵數（從 models/_pre_official_backup/bundle_*.pkl 的
# `cols` 讀出來的，不是推算）。base 兩組換資料源後仍然對得上；v3 三組對不上，
# 原因見下方 check_model_feature_counts 的說明。
BASELINE_FEATURE_COUNTS = {
    "base": 344, "nomkt": 332, "v3": 509, "v3nomkt": 497, "v3nomv": 481,
}


def check_model_feature_counts(report: Report, old_data: Path) -> None:
    """重建後，各模型實際會拿到幾個特徵？跟封存 bundle 的基準對照。

    ⚠️ **v3 三組對不上是預期的，不是 bug。** v3 的 sz/raw 變體選擇依賴
    `data/feature_audit.csv`，那份稽核檔是**資料相依**的（用當時的全市場資料算
    Spearman 相關），換成官方資料源後有 8 個特徵的分類翻轉、另外多出 `_log_xs`
    欄，選出來會是 518/506/490 而不是 509/497/481。

    `spec.py` / `transforms.py` 與舊 repo byte-identical，程式沒被改動 —— 這是
    資料源變更的必然結果。意思是 m3/m4/m5/m8/m9/m10 **無法逐欄重現**封存的模型，
    只能重新訓練出「同一套方法、新資料下的版本」。base 的 m1/m2/m6/m7 不受影響。

    這一項刻意**不**判 FAIL：印出來讓人看見差異，避免有人以為重建成功了。
    """
    print("\n模型特徵數（對照封存 bundle）")
    from engine.models.submodel_config import feature_cols

    vol_file = Path(__file__).resolve().parents[1] / "models" / "config" / "drop_volatility.txt"
    dropped_vol = set(vol_file.read_text().split()) if vol_file.exists() else set()

    # 只檢查現行五個模型用得到的三組。v3nomkt / v3nomv 隨 m4/m5/m9/m10 一起砍了，
    # 基準值留在 BASELINE_FEATURE_COUNTS 供日後復原時對照。
    for parquet, groups in (("features", ("base", "nomkt")),
                            ("features_v3", ("v3",))):
        path = DATA_DIR / f"{parquet}.parquet"
        if not path.exists():
            for g in groups:
                report.skip(f"特徵數 {g}", f"{parquet}.parquet 不存在")
            continue
        all_cols = [c for c in pq.read_schema(path).names if c not in KEYS]
        selected = feature_cols("UP20", all_cols)
        for group in groups:
            cols = list(selected)
            if group != "base" and group != "v3":
                cols = [c for c in cols if not c.startswith("mkt_")]
            if group == "v3nomv":
                cols = [c for c in cols if c not in dropped_vol]
            n, base = len(cols), BASELINE_FEATURE_COUNTS[group]
            if n == base:
                report.ok(f"特徵數 {group}", f"{n}（與封存 bundle 相同）")
            else:
                report.note(f"特徵數 {group}",
                            f"{n}，封存 bundle 是 {base}（差 {n - base:+d}）"
                            + ("　← v3 系列資料相依，預期會漂移，見 doc/EXPERIMENT_STATUS.md"
                               if group.startswith("v3") else "　← ⚠️ base 系列不該漂移，要查"))


def check_labels(report: Report, old_data: Path) -> None:
    print("\nlabel")
    for name, expect_rows in LABEL_FILES.items():
        new = _read(DATA_DIR / f"{name}.parquet")
        old = _read(old_data / f"{name}.parquet")
        if new is None or old is None:
            report.skip(name, "檔案不存在")
            continue
        problems = []
        if len(new) != expect_rows:
            problems.append(f"列數 {len(new):,}，應為 {expect_rows:,}")
        if len(new) != len(old):
            problems.append(f"列數與舊 repo 不同（{len(new):,} vs {len(old):,}）")
        a = new.sort_values(KEYS).reset_index(drop=True)
        b = old.sort_values(KEYS).reset_index(drop=True)
        if list(a.columns) != list(b.columns):
            problems.append("欄位不同")
        elif len(a) == len(b):
            for col in a.columns:
                if col in KEYS:
                    same = bool((a[col].values == b[col].values).all())
                else:
                    same = bool(np.allclose(a[col].astype("float64"), b[col].astype("float64"),
                                            rtol=0, atol=TOLERANCE, equal_nan=True))
                if not same:
                    problems.append(f"{col} 逐列不等")
        (report.fail if problems else report.ok)(name, "；".join(problems) or f"{len(new):,} 列逐列全等")


def check_models(report: Report, old_repo: Path) -> None:
    """RF 有隨機性，**不比 pkl bytes** —— 比 AUC 與 val_sel 門檻曲線。"""
    print("\n模型（比 AUC 與門檻曲線，不比 pkl bytes）")
    old_models = old_repo / "models"
    old_data = old_repo / "data"
    for key in MODEL_KEYS:
        new_pkl = MODEL_DIR / f"bundle_{key}.pkl"
        old_pkl = old_models / f"bundle_{key}.pkl"
        if not new_pkl.exists() or not old_pkl.exists():
            report.skip(key, f"新={'有' if new_pkl.exists() else '無'} 舊={'有' if old_pkl.exists() else '無'}")
            continue
        with open(new_pkl, "rb") as f:
            new_b = pickle.load(f)
        with open(old_pkl, "rb") as f:
            old_b = pickle.load(f)
        problems = []
        if len(new_b.get("cols", [])) != len(old_b.get("cols", [])):
            problems.append(f"特徵數 {len(new_b.get('cols', []))} vs {len(old_b.get('cols', []))}")
        if set(new_b.get("cols", [])) != set(old_b.get("cols", [])):
            problems.append("特徵集合不同")
        aucs = []
        for split in ("val_es", "val_sel", "test", "test2"):
            a = (new_b.get("metrics") or {}).get(f"{split}_auc")
            b = (old_b.get("metrics") or {}).get(f"{split}_auc")
            if a is None or b is None:
                continue
            aucs.append(f"{split} {a:.4f}/{b:.4f}")
            if abs(a - b) > 0.01:
                problems.append(f"{split} AUC 差 {abs(a - b):.4f} > 0.01")
        # 門檻曲線：同一組門檻上的訊號數與勝率
        new_curve = DATA_DIR / f"sigcurve_{key}_val_sel.csv"
        old_curve = old_data / f"sigcurve_{key}_val_sel.csv"
        if new_curve.exists() and old_curve.exists():
            nc, oc = pd.read_csv(new_curve), pd.read_csv(old_curve)
            thr = CHOSEN_THRESHOLDS[key]
            for frame, label in ((nc, "new"), (oc, "old")):
                frame.attrs["label"] = label
            def at(frame):
                above = frame[frame["threshold"] >= thr]
                return None if above.empty else above.iloc[-1]
            rn, ro = at(nc), at(oc)
            if rn is not None and ro is not None:
                aucs.append(f"@{thr} n {int(rn['n'])}/{int(ro['n'])} "
                            f"win {rn['win_rate']:.3f}/{ro['win_rate']:.3f}")
                if abs(rn["win_rate"] - ro["win_rate"]) > 0.05:
                    problems.append(f"門檻 {thr} 勝率差 {abs(rn['win_rate'] - ro['win_rate']):.3f} > 0.05")
        else:
            aucs.append("（缺門檻曲線）")
        (report.fail if problems else report.ok)(key, "；".join(problems) or "　".join(aucs))


CHECKS = {"raw": check_raw, "features": check_features,
          "featcount": check_model_feature_counts, "labels": check_labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD, help="舊 repo 的根目錄（唯讀）")
    parser.add_argument("--only", choices=[*CHECKS, "models"], default=None)
    parser.add_argument("--json", type=Path, default=None, help="另外寫一份 JSON 報告")
    args = parser.parse_args()

    old_repo: Path = args.old
    old_data = old_repo / "data"
    if not old_data.exists():
        raise SystemExit(f"{old_data} 不存在，用 --old 指定舊 repo 根目錄")

    print("=" * 72)
    print(f"新 repo：{DATA_DIR}")
    print(f"舊 repo：{old_data}（唯讀）")
    print("=" * 72)

    report = Report()
    todo = [args.only] if args.only else [*CHECKS, "models"]
    for name in todo:
        if name == "models":
            check_models(report, old_repo)
        else:
            CHECKS[name](report, old_data)

    print("\n" + "=" * 72)
    if report.failures:
        print(f"{FAIL} 有 {len(report.failures)} 項不符：")
        for _, name, detail in report.failures:
            print(f"    {name}: {detail}")
    else:
        notes = sum(1 for r in report.rows if r[0] == NOTE)
        print(f"{OK} 全部相符（跳過 {sum(1 for r in report.rows if r[0] == SKIP)} 項）"
              + (f"，另有 {notes} 項已知差異標 {NOTE}，請看上面的說明" if notes else ""))
    print("=" * 72)

    if args.json:
        args.json.write_text(json.dumps(
            [{"mark": m, "name": n, "detail": d} for m, n, d in report.rows],
            ensure_ascii=False, indent=2), encoding="utf-8")

    raise SystemExit(1 if report.failures else 0)


if __name__ == "__main__":
    main()
