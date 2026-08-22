"""建 v3 特徵集：讀 data/features.parquet，寫出 features_v3.parquet。

只讀不寫 data/ —— 輸出路徑由 --out 指定，預設不指向 repo 內任何既有目錄。

用法：
    .venv/bin/python code/features_v3/build_v3_features.py \
        --audit <feature_audit.csv> --volproxy <volproxy_old.csv> \
        --out <features_v3.parquet>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


from engine.features.v3.spec import (  # noqa: E402
    SUFFIX_SELF_Z,
    SUFFIX_SELF_Z_XS,
    SUFFIX_XS,
    FeaturePlan,
    build_plans,
    plans_to_frame,
)
from engine.features.v3.transforms import (  # noqa: E402
    assert_within_group_ascending,
    cross_section_rank,
    self_zscore,
)
from engine.models.submodel_config import feature_cols  # noqa: E402
from engine.paths import PROJECT_ROOT as REPO  # noqa: E402
DEFAULT_SOURCE = REPO / "data" / "features.parquet"
KEY_COLS = ["date", "stock_id"]
# 一次處理幾個原始欄位。3.4M 列 × float64，20 欄約 550MB，加上轉換後的暫存
# 峰值約 1.5GB —— 在 16GB 機器上安全。
COLUMN_BATCH = 20
OUT_DTYPE = "float32"


def load_keys(source: Path) -> pd.DataFrame:
    keys = pq.read_table(source, columns=KEY_COLS).to_pandas()
    assert_within_group_ascending(keys["date"], keys["stock_id"])
    return keys


def transform_batch(
    source: Path, plans: list[FeaturePlan], keys: pd.DataFrame
) -> dict[str, pd.Series]:
    """讀一批原始欄位，套用各自的計畫，回傳 {輸出欄名: series}。"""
    names = [p.name for p in plans]
    table = pq.read_table(source, columns=names)
    date, sid = keys["date"], keys["stock_id"]
    out: dict[str, pd.Series] = {}

    for plan in plans:
        raw = table.column(plan.name).to_pandas().astype("float64")
        if plan.keep_raw:
            out[plan.name] = raw
        if plan.make_self_z or plan.make_self_z_xs:
            sz = self_zscore(raw, sid)
            if plan.make_self_z:
                out[plan.name + SUFFIX_SELF_Z] = sz
            if plan.make_self_z_xs:
                out[plan.name + SUFFIX_SELF_Z_XS] = cross_section_rank(sz, date)
        if plan.make_xs:
            out[plan.name + SUFFIX_XS] = cross_section_rank(raw, date)

    del table
    return out


def build(source: Path, out_path: Path, audit: Path, volproxy: Path) -> pd.DataFrame:
    all_cols = pq.ParquetFile(source).schema_arrow.names
    selected = set(feature_cols("UP20", [c for c in all_cols if c not in KEY_COLS]))

    plans = [p for p in build_plans(audit, volproxy) if p.name in selected]
    missing = selected - {p.name for p in plans}
    if missing:
        raise ValueError(f"稽核檔缺少 {len(missing)} 個 UP20 特徵，無法決定處理方式: {sorted(missing)[:10]}")

    plan_frame = plans_to_frame(plans)
    print(f"來源 {len(selected)} 欄 → 計畫產出 {int(plan_frame['n_out'].sum())} 欄")

    keys = load_keys(source)
    arrays: dict[str, pa.Array] = {
        "date": pa.array(keys["date"]),
        "stock_id": pa.array(keys["stock_id"]),
    }

    n_batches = -(-len(plans) // COLUMN_BATCH)
    for i in range(0, len(plans), COLUMN_BATCH):
        batch = plans[i : i + COLUMN_BATCH]
        produced = transform_batch(source, batch, keys)
        for name, series in produced.items():
            arrays[name] = pa.array(series.astype(OUT_DTYPE).to_numpy(), type=pa.float32())
        del produced
        print(f"  batch {i // COLUMN_BATCH + 1}/{n_batches} → 累計 {len(arrays) - 2} 欄", flush=True)

    table = pa.table(arrays)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")
    print(f"\n寫出 {out_path}  ({table.num_rows:,} 列 × {table.num_columns} 欄)")
    return plan_frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--volproxy", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan-out", type=Path, default=None)
    args = ap.parse_args()

    if args.out.resolve().is_relative_to(REPO / "data"):
        raise SystemExit("拒絕寫入 data/：v3 產出必須留在 scratchpad")

    plan_frame = build(args.source, args.out, args.audit, args.volproxy)
    if args.plan_out:
        plan_frame.to_csv(args.plan_out)
        print(f"計畫表寫出 {args.plan_out}")


if __name__ == "__main__":
    main()
