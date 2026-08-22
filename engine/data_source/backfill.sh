#!/bin/bash
# 可續跑的長區間回填：把區間切成月段，一段一段推進，完成的段記在標記檔裡。
#
# 為什麼需要：bootstrap 要逐交易日打官方端點 1,994 天（約 3.5 小時），而各
# fetcher 的重試只有 3 次 × 指數退避（合計約 15 秒）。一次 DNS 瞬斷就會讓整支
# 程式 raise、make 中止、前面幾小時的進度雖然有寫檔卻不會自動接續
# —— 2026-08-22 實際發生：抓到第 33/1994 天（2019-02-14）斷在 tpex 的 DNS。
#
# 為什麼是「分段 + 標記檔」而不是「從資料檔的最後一天接著抓」：
# `fetch_chip_tpex` 與 `fetch_chip` 會 upsert 進**同一份** chip.parquet。用資料檔
# 當進度的話，TWSE 那輪跑完後，上櫃那輪的續跑點會被算成「今天」，整段上櫃資料
# 被靜默跳過、還回報成功。標記檔只記「這個模組完成了哪些段」，不受此影響。
#
# 各 fetcher 都是 upsert，同一段重跑無害。
#
# 用法：backfill.sh <模組> <起日> <迄日> [額外參數...]
#   backfill.sh engine.data_source.fetch_price_official 2019-01-01 2026-08-22
#   backfill.sh engine.data_source.fetch_chip_tpex      2019-01-01 2026-08-22

set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
CHUNK_MONTHS=${CHUNK_MONTHS:-3}      # 一段幾個月：失敗時重跑的成本上限
MAX_RETRY=${MAX_RETRY:-10}           # 單一段最多重試幾次
RETRY_WAIT=${RETRY_WAIT:-120}        # 每次重試前等幾秒

MODULE=$1; START=$2; END=$3; shift 3
MARK_DIR=logs/.backfill
MARK="$MARK_DIR/$(echo "$MODULE" | tr '.' '_').done"
mkdir -p "$MARK_DIR"; touch "$MARK"

# 產生月段：[起日, 迄日] → 一行一段 "YYYY-MM-DD YYYY-MM-DD"
chunks=$($PY - "$START" "$END" "$CHUNK_MONTHS" <<'PY'
import sys
import pandas as pd
start, end, months = pd.Timestamp(sys.argv[1]), pd.Timestamp(sys.argv[2]), int(sys.argv[3])
cur = start
while cur <= end:
    nxt = min(cur + pd.DateOffset(months=months) - pd.Timedelta(days=1), end)
    print(f"{cur.date()} {nxt.date()}")
    cur = nxt + pd.Timedelta(days=1)
PY
)

total=$(echo "$chunks" | wc -l | tr -d ' ')
idx=0
while read -r c_start c_end; do
    idx=$((idx + 1))
    tag="$c_start~$c_end"
    if grep -qxF "$tag" "$MARK"; then
        echo "  ⏭  [$idx/$total] $tag 已完成，跳過"
        continue
    fi
    ok=0
    for attempt in $(seq 1 "$MAX_RETRY"); do
        echo "  ▶ [$idx/$total] ${tag}（第 $attempt 次）"
        if $PY -m "$MODULE" --start "$c_start" --end "$c_end" "$@"; then
            echo "$tag" >> "$MARK"; ok=1; break
        fi
        echo "  ✗ $tag 中斷，${RETRY_WAIT}s 後重試這一段"
        sleep "$RETRY_WAIT"
    done
    [ "$ok" -eq 1 ] || { echo "!!!!!! $MODULE 的 $tag 連續 $MAX_RETRY 次失敗，放棄"; exit 1; }
done <<< "$chunks"

echo "  ✓ $MODULE 全部 $total 段完成"
