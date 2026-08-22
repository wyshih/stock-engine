#!/bin/bash
# 5 種特徵集 × 2 種 label = 10 個模型，序列訓練（2026-08-16 使用者指定）。
#
# 特徵集：
#   base      data/features.parquet      344 欄（原本的）
#   nomkt     同上但剔除 mkt_*           332 欄
#   v3        data/features_v3.parquet   509 欄（跨年可比的轉換版）
#   v3nomkt   v3 剔除 mkt_*
#   v3nomv    v3 剔除 mkt_* 與波動度家族 16 欄
#
# ⚠️ v3 沒有「移除」波動度，是把它們換成自身歷史 z-score（natr_14 → natr_14_sz），
#    問的是「相對自己是否異常」而非「絕對值大小」。第 5 組才是真的完全不看波動度。
#
# 調參：非 v3 的兩組沿用 Round 4 搜出的最佳參數（那次就是在這 344 欄上搜的）；
#      v3 三組各自重搜（509 欄，max_features 的最佳值必然不同），
#      分別寫 sweep_round5/6/7_rf.csv。
#
# label：
#   up20      label_up20     原本的（未來20日上漲天數>=10）
#   nobear    label_nobear   同上但「目前空頭排列」的列整列排除（不標 0）
#
# 一次一個，前一個沒跑完不會開始下一個 —— 10 核機器，RF 內層吃 6 核，並行只會更慢。
# 產出已存在就跳過，中斷後直接重跑本腳本即可續跑。
#
# 用法：
#   nohup ./engine/models/train_all.sh > ~/Library/Logs/stock_10.log 2>&1 &
#   tail -f ~/Library/Logs/stock_10.log

set -u
cd "$(dirname "$0")/../.." || exit 1   # repo 根目錄（engine/）
PY=".venv/bin/python"
FEAT_BASE=data/features.parquet
FEAT_V3=data/features_v3.parquet
VOLFILE=engine/models/config/drop_volatility.txt
SWEEP4=engine/models/config/sweep_round4_rf.csv
AUDIT=data/feature_audit.csv
VOLPROXY=data/volproxy.csv

# ── 階段〇：前處理（2026-08-22 補上）────────────────────────────────────────
# 這三樣原本是在 scratchpad 臨時算的，session 一結束就沒了，導致這十個模型
# 在換掉資料源之後**完全無法重建**。現在全部有對應的程式，整條流程可重現。
prep() {
    step "前處理：版控組態檔"
    # 這兩個是人工挑定、程式推導不出來的產物（見 engine/models/config/README.md）。
    # 缺任何一個，m5/m10（波動度清單）或 m1/m2/m6/m7（round 4 組態）就訓不出來，
    # 而且會在跑了幾小時之後才炸 —— 所以在這裡先擋。
    for f in "$VOLFILE" "$SWEEP4"; do
        if [ ! -f "$f" ]; then
            fail "缺少版控組態檔 $f —— 這是人工挑定的產物，不是 data/ 的衍生檔，
      應該跟著 repo 一起來。請確認沒有被誤刪，說明見 engine/models/config/README.md"
        fi
    done
    echo "  ✓ $VOLFILE"
    echo "  ✓ $SWEEP4"

    step "前處理：稽核檔"
    if [ -f "$AUDIT" ] && [ -f "$VOLPROXY" ]; then
        echo "  ⏭  已存在，跳過"
    else
        $PY -m engine.features.v3.audit || fail "audit"
    fi

    step "前處理：label_nobear"
    if [ -f data/labels_nobear.parquet ]; then
        echo "  ⏭  已存在，跳過"
    else
        $PY -m engine.models.build_labels_nobear || fail "labels_nobear"
    fi

    step "前處理：v3 特徵集"
    if [ -f "$FEAT_V3" ]; then
        echo "  ⏭  已存在，跳過"
    else
        # build_v3_features.py 拒絕寫入 data/（產出必須先落在 scratchpad），
        # 所以先產到暫存再複製進來。
        local tmp="${TMPDIR:-/tmp}/features_v3.parquet"
        $PY -m engine.features.v3.build_v3_features \
            --audit "$AUDIT" --volproxy "$VOLPROXY" --out "$tmp" || fail "v3"
        cp "$tmp" "$FEAT_V3"
    fi
}

step() { echo ""; echo "======== [$(date '+%F %T')] $* ========"; }
fail() { echo "!!!!!! [$(date '+%F %T')] 失敗：$* —— 中止"; exit 1; }

# $1=key $2=特徵檔 $3=label檔 $4=label欄 $5=顯示名 $6...=額外參數
train() {
    local key=$1 feats=$2 lfile=$3 lcol=$4 desc=$5; shift 5
    local cfg=4
    if [ "${1:-}" = "--cfg" ]; then cfg=$2; shift 2; fi
    step "${key}  ${desc}"
    if [ -f "models/bundle_${key}.pkl" ]; then
        echo "  ⏭  已存在，跳過訓練"
    else
        $PY -m engine.models.train_label_variant \
            --key "$key" --features "$feats" --label-file "$lfile" --label-col "$lcol" \
            --desc "$desc" --round 4 --config-round "$cfg" "$@" || fail "train $key"
    fi
    if [ -f "data/sigcurve_${key}_val_sel.csv" ]; then
        echo "  ⏭  門檻曲線已存在，跳過"
    else
        $PY -m engine.models.threshold_curve --tag "$key" --split val_sel \
            --trail-trigger 0.15 --trail-pct 0.10 --stop-loss 0.20 || fail "curve $key"
        cp "data/threshold_curve_${key}_val_sel.csv" "data/sigcurve_${key}_val_sel.csv"
    fi
}

echo "10 個模型序列訓練開始 $(date '+%F %T')　PID=$$"

prep


# ── 階段一：v3 三種特徵集各自調參（每組 8 個組態，n_estimators=150）────────
sweep() {   # $1=round $2=名稱 $3...=剔除參數
    local rnd=$1 name=$2; shift 2
    step "調參 round${rnd}  ${name}"
    if [ -f "data/sweep_round${rnd}_rf.csv" ] && \
       [ "$($PY -c "import pandas;print(len(pandas.read_csv('data/sweep_round${rnd}_rf.csv')))" 2>/dev/null)" = "8" ]; then
        echo "  ⏭  已完成 8 組，跳過"
    else
        $PY -m engine.models.sweep_round1 --model rf --round "$rnd" \
            --features "$FEAT_V3" "$@" || fail "sweep round${rnd}"
    fi
}
sweep 5 "v3"            
sweep 6 "v3去大盤"       --drop-prefix mkt_
sweep 7 "v3去大盤去波動"  --drop-prefix mkt_ --drop-file "$VOLFILE"

# ── 階段二：10 個模型 ────────────────────────────────────────────────────
# ── label = label_up20（原本的）──────────────────────────────────────────
train m1_base_up20    "$FEAT_BASE" data/labels.parquet label_up20 "①原特徵·上漲天數"
train m2_nomkt_up20   "$FEAT_BASE" data/labels.parquet label_up20 "②去大盤·上漲天數" \
      --drop-prefix mkt_
train m3_v3_up20      "$FEAT_V3"   data/labels.parquet label_up20 "③v3·上漲天數" --cfg 5
train m4_v3nomkt_up20 "$FEAT_V3"   data/labels.parquet label_up20 "④v3去大盤·上漲天數" --cfg 6 \
      --drop-prefix mkt_
train m5_v3nomv_up20  "$FEAT_V3"   data/labels.parquet label_up20 "⑤v3去大盤去波動·上漲天數" --cfg 7 \
      --drop-prefix mkt_ --drop-file "$VOLFILE"

# ── label = label_nobear（排除空頭排列）──────────────────────────────────
train m6_base_nobear    "$FEAT_BASE" data/labels_nobear.parquet label_nobear "⑥原特徵·去空頭"
train m7_nomkt_nobear   "$FEAT_BASE" data/labels_nobear.parquet label_nobear "⑦去大盤·去空頭" \
      --drop-prefix mkt_
train m8_v3_nobear      "$FEAT_V3"   data/labels_nobear.parquet label_nobear "⑧v3·去空頭" --cfg 5
train m9_v3nomkt_nobear "$FEAT_V3"   data/labels_nobear.parquet label_nobear "⑨v3去大盤·去空頭" --cfg 6 \
      --drop-prefix mkt_
train m10_v3nomv_nobear "$FEAT_V3"   data/labels_nobear.parquet label_nobear "⑩v3去大盤去波動·去空頭" --cfg 7 \
      --drop-prefix mkt_ --drop-file "$VOLFILE"

echo ""
echo "======== [$(date '+%F %T')] 10 個模型全部完成 ========"
ls -1 models/bundle_m*.pkl | sed 's|models/bundle_||;s|\.pkl||' | sed 's/^/  /'
