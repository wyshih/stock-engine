#!/bin/bash
# 2 個模型：m1_base_up20（①）與 m1_mdd10。
#
# 兩者**同一份特徵集、同一個搜尋空間，只差標的** —— 差異只能來自標的，
# 不會混進調參的運氣。
#
# 2026-09-03：m1_steady20 / m1_xsrank20 兩輪修 label_up20 崩跌偏差的實驗都已移除
# （績效比 label_up20 差很多，見 doc/BACKTEST_LOG.md #31 / #32）。程式在 git
# 歷史裡（`git show 5c452f1:engine/models/build_labels_xsrank.py` 等），
# 要復原先讀完那兩篇記錄裡「最貴的教訓」再動手。
#
# 特徵集：
#   base   data/features.parquet   344 欄
#
# label：
#   label_up20      labels.parquet           未來 20 日上漲天數 >= 10
#   label_mdd10     labels_mdd10.parquet     同上，再要求期間最低收盤不跌破 −10%
#
# 2026-09-02 移除：m2（去大盤）/ m3、m8（v3 特徵集）/ m6、m8（label_nobear）。
# 連帶 v3 特徵管線（含 audit / volproxy 前處理）與 `build_labels_nobear` 一併刪除。
# m1_mdd10 原本不在本腳本裡（是 2026-08-28 手動跑 train_label_variant 產的），
# 現在補進來 —— 出貨的模型都必須從這條路徑重建得出來（最高原則）。
#
# ── 調參規定（2026-08-23 使用者指定）───────────────────────────────────────
# **每一個模型都要用自己的特徵集、自己的 label 跑一輪超參數搜尋，讀自己那份
#   sweep_{key}_rf.csv。不得共用組態。**
#
# 即使兩個模型的特徵集相同、只差 label，也各搜各的 —— label 換了，最佳組態
# 就不保證一樣，沿用等於拿別的問題調出來的參數。
#
# 搜尋與訓練用同一個樹數（300 棵），不分兩階段 —— 舊做法用 150 棵篩選、再取前幾名
# 試 300/450，但第二階段從來沒跑過，等於拿篩選階段的暫定值當最終組態。
#
# 一次一個，前一個沒跑完不會開始下一個 —— 10 核機器，RF 內層吃 8 核，並行只會更慢。
# 產出已存在就跳過，中斷後直接重跑本腳本即可續跑。
#
# 用法：
#   nohup ./engine/models/train_all.sh > ~/Library/Logs/stock_train.log 2>&1 &
#   tail -f ~/Library/Logs/stock_train.log

set -u
cd "$(dirname "$0")/../.." || exit 1   # repo 根目錄（engine/）
PY=".venv/bin/python"
FEAT_BASE=data/features.parquet
SWEEP4=engine/models/config/sweep_round4_rf.csv

step() { echo ""; echo "======== [$(date '+%F %T')] $* ========"; }
fail() { echo "!!!!!! [$(date '+%F %T')] 失敗：$* —— 中止"; exit 1; }

# ── 階段〇：前處理 ─────────────────────────────────────────────────────────
# v3 特徵集與 label_nobear 的前處理隨那些模型一起移除（2026-09-02）。
# 剩下的只有「版控組態檔在不在」這道檢查 —— 缺了會在跑了幾小時之後才炸。
prep() {
    step "前處理：版控組態檔"
    # 人工挑定、程式推導不出來的產物（見 engine/models/config/README.md）。
    if [ ! -f "$SWEEP4" ]; then
        fail "缺少版控組態檔 $SWEEP4 —— 這是人工挑定的產物，不是 data/ 的衍生檔，
      應該跟著 repo 一起來。請確認沒有被誤刪，說明見 engine/models/config/README.md"
    fi
    echo "  ✓ $SWEEP4"

    step "前處理：label_mdd10"
    if [ -f data/labels_mdd10.parquet ]; then
        echo "  ⏭  已存在，跳過"
    else
        $PY -m engine.models.build_labels_mdd || fail "labels_mdd10"
    fi

}

# $1=key $2=特徵檔 $3=label檔 $4=label欄 $5=顯示名 $6...=額外參數
train() {
    local key=$1 feats=$2 lfile=$3 lcol=$4 desc=$5; shift 5

    step "${key}  ${desc}"
    if [ -f "models/bundle_${key}.pkl" ]; then
        echo "  ⏭  已存在，跳過訓練"
    else
        $PY -m engine.models.train_label_variant \
            --key "$key" --features "$feats" --label-file "$lfile" --label-col "$lcol" \
            --desc "$desc" --round 4 "$@" || fail "train $key"
    fi
    if [ -f "data/sigcurve_${key}_val_sel.csv" ]; then
        echo "  ⏭  門檻曲線已存在，跳過"
    else
        # 出場參數刻意不傳：threshold_curve 的預設值＝backtest.py 的
        # CURRENT_EXIT_RULES（唯一來源）。寫字面值等於多一份會漂移的副本。
        $PY -m engine.models.threshold_curve --tag "$key" --split val_sel || fail "curve $key"
        cp "data/threshold_curve_${key}_val_sel.csv" "data/sigcurve_${key}_val_sel.csv"
    fi
}

echo "2 個模型序列訓練開始 $(date '+%F %T')　PID=$$"

prep


# ── 階段一：每個模型各自調參 ─────────────────────────────────────────────
# 每個模型一份 sweep_{key}_rf.csv，用該模型自己的特徵集與 label。
sweep() {   # $1=key $2=特徵檔 $3=label檔 $4=label欄 $5=說明 $6...=剔除參數
    local key=$1 feats=$2 lfile=$3 lcol=$4 desc=$5; shift 5
    step "調參 ${key}  ${desc}"
    local out="data/sweep_${key}_rf.csv"
    # 組數問程式而不是寫死數字 —— 空間改了這裡就跟著對
    local want
    want=$($PY -c "
from engine.models.sweep_round1 import search_space
import math
space, _ = search_space('rf', 4, '$key')
print(math.prod(len(v) for v in space.values()))" 2>/dev/null)
    if [ -f "$out" ] && \
       [ "$($PY -c "import pandas;print(len(pandas.read_csv('$out')))" 2>/dev/null)" = "$want" ]; then
        echo "  ⏭  已完成 ${want} 組，跳過"
    else
        $PY -m engine.models.sweep_round1 --model rf --round 5 --key "$key" \
            --features "$feats" --label-file "$lfile" --label-col "$lcol" "$@" \
            || fail "sweep $key"
    fi
}

sweep m1_base_up20 "$FEAT_BASE" data/labels.parquet       label_up20  "①原特徵·上漲天數"
sweep m1_mdd10     "$FEAT_BASE" data/labels_mdd10.parquet label_mdd10 "①原特徵·抗套牢"

# ── 階段二：訓練 2 個模型 ─────────────────────────────────────────────────
# 不傳 --config-round，train_label_variant 就會讀該模型自己的
# sweep_{key}_rf.csv（規定：不得共用組態）。
train m1_base_up20 "$FEAT_BASE" data/labels.parquet       label_up20  "①原特徵·上漲天數"
train m1_mdd10     "$FEAT_BASE" data/labels_mdd10.parquet label_mdd10 "①原特徵·抗套牢"
# ⚠️ 暫定：借 m1_base_up20 的組態（max_features=15/max_depth=20/leaf=200，
# val_sel AUC 0.6088）。轉正時拿掉 --config-key，並把階段一那行取消註解。
train m1_steady20  "$FEAT_BASE" data/labels_steady20.parquet label_steady20 "①原特徵·盤整緩漲" \
      --config-key m1_base_up20

# m1_xsrank20（2026-09-03，第二輪）：同樣先借組態看成效。若成效好，依規則
# 各自調參的步驟與轉正 m1_steady20 完全一樣（見上）。
train m1_xsrank20  "$FEAT_BASE" data/labels_xsrank20.parquet label_xsrank20 "①原特徵·橫斷面風險調整" \
      --config-key m1_base_up20

echo ""
echo "======== [$(date '+%F %T')] 2 個模型全部完成 ========"
ls -1 models/bundle_m*.pkl | sed 's|models/bundle_||;s|\.pkl||' | sed 's/^/  /'
