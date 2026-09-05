"""波段九分類 ground truth（2026-09-05 新增）。
輸出：data/labels_swing.parquet

把每檔每天標成「上漲／下跌／平盤」×「起／中／末」共九類，並由此導出二元
標的 `label_swing_up`：上漲-起 記 1、下跌-起 記 0、其餘全部 NA（不進訓練）。

為什麼只留「起」的兩類：九類混在一起訓練，test AUC 只有 0.53~0.55；縮到
「上漲-起 vs 下跌-起」這一刀，同一批特徵的 test AUC 直接跳到 0.75。差別在於
其餘七類（中段、末段、盤整）本來就沒有方向性，混進去只是稀釋訊號。

三層定義：
  1. ZigZag（門檻 THETA=5%）：反向走超過 5% 才確認轉折，底→頂標「上漲」、
     頂→底標「下跌」。
  2. 平盤覆寫：Kaufman 效率比（置中 21 日窗口）
     ER = |c[t+10] - c[t-10]| / Σ|日變動|，ER < 0.15 覆寫成「平盤」——
     走來走去但沒前進。
  3. 起/中/末：連續同狀態區間按**時間**切三等分。

⚠️ 這是**事後標籤**，用到未來資料，只能當 y 不能當特徵。三個地方會產生
「還不知道」，一律標 NA 而不是猜（同 build_labels.py 踩過的坑，
doc/AUDIT_20260728.md §A-1：pandas 的 `NaN >= x` 回傳 False 不是 NaN，
下游 dropna() 攔不到）：
  a. 最後一個 ZigZag 轉折之後的方向未定
  b. 最後一段連續區間尚未結束，切不出起/中/末
  c. 資料尾端 HALF 天沒有置中 ER

用法：python -m engine.models.build_labels_swing [--full]
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 專案路徑一律走 engine/paths.py（唯一來源），不要各檔自行推導
from engine.paths import DATA_DIR  # noqa: E402

THETA = 0.05            # ZigZag 反轉門檻
ER_WINDOW = 21          # Kaufman 效率比的置中窗口（必須是奇數）
ER_HALF = ER_WINDOW // 2
ER_FLAT = 0.15          # 低於此值判為平盤
# ⚠️ 2026-09-05：這個參數曾經設成 10（把短段落併入前一段，讓連續同狀態區間的
# 中位數從 3 天變成 36 天，看起來比較像「波段」）。**那是錯的，已改回 1（關閉）。**
# 實測比對高分訊號的「過去 20 日報酬」：
#   MIN_RUN_DAYS=1  最高分組 -7.98%（2024Q4~2025）  → 買的是剛跌下來的，未來 +2.31%
#   MIN_RUN_DAYS=10 最高分組 +32.56%（2026）        → 買的是已經漲很多的，未來 +1.13%
# 拉長「起」之後，前面 11 天的漲勢已經寫在 return_5d 這類特徵裡，模型改去讀
# 已發生的動能。AUC 從 0.75 升到 0.87，但抓的東西從轉折點換成動能，而且更不穩
# （val_sel 的未來報酬變成反向）。這是 BACKTEST_LOG #31/#32 的同一個教訓：
# label 側統計變好不代表高分訊號變好。
MIN_RUN_DAYS = 1
MIN_BARS = 300          # 歷史太短不切波段
# ⚠️ 2026-09-05：只用 ZigZag 的方向當標的是不夠的。實測 164,900 個上升段裡
# **60.7% 漲不到 2%、75% 漲不到 5%**，中位漲幅只有 +0.7%、中位 3 天；扣掉台股
# 來回成本 0.585% 之後只剩一半是正的。也就是說模型很努力在學「怎麼分辨起漲」，
# 但四分之三的「起漲」根本沒有肉——預測得準，預測的卻是不值錢的事。
# 直覺的修法是把「值得交易」寫進 ground truth（幅度不到門檻就不給標籤），
# **但實測失敗，已關閉（MIN_MOVE = 0）**：設 0.10 之後
#   抓到起漲的倍數 2.80x → 1.55x、好壞比 11.5 → 4.99、
#   未來 20 日分辨力 +2.99/-2.33/+1.65pp → -0.38/-0.87/+1.36pp
# 原因是「會往哪邊走」和「會走多大」是兩種難度，後者難很多；要模型同時猜對
# 兩件事的結果是兩邊都變差，訓練樣本也從 72 萬掉到 22.5 萬。
# 這個常數留著是為了記錄這條走過的路，不要再試一次。
MIN_MOVE = 0.0          # 0 = 不看幅度，只看方向
# 增量模式每次一併重算的尾端交易日數。ZigZag 的轉折確認延遲中位數 6 天、
# P90 21 天，但實測最長到 445 天，取 250 留足夠緩衝（只增不減）。
RECOMPUTE_TAIL_DAYS = 250

STATE_UP, STATE_DOWN, STATE_FLAT = "up", "down", "flat"
PHASE_EARLY, PHASE_MID, PHASE_LATE = "early", "mid", "late"


def _read(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    p = DATA_DIR / f"{name}.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p, columns=columns) if columns else pd.read_parquet(p)


def adjust_factor(dates: np.ndarray, events: pd.DataFrame | None) -> np.ndarray:
    """還原權值係數：事件日**之前**的價格乘上 ratio，多個事件連乘。

    ratio = 事件後參考價 ÷ 事件前收盤價（見 fetch_exright.py:124）。不還原的話
    分割／減資會被 ZigZag 當成崩跌（實測 8422 的 10:1 分割會變成 -90%）。

    ⚠️ exright.parquet 只涵蓋上市（TWSE），上櫃（TPEX）的除權息不在裡面
    （validate_data.py:139）。上櫃股的配息日會被當成小幅下跌，是已知缺口。
    """
    factor = np.ones(len(dates), dtype="float64")
    if events is None or events.empty:
        return factor
    for event_date, ratio in zip(events["date"].to_numpy(), events["ratio"].to_numpy()):
        if np.isfinite(ratio) and ratio > 0:
            factor[dates < event_date] *= ratio
    return factor


def zigzag(close: np.ndarray, theta: float = THETA) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (轉折點索引, 種類)。種類 +1 = 波段底、-1 = 波段頂。

    只有「反向走超過 theta」才確認前一個極值是轉折點，所以每個轉折都是事後
    才成立的；最後一個轉折之後的方向永遠未定，由呼叫端標 NA。
    """
    n = len(close)
    pivot_idx: list[int] = []
    pivot_kind: list[int] = []
    direction = 0          # 0 未定 / 1 上升中 / -1 下降中
    extreme = 0            # 目前這段的極值索引
    for i in range(1, n):
        price = close[i]
        if direction == 1:
            if price >= close[extreme]:
                extreme = i
            elif price <= close[extreme] * (1 - theta):
                pivot_idx.append(extreme); pivot_kind.append(-1)
                direction, extreme = -1, i
        elif direction == -1:
            if price <= close[extreme]:
                extreme = i
            elif price >= close[extreme] * (1 + theta):
                pivot_idx.append(extreme); pivot_kind.append(1)
                direction, extreme = 1, i
        else:
            if price >= close[extreme] * (1 + theta):
                pivot_idx.append(extreme); pivot_kind.append(1)
                direction, extreme = 1, i
            elif price <= close[extreme] * (1 - theta):
                pivot_idx.append(extreme); pivot_kind.append(-1)
                direction, extreme = -1, i
    return np.array(pivot_idx, dtype="int64"), np.array(pivot_kind, dtype="int8")


def efficiency_ratio(close: np.ndarray, window: int = ER_WINDOW) -> np.ndarray:
    """Kaufman 效率比（置中窗口）。接近 1 = 單向趨勢，接近 0 = 原地震盪。

    分母是路徑總長度，不需要估波動度，所以沒有「分母選錯」的問題——先前用
    「區間寬度 ÷ 隨機漫步預期寬度」時，因為分母用全期間波動而讓低波動期整段
    被誤判成盤整（門檻 1.5 就吃掉 75% 的樣本）。
    """
    half = window // 2
    s = pd.Series(close)
    net = (s.shift(-half) - s.shift(half)).abs()
    path = s.diff().abs().rolling(window - 1).sum().shift(-half)
    return (net / path.replace(0, np.nan)).to_numpy()


def _state_array(close: np.ndarray) -> np.ndarray:
    """ZigZag 方向 + 效率比覆寫，回傳每天的狀態（未定為空字串）。"""
    n = len(close)
    state = np.full(n, "", dtype=object)
    pivot_idx, pivot_kind = zigzag(close)
    if len(pivot_idx) < 2:
        return state
    for a in range(len(pivot_idx) - 1):
        lo, hi = pivot_idx[a], pivot_idx[a + 1]
        state[lo:hi + 1] = STATE_UP if pivot_kind[a] == 1 else STATE_DOWN
    # 最後一個轉折之後方向未定 → 留空（呼叫端會標 NA）
    er = efficiency_ratio(close)
    # 效率比算不出來（序列頭尾各 ER_HALF 天）就不知道是不是盤整，一律留空不猜
    state[~np.isfinite(er)] = ""
    flat = np.isfinite(er) & (er < ER_FLAT) & (state != "")
    state[flat] = STATE_FLAT
    return _absorb_short_runs(state)


def _absorb_short_runs(state: np.ndarray, min_len: int = MIN_RUN_DAYS) -> np.ndarray:
    """把短於 min_len 的段落併入前一段（開頭那段沒有前一段，併入後一段）。

    不做這件事的話效率比在門檻附近的抖動會把波段剁碎（見 MIN_RUN_DAYS 的說明）。
    """
    out = state.copy()
    runs = _runs(out)
    changed = True
    while changed:
        changed = False
        runs = _runs(out)
        for k, (lo, hi, val) in enumerate(runs):
            if val == "" or hi - lo + 1 >= min_len:
                continue
            if k > 0 and runs[k - 1][2] != "":
                out[lo:hi + 1] = runs[k - 1][2]
            elif k + 1 < len(runs) and runs[k + 1][2] != "":
                out[lo:hi + 1] = runs[k + 1][2]
            else:
                continue
            changed = True
            break
    return out


def _runs(arr: np.ndarray) -> list[tuple[int, int, object]]:
    """把陣列切成 (起, 迄, 值) 的連續區間清單。"""
    out: list[tuple[int, int, object]] = []
    i = 0
    n = len(arr)
    while i < n:
        j = i
        while j + 1 < n and arr[j + 1] == arr[i]:
            j += 1
        out.append((i, j, arr[i]))
        i = j + 1
    return out


def _phase_array(state: np.ndarray) -> np.ndarray:
    """連續同狀態區間按時間切三等分。最後一段尚未結束 → 留空。"""
    n = len(state)
    phase = np.full(n, "", dtype=object)
    i = 0
    while i < n:
        if state[i] == "":
            i += 1
            continue
        j = i
        while j + 1 < n and state[j + 1] == state[i]:
            j += 1
        is_last_run = (j == n - 1) or all(state[k] == "" for k in range(j + 1, n))
        if not is_last_run:                       # 尚未結束的區間切不出起/中/末
            length = j - i + 1
            for k in range(i, j + 1):
                pos = (k - i) / length
                phase[k] = (PHASE_EARLY if pos < 1 / 3
                            else PHASE_MID if pos < 2 / 3 else PHASE_LATE)
        i = j + 1
    return phase


def run_amplitude(close: np.ndarray, state: np.ndarray) -> np.ndarray:
    """每一天所屬連續區間的整段幅度（該段最後一天 ÷ 第一天 − 1）。

    用來把「漲不到門檻的小波段」擋在標籤外，見 MIN_MOVE 的說明。
    """
    amp = np.full(len(close), np.nan)
    for lo, hi, val in _runs(state):
        if val == "":
            continue
        amp[lo:hi + 1] = close[hi] / close[lo] - 1
    return amp


def label_one_stock(close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """單檔的 (state, phase)。兩者任一為空字串代表「還不知道」。

    ⚠️ 壞值（NaN／非正數）只切斷該處，**不是整檔否決**。舊版用
    `np.isfinite(close).all()` 一票否決，實測 2,071 檔裡有 799 檔因此整檔消失
    （其中 376 檔只有 1~5 根壞值，壞值佔比中位數 0.48%），而被砍掉的正是停牌、
    低流動性那些——那跟結果高度相關，等於在 label 階段就做了選擇偏誤，
    而推論卻是對全市場打分。
    """
    n = len(close)
    state = np.full(n, "", dtype=object)
    phase = np.full(n, "", dtype=object)
    good = np.isfinite(close) & (close > 0)
    for lo, hi, ok in _runs(good.astype(object)):
        if not ok or hi - lo + 1 < MIN_BARS:
            continue
        seg = close[lo:hi + 1]
        seg_state = _state_array(seg)
        seg_phase = _phase_array(seg_state)
        state[lo:hi + 1] = np.where(seg_phase == "", "", seg_state)
        phase[lo:hi + 1] = seg_phase
    return state, phase


def _upsert(df: pd.DataFrame, full: bool = False) -> None:
    """`full=True` 時**整檔覆寫**。

    這一支跟 build_labels.py 不同：未定的列是整列不輸出（不是留列標 NA），所以
    `keep="last"` 的合併蓋不掉「這次算不出來、上次算得出來」的舊列，全量重算之後
    檔案裡會留著程式現在產不出來的標籤。全量模式一律覆寫，不留殘骸。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if full:
        df = (df.drop_duplicates(subset=["date", "stock_id"], keep="last")
                .sort_values(["date", "stock_id"]).reset_index(drop=True))
        df.to_parquet(DATA_DIR / "labels_swing.parquet", index=False, engine="pyarrow")
        logger.info(f"labels_swing.parquet 全量覆寫：{len(df)} 筆")
        return
    existing = _read("labels_swing")
    if not existing.empty:
        orphan = [c for c in existing.columns if c not in df.columns]
        if orphan:
            logger.warning(f"清除孤兒欄位（目前程式碼已不再產生）：{orphan}")
            existing = existing.drop(columns=orphan)
    combined = pd.concat([existing, df], ignore_index=True) if not existing.empty else df
    combined = (combined
                .drop_duplicates(subset=["date", "stock_id"], keep="last")
                .sort_values(["date", "stock_id"])
                .reset_index(drop=True))
    combined.to_parquet(DATA_DIR / "labels_swing.parquet", index=False, engine="pyarrow")
    logger.info(f"labels_swing.parquet 寫入：{len(combined)} 筆")


def run(full: bool = False) -> pd.DataFrame:
    price = _read("price", ["date", "stock_id", "close"])
    if price.empty:
        raise RuntimeError("price.parquet 不存在")
    price["date"] = pd.to_datetime(price["date"])
    price = price.sort_values(["stock_id", "date"]).reset_index(drop=True)

    exright = _read("exright", ["date", "stock_id", "ratio"])
    if not exright.empty:
        exright["date"] = pd.to_datetime(exright["date"])
        events = dict(tuple(exright.dropna(subset=["ratio"]).groupby("stock_id")))
    else:
        # CLAUDE.md 明訂還原權值是必要步驟。缺檔時整份 label 會靜默變成垃圾
        # （8422 的 10:1 分割會被 ZigZag 當成 -90% 崩跌），不能只警告。
        raise RuntimeError("exright.parquet 不存在，無法還原權值；先跑 make data")

    target_dates: set | None = None
    if not full:
        existing = _read("labels_swing")
        if not existing.empty:
            existing["date"] = pd.to_datetime(existing["date"])
            done = set(existing["date"].dt.normalize().unique())
            new = set(price["date"].dt.normalize().unique()) - done
            recent = sorted(done)[-RECOMPUTE_TAIL_DAYS:] if done else []
            target_dates = new | set(recent)
            logger.info(f"增量：新資料 {len(new)} 天 + 重算尾端 {len(recent)} 天")

    frames = []
    for stock_id, g in price.groupby("stock_id", sort=False):
        dates = g["date"].to_numpy()
        close = g["close"].to_numpy("float64") * adjust_factor(dates, events.get(stock_id))
        state, phase = label_one_stock(close)
        frames.append(pd.DataFrame({"date": dates, "stock_id": stock_id,
                                    "swing_state": state, "swing_phase": phase,
                                    "swing_amp": run_amplitude(close, state)}))
    out = pd.concat(frames, ignore_index=True)
    out = out[out["swing_state"] != ""].reset_index(drop=True)

    # 二元標的：上漲-起 = 1、下跌-起 = 0、其餘 NA（可空整數，不能用 0 佔位）
    # 幅度不到 MIN_MOVE 的段落一律 NA —— 那種波段賺不到成本，不該進訓練。
    is_early = out["swing_phase"] == PHASE_EARLY
    big = out["swing_amp"].abs() >= MIN_MOVE
    up = is_early & big & (out["swing_state"] == STATE_UP)
    down = is_early & big & (out["swing_state"] == STATE_DOWN)
    out["label_swing_up"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    out.loc[up, "label_swing_up"] = 1
    out.loc[down, "label_swing_up"] = 0

    if target_dates is not None:
        out = out[out["date"].dt.normalize().isin(target_dates)]

    logger.info(f"labels_swing：{len(out)} 筆")
    lab = out["label_swing_up"].dropna()
    if len(lab):
        logger.info(f"可訓練樣本 {len(lab)} 筆，正例率 {lab.mean():.3f}")
    _upsert(out, full=full)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    df = run(full=args.full)
    if not df.empty:
        print("\n九分類分布：")
        mix = (df["swing_state"] + "-" + df["swing_phase"]).value_counts(normalize=True)
        for k, v in mix.items():
            print(f"  {k:<12} {v * 100:5.2f}%")
