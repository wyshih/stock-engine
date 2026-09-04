"""型態規則篩選頁：瀏覽 fpm 專案（平盤起漲點型態探勘）挖出的規則，在樣本外測試期
的歷史命中買點，可依規則過濾、點選任一筆跳到走勢圖。

資料來源只讀不寫：`data/fpm_rules/{rules_registry.yaml, rules_hitlist_oos.csv}`，
由 fpm 專案（獨立於本 repo）產出後複製進來，不在這裡重新計算或訓練。
規則清單是純資料，之後要新增規則只要在 rules_registry.yaml 加一筆、
rules_hitlist_oos.csv 補上對應買點，這頁不用改一行程式碼。

只顯示樣本外（測試期）買點：fpm 那邊產出 rules_hitlist_oos.csv 時就已經只收
walk-forward 每個視窗的測試期資料，這裡不用再篩一次訓練/測試。

跳轉走勢圖刻意借用「個股歷史預測」而不是「技術面分析」（2026-09-04 討論過）：
後者的「分析日期」是只看到當天為止的截止線，看不到訊號日之後股價有沒有真的漲；
前者雖然是給模型分數用的頁面，但視窗設計是訊號日前後各 40 個交易日，才看得到
「進場後到底怎樣」這個我們真正要驗證的東西。畫面上會多一條不相關的模型分數線，
是刻意接受的代價，不是忘了改。
"""
from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st
import yaml

from engine.paths import DATA_DIR

RULES_DIR = DATA_DIR / "fpm_rules"
REGISTRY_PATH = RULES_DIR / "rules_registry.yaml"
HITLIST_PATH = RULES_DIR / "rules_hitlist_oos.csv"


@st.cache_data(ttl=3600)
def _load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or []


@st.cache_data(ttl=3600)
def _load_hitlist() -> pd.DataFrame:
    if not HITLIST_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(HITLIST_PATH, parse_dates=["date"])
    df["stock_id"] = df["stock_id"].astype(str)
    return df


def render(load_stock_list: Callable[[], pd.DataFrame]) -> None:
    st.title("🧩 型態規則篩選")
    st.caption("fpm 專案挖出的「平盤起漲點」規則，以下只顯示樣本外（walk-forward 測試期）"
               "的歷史命中買點——規則沒看過這些資料，才能當驗證用。")

    registry = _load_registry()
    hitlist = _load_hitlist()
    if not registry or hitlist.empty:
        st.warning(f"找不到規則資料，預期在 `{RULES_DIR}`（rules_registry.yaml / "
                   f"rules_hitlist_oos.csv）。請先從 fpm 專案的 outputs/ 複製過來。")
        st.stop()

    name_by_id = {r["id"]: r["name"] for r in registry}
    stats_by_id = {r["id"]: r["stats"] for r in registry}

    st.subheader("規則過濾")
    selected_ids = st.multiselect(
        "選規則（可複選，預設全選）",
        options=list(name_by_id),
        default=list(name_by_id),
        format_func=lambda rid: f"{rid}：{name_by_id[rid][:40]}",
    )
    if not selected_ids:
        st.info("至少選一條規則")
        st.stop()

    levels = sorted(hitlist["validation_level"].unique()) if "validation_level" in hitlist.columns else []
    selected_levels = st.multiselect(
        "驗證等級（可複選，預設全選）", options=levels, default=levels,
        help="「window顯著性驗證過」：規則在該筆所屬的 walk-forward 視窗有通過統計檢定。"
             "「直接套規則」：2026 年資料因為沒有任何規則通過 window 4 的檢定，"
             "是拿定案的規則直接套用算出來的，嚴謹度較低，不是同一個等級。",
    ) if levels else []

    with st.expander("規則統計（勝率 / lift / 樣本外期望報酬）", expanded=False):
        stats_df = pd.DataFrame([
            {"規則": rid, "名稱": name_by_id[rid], **stats_by_id[rid]}
            for rid in selected_ids
        ])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    filtered = hitlist[hitlist["rule_id"].isin(selected_ids)]
    if selected_levels:
        filtered = filtered[filtered["validation_level"].isin(selected_levels)]
    filtered = filtered.sort_values("date", ascending=False)

    st.subheader("依買點聚合（同一天同一檔股票，符合了幾條規則）")
    st.caption("同時符合越多規則的買點，樣本外表現通常越好——這是型態疊加的訊號，"
               "不是單一規則各自獨立的訊號。")
    agg = filtered.groupby(["date", "stock_id"]).agg(
        符合規則數=("rule_id", "nunique"),
        規則清單=("rule_id", lambda s: ", ".join(sorted(s))),
        r_end=("r_end", "first"),
        label=("label", "first"),
    ).reset_index().sort_values(["符合規則數", "date"], ascending=[False, False])
    agg_display = agg.rename(columns={
        "date": "進場決策日", "stock_id": "股票", "r_end": "20日後超額報酬", "label": "是否起漲",
    })
    st.dataframe(
        agg_display.head(500).style.format({"20日後超額報酬": "{:.2%}"}),
        use_container_width=True, hide_index=True, height=300,
    )

    st.subheader(f"歷史命中買點（樣本外，共 {len(filtered)} 筆，每條規則各一列）")

    sl = load_stock_list()
    filtered = filtered.copy()
    if not sl.empty:
        name_map = dict(zip(sl["stock_id"], sl["stock_name"]))
        filtered["stock_name"] = filtered["stock_id"].map(name_map).fillna("")
    else:
        filtered["stock_name"] = ""

    show_cols = ["date", "stock_id", "stock_name", "rule_id", "r_end", "mdd", "label", "validation_level"]
    display_df = filtered[show_cols].rename(columns={
        "date": "進場決策日", "stock_id": "股票", "stock_name": "名稱",
        "rule_id": "規則", "r_end": "20日後超額報酬", "mdd": "期間最大回檔", "label": "是否起漲",
        "validation_level": "驗證等級",
    })
    st.dataframe(
        display_df.style.format({"20日後超額報酬": "{:.2%}", "期間最大回檔": "{:.2%}"}),
        use_container_width=True, hide_index=True, height=400,
    )

    st.subheader("跳轉走勢圖")
    if filtered.empty:
        return
    options = list(filtered.index)
    pick_idx = st.selectbox(
        "選一筆看走勢圖（可打字搜尋股票代號）",
        options,
        format_func=lambda i: (
            f"{filtered.loc[i,'date'].date()}　{filtered.loc[i,'stock_id']} "
            f"{filtered.loc[i,'stock_name']}　{filtered.loc[i,'rule_id']}　"
            f"20日後{filtered.loc[i,'r_end']:+.1%}"
        ),
    )
    if st.button("跳轉 →", key="rule_screener_jump"):
        row = filtered.loc[pick_idx]
        st.session_state["hist_jump"] = (row["stock_id"], row["date"].date())
        st.session_state["page"] = "個股歷史預測"
        st.rerun()
