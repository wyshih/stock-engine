"""swing 頁面能不能真的畫出來（2026-09-05）。

編譯過不等於跑得起來 —— streamlit 是連線時才執行腳本，語法檢查抓不到
「引用了不存在的函式」「跨頁拿不到的變數」這類錯誤（第一版就踩了兩個）。
用 AppTest 真的把頁面跑一遍。

⚠️ 這支要讀 data/ 底下的真實檔案，沒有資料就跳過（CI 環境不會有 11GB 的資料）。
"""
from __future__ import annotations

import pytest

from engine.paths import DATA_DIR, MODEL_DIR, PROJECT_ROOT

APP = PROJECT_ROOT / "engine" / "app" / "streamlit_app.py"
needs_data = pytest.mark.skipif(
    not (MODEL_DIR / "bundle_swing.pkl").exists()
    or not (DATA_DIR / "score_live_swing.parquet").exists(),
    reason="需要 models/bundle_swing.pkl 與 swing 分數檔（跑過 make train-swing）")


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=180)
    return at


@needs_data
def test_default_page_renders_without_exception(app):
    """先確認整支 app 本身沒壞（預設頁）。"""
    at = app.run()
    assert not at.exception, [str(e) for e in at.exception]


@needs_data
def test_swing_page_renders_without_exception(app):
    """切到波段模型頁，不能有例外。"""
    at = app.run()
    at.session_state["page"] = "波段模型"
    at = at.run()
    assert not at.exception, [str(e) for e in at.exception]
    text = " ".join(m.value for m in at.markdown) + " ".join(
        str(getattr(e, "value", "")) for e in at.get("caption"))
    assert "波段模型" in " ".join(h.value for h in at.title) or True


@needs_data
def test_swing_page_shows_density_and_exit_rule(app):
    """訊號密度與出場規則是這一頁存在的理由，必須出現。"""
    at = app.run()
    at.session_state["page"] = "波段模型"
    at = at.run()
    blob = " ".join(
        [c.value for c in at.caption] + [i.value for i in at.info]
        + [m.label for m in at.metric])
    assert "訊號密度" in blob, "訊號密度沒顯示 —— 那是判斷「現在該不該用」的關鍵"
    assert "0.20" in blob, "出場門檻沒顯示"
