"""買賣點規則頁能不能真的畫出來（2026-09-06）。

理由同 test_swing_page_renders.py：編譯過不等於跑得起來，streamlit 是連線時才
執行腳本，語法檢查抓不到「引用了不存在的函式」這類錯誤。
"""
from __future__ import annotations

import pytest

from engine.app.frontend import trade_rule_panel as trp
from engine.paths import PROJECT_ROOT

APP = PROJECT_ROOT / "engine" / "app" / "streamlit_app.py"
PAGE = "每日買賣點"
needs_data = pytest.mark.skipif(
    not trp.REGISTRY_PATH.exists() or not trp.HITLIST_PATH.exists(),
    reason="需要 data/fpm_rules/trade_rules_*（由 fpm 的 target_rule.py 產出）")


@pytest.fixture(scope="module")
def at():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(APP), default_timeout=180)


@needs_data
def test_page_renders_without_exception(at):
    at.run()
    at.session_state["page"] = PAGE
    at.run()
    assert not at.exception, [str(e) for e in at.exception]


@needs_data
def test_page_shows_the_three_daily_sections(at):
    """選定一天要能看到「買進 / 賣出 / 收盤後選出」三段。"""
    at.run()
    at.session_state["page"] = PAGE
    at.run()
    heads = " ".join(s.value for s in at.subheader)
    assert "開盤買進" in heads
    assert "開盤賣出" in heads
    assert "收盤後選出" in heads


@needs_data
def test_page_has_a_date_picker(at):
    at.run()
    at.session_state["page"] = PAGE
    at.run()
    assert any(d.label == "看哪一天" for d in at.date_input)


@needs_data
def test_page_still_states_the_no_stop_loss_risk(at):
    at.run()
    at.session_state["page"] = PAGE
    at.run()
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "停損" in text


def test_page_is_registered_in_page_list():
    src = APP.read_text(encoding="utf-8")
    assert f'"{PAGE}"' in src
    assert f'elif page == "{PAGE}"' in src
