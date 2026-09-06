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
def test_default_source_is_the_swing_model(at):
    """使用者要的是「這頁看 swing 模型的買賣點」，模型必須是預設來源。"""
    at.run()
    at.session_state["page"] = PAGE
    at.run()
    sel = at.selectbox(key="trade_rule_select")
    assert sel.value == "swing"
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "模型分數" in text


@needs_data
def test_swing_source_exposes_both_thresholds(at):
    at.run()
    at.session_state["page"] = PAGE
    at.run()
    labels = [s.label for s in at.slider]
    assert any("買進門檻" in l for l in labels)
    assert any("賣出門檻" in l for l in labels)


@needs_data
def test_rule_source_still_states_the_no_stop_loss_risk(at):
    """切到規則來源時，「不設停損」的風險一定要出現在畫面上。"""
    at.run()
    at.session_state["page"] = PAGE
    at.run()
    at.selectbox(key="trade_rule_select").set_value("target_01").run()
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "停損" in text


def test_page_is_registered_in_page_list():
    src = APP.read_text(encoding="utf-8")
    assert f'"{PAGE}"' in src
    assert f'elif page == "{PAGE}"' in src


@needs_data
def test_all_view_lists_every_trade(at):
    """「全部」檢視要看得到整份紀錄與績效表，不是只有單日。"""
    at.run()
    at.session_state["page"] = PAGE
    at.run()
    at.radio(key="trade_rule_view").set_value("全部").run()
    assert not at.exception, [str(e) for e in at.exception]

    text = " ".join(c.value for c in at.caption)
    assert "所有買賣紀錄" in text
    assert "共" in text and "筆" in text


@needs_data
def test_view_switch_offers_both_modes(at):
    at.run()
    at.session_state["page"] = PAGE
    at.run()
    assert list(at.radio(key="trade_rule_view").options) == ["單日", "全部"]
