"""門檻刻度的契約測試。

曲線本身整條匯出、不抽樣；這裡釘住的是「挑定的門檻必須落在前端滑桿的刻度上」，
以及 engine 與 dashboard 兩邊的刻度定義不能各自飄掉。
"""
from __future__ import annotations

from pathlib import Path

from engine.export.build_public_bundle import (
    SLIDER_GRID,
    SLIDER_MAX,
    SLIDER_MIN,
    SLIDER_STEP,
)
from engine.models.bundle import CHOSEN_THRESHOLDS


def test_grid_is_51_ticks_of_one_percent():
    assert SLIDER_GRID[0] == SLIDER_MIN == 0.50
    assert SLIDER_GRID[-1] == SLIDER_MAX == 1.00
    assert len(SLIDER_GRID) == 51
    assert all(round(b - a, 10) == SLIDER_STEP
               for a, b in zip(SLIDER_GRID, SLIDER_GRID[1:]))


def test_every_chosen_threshold_lands_on_a_slider_tick():
    """門檻不在刻度上，前端就拉不到它 —— m6 的 0.625 就是這樣改成 0.62 的。"""
    off = {k: v for k, v in CHOSEN_THRESHOLDS.items() if v not in SLIDER_GRID}
    assert not off, f"這些門檻不在滑桿刻度上：{off}"


def test_dashboard_slider_constants_match_export_grid():
    """兩份程式各自寫一份刻度，飄掉的話滑桿會查不到值 —— 這裡釘住。

    比對實際數值而不是字面字串：`"SLIDER_MIN = 0.5"` 是 `"...= 0.55"` 的子字串，
    用 `in` 比對的話刻度飄掉了測試還是綠的（2026-08-27 稽核抓到的假護欄）。
    """
    import ast

    app = (Path(__file__).resolve().parents[2]
           / "dashboard" / "streamlit_app.py").read_text()
    consts = {t.id: ast.literal_eval(node.value)
              for node in ast.parse(app).body if isinstance(node, ast.Assign)
              for t in node.targets
              if isinstance(t, ast.Name) and t.id.startswith("SLIDER_")}
    for name, value in (("SLIDER_MIN", SLIDER_MIN),
                        ("SLIDER_MAX", SLIDER_MAX),
                        ("SLIDER_STEP", SLIDER_STEP)):
        assert name in consts, f"dashboard 沒有定義 {name}"
        assert consts[name] == value, (
            f"dashboard 的 {name} = {consts[name]}，engine 是 {value}")


def test_export_does_not_sample_the_curve():
    """資料包必須是整條曲線 —— 使用者明確要求不抽樣。

    改成 gzip 之後仍然是「整條」：壓的是原始位元組，解開逐位元組相同。
    這裡釘住「沒有抽樣邏輯」與「壓縮後有驗回」兩件事。
    """
    src = Path(__file__).resolve().parents[1] / "engine" / "export" / "build_public_bundle.py"
    code = src.read_text()
    assert "resample" not in code, "匯出程式又出現抽樣邏輯"
    assert "gzip.compress(raw" in code, "曲線必須整條壓縮，不是重寫內容"
    assert "gzip.decompress(dst.read_bytes()) != raw" in code, "壓完必須驗回原檔"


def test_gzip_roundtrip_is_byte_identical():
    """壓縮不是取樣 —— 解開必須逐位元組等於原檔。"""
    import gzip
    original = b"threshold,n,win_rate\n" + b"".join(
        f"{0.5 + i * 1e-6:.6f},{i},0.55\n".encode() for i in range(5000))
    assert gzip.decompress(gzip.compress(original, compresslevel=6)) == original
