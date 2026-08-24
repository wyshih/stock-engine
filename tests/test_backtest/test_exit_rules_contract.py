"""回測口徑的契約：dedup=False 與出場規則必須來自唯一來源。

## 這支在防哪些回歸

2026-08-23~24 兩天內，這個 repo 發生五次同型問題 —— 改了某個約定，卻沒把所有
依賴它的地方找齊。其中兩次直接關係到回測口徑：

**#4（2026-08-24）** `threshold_curve.py` 呼叫 `simulate()` 沒傳 `dedup`，吃到
預設值 `True`（同股不重複進場）。但 CLAUDE.md 規則 8/9、`bundle.py` 的
`CHOSEN_THRESHOLDS` 註解、`make backtest` 走的 `summary.py` 全部是
`dedup=False`（訊號層級）。「在去重曲線上挑門檻、拿去跑無去重回測」正是
BACKTEST_LOG #24 vs #25 明文警告過的兩把尺。實測落差 20~30 個百分點：
m1 在 10% 訊號率下，去重曲線是勝率 83.9% / +10.70%，訊號層級是 49.5% / −2.63%。

**#5（2026-08-24）** `threshold_curve.py` 的 `EXIT_DEFAULTS` 是一份自己寫死的
副本，且已漂移（`trail_trigger=0.25` / `stop_loss=None` vs 權威的 0.15 / 0.20），
只因為 Makefile 與 train_all.sh 用字面值覆寫才沒出事 —— 那等於再多兩份副本。

稽核者當時的評語：**「同樣的回歸明天再發生一次，測試會全綠。」** 這支就是補那個洞。

## 為什麼用 monkeypatch 攔 kwargs

不跑真的回測（21.9 萬筆訊號要好幾分鐘），只攔截 `simulate` 看它**收到什麼**。
契約是「呼叫端有沒有按規則傳參數」，不是「回測算得對不對」——後者由
`backtest.py` 自己的邏輯負責。
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.backtest.backtest import CURRENT_EXIT_RULES

EXIT_KEYS = ("trail_trigger", "trail_pct", "stop_loss", "take_profit", "stop_ma")


class _Spy:
    """記錄 simulate 收到的 kwargs，回傳空交易表讓呼叫端能跑完。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        trades = pd.DataFrame(columns=["score", "return", "date", "stock_id"])
        price = pd.DataFrame(columns=["date", "stock_id", "close"])
        return trades, price


class TestThresholdCurve:
    """挑門檻的曲線 —— 必須與 make backtest 同一把尺。"""

    def test_passes_dedup_false(self, monkeypatch, tmp_path):
        # Arrange
        from engine.models import threshold_curve as tc
        spy = _Spy()
        monkeypatch.setattr(tc, "simulate", spy)

        # Act
        tc.backtest_trades("m1_base_up20", "val_sel", 0.0, dict(CURRENT_EXIT_RULES))

        # Assert
        assert spy.calls, "simulate 沒有被呼叫"
        assert spy.calls[0].get("dedup") is False, (
            "門檻曲線必須用 dedup=False（訊號層級）—— 與 make backtest 同口徑。"
            "見 BACKTEST_LOG #24 vs #25")

    def test_default_exit_rules_are_the_single_source(self):
        """EXIT_DEFAULTS 不得是自己寫死的副本。"""
        # Arrange / Act
        from engine.models.threshold_curve import EXIT_DEFAULTS

        # Assert
        assert EXIT_DEFAULTS == CURRENT_EXIT_RULES, (
            "threshold_curve 的出場預設值與 backtest.CURRENT_EXIT_RULES 不同 —— "
            "2026-08-24 就是這樣漂移到 trail_trigger=0.25 / stop_loss=None 的")

    def test_cli_defaults_come_from_exit_defaults(self):
        """argparse 的 default 也不能寫字面值。"""
        # Arrange
        import re
        from pathlib import Path
        source = Path("engine/models/threshold_curve.py").read_text()

        # Act：找 add_argument 裡對出場參數寫死數字的 default
        offenders = re.findall(
            r'add_argument\("--(?:stop-loss|trail-trigger|trail-pct|take-profit|stop-ma)"'
            r'[^)]*?default=(?!EXIT_DEFAULTS)([0-9][^,)\s]*)', source, re.S)

        # Assert
        assert not offenders, f"這些出場參數的 CLI 預設值是字面值：{offenders}"


class TestBacktestSummary:
    """make backtest 走的唯一實作。"""

    def test_run_one_passes_dedup_false_and_exit_rules(self, monkeypatch, tmp_path):
        # Arrange
        from engine.backtest import summary as sm
        spy = _Spy()
        monkeypatch.setattr("engine.backtest.backtest.simulate", spy)
        score_file = tmp_path / "score.parquet"
        pd.DataFrame({"date": [], "stock_id": [], "score": []}).to_parquet(score_file)

        # Act
        sm.run_one("m1_base_up20", score_file, 0.6, "absolute")

        # Assert
        assert spy.calls, "simulate 沒有被呼叫"
        kwargs = spy.calls[0]
        assert kwargs.get("dedup") is False, "回測必須用 dedup=False（CLAUDE.md 規則 8）"
        for key in EXIT_KEYS:
            assert kwargs.get(key) == CURRENT_EXIT_RULES[key], (
                f"出場參數 {key} 是 {kwargs.get(key)}，"
                f"應為 CURRENT_EXIT_RULES 的 {CURRENT_EXIT_RULES[key]}")


class TestNoLiteralCopies:
    """出場規則的字面值不得散落在 Makefile / shell 腳本裡。"""

    @pytest.mark.parametrize("path", ["Makefile", "engine/models/train_all.sh"])
    def test_no_hardcoded_exit_params(self, path):
        # Arrange
        from pathlib import Path
        source = Path(path).read_text()

        # Act：註解行不算（那是在解釋為什麼不要寫字面值）
        code = "\n".join(line for line in source.splitlines()
                         if not line.strip().startswith("#")
                         and not line.strip().startswith("@#"))

        # Assert
        for flag in ("--trail-trigger", "--trail-pct", "--stop-loss", "--take-profit"):
            assert flag not in code, (
                f"{path} 用字面值覆寫 {flag} —— 那是第三、第四份會漂移的副本。"
                f"預設值已經是 CURRENT_EXIT_RULES，不需要傳")
