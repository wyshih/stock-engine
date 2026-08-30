"""Makefile 的清理契約：重建路徑必須真的重建。

## 這支在防哪一次的回歸

2026-08-24 稽核抓到：`clean-derived` 刪 features / labels / score / sigcurve，
但**不刪** `data/sweep_m*_rf.csv` 與 `models/bundle_*.pkl`。而 `train_all.sh` 的
跳過判斷只看「CSV 列數 == 組合數」與「bundle 檔存在」。

後果：`make rebuild-full && make train` 在特徵重建之後，會全部印
「⏭ 已存在，跳過」，留下用**舊特徵資料**調出來的組態與舊模型 —— 直接抵觸
CLAUDE.md 的最高原則「任何改動都不可以讓這 5 個模型變得無法重建」。

調參結果是衍生檔：它是「用某一份特徵資料調出來的組態」，資料換了就該作廢。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

MAKEFILE = Path("Makefile").read_text()


def _target_body(name: str) -> str:
    """取出某個 target 的內容（到下一個頂層 target 為止）。"""
    match = re.search(rf"^{name}:.*?$(.*?)(?=^\S+:|\Z)", MAKEFILE, re.M | re.S)
    assert match, f"Makefile 裡找不到 {name} target"
    return match.group(1)


class TestCleanDerived:
    def test_removes_sweep_csvs(self):
        """調參結果是衍生檔，特徵重建後必須跟著作廢。"""
        # Act
        body = _target_body("clean-derived")

        # Assert
        assert "data/sweep_m" in body, (
            "clean-derived 沒有刪 data/sweep_m*_rf.csv —— "
            "重建特徵後 train_all.sh 會跳過調參，留下用舊資料調出來的組態")

    def test_does_not_remove_versioned_config(self):
        """engine/models/config/ 底下是人工挑定的產物，刪了就重建不出模型。"""
        # Act
        body = _target_body("clean-derived")
        commands = [ln for ln in body.splitlines()
                    if "rm " in ln and not ln.strip().startswith("@#")]

        # Assert
        for line in commands:
            assert "engine/models/config" not in line, (
                f"clean-derived 會刪到版控的組態檔：{line.strip()}")

    def test_removes_features_and_labels(self):
        """既有行為不能因為這次改動而掉了。"""
        # Act
        body = _target_body("clean-derived")

        # Assert
        for expected in ("features.parquet", "features_v3.parquet",
                         "labels.parquet", "labels_nobear.parquet"):
            assert expected in body, f"clean-derived 不再刪 {expected}"

    def test_warns_about_bundles(self):
        """不預設刪 bundle（重訓很貴），但必須講清楚後果與出路。"""
        # Act
        body = _target_body("clean-derived")

        # Assert
        assert "clean-models" in body, (
            "clean-derived 沒有提示 make clean-models —— "
            "使用者會以為 rebuild-full 之後 make train 真的會重訓")


class TestCurve:
    """門檻曲線的重跑成本。

    2026-08-26：`make curve` 舊版無條件重跑，而 `train_all.sh` 在訓練每個模型時
    **已經產出曲線了**（第 100 行就有跳過判斷）。所以 P1 重建流程跑完 train 再跑
    curve，等於把五條曲線重算一遍 —— 實測白花約 30 分鐘，結果完全相同。
    """

    def test_skips_existing_curves(self):
        # Act
        body = _target_body("curve")

        # Assert
        assert "sigcurve_" in body and "-f " in body, (
            "make curve 沒有檢查曲線是否已存在 —— 會無條件重跑，"
            "而 train_all.sh 已經產過一次了")

    def test_force_flag_exists(self):
        """改了出場規則或換了分數檔時要能強制重算。"""
        # Act
        body = _target_body("curve")

        # Assert
        assert "FORCE" in body, "少了強制重算的出路"

    def test_still_reminds_human_to_pick(self):
        """規則 7：門檻由人挑。這個提示不能因為加跳過而掉了。"""
        # Act
        body = _target_body("curve")

        # Assert
        assert "CHOSEN_THRESHOLDS" in body


class TestCleanModels:
    def test_target_exists_and_removes_bundles(self):
        # Act
        body = _target_body("clean-models")

        # Assert
        assert "models/bundle_" in body
        assert "sigcurve_" in body, "刪了模型卻留著它的門檻曲線，會對不起來"


class TestUpdateCoversEveryModel:
    """`make update` 必須把每個模型要用的東西都更新到。

    2026-08-29 稽核抓到兩個漏登記：`features-v3` 不在 update 裡，於是 m3/m8 靜默
    停在舊日期（score_recent 只會說「已是最新」，不報錯）；`labels_mdd10` 不在
    labels 裡。這類漏登記不會有任何錯誤訊息，只能靠測試擋。
    """

    def _target(self, name: str) -> str:
        import re
        text = MAKEFILE
        m = re.search(rf"^{name}:.*?$\n((?:\t.*\n|\n)*)", text, re.M)
        assert m, f"Makefile 沒有 {name} target"
        return m.group(0)

    def test_update_builds_every_feature_file_models_depend_on(self):
        from engine.backtest.summary import MODEL_KEYS
        from engine.models.bundle import features_file_for_key

        needed = {features_file_for_key(k) for k in MODEL_KEYS}
        update = self._target("update").splitlines()[0]
        for path in needed:
            if "v3" in path:
                assert "features-v3" in update, (
                    f"有模型用 {path}，但 update 沒有 features-v3 —— "
                    "那些模型會靜默停在舊日期")
            else:
                assert "features" in update

    def test_labels_target_builds_every_label_file(self):
        """每個 build_labels* 模組都要被 labels target 呼叫到。"""
        from pathlib import Path
        modules = sorted(p.stem for p in
                         (Path(__file__).resolve().parents[1] / "engine" / "models")
                         .glob("build_labels*.py"))
        body = self._target("labels")
        missing = [m for m in modules if m not in body]
        assert not missing, f"這些標的沒有被 `make labels` 建到：{missing}"
