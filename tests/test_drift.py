"""漂移监测核心逻辑测试：加一平滑、累计似然比、首次过线与重算语义。"""
from __future__ import annotations

import json
import math

import pytest

from app.drift import (
    THRESHOLD,
    DriftError,
    compute_drift,
    validate_reference_matrix,
)
from app.estimation import CATEGORIES, CAT_INDEX

# 行=实检，列=初判，每行和为 1、每项为正。
REF = [
    [0.80, 0.13, 0.07],
    [0.12, 0.78, 0.10],
    [0.05, 0.15, 0.80],
]


def _rec(mid: str, initial: str, inspected: str) -> dict:
    return {"id": mid, "initial": initial, "inspected": inspected}


def test_validate_reference_matrix_rules() -> None:
    good = validate_reference_matrix([row[:] for row in REF])
    assert good == REF

    # 非 3×3
    with pytest.raises(DriftError):
        validate_reference_matrix([[1, 0, 0], [0, 1, 0]])
    with pytest.raises(DriftError):
        validate_reference_matrix([[1, 0], [0, 1], [0, 0]])
    # 含非正数
    with pytest.raises(DriftError):
        validate_reference_matrix(
            [[1.0, 0.0, 0.0], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]]
        )
    with pytest.raises(DriftError):
        validate_reference_matrix(
            [[-0.1, 0.6, 0.5], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]]
        )
    # 行和不为 1
    with pytest.raises(DriftError):
        validate_reference_matrix(
            [[0.8, 0.13, 0.08], [0.12, 0.78, 0.1], [0.05, 0.15, 0.8]]
        )
    # 非数值 / 布尔
    with pytest.raises(DriftError):
        validate_reference_matrix(
            [["0.8", 0.13, 0.07], [0.12, 0.78, 0.1], [0.05, 0.15, 0.8]]
        )
    with pytest.raises(DriftError):
        validate_reference_matrix(
            [[True, 0.13, 0.07], [0.12, 0.78, 0.1], [0.05, 0.15, 0.8]]
        )
    # 行和在容差内可通过
    tol_rows = [
        [0.8 + 1e-10, 0.13, 0.07 - 1e-10],
        [0.12, 0.78, 0.10],
        [0.05, 0.15, 0.80],
    ]
    assert validate_reference_matrix(tol_rows) is not None


def test_lr_starts_at_one_and_matches_manual_add_one_smoothing() -> None:
    # 两台 power 实检：先初判 power，再初判 control，手工核对乘积。
    records = [
        _rec("M1", "power", "power"),
        _rec("M2", "control", "power"),
        _rec("M3", "drive", "drive"),
    ]
    res = compute_drift(records, REF)
    # M1：此前 power 类 0 个样本，平滑概率 1/3
    step1 = (1.0 / 3.0) / REF[CAT_INDEX["power"]][CAT_INDEX["power"]]
    # M2：此前 power 类 1 个（power 列计数 1、control 列 0、总数 1）
    # control 列平滑概率 = (1+0)/(3+1) = 0.25
    step2 = 0.25 / REF[CAT_INDEX["power"]][CAT_INDEX["control"]]
    # M3：drive 类首样本，初判 drive -> (1/3)/0.78
    step3 = (1.0 / 3.0) / REF[CAT_INDEX["drive"]][CAT_INDEX["drive"]]
    assert res.likelihood_ratios["power"] == pytest.approx(step1 * step2)
    assert res.likelihood_ratios["drive"] == pytest.approx(step3)
    assert res.likelihood_ratios["control"] == 1.0
    assert res.alert is None
    assert res.order_ids == ["M1", "M2", "M3"]
    assert res.counts["power"] == {"power": 1, "drive": 0, "control": 1}


def test_smoothing_uses_only_prior_samples_of_same_category() -> None:
    # drive 类的登记不应影响 power 类的平滑分母/分子。
    records = [
        _rec("D1", "drive", "drive"),
        _rec("D2", "control", "drive"),
        _rec("P1", "control", "power"),  # power 类首样本：仍为 1/3
    ]
    res = compute_drift(records, REF)
    expected = (1.0 / 3.0) / REF[CAT_INDEX["power"]][CAT_INDEX["control"]]
    assert res.likelihood_ratios["power"] == pytest.approx(expected)


def test_first_crossing_machine_alert_and_sticky_after_drop() -> None:
    # 连续登记 power 实检、初判 control 的机器：参考概率仅 0.07，
    # 第 1 台 LR=4.76，第 2 台 34.01，第 3 台 291.55 -> M3 首个过线。
    records = [_rec(f"M{i}", "control", "power") for i in range(1, 6)]
    res = compute_drift(records, REF)
    assert res.alert is not None
    assert res.alert.order == 3
    assert res.alert.machine_id == "M3"
    assert res.alert.category == "power"
    assert res.alert.lr == pytest.approx(291.54518950437307)

    # 追加多台 power 实检、初判 power 的机器：参考概率 0.8 很高，
    # 平滑预测概率 <=1，每台比值 < 1，当前 LR 被拉到 60 以下；
    # 告警仍保留首个过线点（黏性，回落不撤销）。
    dropped = records + [
        _rec(f"B{i}", "power", "power") for i in range(1, 8)
    ]
    res2 = compute_drift(dropped, REF)
    assert res2.likelihood_ratios["power"] < THRESHOLD
    assert res2.alert is not None
    assert res2.alert.machine_id == "M3"
    assert res2.alert.order == 3


def test_no_alert_when_all_ratios_below_threshold() -> None:
    # 完全贴合参考分布的少量样本，LR 在 1 附近波动，不会过线。
    records = [
        _rec("M1", "power", "power"),
        _rec("M2", "drive", "drive"),
        _rec("M3", "control", "control"),
    ]
    res = compute_drift(records, REF)
    assert res.alert is None
    assert all(v < THRESHOLD for v in res.likelihood_ratios.values())


def test_correction_keeps_position_and_recomputes_alert() -> None:
    # M1..M3 均为 power 实检/control 初判，M3 首个过线（291.55）。
    records = [_rec(f"M{i}", "control", "power") for i in range(1, 4)]
    before = compute_drift(records, REF)
    assert before.alert is not None and before.alert.machine_id == "M3"

    # 更正 M3（保留第 3 个位置）为 control 实检：power 类只剩 2 条，
    # 重算后最大 LR 为 34.01（M2 处）-> 无过线点，告警撤下。
    corrected = [dict(r) for r in records]
    corrected[2]["inspected"] = "control"
    after = compute_drift(corrected, REF)
    assert after.order_ids == ["M1", "M2", "M3"]  # 位置不变
    assert after.alert is None
    assert after.likelihood_ratios["power"] == pytest.approx(34.01360544217687)
    # M3 现在计入 control 类（首样本，初判 control）。
    assert after.counts["control"]["control"] == 1


def test_withdraw_removes_and_reregister_appends_then_alert_recomputed() -> None:
    # 三台 power/control：LR 依次 4.76、34.01、291.55，M3 过线。
    records = [_rec(f"M{i}", "control", "power") for i in range(1, 4)]
    before = compute_drift(records, REF)
    assert before.alert is not None and before.alert.machine_id == "M3"

    # 撤回 M2：序列只剩 M1、M3 两台，重算 LR = 4.76 * 7.14 = 34.01 < 60。
    withdrawn = [dict(r) for r in records if r["id"] != "M2"]
    mid = compute_drift(withdrawn, REF)
    assert mid.order_ids == ["M1", "M3"]
    assert mid.alert is None
    assert mid.likelihood_ratios["power"] == pytest.approx(34.01360544217687)

    # 再登记 M2：排到末尾。此时第 3 条 power 记录（即末尾的 M2）才过线，
    # 首个过线点随之变为末尾的 M2（顺序 #3）。
    reappended = withdrawn + [_rec("M2", "control", "power")]
    after = compute_drift(reappended, REF)
    assert after.order_ids == ["M1", "M3", "M2"]
    assert after.alert is not None
    assert after.alert.machine_id == "M2"
    assert after.alert.order == 3


def test_withdraw_then_reregister_moves_first_crossing_to_end() -> None:
    """撤回的恰是首个过线机器：再登记后它排末尾并在末尾重新过线。"""
    records = [_rec(f"M{i}", "control", "power") for i in range(1, 4)]
    # 撤回 M3（首个过线点），剩余 M1、M2 的 LR=34.01 不过线
    withdrawn = [dict(r) for r in records if r["id"] != "M3"]
    assert compute_drift(withdrawn, REF).alert is None
    # 再登记 M3 到末尾：第 3 台再次过线，告警指向末尾的 M3
    after = compute_drift(withdrawn + [_rec("M3", "control", "power")], REF)
    assert after.order_ids == ["M1", "M2", "M3"]
    assert after.alert is not None
    assert after.alert.machine_id == "M3" and after.alert.order == 3


def test_change_reference_matrix_recomputes_batch() -> None:
    records = [_rec(f"M{i}", "control", "power") for i in range(1, 5)]
    assert compute_drift(records, REF).alert is not None
    # 换成让 control 初判在 power 行概率很高的矩阵（0.9），比值不再过线。
    ref2 = [
        [0.05, 0.05, 0.90],
        [0.12, 0.78, 0.10],
        [0.05, 0.15, 0.80],
    ]
    res2 = compute_drift(records, ref2)
    assert res2.alert is None
    assert res2.reference_matrix == ref2


def test_without_reference_matrix_no_lr_no_alert_but_order_kept() -> None:
    records = [_rec("M1", "control", "power"), _rec("M2", "power", "power")]
    res = compute_drift(records, None)
    assert res.reference_matrix is None
    assert res.alert is None
    assert res.likelihood_ratios == {c: 1.0 for c in CATEGORIES}
    assert res.order_ids == ["M1", "M2"]
    assert res.counts["power"]["control"] == 1


def test_payload_shape_json_safe() -> None:
    res = compute_drift([_rec("M1", "control", "power")], REF)
    payload = res.to_payload()
    assert payload["configured"] is True
    assert payload["threshold"] == THRESHOLD
    assert payload["alert"] is None
    assert set(payload["likelihood_ratios"]) == set(CATEGORIES)
    assert payload["sample_counts"]["power"]["control"] == 1
    json.dumps(payload)  # 全部为 JSON 可序列化的基础类型
    assert math.isfinite(payload["likelihood_ratios"]["power"])
