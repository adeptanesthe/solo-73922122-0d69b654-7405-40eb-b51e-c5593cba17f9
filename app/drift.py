"""漂移监测核心逻辑（纯函数，无外部依赖）。

业务口径
========
维修班组需要知道**哪一类实检故障最先出现偏离出厂校准的证据**。

* 出厂校准用一张 3×3 参考混淆概率矩阵 ``R`` 描述：行=实检类别，
  列=初判类别，每项概率均为正，每行和为 1。
* 对每个实检类别，按登记顺序单独维护一条累计似然比（LR），初值为 1。
* 每登记一台（实检类别 ``r``、初判类别 ``c``），只更新 ``r`` 类的 LR，
  乘上「本台初判的加一平滑预测概率」与「参考概率」之比：

      LR_r <- LR_r * smooth(r, c) / R[r][c]

  其中加一平滑只使用该实检类别**此前**已经登记的样本（不含本台）：

      smooth(r, c) = (1 + count_before(r, c)) / (3 + n_before(r))

* 告警：记录**首个**使任一类别 LR 达到或超过阈值 60 的机器编号与实检
  类别；之后即便该比值回落到 60 以下，告警仍保留（只认首次过线点）。
* 更正、撤回、再登记、更换参考矩阵、新批次等场景下，全部 LR 与首次
  过线点都基于当前记录**重算**：没有过线点就撤下告警，因此迟到的旧
  快照不可能恢复已经失效的告警。

注意：登记顺序由调用方（存储层）维护——更正保留原位置，撤回移除记录，
再登记排到末尾；本模块只接收有序的记录序列做纯计算。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .estimation import CAT_INDEX, CATEGORIES, CATEGORY_LABELS

THRESHOLD = 60.0
ROW_SUM_TOL = 1e-8


class DriftError(ValueError):
    """参考矩阵输入非法，映射为 HTTP 400。"""


@dataclass
class DriftAlert:
    """首次过线告警（只要曾经过线即存在，回落后不自动撤销）。"""

    machine_id: str
    category: str
    lr: float
    order: int  # 该记录在登记序列中的位置（从 1 开始）

    def to_payload(self) -> dict:
        return {
            "machine_id": self.machine_id,
            "category": self.category,
            "category_label": CATEGORY_LABELS[self.category],
            "lr": self.lr,
            "order": self.order,
            "threshold": THRESHOLD,
        }


@dataclass
class DriftResult:
    """一次漂移监测重算的完整结果。"""

    reference_matrix: list[list[float]] | None
    likelihood_ratios: dict[str, float]
    counts: dict[str, dict[str, int]]
    order_ids: list[str]
    alert: DriftAlert | None
    threshold: float = THRESHOLD

    def to_payload(self) -> dict:
        return {
            "configured": self.reference_matrix is not None,
            "reference_matrix": self.reference_matrix,
            "categories": list(CATEGORIES),
            "labels": dict(CATEGORY_LABELS),
            "threshold": self.threshold,
            "likelihood_ratios": dict(self.likelihood_ratios),
            "sample_counts": {
                cat: dict(row) for cat, row in self.counts.items()
            },
            "order": list(self.order_ids),
            "alert": self.alert.to_payload() if self.alert else None,
        }


def validate_reference_matrix(matrix: object) -> list[list[float]]:
    """校验 3×3 参考混淆概率矩阵，返回规整后的矩阵。

    规则：必须是 3×3 数值；每项概率均为正；每行和为 1（容差
    ``ROW_SUM_TOL``）。非法时抛 :class:`DriftError`，调用方据此保留
    当前矩阵与告警、不做任何状态变更。
    """
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise DriftError("参考矩阵必须是 3×3 嵌套数组（3 行，每行 3 个概率）")
    cleaned: list[list[float]] = []
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != 3:
            raise DriftError(
                f"参考矩阵第 {i + 1} 行必须包含 3 个概率值"
            )
        cleaned_row: list[float] = []
        for j, value in enumerate(row):
            # 拒绝 bool（bool 是 int 的子类）与非数值类型。
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DriftError(
                    f"参考矩阵第 {i + 1} 行第 {j + 1} 列必须是数值"
                )
            fv = float(value)
            if fv <= 0.0:
                raise DriftError(
                    "参考矩阵每项概率均须为正数，"
                    f"第 {i + 1} 行第 {j + 1} 列为 {fv:g}"
                )
            cleaned_row.append(fv)
        row_sum = sum(cleaned_row)
        if abs(row_sum - 1.0) > ROW_SUM_TOL:
            raise DriftError(
                f"参考矩阵第 {i + 1} 行和为 {row_sum:.6f}，"
                "每行概率之和必须为 1"
            )
        cleaned.append(cleaned_row)
    return cleaned


def _empty_counts() -> dict[str, dict[str, int]]:
    return {cat: {c: 0 for c in CATEGORIES} for cat in CATEGORIES}


def compute_drift(
    records: list[dict],
    reference_matrix: list[list[float]] | None,
) -> DriftResult:
    """按登记顺序重算各类别累计似然比与首次过线告警。

    ``records`` 为有序字典列表，每项含 ``id``、``initial``、
    ``inspected``（实检类别，序列中不应为 None）。``reference_matrix``
    为 None 表示尚未录入有效参考矩阵，此时不计算比值、不产生告警。
    """
    order_ids = [r["id"] for r in records]
    counts = _empty_counts()
    n_before = {c: 0 for c in CATEGORIES}
    lr = {c: 1.0 for c in CATEGORIES}
    alert: DriftAlert | None = None

    if reference_matrix is None:
        # 未配置参考矩阵：仍返回登记顺序与计数，供页面展示，但不算 LR。
        for rec in records:
            r = rec["inspected"]
            if r in counts:
                counts[r][rec["initial"]] += 1
                n_before[r] += 1
        return DriftResult(
            reference_matrix=None,
            likelihood_ratios=dict(lr),
            counts=counts,
            order_ids=order_ids,
            alert=None,
        )

    for order, rec in enumerate(records, start=1):
        r, c = rec["inspected"], rec["initial"]
        if r not in CAT_INDEX or c not in CAT_INDEX:
            continue
        ri, ci = CAT_INDEX[r], CAT_INDEX[c]
        # 加一平滑：只用该实检类别此前登记的样本。
        smoothed = (1.0 + counts[r][c]) / (3.0 + n_before[r])
        lr[r] *= smoothed / reference_matrix[ri][ci]
        # 首个使任一比值达到阈值的记录；一旦锁定不再更改（回落保留）。
        if alert is None and lr[r] >= THRESHOLD:
            alert = DriftAlert(
                machine_id=rec["id"], category=r, lr=lr[r], order=order
            )
        counts[r][c] += 1
        n_before[r] += 1

    return DriftResult(
        reference_matrix=reference_matrix,
        likelihood_ratios=lr,
        counts=counts,
        order_ids=order_ids,
        alert=alert,
    )
