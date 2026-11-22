"""批次、拆检记录与漂移监测状态的内存存储。

演示环境使用单进程内存存储，所有读写经过同一把异步锁，保证两个浏览器
窗口同时登记/更正/撤回（即便操作同一台机器）时两笔修改都基于同一份
拆检记录顺序生效，实检数、估数与漂移告警始终来自同一快照。

登记顺序（漂移监测的时序依据）
------------------------------
``_order`` 保存已登记（未撤回）机器编号的有序序列：

* 首次登记：追加到末尾；
* 更正：只改实检类别，编号保留在原位置；
* 撤回：从序列移除，该机器回到未检；
* 撤回后再登记：作为新记录追加到末尾。

新批次（``reset_locked``）清空登记序列与参考矩阵，漂移监测从零开始。

注意：该实现与 uvicorn 单 worker 绑定（docker-compose 已固定
``--workers 1``）；多进程部署需替换为数据库等共享存储。
"""
from __future__ import annotations

import asyncio

from .drift import compute_drift, validate_reference_matrix
from .estimation import CATEGORIES, CAT_INDEX, estimate


class StoreError(Exception):
    """业务校验错误，映射为 HTTP 400。"""


class BatchStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._machines: list[dict] = []
        self._id_index: dict[str, int] = {}
        self._order: list[str] = []
        self._reference_matrix: list[list[float]] | None = None

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def reset_locked(self, machines: list[dict]) -> None:
        """载入新批次（调用方须持锁）：清空登记序列、参考矩阵与告警。"""
        self._machines = machines
        self._id_index = {m["id"]: i for i, m in enumerate(machines)}
        self._order = []
        self._reference_matrix = None

    def _ordered_records_locked(self) -> list[dict]:
        """按登记顺序返回已登记机器的记录引用。"""
        return [self._machines[self._id_index[mid]] for mid in self._order]

    def snapshot_locked(self) -> dict:
        """基于同一份拆检记录生成机器清单、估算结果与漂移监测（持锁调用）。"""
        result = estimate(self._machines)
        drift = compute_drift(
            self._ordered_records_locked(), self._reference_matrix
        )
        return {
            "machines": [dict(m) for m in self._machines],
            "estimate": result.to_payload(),
            "drift": drift.to_payload(),
        }

    def update_inspection_locked(
        self, machine_id: str, category: str | None
    ) -> dict:
        """登记/更正（category 为三类之一）或撤回（None）一台机器。"""
        if machine_id not in self._id_index:
            raise StoreError(f"机器编号 {machine_id} 不存在于当前批次")
        if category is not None and category not in CAT_INDEX:
            raise StoreError(
                f"非法实检类别 {category!r}，允许：{', '.join(CATEGORIES)}"
            )
        machine = self._machines[self._id_index[machine_id]]
        if category is None:
            # 撤回：移除登记记录，该机器重新计入未检。
            machine["inspected"] = None
            try:
                self._order.remove(machine_id)
            except ValueError:
                pass
        else:
            if machine_id not in self._order:
                # 首次登记、或撤回后的再登记：排到序列末尾。
                self._order.append(machine_id)
            # 已在序列中则为更正：保留原位置，仅替换实检类别。
            machine["inspected"] = category
        return self.snapshot_locked()

    def set_reference_matrix_locked(self, raw_matrix: object) -> dict:
        """校验并更换参考矩阵；非法输入抛错且不改动当前矩阵与告警。

        校验（可能抛 ``DriftError``）发生在任何状态写入之前。
        """
        matrix = validate_reference_matrix(raw_matrix)
        self._reference_matrix = matrix
        # 换一份有效参考矩阵：基于当前本批登记记录整体重算。
        return self.snapshot_locked()

    def is_empty_locked(self) -> bool:
        return not self._machines


store = BatchStore()
