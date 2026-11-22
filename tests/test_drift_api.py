"""漂移监测 API 端到端测试：参考矩阵录入/校验、告警随登记更正撤回重算、
新批次清空、快照一致性。"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.storage import store

REF = [
    [0.80, 0.13, 0.07],
    [0.12, 0.78, 0.10],
    [0.05, 0.15, 0.80],
]


@pytest.fixture(autouse=True)
async def _reset_store():
    async with store.lock:
        store.reset_locked([])
    yield
    async with store.lock:
        store.reset_locked([])


def _csv(text: str) -> bytes:
    return text.encode("utf-8")


async def _upload(client: AsyncClient, rows: list[tuple[str, str]]) -> dict:
    lines = ["机器编号,初判"] + [f"{mid},{ini}" for mid, ini in rows]
    content = _csv("\n".join(lines) + "\n")
    r = await client.post(
        "/api/batch", files={"file": ("b.csv", content, "text/csv")}
    )
    assert r.status_code == 200
    return r.json()


async def test_reference_validation_keeps_current_matrix_and_alert() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        await _upload(client, [(f"M{i}", "control") for i in range(1, 7)])
        # 未录入矩阵时 drift.configured 为 False
        snap = (await client.get("/api/batch")).json()
        assert snap["drift"]["configured"] is False
        assert snap["drift"]["alert"] is None

        # 无效矩阵：非正数、行和错误、形状错误 -> 400
        bad_matrices = [
            [[0, 0.3, 0.7], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]],
            [[0.8, 0.13, 0.08], [0.12, 0.78, 0.1], [0.05, 0.15, 0.8]],
            [[1, 0], [0, 1], [0, 0]],
        ]
        for bad in bad_matrices:
            r = await client.post("/api/drift/reference", json={"matrix": bad})
            assert r.status_code == 400
        r = await client.post("/api/drift/reference", json={"nope": REF})
        assert r.status_code == 400

        # 录入有效矩阵
        r = await client.post("/api/drift/reference", json={"matrix": REF})
        assert r.status_code == 200
        assert r.json()["drift"]["configured"] is True

        # 制造一个过线告警：power 实检 + control 初判连登（参考概率仅 0.07）
        alert_id = None
        for i in range(1, 7):
            r = await client.post(
                f"/api/machines/M{i}/inspect", json={"category": "power"}
            )
            assert r.status_code == 200
            a = r.json()["drift"]["alert"]
            if a is not None and alert_id is None:
                alert_id = a["machine_id"]
        assert alert_id is not None
        snap = (await client.get("/api/batch")).json()
        assert snap["drift"]["alert"]["machine_id"] == alert_id

        # 再提交无效矩阵：400，矩阵与告警原样保留
        r = await client.post(
            "/api/drift/reference",
            json={"matrix": [[0.0, 1.0, 0.0], [0.2, 0.7, 0.1],
                             [0.1, 0.1, 0.8]]},
        )
        assert r.status_code == 400
        snap = (await client.get("/api/batch")).json()
        assert snap["drift"]["configured"] is True
        assert snap["drift"]["alert"]["machine_id"] == alert_id


async def test_alert_lifecycle_correct_withdraw_reregister() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        await _upload(client, [(f"M{i}", "control") for i in range(1, 5)])
        await client.post("/api/drift/reference", json={"matrix": REF})

        # 三台 power/control：LR 依次 4.76、34.01、291.55，M3 首个过线。
        for i in range(1, 4):
            r = await client.post(
                f"/api/machines/M{i}/inspect", json={"category": "power"}
            )
        first_alert = r.json()["drift"]["alert"]
        assert first_alert is not None and first_alert["machine_id"] == "M3"

        # 快照一致性：机器清单与告警/顺序来自同一份记录
        snap = (await client.get("/api/batch")).json()
        order = snap["drift"]["order"]
        inspected_ids = [
            m["id"] for m in snap["machines"] if m["inspected"] is not None
        ]
        assert order == inspected_ids == ["M1", "M2", "M3"]

        # 更正首个过线机器 M3 为 control（保留原位置）：power 类只剩两台，
        # 重算最大 LR=34.01 < 60 -> 无过线点，告警撤下。
        r = await client.post(
            "/api/machines/M3/inspect", json={"category": "control"}
        )
        assert r.json()["drift"]["order"][2] == "M3"  # 原位置保留
        assert r.json()["drift"]["alert"] is None
        assert r.json()["drift"]["order"] == ["M1", "M2", "M3"]

        # 撤回 M3：记录移除，仍无过线点
        r = await client.post("/api/machines/M3/withdraw")
        assert r.json()["drift"]["order"] == ["M1", "M2"]
        assert r.json()["drift"]["alert"] is None

        # 再登记 M3：排到末尾；power 类重新累乘至过线，首个过线点落在
        # 末尾的 M3（顺序 #3）。
        r = await client.post(
            "/api/machines/M3/inspect", json={"category": "power"}
        )
        drift = r.json()["drift"]
        assert drift["order"] == ["M1", "M2", "M3"]
        assert drift["alert"] is not None
        assert drift["alert"]["machine_id"] == "M3"
        assert drift["alert"]["order"] == 3


async def test_new_batch_clears_monitoring_but_needs_new_matrix() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        await _upload(client, [(f"M{i}", "control") for i in range(1, 5)])
        await client.post("/api/drift/reference", json={"matrix": REF})
        await client.post("/api/machines/M1/inspect", json={"category": "power"})
        snap = (await client.get("/api/batch")).json()
        assert snap["drift"]["configured"] is True
        assert snap["drift"]["order"] == ["M1"]

        # 上传新批次：监测（参考矩阵、登记序列、告警）全部清空
        await _upload(client, [("A1", "power"), ("A2", "drive")])
        snap = (await client.get("/api/batch")).json()
        assert snap["drift"]["configured"] is False
        assert snap["drift"]["order"] == []
        assert snap["drift"]["alert"] is None
        assert all(v == 1.0 for v in snap["drift"]["likelihood_ratios"].values())
        assert [m["id"] for m in snap["machines"]] == ["A1", "A2"]


async def test_reference_endpoint_requires_batch() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        r = await client.post("/api/drift/reference", json={"matrix": REF})
        assert r.status_code == 400
        # 非 JSON 请求体
        r = await client.post(
            "/api/drift/reference",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400


async def test_stale_snapshot_cannot_revive_alert() -> None:
    """告警完全由当前记录重算得到：旧响应里的告警不会被任何接口恢复。"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        await _upload(client, [(f"M{i}", "control") for i in range(1, 7)])
        await client.post("/api/drift/reference", json={"matrix": REF})
        stale = None
        for i in range(1, 6):
            stale = (
                await client.post(
                    f"/api/machines/M{i}/inspect",
                    json={"category": "power"},
                )
            ).json()
        assert stale["drift"]["alert"] is not None
        # 撤回全部记录 -> 服务端无过线点
        for i in range(1, 6):
            await client.post(f"/api/machines/M{i}/withdraw")
        fresh = (await client.get("/api/batch")).json()
        assert fresh["drift"]["alert"] is None
        assert fresh["drift"]["order"] == []
        # 旧快照只是历史数据，服务端状态不受其影响
        again = (await client.get("/api/batch")).json()
        assert again["drift"]["alert"] is None
