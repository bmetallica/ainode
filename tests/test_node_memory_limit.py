"""A node's memory limit, and what it uses with nothing loaded.

Phase 0 of wizzard.md. The limit caps a node's total use for planning and
admission (E2) — every budget goes through node_budgets, so the launch form,
the admission gate and the gmu ceiling all hold to it. The idle baseline is
what the wizard subtracts from a node's total when it plans a whole profile.
"""

from __future__ import annotations

import json

import pytest

from ainode.core.config import NodeConfig
from ainode.core.units import gb_from_mib


def _node(node_id="n2", limit=0.0, total_mib=122070, used_mib=14 * 1024):
    return type("N", (), {
        "node_id": node_id, "node_name": node_id, "status": "online",
        "gpu_memory_total_mb": total_mib, "gpu_memory_used_mb": used_mib,
        "gpu_memory_gb": 0, "instances": [], "memory_limit_gb": limit,
    })()


def _budgets(*nodes, config=None):
    from ainode.planner.api_routes import node_budgets

    cluster = type("C", (), {"members": lambda self: list(nodes)})()
    app = {"cluster_state": cluster}
    if config is not None:
        app["config"] = config
    return node_budgets(app)


class TestTheLimitCapsWhatIsFree:
    def test_no_limit_is_what_it_was(self):
        (budget,) = _budgets(_node())
        assert budget.free_gb == pytest.approx(gb_from_mib(122070 - 14 * 1024), abs=0.1)
        assert budget.limit_gb == 0.0

    def test_a_limit_leaves_limit_minus_use(self):
        (budget,) = _budgets(_node(limit=100.0))
        assert budget.free_gb == pytest.approx(100.0 - gb_from_mib(14 * 1024), abs=0.1)
        assert budget.limit_gb == 100.0
        # The total stays the machine's: gmu is a fraction of it.
        assert budget.total_gb == pytest.approx(128.0, abs=0.1)

    def test_a_node_over_its_limit_has_nothing_free(self):
        (budget,) = _budgets(_node(limit=20.0, used_mib=30 * 1024))
        assert budget.free_gb == 0.0

    def test_its_own_limit_comes_from_the_config(self):
        config = NodeConfig(node_id="n1", memory_limit_gb=64.0)
        (budget,) = _budgets(_node("n1", limit=0.0), config=config)
        assert budget.limit_gb == 64.0


class TestItIsCarriedBetweenNodes:
    def test_the_announcement_has_it(self):
        from ainode.discovery.broadcast import NodeAnnouncement
        from ainode.discovery.cluster import ClusterNode

        ann = NodeAnnouncement(node_id="n2", node_name="n2", gpu_name="GB10",
                               gpu_memory_gb=128, unified_memory=True, model="",
                               status="online", api_port=8000, web_port=3000,
                               memory_limit_gb=96.0, baseline_used_mb=9000.0)
        again = NodeAnnouncement.from_json(ann.to_json())
        node = ClusterNode.from_announcement(again, "online")
        assert node.memory_limit_gb == 96.0 and node.baseline_used_mb == 9000.0


class TestSettingIt:
    @pytest.mark.asyncio
    async def test_on_this_node(self, tmp_path, monkeypatch):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from ainode.safety.limit_routes import register_limit_routes

        config = NodeConfig(node_id="n1")
        monkeypatch.setattr(config, "save", lambda *a, **k: None)
        app = web.Application()
        app["config"] = config
        register_limit_routes(app)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/nodes/n1/memory-limit", json={"gb": 100})
            assert resp.status == 200
            assert config.memory_limit_gb == 100.0
            resp = await client.post("/api/nodes/n1/memory-limit", json={"gb": 3})
            assert resp.status == 400
            resp = await client.post("/api/nodes/n1/memory-limit", json={"gb": 0})
            assert resp.status == 200 and config.memory_limit_gb == 0.0

    @pytest.mark.asyncio
    async def test_an_unknown_node_is_a_404(self):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from ainode.safety.limit_routes import register_limit_routes

        app = web.Application()
        app["config"] = NodeConfig(node_id="n1")
        app["cluster_state"] = type("C", (), {"members": lambda self: []})()
        register_limit_routes(app)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/nodes/nx/memory-limit", json={"gb": 90})
            assert resp.status == 404


class TestTheIdleBaseline:
    def _app(self, instances=(), embeddings=(), used_mb=9000.0):
        manager = type("M", (), {"instances": lambda self: list(instances)})()
        emb = type("E", (), {"list_loaded": lambda self: list(embeddings)})()
        collector = type("C", (), {"get_gpu_metrics":
                                   lambda self: {"memory_used_mb": used_mb}})()
        return {"instances": manager, "embedding_manager": emb,
                "metrics_collector": collector}

    def test_an_idle_node_is_written_down(self, tmp_path, monkeypatch):
        from ainode.safety import baseline

        monkeypatch.setattr(baseline, "_engine_containers_running", lambda: False)
        record = baseline.IdleBaseline(self._app(), tmp_path / "b.json")
        record.poll(now=1000.0)
        assert record.used_mb() == 9000.0
        assert json.loads((tmp_path / "b.json").read_text())["used_mb"] == 9000.0

    @pytest.mark.parametrize("busy", ["instance", "embedding", "container", "docker?"])
    def test_a_busy_node_is_not(self, tmp_path, monkeypatch, busy):
        from ainode.safety import baseline

        running = {"container": True, "docker?": None}.get(busy, False)
        monkeypatch.setattr(baseline, "_engine_containers_running", lambda: running)
        app = self._app(instances=["x"] if busy == "instance" else (),
                        embeddings=[{"id": "e"}] if busy == "embedding" else ())
        record = baseline.IdleBaseline(app, tmp_path / "b.json")
        record.poll(now=1000.0)
        assert record.used_mb() == 0.0

    def test_it_looks_once_a_minute(self, tmp_path, monkeypatch):
        from ainode.safety import baseline

        calls = []
        monkeypatch.setattr(baseline, "_engine_containers_running",
                            lambda: calls.append(1) or False)
        record = baseline.IdleBaseline(self._app(), tmp_path / "b.json")
        record.poll(now=1000.0)
        record.poll(now=1010.0)
        record.poll(now=1070.0)
        assert len(calls) == 2
