"""tok/s for the launch that served, on the node that runs it; the vLLM
version from the engine.

The export showed nvidia/Qwen3.8-Flash-Next-NVFP4 at 0 tok/s on spark-1432,
the node that led it, and 20.2 on the head's entry — an older launch in which
the head had taken part. Requests are counted where they arrive, and the
head wrote the lifetime average into its own store. The vLLM version was
empty everywhere: the log line naming it had scrolled out of the tail read.
"""

from __future__ import annotations

import pytest
from aiohttp import web

from ainode.core.config import NodeConfig
from ainode.measure.recorder import Recorder
from ainode.measure.store import MeasurementStore


class _Collector:
    def __init__(self):
        self.stats = {}

    def model_stats(self):
        return self.stats


class _Peer:
    def __init__(self, node_id, instances, fabric_ip="10.0.0.2"):
        self.node_id = node_id
        self.instances = instances
        self.fabric_ip = fabric_ip
        self.web_port = 3000


class _Cluster:
    def __init__(self, *nodes):
        self.nodes = list(nodes)

    def members(self):
        return self.nodes


def _flash(load_seconds=623.1, status="serving", head="p1"):
    return {"model": "q/flash", "instance_id": "8000", "status": status,
            "head_node_id": head, "load_seconds": load_seconds, "kind": "llm"}


def _app(tmp_path, *peers):
    store = MeasurementStore(tmp_path / "m.json")
    return {"config": NodeConfig(node_id="head"), "measurement_store": store,
            "metrics_collector": _Collector(), "cluster_state": _Cluster(*peers)}


class TestAPeerLedModel:
    def test_its_speed_goes_to_the_node_that_runs_it(self, tmp_path):
        app = _app(tmp_path, _Peer("p1", [_flash()]), _Peer("p2", [_flash()]))
        store = app["measurement_store"]
        store.record_launch("q/flash", ok=True, memory_gb=107.5)   # the head's old entry
        recorder = Recorder(app)
        app["metrics_collector"].stats = {"q/flash": {"requests": 2, "tokens_generated": 100,
                                                      "avg_latency_ms": 1000.0}}
        recorder._record_speeds()
        app["metrics_collector"].stats = {"q/flash": {"requests": 12,
                                                      "tokens_generated": 100 + 10 * 400,
                                                      "avg_latency_ms": (2000 + 10 * 20000) / 12}}
        recorder._record_speeds()
        assert recorder._outbox == [("p1", {"model": "q/flash", "tokens_per_second": 20.0})]
        # Not written into the head's own entry for it.
        assert store.get("q/flash").tokens_per_second == 0.0

    def test_unchanged_it_is_not_sent_again_within_a_minute(self, tmp_path):
        app = _app(tmp_path, _Peer("p1", [_flash()]))
        recorder = Recorder(app)
        app["metrics_collector"].stats = {"q/flash": {"requests": 0}}
        recorder._record_speeds()
        app["metrics_collector"].stats = {"q/flash": {"requests": 10, "tokens_generated": 4000,
                                                      "avg_latency_ms": 20000.0}}
        recorder._record_speeds()
        recorder._record_speeds()
        assert len(recorder._outbox) == 1

    def test_a_new_launch_counts_from_zero(self, tmp_path):
        peer = _Peer("p1", [_flash()])
        app = _app(tmp_path, peer)
        recorder = Recorder(app)
        app["metrics_collector"].stats = {"q/flash": {"requests": 100, "tokens_generated": 50000,
                                                      "avg_latency_ms": 5000.0}}
        recorder._record_speeds()
        peer.instances = [_flash(status="starting")]             # relaunched…
        recorder._record_speeds()
        peer.instances = [_flash()]                               # …and serving again
        app["metrics_collector"].stats = {"q/flash": {"requests": 104, "tokens_generated": 52000,
                                                      "avg_latency_ms": 5000.0}}
        recorder._record_speeds()
        assert recorder._outbox == []            # counted from the relaunch: 4 < 5

    def test_the_time_since_start_growing_does_not_restart_the_count(self, tmp_path):
        # The bug: the identity carried load_seconds, which on a serving
        # instance grows with every announcement.
        peer = _Peer("p1", [_flash(load_seconds=100.0)])
        app = _app(tmp_path, peer)
        recorder = Recorder(app)
        app["metrics_collector"].stats = {"q/flash": {"requests": 0}}
        recorder._record_speeds()
        peer.instances = [_flash(load_seconds=160.0)]
        app["metrics_collector"].stats = {"q/flash": {"requests": 5, "tokens_generated": 5000,
                                                      "avg_latency_ms": 28000.0}}
        recorder._record_speeds()
        assert recorder._outbox == [("p1", {"model": "q/flash", "tokens_per_second": 35.7})]

    def test_a_member_that_does_not_lead_is_not_the_target(self, tmp_path):
        app = _app(tmp_path, _Peer("p2", [_flash(head="p1")]))
        recorder = Recorder(app)
        assert "q/flash" not in recorder._serving_where()


class TestTheStore:
    def test_a_new_launch_forgets_the_last_ones_speed_and_version(self, tmp_path):
        store = MeasurementStore(tmp_path / "m.json")
        store.record_launch("m", ok=True, memory_gb=10.0)
        store.record_speed("m", tokens_per_second=12.3)
        store.record_engine_version("m", "0.27.1")
        store.record_launch("m", ok=True, memory_gb=10.0)
        entry = store.get("m")
        assert entry.tokens_per_second == 0.0 and entry.engine_version == ""
        # The first launch keeps both in its history.
        assert entry.history[0]["tokens_per_second"] == 12.3
        assert entry.history[0]["engine_version"] == "0.27.1"


class TestTheRoute:
    @pytest.fixture
    def app(self, tmp_path, monkeypatch):
        from ainode.measure.api_routes import register_measurement_routes

        monkeypatch.setattr("ainode.auth.cluster_key.cluster_key", lambda create=False: "k")
        app = web.Application()
        app["measurement_store"] = MeasurementStore(tmp_path / "m.json")
        app["measurement_store"].record_launch("q/flash", ok=True, memory_gb=114.6)
        register_measurement_routes(app)
        return app

    @pytest.mark.asyncio
    async def test_a_node_can_tell_another(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.post("/api/measurements/speed",
                                 json={"model": "q/flash", "tokens_per_second": 20.2},
                                 headers={"X-AINode-Cluster-Key": "k"})
        assert resp.status == 200
        assert app["measurement_store"].get("q/flash").tokens_per_second == 20.2

    @pytest.mark.asyncio
    async def test_a_browser_cannot(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.post("/api/measurements/speed",
                                 json={"model": "q/flash", "tokens_per_second": 999})
        assert resp.status == 403


class TestTheVersion:
    def test_a_launch_without_it_in_the_log_asks_the_engine(self, tmp_path):
        store = MeasurementStore(tmp_path / "m.json")
        config = NodeConfig(node_id="n", model="org/m", api_port=8003)
        backend = type("B", (), {"load_phase": "ready", "config": config,
                                 "logs": lambda self, n: "no version here"})()
        instance = type("I", (), {"backend": backend,
                                  "record": type("R", (), {"model": "org/m", "kind": "llm"})()})()
        app = {"config": config, "measurement_store": store}
        recorder = Recorder(app)
        recorder._record("org/m", instance, "ready", 0.0)
        assert recorder._want_version == {"org/m": 8003}

    @pytest.mark.asyncio
    async def test_flush_writes_what_the_engine_says(self, tmp_path, aiohttp_server):
        async def version(request):
            return web.json_response({"version": "0.27.1"})

        engine = web.Application()
        engine.router.add_get("/version", version)
        server = await aiohttp_server(engine)

        import aiohttp

        store = MeasurementStore(tmp_path / "m.json")
        store.record_launch("org/m", ok=True, memory_gb=10.0)
        async with aiohttp.ClientSession() as session:
            recorder = Recorder({"measurement_store": store, "client_session": session})
            recorder._want_version = {"org/m": server.port}
            await recorder.flush()
        assert store.get("org/m").engine_version == "0.27.1"
