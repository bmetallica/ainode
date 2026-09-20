"""The reserve, per node, from one page.

The guard is a per-node setting and the node that runs out is rarely the one
whose UI is open — two nodes were lost, and neither was the head. A settings
page that could only show and set this node's figure would be the least useful
place to look after that.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
          "static" / "js" / "app.js").read_text()
INDEX = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
         "templates" / "index.html").read_text()


class _Node:
    def __init__(self, node_id, fabric_ip="10.0.0.2"):
        self.node_id = node_id
        self.node_name = node_id.upper()
        self.status = "online"
        self.fabric_ip = fabric_ip
        self.web_port = 3000


class _Cluster:
    def __init__(self, nodes):
        self._nodes = nodes

    def members(self):
        return list(self._nodes)


class _Config:
    node_id = "head"
    node_name = "HEAD"
    host_memory_warn_gb = 8.0
    host_memory_critical_gb = 4.0
    host_memory_guard = True

    def save(self):
        pass


class _Guard:
    warn_mb = 8192
    critical_mb = 4096
    enabled = True

    def __init__(self):
        self.configured = None

    def read(self):
        from ainode.safety.memory_guard import MemoryReading

        return MemoryReading(available_mb=60000, total_mb=128000,
                             warn_mb=self.warn_mb, critical_mb=self.critical_mb)

    def configure(self, **kw):
        self.configured = kw
        if kw.get("warn_gb"):
            self.warn_mb = kw["warn_gb"] * 1024
        if kw.get("critical_gb"):
            self.critical_mb = kw["critical_gb"] * 1024


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self, content_type=None):
        return self._payload


class _Session:
    def __init__(self, payload=None, fail=False):
        self._payload = payload or {}
        self._fail = fail
        self.puts = []

    def get(self, url, timeout=None):
        if self._fail:
            raise OSError("unreachable")
        return _Resp(self._payload)

    def put(self, url, json=None, timeout=None):
        self.puts.append((url, json))
        if self._fail:
            raise OSError("unreachable")
        return _Resp({"ok": True})


class _Req:
    def __init__(self, app, body=None):
        self.app = app
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _app(session=None):
    return {
        "config": _Config(),
        "memory_guard": _Guard(),
        "cluster_state": _Cluster([_Node("head", "10.0.0.1"), _Node("n3")]),
        "client_session": session or _Session(
            {"enabled": True, "available": True, "warn_gb": 8.0,
             "critical_gb": 4.0, "available_mb": 40000, "total_mb": 128000,
             "warn_mb": 8192, "critical_mb": 4096}),
    }


class TestReadingTheFleet:
    def test_every_node_is_reported(self):
        from ainode.api.server import handle_cluster_memory_get

        body = json.loads(asyncio.run(
            handle_cluster_memory_get(_Req(_app()))).body)
        assert [r["node_id"] for r in body["nodes"]] == ["head", "n3"]
        assert all(r["reachable"] for r in body["nodes"])

    def test_this_node_reports_its_own_enforced_lines(self):
        from ainode.api.server import handle_cluster_memory_get

        body = json.loads(asyncio.run(
            handle_cluster_memory_get(_Req(_app()))).body)
        head = body["nodes"][0]
        assert head["warn_gb"] == 8.0 and head["critical_gb"] == 4.0
        assert head["warn_mb"] == 8192

    def test_an_unreachable_node_is_said_to_be_unreachable(self):
        # Not silently dropped: a node missing from the list reads as a node
        # that does not exist, and this page is looked at after one died.
        from ainode.api.server import handle_cluster_memory_get

        body = json.loads(asyncio.run(
            handle_cluster_memory_get(_Req(_app(_Session(fail=True))))).body)
        peer = next(r for r in body["nodes"] if r["node_id"] == "n3")
        assert peer["reachable"] is False


class TestSettingIt:
    def test_one_node_is_set_on_its_own(self):
        from ainode.api.server import handle_cluster_memory_put

        app = _app()
        body = json.loads(asyncio.run(handle_cluster_memory_put(
            _Req(app, {"node_id": "n3", "warn_gb": 12, "critical_gb": 6}))).body)
        assert [r["node_id"] for r in body["results"]] == ["n3"]
        assert app["client_session"].puts[0][1] == {"warn_gb": 12,
                                                    "critical_gb": 6}
        # And this node was NOT touched.
        assert app["memory_guard"].configured is None

    def test_the_head_is_set_when_it_is_the_target(self):
        from ainode.api.server import handle_cluster_memory_put

        app = _app()
        asyncio.run(handle_cluster_memory_put(
            _Req(app, {"node_id": "head", "warn_gb": 10, "critical_gb": 5})))
        assert app["memory_guard"].configured["warn_gb"] == 10

    def test_all_means_every_node_including_this_one(self):
        # Setting three nodes by opening three UIs is how the setting ends up
        # inconsistent across a cluster.
        from ainode.api.server import handle_cluster_memory_put

        app = _app()
        body = json.loads(asyncio.run(handle_cluster_memory_put(
            _Req(app, {"all": True, "preset": "dgx-spark"}))).body)
        assert sorted(r["node_id"] for r in body["results"]) == ["head", "n3"]
        assert app["memory_guard"].configured is not None
        assert app["client_session"].puts[0][1] == {"preset": "dgx-spark"}

    def test_a_node_that_refuses_is_named(self):
        from ainode.api.server import handle_cluster_memory_put

        app = _app(_Session(fail=True))
        body = json.loads(asyncio.run(handle_cluster_memory_put(
            _Req(app, {"node_id": "n3", "warn_gb": 12}))).body)
        assert body["results"][0]["ok"] is False

    def test_the_routes_exist(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="head"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/api/cluster/safety/memory" in paths


class TestThePage:
    def test_it_is_in_the_settings_sidebar(self):
        assert 'data-section="memory"' in INDEX
        assert "renderConfigMemory" in APP_JS

    def test_it_explains_why_this_hardware_needs_it(self):
        # "Out of memory" is not why this matters here.
        block = APP_JS.split("async renderConfigMemory() {")[1][:3000]
        assert "share one pool" in block
        assert "power-cycled" in block

    def test_both_lines_are_editable_per_node(self):
        block = APP_JS.split("async renderConfigMemory() {")[1][:6000]
        assert "mem-warn-" in block and "mem-crit-" in block
        assert "data-mem-save=" in block

    def test_the_presets_are_offered(self):
        block = APP_JS.split("async renderConfigMemory() {")[1][:6000]
        assert 'data-mem-preset="dgx-spark"' in block
        assert 'data-mem-preset="generic"' in block

    def test_the_enforced_value_is_shown_next_to_the_configured_one(self):
        # They differ only on a machine too small to hold the reserve back,
        # and a difference the page hides is a difference that surprises.
        block = APP_JS.split("async renderConfigMemory() {")[1][:6000]
        assert "enforcing <strong>" in block
        assert "capped to the enforced values" in block

    def test_a_node_can_be_switched_off_individually(self):
        block = APP_JS.split("async renderConfigMemory() {")[1][:6000]
        assert "mem-on-" in block
