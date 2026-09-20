"""The Server view: every probe at once, and one round trip fewer.

The page an operator opens to find out why something is not answering was the
slowest page in the UI, and for the same reason: it waited, serially, on the
instances that were not answering. Four local instances of which two were dead
cost four seconds before anything was drawn.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from pathlib import Path

from ainode.api import server_routes

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
          "static" / "js" / "app.js").read_text()


class _Record:
    def __init__(self, model, port):
        self.model = model
        self.api_port = port
        self.status = "serving"


class _Instance:
    def __init__(self, model, port):
        self.record = _Record(model, port)


class _Manager:
    def __init__(self, models):
        self._instances = [_Instance(m, 8001 + i) for i, m in enumerate(models)]

    def instances(self):
        return list(self._instances)


class _Config:
    node_id = "head"
    node_name = "HEAD"
    api_port = 8000
    web_port = 3000
    host = "0.0.0.0"


class _Req:
    def __init__(self, app):
        self.app = app


class TestTheProbesRunAtOnce:
    def test_a_dead_instance_does_not_delay_the_others(self, monkeypatch):
        # Serially, four probes at one second each is four seconds. In
        # parallel it is one — and this page is the one that has to load when
        # things are broken.
        started = []

        async def _slow(session, api_port):
            started.append(time.monotonic())
            await asyncio.sleep(0.2)
            return [f"model-{api_port}"]

        monkeypatch.setattr(server_routes, "_probe_loaded_models", _slow)

        app = {"config": _Config(), "start_time": time.time(),
               "client_session": object(),
               "instances": _Manager(["a", "b", "c"])}

        async def _run():
            began = time.monotonic()
            resp = await server_routes.handle_server_status(_Req(app))
            return time.monotonic() - began, json.loads(resp.body)

        elapsed, body = asyncio.run(_run())
        assert len(started) == 4                 # primary + three stacked
        assert elapsed < 0.5, f"probes ran serially: {elapsed:.2f}s"
        assert len(body["loaded_models"]) >= 4

    def test_a_probe_that_raises_costs_its_own_row_only(self, monkeypatch):
        async def _boom(session, api_port):
            if api_port == 8002:
                raise OSError("connection refused")
            return [f"model-{api_port}"]

        monkeypatch.setattr(server_routes, "_probe_loaded_models", _boom)
        app = {"config": _Config(), "start_time": time.time(),
               "client_session": object(), "instances": _Manager(["a", "b"])}
        body = json.loads(asyncio.run(
            server_routes.handle_server_status(_Req(app))).body)
        ports = {m["port"] for m in body["loaded_models"]}
        assert 8001 in ports and 8002 in ports
        dead = next(m for m in body["loaded_models"] if m["port"] == 8002)
        assert dead["ready"] is False
        # Still ejectable: a dead instance is exactly what needs cleaning up.
        assert dead["ejectable"] is True

    def test_the_probe_timeout_is_a_second(self):
        # localhost. An engine that needs longer than that to answer
        # /v1/models is not answering, and waiting only delays the page that
        # exists to say so.
        source = inspect.getsource(server_routes._probe_loaded_models)
        assert "total=1" in source


class TestOneRoundTripFewer:
    def test_the_endpoint_catalog_travels_with_the_status(self):
        app = {"config": _Config(), "start_time": time.time(),
               "client_session": None}
        body = json.loads(asyncio.run(
            server_routes.handle_server_status(_Req(app))).body)
        assert body["endpoints"] == server_routes.ENDPOINT_CATALOG

    def test_the_separate_route_still_exists(self):
        # An older node in the fleet still answers only there.
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="head"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/api/server/endpoints" in paths

    def test_the_view_fetches_what_it_needs_in_parallel(self):
        block = APP_JS.split("async renderServer() {")[1][:1600]
        assert "Promise.all([" in block
        assert "status.endpoints" in block
