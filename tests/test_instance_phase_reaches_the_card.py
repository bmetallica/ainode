"""The phase a node computes has to survive the trip to the dashboard.

Reported: Qwen on the third node sat at "STARTING · 12%" for eight minutes
while its own /api/status read

    load_phase: loading_weights
    load_detail: reading the weights from disk

and the engine log showed a healthy launch. Nothing was stuck; the dashboard
simply had no phase to draw, because /api/nodes projected each advertised
instance down to three keys and load_phase was not one of them. The card then
fell back to the status word, and "starting" is 12%.

The same card also read "STACKED · :8000" on a node running exactly one
model — see TestThePrimaryIsNotStacked.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import create_app
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeStatus
from ainode.discovery.cluster import ClusterNode
from ainode.discovery.instance import InstanceRecord

APP_JS = Path(__file__).resolve().parent.parent / "ainode" / "web" / "static" / "js" / "app.js"


@pytest.fixture
def app():
    return create_app(config=NodeConfig(node_id="head", node_name="Head",
                                        model="head-model"), engine=None)


@pytest_asyncio.fixture
async def client(app):
    async with TestClient(TestServer(app)) as c:
        yield c


def _add_node(app, instances):
    app["cluster_state"].add_node(ClusterNode(
        node_id="n3", node_name="spark-1432", gpu_name="GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="",
        status=NodeStatus.ONLINE, api_port=8000, web_port=3000,
        last_seen=0.0, instances=instances,
    ))


async def _instances(client, app, instances):
    _add_node(app, instances)
    data = await (await client.get("/api/nodes")).json()
    return next(n for n in data["nodes"] if n["node_id"] == "n3")["instances"]


class TestTheLoadStateSurvivesTheProjection:
    @pytest.mark.asyncio
    async def test_the_phase_arrives(self, client, app):
        got = await _instances(client, app, [
            {"model": "unsloth/Qwen3.8-27B-NVFP4", "api_port": 8000,
             "status": "starting", "load_phase": "loading_weights",
             "load_detail": "reading the weights from disk", "load_error": ""},
        ])
        assert got[0]["load_phase"] == "loading_weights"
        assert got[0]["load_detail"] == "reading the weights from disk"

    @pytest.mark.asyncio
    async def test_a_failure_reason_arrives(self, client, app):
        got = await _instances(client, app, [
            {"model": "m", "api_port": 8001, "status": "failed",
             "load_phase": "failed", "load_error": "the launcher exited (code 2)"},
        ])
        assert got[0]["load_error"] == "the launcher exited (code 2)"

    @pytest.mark.asyncio
    async def test_an_old_record_without_them_still_parses(self, client, app):
        # A node on the previous build advertises three keys and nothing else.
        got = await _instances(client, app, [
            {"model": "m", "api_port": 8000, "status": "serving"},
        ])
        assert got[0]["load_phase"] == ""
        assert got[0]["load_error"] == ""

    def test_every_field_the_record_carries_is_projected(self):
        """The projection is a hand-written list of keys, which is how the
        phase went missing in the first place. If a load_* field is added to
        InstanceRecord, this fails until /api/nodes carries it too."""
        import inspect

        from ainode.api import server

        source = inspect.getsource(server.handle_nodes)
        for field in InstanceRecord.__dataclass_fields__:
            if field.startswith("load_"):
                assert f'"{field}"' in source, field


class TestThePrimaryIsNotStacked:
    """A node blanks its advertised `model` until the engine answers, so
    during a load the UI's "same model on the main port" test cannot match and
    the node's only model was drawn as a stacked card."""

    def _block(self) -> str:
        source = APP_JS.read_text()
        start = source.index("// Stacked instances (2nd+ model on this node")
        return source[start:source.index("// Collect from sharding status", start)]

    def test_the_port_decides_not_the_model_name(self):
        block = self._block()
        assert re.search(r"var isPrimary = inst\.api_port == null \|\| "
                         r"inst\.api_port === n\.api_port;", block)

    def test_the_badge_follows_it(self):
        block = self._block()
        assert "isPrimary ? 'SINGLE'" in block

    def test_the_primary_is_still_skipped_when_already_drawn(self):
        # Otherwise a serving node shows its model twice.
        assert "if (isPrimary && im === n.model) return;" in self._block()

    def test_a_real_stacked_instance_keeps_its_port_badge(self):
        assert "'STACKED' + (inst.api_port ? ' · :' + inst.api_port : '')" in self._block()
