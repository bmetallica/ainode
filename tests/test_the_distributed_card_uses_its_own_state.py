"""A card must show the state of ITS instance, not of the browser's node.

Reported: Gemma 4 26B was serving on spark-659b and answering requests, while
the head's dashboard read

    nvidia/Gemma-4-26B-A4B-NVFP4
    SOLO · TP=1   spark-659b
    STARTING · 8%

The card for an instance advertised through /api/cluster/resources took its
readiness from `live` — the LOCAL node's engine_ready — and its phase from
the local node's load_phase. Pointed at the head, which had no engine of its
own running, both read "not ready" and "idle", which is 8%.

This is the third card in this panel to make the same mistake: the stacked
card had it (fixed by carrying load_* through /api/nodes) and the node-level
one is documented as never carrying its own state. This one was missed.
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

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" / "static" /
          "js" / "app.js").read_text()
MODEL = "nvidia/Gemma-4-26B-A4B-NVFP4"


@pytest.fixture
def app():
    return create_app(config=NodeConfig(node_id="head", node_name="spark-13e1"),
                      engine=None)


@pytest_asyncio.fixture
async def client(app):
    async with TestClient(TestServer(app)) as c:
        yield c


def _add(app, instance):
    app["cluster_state"].add_node(ClusterNode(
        node_id="n3", node_name="spark-659b", gpu_name="GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="",
        status=NodeStatus.ONLINE, api_port=8000, web_port=3000,
        last_seen=0.0, fabric_ip="10.0.0.3", instances=[instance],
    ))


class TestTheLoadStateReachesTheCard:
    @pytest.mark.asyncio
    async def test_the_phase_is_carried(self, client, app):
        _add(app, {"instance_id": "n3:" + MODEL, "model": MODEL,
                   "status": "starting", "load_phase": "loading_weights",
                   "load_detail": "reading the weights from disk"})
        payload = await (await client.get("/api/cluster/resources")).json()
        instance = payload["distributed_instance"]
        assert instance["load_phase"] == "loading_weights"
        assert instance["load_detail"] == "reading the weights from disk"

    @pytest.mark.asyncio
    async def test_the_failure_reason_is_carried(self, client, app):
        _add(app, {"instance_id": "n3:" + MODEL, "model": MODEL,
                   "status": "failed", "load_error": "the launcher exited (code 1)"})
        payload = await (await client.get("/api/cluster/resources")).json()
        assert payload["distributed_instance"]["load_error"] == \
            "the launcher exited (code 1)"

    @pytest.mark.asyncio
    async def test_an_older_record_still_parses(self, client, app):
        _add(app, {"instance_id": "n3:" + MODEL, "model": MODEL})
        instance = (await (await client.get("/api/cluster/resources")).json())[
            "distributed_instance"]
        assert instance["load_phase"] == ""
        # No status advertised means serving, which is how those behaved.
        assert instance["status"] == "serving"


class TestTheCardReadsIt:
    def _block(self) -> str:
        start = APP_JS.index("if (cr && cr.distributed_instance && cr.distributed_instance.model)")
        return APP_JS[start:APP_JS.index("// Fleet-wide:", start)]

    def test_readiness_comes_from_the_instance(self):
        block = self._block()
        assert "di.status || 'serving'" in block
        assert re.search(r"status:\s*live\s*\?", block) is None

    def test_the_phase_comes_from_the_instance(self):
        assert "phase: di.status === 'failed' ? 'failed' : (di.load_phase || '')" \
            in self._block()

    def test_the_error_and_detail_come_with_it(self):
        block = self._block()
        assert "error: di.load_error || ''" in block
        assert "detail: di.load_detail || ''" in block

    def test_having_a_phase_key_switches_the_card_to_its_own_state(self):
        """renderInstances decides by `inst.phase !== undefined`, so setting
        the key is what stops it borrowing the node's values."""
        assert "var hasOwnState = inst.phase !== undefined;" in APP_JS
        assert "phase: di.status === 'failed'" in self._block()
