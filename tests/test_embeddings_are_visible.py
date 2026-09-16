"""An embedding model on another node ran, answered, and appeared nowhere.

Reported: loaded on node 3, serving there, absent from *Loaded Models* on the
head and from the cluster graphic's hover.

Two places, one cause. Embedding models run in-process through
sentence-transformers, so they appear in no InstanceRecord — and both views
are built from instance records:

  * /api/server/status listed the LOCAL embedding models and every member's
    instances, but never a member's embeddings. The view was right about this
    node and silent about every other.
  * /api/nodes did not carry them at all, and the topology tooltip showed a
    single `model` field, so a node serving a stacked second model or an
    embedding model looked identical to an idle one.
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

TOPOLOGY_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
               "static" / "js" / "topology.js").read_text()
EMBED = "nomic-ai/nomic-embed-text-v1.5"


@pytest.fixture
def app():
    return create_app(config=NodeConfig(node_id="head", node_name="spark-13e1"),
                      engine=None)


@pytest_asyncio.fixture
async def client(app):
    async with TestClient(TestServer(app)) as c:
        yield c


def _peer(app, **kw):
    app["cluster_state"].add_node(ClusterNode(
        node_id="n3", node_name="spark-659b", gpu_name="GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="",
        status=NodeStatus.ONLINE, api_port=8000, web_port=3000,
        last_seen=0.0, fabric_ip="10.0.0.3", **kw))


class TestLoadedModels:
    @pytest.mark.asyncio
    async def test_a_peers_embedding_model_is_listed(self, client, app):
        _peer(app, embedding_models=[EMBED])
        payload = await (await client.get("/api/server/status")).json()
        rows = [m for m in payload["loaded_models"] if m["id"] == EMBED]
        assert rows, payload["loaded_models"]
        assert rows[0]["type"] == "embed"
        assert rows[0]["node_hostname"] == "spark-659b"

    @pytest.mark.asyncio
    async def test_it_is_not_offered_for_ejection_from_here(self, client, app):
        """The eject endpoint only targets this node's own manager."""
        _peer(app, embedding_models=[EMBED])
        row = next(m for m in (await (await client.get("/api/server/status")).json())
                   ["loaded_models"] if m["id"] == EMBED)
        assert row["ejectable"] is False

    @pytest.mark.asyncio
    async def test_a_peer_with_none_adds_nothing(self, client, app):
        _peer(app, embedding_models=[])
        payload = await (await client.get("/api/server/status")).json()
        assert all(m["type"] != "embed" for m in payload["loaded_models"])

    @pytest.mark.asyncio
    async def test_an_older_peer_without_the_field_is_survivable(self, client, app):
        _peer(app)
        assert (await client.get("/api/server/status")).status == 200


class TestTheClusterGraphic:
    @pytest.mark.asyncio
    async def test_api_nodes_carries_them(self, client, app):
        _peer(app, embedding_models=[EMBED])
        data = await (await client.get("/api/nodes")).json()
        node = next(n for n in data["nodes"] if n["node_id"] == "n3")
        assert node["embedding_models"] == [EMBED]

    @pytest.mark.asyncio
    async def test_a_node_without_them_reports_an_empty_list(self, client, app):
        _peer(app)
        data = await (await client.get("/api/nodes")).json()
        node = next(n for n in data["nodes"] if n["node_id"] == "n3")
        assert node["embedding_models"] == []

    def test_the_tooltip_lists_every_model(self):
        assert "_servedModels(d)" in TOPOLOGY_JS
        assert "d.embedding_models" in TOPOLOGY_JS

    def test_stacked_instances_are_named_with_their_port(self):
        """Two models on one node are otherwise indistinguishable."""
        assert "' :' + inst.api_port" in TOPOLOGY_JS

    def test_embeddings_are_marked_as_such(self):
        assert "' (embed)'" in TOPOLOGY_JS

    def test_the_primary_is_not_listed_twice(self):
        """A node's primary also appears in its own instance list."""
        block = TOPOLOGY_JS[TOPOLOGY_JS.index("_servedModels(d)"):]
        block = block[:block.index("_shortText(")]
        assert "seen.has(name)" in block

    def test_a_long_list_is_truncated_rather_than_overflowing(self):
        assert re.search(r"served\.slice\(0, \d\)", TOPOLOGY_JS)
        assert "' more'" in TOPOLOGY_JS
