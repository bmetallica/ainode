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


class TestCaptureSeesEveryNode:
    """A saved profile listed two of four models.

    Reported: MiniMax-M2.7 across nodes 1+2, Qwen3.8 and an embedding model on
    node 3. The profile held MiniMax and the embedding model — Qwen3.8 was
    gone, so applying it would bring the cluster back one model short.

    capture_profile read this node's InstanceManager and the cluster's
    embedding models, and never a peer's LLM instances. The same blind spot
    the Loaded Models view had: the head describing itself and calling it the
    cluster.
    """

    def _app(self, peer_instances):
        from ainode.core.config import NodeConfig

        class _Cluster:
            def __init__(self, nodes):
                self._nodes = nodes

            def get_nodes(self, include_offline=False):
                return self._nodes

        class _Node:
            node_id = "n3"
            node_name = "spark-659b"
            fabric_ip = "10.0.0.3"
            embedding_models = []

            def __init__(self, instances):
                self.instances = instances

        return {
            "config": NodeConfig(node_id="head", node_name="spark-13e1"),
            "instances": None,
            "embedding_manager": None,
            "cluster_state": _Cluster([_Node(peer_instances)]),
        }

    def _capture(self, peer_instances):
        from ainode.profiles.apply import capture_profile

        return capture_profile(self._app(peer_instances), "def")

    def test_a_peers_solo_model_is_captured(self):
        profile = self._capture([{"model": "unsloth/Qwen3.8-27B-NVFP4",
                                  "api_port": 8000, "status": "serving"}])
        entry = next(e for e in profile.entries
                     if e.model == "unsloth/Qwen3.8-27B-NVFP4")
        assert entry.node_ids == ["n3"]
        assert entry.strategy == ""

    def test_a_peers_distributed_model_keeps_its_axis(self):
        profile = self._capture([{"model": "org/big", "status": "serving",
                                  "tensor_parallel_size": 2,
                                  "peer_ips": ["10.0.0.9"]}])
        entry = next(e for e in profile.entries if e.model == "org/big")
        assert entry.strategy == "tensor"
        assert entry.node_ids[0] == "n3" and len(entry.node_ids) == 2

    def test_a_pipeline_split_is_recorded_as_such(self):
        profile = self._capture([{"model": "org/pp", "status": "serving",
                                  "pipeline_parallel_size": 3,
                                  "peer_ips": ["10.0.0.8", "10.0.0.9"]}])
        assert next(e for e in profile.entries
                    if e.model == "org/pp").strategy == "pipeline"

    def test_a_failed_instance_is_not_captured(self):
        """A profile is what SHOULD be running. Capturing a launch that died
        would restore the failure."""
        profile = self._capture([{"model": "org/broken", "status": "failed"}])
        assert all(e.model != "org/broken" for e in profile.entries)

    def test_a_malformed_record_is_skipped(self):
        profile = self._capture(["not a dict", {"api_port": 8000}])
        assert profile.entries == []

    def test_the_same_model_is_not_captured_twice(self):
        """Two nodes serving the same model is one entry, not two — the node
        set says where it runs."""
        profile = self._capture([{"model": "org/x", "status": "serving"},
                                 {"model": "org/x", "status": "serving"}])
        assert len([e for e in profile.entries if e.model == "org/x"]) == 1

    def test_overrides_are_left_empty_rather_than_guessed(self):
        """They live in that node's own config and do not cross the wire.
        A guess would look exact."""
        profile = self._capture([{"model": "org/x", "status": "serving"}])
        entry = next(e for e in profile.entries if e.model == "org/x")
        assert entry.gpu_memory_utilization is None
        assert entry.extra_vllm_args == []
