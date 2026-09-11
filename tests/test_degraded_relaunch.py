"""A distributed instance that loses a member node: detection and relaunch.

When a member node goes away, the instance keeps running on the head but has
lost the ranks Ray placed on that node, so it cannot serve. Nothing recovers
it automatically — a model spread across three nodes usually does not fit on
two, so the head refuses to guess. These tests cover the two halves of the
answer: reporting it as degraded, and relaunching it on what is left when
that is actually possible.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeAnnouncement, NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState
from ainode.discovery.instance import InstanceRecord
from ainode.engine.instance_manager import InstanceManager
from ainode.engine.sharding_routes import handle_sharding_relaunch
import ainode.engine.backends as backends


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _member(node_id, fabric_ip, gpu_gb=128.0, used_mb=0.0,
            status=NodeStatus.ONLINE):
    return ClusterNode(
        node_id=node_id, node_name=f"spark-{node_id}", gpu_name="NVIDIA GB10",
        gpu_memory_gb=gpu_gb, unified_memory=True, model="", status=status,
        api_port=8000, web_port=3000, last_seen=0.0,
        distributed_mode="member", fabric_ip=fabric_ip,
        gpu_memory_used_mb=used_mb,
    )


def _head_node(gpu_gb=128.0, used_mb=0.0):
    return ClusterNode(
        node_id="head", node_name="spark-head", gpu_name="NVIDIA GB10",
        gpu_memory_gb=gpu_gb, unified_memory=True, model="", status=NodeStatus.ONLINE,
        api_port=8000, web_port=3000, last_seen=0.0,
        distributed_mode="head", fabric_ip="10.0.0.11", gpu_memory_used_mb=used_mb,
    )


class _FakeBackend:
    last = {}

    def __init__(self, config, on_ready=None, instance_id=""):
        _FakeBackend.last = {"config": config, "launched": False}

    def is_running(self):
        return False

    def stop(self):
        _FakeBackend.last["stopped"] = True

    def start_distributed(self):
        _FakeBackend.last["launched"] = True
        return True

    @property
    def config(self):
        return _FakeBackend.last["config"]


def _app(model, launched_peers, online_members, head=None):
    """A head with one running distributed instance and a given online set."""
    _FakeBackend.last = {}
    config = NodeConfig(node_id="head", model=model)
    config.save = lambda *a, **k: None

    cluster = ClusterState(local_announcement=NodeAnnouncement(
        node_id="head", node_name="spark-head", gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True, model=model, status="serving",
        api_port=8000, web_port=3000, distributed_mode="head",
    ))
    cluster.add_node(head or _head_node())
    for m in online_members:
        cluster.add_node(m)

    manager = InstanceManager(base_port=8000)
    manager.add(
        InstanceRecord(
            instance_id=f"head:{model}", model=model, head_node_id="head",
            peer_ips=list(launched_peers), api_port=8000,
            tensor_parallel_size=1, pipeline_parallel_size=1 + len(launched_peers),
            status="serving",
        ),
        _FakeBackend(config),
    )
    return {"cluster_state": cluster, "config": config, "instances": manager,
            "engine": None}


def _relaunch(app, body):
    class _Req:
        def __init__(self):
            self.app = app

        async def json(self):
            return body

    with patch.object(backends, "get_backend", _FakeBackend):
        return asyncio.run(handle_sharding_relaunch(_Req()))


# ---------------------------------------------------------------------------
# Detection — /api/cluster/resources reports the loss
# ---------------------------------------------------------------------------


class TestDegradedDetection:
    def test_all_peers_online_is_not_degraded(self, monkeypatch):
        payload = _resources_payload(
            launched_peers=["10.0.0.12", "10.0.0.13"],
            online_members=[_member("m1", "10.0.0.12"), _member("m2", "10.0.0.13")],
        )
        assert payload["degraded"] is False
        assert payload["missing_peer_ips"] == []
        assert sorted(payload["surviving_node_ids"]) == ["head", "m1", "m2"]

    def test_a_lost_peer_marks_the_instance_degraded(self):
        payload = _resources_payload(
            launched_peers=["10.0.0.12", "10.0.0.13"],
            online_members=[_member("m1", "10.0.0.12")],  # m2 gone
        )
        assert payload["degraded"] is True
        assert payload["missing_peer_ips"] == ["10.0.0.13"]
        assert payload["surviving_node_ids"] == ["head", "m1"]

    def test_every_peer_lost_is_still_reported_not_hidden(self):
        payload = _resources_payload(
            launched_peers=["10.0.0.12", "10.0.0.13"], online_members=[],
        )
        assert payload["degraded"] is True
        assert len(payload["missing_peer_ips"]) == 2
        assert payload["surviving_node_ids"] == ["head"]


def _resources_payload(launched_peers, online_members):
    """Run the resources route and return its single distributed instance."""
    from ainode.api import server

    cluster = ClusterState()
    head = _head_node()
    head.instances = [InstanceRecord(
        instance_id="head:m", model="m", head_node_id="head",
        peer_ips=list(launched_peers), api_port=8000,
        tensor_parallel_size=1, pipeline_parallel_size=1 + len(launched_peers),
    ).to_dict()]
    cluster.add_node(head)
    for m in online_members:
        cluster.add_node(m)

    config = NodeConfig(node_id="head", model="m")
    app = {"cluster_state": cluster, "config": config, "engine": None,
           "instances": None}

    class _Req:
        def __init__(self):
            self.app = app
            self.query = {}

    with patch("ainode.engine.ray_setup.get_ray_status") as ray:
        ray.return_value = type("S", (), {
            "is_head": True, "running": True, "total_cpus": 0, "total_gpus": 0,
            "to_dict": lambda self: {},
        })()
        resp = asyncio.run(server.handle_cluster_resources(_Req()))
    body = json.loads(resp.body)
    return body["distributed_instances"][0]


# ---------------------------------------------------------------------------
# Relaunch
# ---------------------------------------------------------------------------


class TestRelaunch:
    def test_relaunches_on_the_surviving_nodes(self):
        app = _app("tiny-model-1B", ["10.0.0.12", "10.0.0.13"],
                   [_member("m1", "10.0.0.12")])
        resp = _relaunch(app, {"model": "tiny-model-1B"})
        assert resp.status == 200
        body = json.loads(resp.body)
        # Three nodes down to two: re-planned, not carried over.
        assert body["tensor_parallel_size"] == 2
        assert body["pipeline_parallel_size"] == 1
        assert body["relaunched_from"]["missing_peer_ips"] == ["10.0.0.13"]
        assert _FakeBackend.last["launched"] is True

    def test_replans_the_axis_for_the_smaller_node_set(self):
        """A 4-node TP instance losing one node must not come back as TP=3."""
        app = _app("tiny-model-1B",
                   ["10.0.0.12", "10.0.0.13", "10.0.0.14"],
                   [_member("m1", "10.0.0.12"), _member("m2", "10.0.0.13")])
        resp = _relaunch(app, {"model": "tiny-model-1B"})
        assert resp.status == 200
        body = json.loads(resp.body)
        assert body["strategy"] == "pipeline"
        assert body["pipeline_parallel_size"] == 3

    def test_a_healthy_instance_is_refused(self):
        app = _app("tiny-model-1B", ["10.0.0.12"], [_member("m1", "10.0.0.12")])
        resp = _relaunch(app, {"model": "tiny-model-1B"})
        assert resp.status == 409
        assert "not degraded" in json.loads(resp.body)["error"]
        assert _FakeBackend.last.get("launched") is not True

    def test_an_unknown_model_is_a_404(self):
        app = _app("tiny-model-1B", ["10.0.0.12"], [])
        resp = _relaunch(app, {"model": "some-other-model"})
        assert resp.status == 404

    def test_missing_model_field_is_a_400(self):
        app = _app("tiny-model-1B", ["10.0.0.12"], [])
        assert _relaunch(app, {}).status == 400

    def test_refuses_when_the_weights_no_longer_fit(self):
        """The case the whole design hinges on: three nodes' worth of weights
        cannot be squeezed onto two, and the operator must be told why rather
        than watching a launch OOM."""
        app = _app(
            "meta-llama/Llama-3.1-405B-Instruct",   # ~810 GB in the size table
            ["10.0.0.12", "10.0.0.13"],
            [_member("m1", "10.0.0.12")],
        )
        resp = _relaunch(app, {"model": "meta-llama/Llama-3.1-405B-Instruct"})
        assert resp.status == 422
        body = json.loads(resp.body)
        assert "GB per node" in body["error"]
        assert body["surviving_node_ids"] == ["head", "m1"]
        assert body["missing_peer_ips"] == ["10.0.0.13"]
        assert _FakeBackend.last.get("launched") is not True

    def test_refusal_names_the_nodes_that_are_short(self):
        app = _app(
            "meta-llama/Llama-3.1-70B-Instruct",   # ~140 GB
            ["10.0.0.12", "10.0.0.13"],
            [_member("m1", "10.0.0.12", gpu_gb=128.0, used_mb=120 * 1024)],
        )
        resp = _relaunch(app, {"model": "meta-llama/Llama-3.1-70B-Instruct"})
        assert resp.status == 422
        assert "spark-m1" in json.loads(resp.body)["error"]

    def test_a_solo_instance_has_nothing_to_relaunch_across(self):
        app = _app("tiny-model-1B", [], [])
        resp = _relaunch(app, {"model": "tiny-model-1B"})
        assert resp.status == 409
        assert "single-node" in json.loads(resp.body)["error"]

    def test_explicit_strategy_is_honoured(self):
        app = _app("tiny-model-1B", ["10.0.0.12", "10.0.0.13"],
                   [_member("m1", "10.0.0.12")])
        resp = _relaunch(app, {"model": "tiny-model-1B", "strategy": "pipeline"})
        assert resp.status == 200
        assert json.loads(resp.body)["pipeline_parallel_size"] == 2

    def test_an_impossible_axis_is_refused_with_the_reason(self):
        app = _app("tiny-model-1B",
                   ["10.0.0.12", "10.0.0.13", "10.0.0.14"],
                   [_member("m1", "10.0.0.12"), _member("m2", "10.0.0.13")])
        resp = _relaunch(app, {"model": "tiny-model-1B", "strategy": "tensor"})
        assert resp.status == 422
        assert "pipeline" in json.loads(resp.body)["error"]
        assert _FakeBackend.last.get("launched") is not True

    def test_the_degraded_instance_is_stopped_before_the_replacement_starts(self):
        """Two engines on the same port fight over it, so the old one has to go
        down first. handle_sharding_launch owns that; this pins the contract."""
        app = _app("tiny-model-1B", ["10.0.0.12", "10.0.0.13"],
                   [_member("m1", "10.0.0.12")])
        old = app["instances"].by_model("tiny-model-1B")
        stopped = {}
        old.backend.stop = lambda: stopped.setdefault("yes", True)

        resp = _relaunch(app, {"model": "tiny-model-1B"})
        assert resp.status == 200
        assert stopped.get("yes") is True

    def test_losing_every_peer_falls_back_to_a_solo_relaunch(self):
        """A model that still fits one node should come back on the head rather
        than stay dead because its peers are gone."""
        app = _app("tiny-model-1B", ["10.0.0.12", "10.0.0.13"], [])
        with patch("ainode.models.api_routes.handle_model_load") as load:
            async def _ok(req):
                from aiohttp import web
                return web.json_response({"status": "loading", "solo": True})
            load.side_effect = _ok
            resp = _relaunch(app, {"model": "tiny-model-1B"})
        assert resp.status == 200
        assert json.loads(resp.body)["solo"] is True

    def test_losing_every_peer_is_refused_when_the_model_cannot_fit_one_node(self):
        app = _app("meta-llama/Llama-3.1-405B-Instruct",
                   ["10.0.0.12", "10.0.0.13"], [])
        resp = _relaunch(app, {"model": "meta-llama/Llama-3.1-405B-Instruct"})
        assert resp.status == 422
        assert "cannot hold that" in json.loads(resp.body)["error"]
