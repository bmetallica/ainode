"""BUG D + node-selection: /api/sharding/launch must resolve participating peers
to their FABRIC IPs (never the mgmt-LAN UDP peer_ip) and honor explicit node_ids.

Calls the handler directly with a fake request (like test_distributed) and a
patched backend so nothing actually launches.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeAnnouncement, NodeStatus
from ainode.discovery.cluster import ClusterNode, ClusterState
from ainode.engine.sharding_routes import handle_sharding_launch
import ainode.engine.backends as backends


def _member(node_id, fabric_ip, peer_ip="192.168.0.99", ib_ips=None):
    return ClusterNode(
        node_id=node_id, node_name=f"host-{node_id}", gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="", status=NodeStatus.ONLINE,
        api_port=8000, web_port=3000, last_seen=0.0,
        distributed_mode="member", peer_ip=peer_ip, fabric_ip=fabric_ip,
        ib_ips=list(ib_ips or []),
    )


def _local_ann(node_id="head"):
    return NodeAnnouncement(
        node_id=node_id, node_name="head", gpu_name="NVIDIA GB10", gpu_memory_gb=128.0,
        unified_memory=True, model="", status="starting", api_port=8000, web_port=3000,
        distributed_mode="head",
    )


class _FakeBackend:
    """Stand-in for NvidiaBackend — records launch, never touches docker."""
    last = {}

    def __init__(self, config, on_ready=None, instance_id=""):
        _FakeBackend.last = {"config": config, "launched": False, "instance_id": instance_id}

    def is_running(self):
        return False

    def start_distributed(self):
        _FakeBackend.last["launched"] = True
        _FakeBackend.last["peer_ips"] = list(self.config.peer_ips)
        return True

    @property
    def config(self):
        return _FakeBackend.last["config"]


def _run(body, members):
    _FakeBackend.last = {}  # reset cross-test state
    config = NodeConfig(node_id="head")
    config.save = lambda *a, **k: None  # no disk writes
    cluster = ClusterState(local_announcement=_local_ann("head"))
    for m in members:
        cluster.add_node(m)
    app = {"cluster_state": cluster, "config": config, "engine": None}

    class _Req:
        def __init__(self):
            self.app = app
        async def json(self):
            return body

    with patch.object(backends, "get_backend", _FakeBackend):
        resp = asyncio.run(handle_sharding_launch(_Req()))
    return config, resp


def test_count_mode_uses_fabric_not_mgmt():
    # min_nodes=2 → 1 peer; it must be the member's FABRIC ip, not its mgmt peer_ip.
    config, resp = _run(
        {"model": "nvidia/Llama-3.3-70B-Instruct-NVFP4", "min_nodes": 2},
        [_member("m1", fabric_ip="10.100.0.13", peer_ip="192.168.0.13")],
    )
    assert resp.status == 200
    assert config.peer_ips == ["10.100.0.13"]
    assert _FakeBackend.last["launched"] is True


def test_explicit_node_ids_selects_exactly_those():
    # Choose head + m2 → peer_ips is exactly m2's fabric IP (not m1's).
    config, resp = _run(
        {"model": "m", "node_ids": ["head", "m2"]},
        [_member("m1", "10.100.0.13"), _member("m2", "10.100.0.15"), _member("m3", "10.100.0.17")],
    )
    assert resp.status == 200
    assert config.peer_ips == ["10.100.0.15"]


def test_missing_fabric_ip_is_rejected():
    # A selected node with no fabric IP must 422, not silently fall back to mgmt.
    config, resp = _run(
        {"model": "m", "node_ids": ["head", "m1"]},
        [_member("m1", fabric_ip="", peer_ip="192.168.0.13")],
    )
    assert resp.status == 422
    assert _FakeBackend.last.get("launched") is not True


def test_unknown_node_id_is_rejected():
    config, resp = _run(
        {"model": "m", "node_ids": ["head", "ghost"]},
        [_member("m1", "10.100.0.13")],
    )
    assert resp.status == 422


def test_distributed_launch_honours_gpu_memory_utilization():
    # The #launch-gmu box is sent for BOTH branches; a TP>1 launch must apply the
    # typed value to the distributed backend, not silently drop it to the 0.5 default.
    config, resp = _run(
        {"model": "m", "node_ids": ["head", "m1"], "gpu_memory_utilization": 0.4},
        [_member("m1", "10.100.0.13")],
    )
    assert resp.status == 200
    assert _FakeBackend.last["config"].gpu_memory_utilization == 0.4


def test_distributed_launch_ignores_out_of_range_gmu():
    # Junk / out-of-range input falls back to the config default rather than
    # forwarding a nonsense fraction to vLLM.
    config, resp = _run(
        {"model": "m", "node_ids": ["head", "m1"], "gpu_memory_utilization": 5},
        [_member("m1", "10.100.0.13")],
    )
    assert resp.status == 200
    assert _FakeBackend.last["config"].gpu_memory_utilization == NodeConfig().gpu_memory_utilization


@pytest.mark.xfail(reason="F1 (0.4.26) removed proxy_to_vllm's 503 visible-loading guard "
                          "because engine.ready is stale-False even while serving; a correct "
                          "loading-state is an owned follow-up (restore once readiness is fixed).",
                   strict=False)
def test_proxy_returns_503_loading_during_swap():
    """While the engine is mid-swap (not ready), :3000 returns a clear loading
    state with the phase — not a hang/opaque proxy error."""
    import json as _json
    from ainode.api.server import proxy_to_vllm

    class _Engine:
        ready = False
        load_phase = "loading_weights"

    config = NodeConfig(node_id="head")
    config.model = "nvidia/Llama-3.3-70B-Instruct-NVFP4"
    app = {"config": config, "client_session": None, "metrics_collector": None, "engine": _Engine()}

    class _Req:
        method = "GET"
        path = "/v1/models"
        headers = {}
        def __init__(self):
            self.app = app

    resp = asyncio.run(proxy_to_vllm(_Req()))
    assert resp.status == 503
    data = _json.loads(resp.body.decode())
    assert data["error"]["load_phase"] == "loading_weights"
    assert resp.headers.get("Retry-After") == "10"


def test_distributed_instance_resolves_peers_and_model():
    """cluster/resources must report the running model (not stale "") and resolve
    fabric-IP peers back to member nodes so the UI shows DISTRIBUTED, not SINGLE."""
    import json as _json
    from ainode.api.server import handle_cluster_resources
    from ainode.engine.ray_autostart import RayAutostartState

    head = ClusterNode(
        node_id="headid", node_name="Spark-1-DGX", gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="",  # stale (idle-start)
        status=NodeStatus.ONLINE, api_port=8000, web_port=3000, last_seen=0.0,
        distributed_mode="head",
        distributed_instance_id="headid:nvidia/Llama-3.3-70B-Instruct-NVFP4",
        distributed_peers=["10.100.0.13"],  # FABRIC IP
    )
    member = _member("memberid", fabric_ip="10.100.0.13")
    cluster = ClusterState()
    cluster.add_node(head)
    cluster.add_node(member)
    app = {"cluster_state": cluster,
           "ray_autostart_state": RayAutostartState(is_head=True, head_address="x:6379")}

    class _Req:
        def __init__(self):
            self.app = app

    resp = asyncio.run(handle_cluster_resources(_Req()))
    di = _json.loads(resp.body.decode())["distributed_instance"]
    assert di["model"] == "nvidia/Llama-3.3-70B-Instruct-NVFP4"  # from iid, not ""
    assert "memberid" in di["peer_node_ids"]                      # fabric IP → node_id
    assert di["tensor_parallel_size"] == 2
    assert "Spark-1-DGX" in di["member_names"] and "host-memberid" in di["member_names"]


# ---------------------------------------------------------------------------
# Part B — a direct RoCE cable is used for weights, never for coordination
# ---------------------------------------------------------------------------


def _cx7_link(cidr):
    from ainode.cluster.topology import CX7Link

    return CX7Link(hca="rocep1s0f1", netdev="enp1s0f1np1",
                   ipv4=cidr.split("/")[0], cidr=cidr)


def test_direct_link_peer_gets_a_transfer_address():
    """Head sits on 192.168.177.0/24; the peer announces an address there, so
    weights go over the cable while Ray stays on the coordination IP."""
    with patch("ainode.cluster.topology.detect_cx7_links",
               return_value=[_cx7_link("192.168.177.11/24")]):
        config, resp = _run(
            {"model": "m", "node_ids": ["head", "m1"]},
            [_member("m1", fabric_ip="10.0.0.12",
                     ib_ips=["192.168.177.12", "192.168.197.12"])],
        )
    assert resp.status == 200
    assert config.peer_ips == ["10.0.0.12"]  # coordination unchanged
    launched = _FakeBackend.last["config"]
    assert launched.peer_transfer_ips == {"10.0.0.12": "192.168.177.12"}


def test_peer_without_a_shared_subnet_gets_no_transfer_address():
    with patch("ainode.cluster.topology.detect_cx7_links",
               return_value=[_cx7_link("192.168.177.11/24")]):
        config, resp = _run(
            {"model": "m", "node_ids": ["head", "m1"]},
            [_member("m1", fabric_ip="10.0.0.13", ib_ips=["192.168.197.13"])],
        )
    assert resp.status == 200
    assert _FakeBackend.last["config"].peer_transfer_ips == {}


def test_peer_from_an_older_build_gets_no_transfer_address():
    with patch("ainode.cluster.topology.detect_cx7_links",
               return_value=[_cx7_link("192.168.177.11/24")]):
        config, resp = _run(
            {"model": "m", "node_ids": ["head", "m1"]},
            [_member("m1", fabric_ip="10.0.0.12")],  # announces no ib_ips
        )
    assert resp.status == 200
    assert _FakeBackend.last["config"].peer_transfer_ips == {}


def test_transfer_resolution_failure_does_not_block_the_launch():
    """Address selection is an optimisation. If it raises, the launch must
    still proceed over the coordination path."""
    with patch("ainode.cluster.topology.detect_cx7_links",
               side_effect=OSError("sysfs exploded")):
        config, resp = _run(
            {"model": "m", "node_ids": ["head", "m1"]},
            [_member("m1", fabric_ip="10.0.0.12", ib_ips=["192.168.177.12"])],
        )
    assert resp.status == 200
    assert _FakeBackend.last["launched"] is True
    assert _FakeBackend.last["config"].peer_transfer_ips == {}


# ---------------------------------------------------------------------------
# Part C — the strategy is honoured instead of always building TP
# ---------------------------------------------------------------------------


def _members(n):
    """n member nodes, so the cluster has n+1 counting the head."""
    return [_member(f"m{i}", fabric_ip=f"10.0.0.{12 + i}") for i in range(n)]


def _ids(n):
    return ["head"] + [f"m{i}" for i in range(n)]


def test_two_nodes_still_launch_tensor_parallel():
    """Regression guard: the existing 2-node setups must be untouched."""
    config, resp = _run({"model": "m", "node_ids": _ids(1)}, _members(1))
    assert resp.status == 200
    body = json.loads(resp.body)
    assert body["tensor_parallel_size"] == 2
    assert body["pipeline_parallel_size"] == 1
    assert body["strategy"] == "tensor"
    assert _FakeBackend.last["config"].tensor_parallel_size == 2


def test_four_nodes_still_launch_tensor_parallel():
    config, resp = _run({"model": "m", "node_ids": _ids(3)}, _members(3))
    assert resp.status == 200
    assert json.loads(resp.body)["tensor_parallel_size"] == 4


def test_three_nodes_auto_resolves_to_pipeline():
    """TP=3 has no models behind it, so auto must pick pipeline."""
    config, resp = _run({"model": "m", "node_ids": _ids(2)}, _members(2))
    assert resp.status == 200
    body = json.loads(resp.body)
    assert body["strategy"] == "pipeline"
    assert body["pipeline_parallel_size"] == 3
    assert body["tensor_parallel_size"] == 1
    assert body["parallel_plan"]["label"] == "PP=3"


def test_three_nodes_explicit_tensor_is_refused_before_launching():
    config, resp = _run(
        {"model": "m", "node_ids": _ids(2), "strategy": "tensor"}, _members(2)
    )
    assert resp.status == 422
    body = json.loads(resp.body)
    assert "pipeline" in body["error"]
    assert body["node_count"] == 3
    # Nothing was started — the point of validating before the backend runs.
    assert _FakeBackend.last.get("launched") is not True


def test_three_nodes_explicit_data_parallel():
    config, resp = _run(
        {"model": "m", "node_ids": _ids(2), "strategy": "data"}, _members(2)
    )
    assert resp.status == 200
    body = json.loads(resp.body)
    assert body["data_parallel_size"] == 3
    assert body["tensor_parallel_size"] == 1
    assert _FakeBackend.last["config"].parallel_strategy == "data"


def test_docs_spelling_of_the_strategy_is_accepted():
    """The UI posts "tensor", the API docs say "tensor_parallel"; both have
    been sent for releases and neither was ever acted on."""
    config, resp = _run(
        {"model": "m", "node_ids": _ids(1), "strategy": "tensor_parallel"},
        _members(1),
    )
    assert resp.status == 200
    assert json.loads(resp.body)["tensor_parallel_size"] == 2


def test_unknown_strategy_is_a_400():
    config, resp = _run(
        {"model": "m", "node_ids": _ids(1), "strategy": "megatron"}, _members(1)
    )
    assert resp.status == 400
    assert "tensor, pipeline, data, auto" in json.loads(resp.body)["error"]


def test_instance_record_carries_every_axis():
    config, resp = _run({"model": "m", "node_ids": _ids(2)}, _members(2))
    assert resp.status == 200
    cfg = _FakeBackend.last["config"]
    assert (cfg.tensor_parallel_size, cfg.pipeline_parallel_size,
            cfg.data_parallel_size) == (1, 3, 1)
    assert cfg.parallel_strategy == "pipeline"


# --- Eligibility ------------------------------------------------------------
#
# From the cluster: "Selected node(s) not available as members: ['10c0520d']",
# whichever nodes were picked. Every node reported distributed_mode "solo",
# because loading one solo model sets that field and nothing sets it back — so
# a node that had ever served anything could never join a distributed launch
# again. On a cluster that has been used, that is every node.


def _solo_member(node_id, fabric_ip, model="", status=NodeStatus.ONLINE):
    node = _member(node_id, fabric_ip)
    node.distributed_mode = "solo"
    node.model = model
    node.status = status
    return node


def test_a_node_that_has_served_solo_can_still_be_chosen():
    config, resp = _run(
        {"model": "m", "node_ids": ["head", "m1"]},
        [_solo_member("m1", "10.100.0.13", model="some/model")],
    )
    assert resp.status == 200
    assert config.peer_ips == ["10.100.0.13"]


def test_the_head_is_never_its_own_peer():
    # Eligibility by reachability has to keep excluding this node, or a
    # count-mode launch picks the head as its own worker.
    config, resp = _run(
        {"model": "m", "min_nodes": 2},
        [_solo_member("m1", "10.100.0.13")],
    )
    assert resp.status == 200
    assert config.peer_ips == ["10.100.0.13"]


def test_an_offline_node_is_still_refused():
    config, resp = _run(
        {"model": "m", "node_ids": ["head", "m1"]},
        [_solo_member("m1", "10.100.0.13", status=NodeStatus.OFFLINE)],
    )
    assert resp.status == 422
    assert "cannot take part" in json.loads(resp.body)["error"]


def test_the_refusal_names_the_node_and_the_reason():
    # Three causes need three different actions; "not available" named none.
    config, resp = _run({"model": "m", "node_ids": ["head", "ghost"]}, [])
    assert resp.status == 422
    error = json.loads(resp.body)["error"]
    assert "ghost" in error
    assert "not discovered" in error


def test_a_busy_peer_is_a_warning_not_a_refusal():
    # The operator may be about to unload it, and the engine reports an
    # out-of-memory far more precisely than a guess here could.
    config, resp = _run(
        {"model": "m", "node_ids": ["head", "m1"]},
        [_solo_member("m1", "10.100.0.13", model="other/model")],
    )
    assert resp.status == 200
    note = json.loads(resp.body)["note"]
    assert "Already serving" in note and "other/model" in note


def test_an_idle_peer_produces_no_warning():
    config, resp = _run(
        {"model": "m", "node_ids": ["head", "m1"]},
        [_solo_member("m1", "10.100.0.13")],
    )
    assert json.loads(resp.body)["note"] == ""
