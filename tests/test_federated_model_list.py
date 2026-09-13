"""A model in /v1/models is a model a client may call.

Advertising instances that are still loading was right for the dashboard — it
is what makes a twenty-minute launch visible instead of looking like nothing
happened. It was wrong for /v1/models, which is the list OpenWebUI builds its
dropdown from: a model appeared there minutes before it could answer, and
picking it produced a 502.

Same data, two audiences. The dashboard reads the instances directly; routing
goes through _instance_can_answer.
"""

from __future__ import annotations

import pytest

from ainode.api.server import _instance_can_answer, _routing_candidates, _routing_table


class _Node:
    def __init__(self, node_id, host, model="", instances=(), status="online",
                 port=8000):
        self.node_id = node_id
        self.fabric_ip = host
        self.model = model
        self.instances = list(instances)
        self.status = status
        self.api_port = port
        self.hostname = node_id


class _Cluster:
    def __init__(self, nodes):
        self._nodes = list(nodes)

    def members(self):
        return self._nodes


class TestWhatCanAnswer:
    @pytest.mark.parametrize("status,expected", [
        ("serving", True),
        ("member-ready", True),
        ("starting", False),
        ("failed", False),
        ("distributing", False),
    ])
    def test_status_decides(self, status, expected):
        assert _instance_can_answer({"status": status}) is expected

    def test_a_record_without_status_stays_routable(self):
        # An older node sends none, and it behaved as routable before.
        assert _instance_can_answer({"model": "a/b"}) is True


class TestTheModelList:
    def test_a_loading_model_is_not_offered(self):
        cluster = _Cluster([_Node("n2", "10.0.0.2", instances=[
            {"model": "gemma", "api_port": 8000, "status": "starting"}])])
        assert _routing_table(cluster, "head", 8000) == {}

    def test_a_serving_model_is(self):
        cluster = _Cluster([_Node("n2", "10.0.0.2", instances=[
            {"model": "gemma", "api_port": 8000, "status": "serving"}])])
        assert _routing_table(cluster, "head", 8000) == {"gemma": ("10.0.0.2", 8000)}

    def test_the_two_states_coexist(self):
        # Exactly the cluster in the report: one serving, one loading.
        cluster = _Cluster([
            _Node("head", "10.0.0.1", instances=[
                {"model": "qwen", "api_port": 8000, "status": "serving"}]),
            _Node("n3", "10.0.0.3", instances=[
                {"model": "gemma", "api_port": 8000, "status": "starting"}]),
        ])
        assert list(_routing_table(cluster, "head", 8000)) == ["qwen"]


class TestRoutingACall:
    def test_a_loading_instance_is_not_a_candidate(self):
        cluster = _Cluster([_Node("n3", "10.0.0.3", instances=[
            {"model": "gemma", "api_port": 8000, "status": "starting"}])])
        assert _routing_candidates(cluster, "gemma", "head", 8000) == []

    def test_a_serving_one_is(self):
        cluster = _Cluster([_Node("n3", "10.0.0.3", instances=[
            {"model": "gemma", "api_port": 8001, "status": "serving"}])])
        assert _routing_candidates(cluster, "gemma", "head", 8000) == [
            ("10.0.0.3", 8001)]

    def test_a_failed_instance_is_never_routed_to(self):
        cluster = _Cluster([_Node("n3", "10.0.0.3", instances=[
            {"model": "gemma", "api_port": 8000, "status": "failed"}])])
        assert _routing_candidates(cluster, "gemma", "head", 8000) == []

    def test_the_nodes_primary_model_still_routes(self):
        # That path is gated on the node's own liveness, not on an instance
        # record, and must keep working.
        cluster = _Cluster([_Node("n2", "10.0.0.2", model="qwen")])
        assert _routing_candidates(cluster, "qwen", "head", 8000) == [
            ("10.0.0.2", 8000)]


class TestTheDashboardStillSeesEverything:
    def test_the_instance_list_is_not_filtered(self):
        # The filter belongs to routing. /api/nodes must keep showing a
        # loading instance, or a long launch looks like nothing happened.
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "ainode" / "api" /
               "server.py").read_text()
        node_list = src[src.index('"instances": ['):]
        node_list = node_list[:node_list.index("]")]
        assert "_instance_can_answer" not in node_list
        assert '"status": inst.get("status")' in node_list
