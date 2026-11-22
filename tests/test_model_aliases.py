"""A model served under another name — its predecessor's, so no client's
opencode.json has to change.

The API-name field existed in the wizard and the launch dialog, and vLLM was
given the alias; but the proxy routed by repo id only (a request for the
alias found "no node"), vLLM answered to the alias alone (a request for the
repo id 404'd), and the opencode export listed the repo id.
"""

from __future__ import annotations

from ainode.api.server import _routing_candidates, _routing_table
from ainode.core.config import NodeConfig

OLD = "nvidia/Qwen3.8-Flash-Next-NVFP4"
NEW = "nvidia/Qwen3.8-Flash-Next-FP4-Local"


class _Node:
    def __init__(self, node_id, instances):
        self.node_id = node_id
        self.status = "online"
        self.fabric_ip = "10.0.0.2"
        self.api_port = 8000
        self.model = ""
        self.instances = instances


def _cluster(*nodes):
    return type("C", (), {"members": lambda self: list(nodes)})()


INST = {"model": NEW, "served_names": [OLD], "api_port": 8000, "status": "serving"}


class TestTheProxy:
    def test_it_routes_the_alias(self):
        assert _routing_candidates(_cluster(_Node("p1", [INST])), OLD, "head", 8000) == \
            [("10.0.0.2", 8000)]

    def test_and_still_the_repo_id(self):
        assert _routing_candidates(_cluster(_Node("p1", [INST])), NEW, "head", 8000) == \
            [("10.0.0.2", 8000)]

    def test_v1_models_lists_both(self):
        table = _routing_table(_cluster(_Node("p1", [INST])), "head", 8000)
        assert OLD in table and NEW in table


class TestTheEngineAnswersToBoth:
    def _script(self, tmp_path, implied):
        from unittest import mock

        from ainode.engine.backends import eugr
        from ainode.engine.parallelism import ParallelPlan

        backend = eugr.EugrBackend(NodeConfig(node_id="n", model=NEW, served_model_name=[OLD]))
        with mock.patch.object(eugr, "detect_gpu", return_value=None), \
             mock.patch.object(eugr, "EUGR_LAUNCHER", tmp_path / "launch-cluster.sh"), \
             mock.patch.object(type(backend), "_serve_target_and_name",
                               lambda s: (("/models/x", NEW) if implied else (NEW, ""))):
            return backend._write_launch_script(ParallelPlan(), solo=True).read_text()

    def test_eugr_serves_repo_id_then_alias_from_a_path(self, tmp_path):
        assert f"--served-model-name {NEW} {OLD}" in self._script(tmp_path, True)

    def test_and_by_repo_id(self, tmp_path):
        assert f"--served-model-name {NEW} {OLD}" in self._script(tmp_path, False)

    def test_the_record_carries_the_alias(self):
        from ainode.api.server import _stamp_load_state
        from ainode.discovery.instance import InstanceRecord

        record = InstanceRecord(model=NEW)
        backend = type("B", (), {"config": NodeConfig(served_model_name=[OLD, NEW])})()
        _stamp_load_state(type("I", (), {"record": record, "backend": backend})())
        assert record.served_names == [OLD]


class TestTheClientConfigKeepsTheName:
    def test_opencode_uses_the_alias(self):
        from ainode.clients.opencode import build_opencode_config
        from ainode.profiles.store import ProfileEntry

        entry = ProfileEntry(model=NEW, served_model_name=[OLD], max_model_len=262144)
        config = build_opencode_config({}, "http://h:3000", served=[(NEW, entry)],
                                       cache_tokens={NEW: 1_000_000})
        assert list(config["config"]["provider"]["vllm"]["models"]) == [OLD]


class TestSpeedFollowsTheAlias:
    def test_requests_to_the_alias_count_for_the_model(self, tmp_path):
        from ainode.measure.recorder import Recorder

        app = {"config": NodeConfig(node_id="head"), "cluster_state": _cluster(_Node("p1", [INST]))}
        folded = Recorder(app)._fold_aliases({
            OLD: {"requests": 4, "tokens_generated": 400, "avg_latency_ms": 1000.0},
            NEW: {"requests": 1, "tokens_generated": 100, "avg_latency_ms": 1000.0}})
        assert folded[NEW]["requests"] == 5 and folded[NEW]["tokens_generated"] == 500
