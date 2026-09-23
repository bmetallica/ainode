"""A profile has to carry HOW each model was started, not just that it was.

"beim speichern eines profiels muss natürlich die genaue config der einzelnen
modelle mitgenommen werden startparameter nodes etc."

#100 captured the peers' models from their announcements, which name the
model, the node set and the parallel axis — and nothing else. The launch
parameters live in each node's own InstanceManager and appear in no
broadcast. A profile that omits them looks complete and restores a model
without the flags it needs, which nobody discovers until the first request:
GLM without --attention-backend B12X, Gemma with the fp8 KV cache that
corrupts its output.

So the head asks. One request per peer, on an action a person performs
deliberately and rarely.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import create_app
from ainode.core.config import NodeConfig
from ainode.profiles.apply import capture_profile, local_launch_specs

GLM = "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark"


class _Record:
    def __init__(self, model, peer_ips=()):
        self.model = model
        self.peer_ips = list(peer_ips)


class _Instance:
    def __init__(self, record, config):
        self.record = record
        self.backend = type("B", (), {"config": config})()


class _Manager:
    def __init__(self, instances):
        self._instances = instances

    def instances(self):
        return self._instances


def _config(**kw):
    return NodeConfig(node_id="n3", model=GLM, **kw)


def _app_with_local(instances):
    return {"config": NodeConfig(node_id="n3", node_name="spark-659b"),
            "instances": _Manager(instances), "embedding_manager": None,
            "cluster_state": None}


class TestWhatALocalInstanceContributes:
    def _spec(self):
        config = _config(
            gpu_memory_utilization=0.87, max_model_len=131072,
            kv_cache_dtype="fp8", kv_cache_dtype_explicit=True,
            trust_remote_code=True,
            extra_vllm_args=["--attention-backend", "B12X", "--block-size", "256"],
            extra_env={"CUTE_DSL_ARCH": "sm_121a"},
            engine_image="vllm-node-b12x", parallel_strategy="tensor")
        app = _app_with_local([_Instance(_Record(GLM, ["10.0.0.3"]), config)])
        return local_launch_specs(app)[0]

    def test_every_launch_field_is_carried(self):
        spec = self._spec()
        assert spec["gpu_memory_utilization"] == 0.87
        assert spec["max_model_len"] == 131072
        # Explicit above, so it is the operator's choice and travels.
        assert spec["kv_cache_dtype"] == "fp8"
        assert spec["trust_remote_code"] is True
        assert spec["engine_image"] == "vllm-node-b12x"
        assert spec["strategy"] == "tensor"

    def test_the_extra_flags_survive_in_order(self):
        """--attention-backend B12X is not optional for GLM; a profile that
        drops it restores a model that cannot start."""
        assert self._spec()["extra_vllm_args"] == [
            "--attention-backend", "B12X", "--block-size", "256"]

    def test_the_environment_survives(self):
        assert self._spec()["extra_env"] == {"CUTE_DSL_ARCH": "sm_121a"}

    def test_the_lists_are_copies(self):
        """A profile holding a reference into a live config would change
        underneath the operator when the model is relaunched."""
        config = _config(extra_vllm_args=["--enforce-eager"])
        app = _app_with_local([_Instance(_Record(GLM), config)])
        spec = local_launch_specs(app)[0]
        spec["extra_vllm_args"].append("--mutated")
        assert config.extra_vllm_args == ["--enforce-eager"]


class _Node:
    node_id = "n3"
    node_name = "spark-659b"
    fabric_ip = "10.0.0.3"
    web_port = 3000
    embedding_models: list = []

    def __init__(self, instances=()):
        self.instances = list(instances)


class _Cluster:
    def __init__(self, nodes):
        self._nodes = nodes

    def get_nodes(self, include_offline=False):
        return self._nodes


def _head(nodes):
    return {"config": NodeConfig(node_id="head", node_name="spark-13e1"),
            "instances": None, "embedding_manager": None,
            "cluster_state": _Cluster(nodes)}


def _answer(payload):
    class _R:
        def read(self):
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return _R()


class TestThePeerIsAsked:
    PEER_SPEC = {"model": "unsloth/Qwen3.8-27B-NVFP4", "kind": "llm",
                 "node_ids": [], "strategy": "",
                 "gpu_memory_utilization": 0.6, "max_model_len": 262144,
                 "kv_cache_dtype": "auto", "quantization": "",
                 "served_model_name": [], "trust_remote_code": None,
                 "extra_vllm_args": ["--load-format", "instanttensor"],
                 "extra_env": {}, "engine_image": "vllm/vllm-openai:v0.27.1"}

    def _capture(self, payload, announced=()):
        app = _head([_Node(announced)])
        with mock.patch("urllib.request.urlopen", return_value=_answer(payload)):
            return capture_profile(app, "def")

    def test_the_parameters_come_back_intact(self):
        profile = self._capture({"instances": [self.PEER_SPEC]})
        entry = next(e for e in profile.entries
                     if e.model == "unsloth/Qwen3.8-27B-NVFP4")
        assert entry.gpu_memory_utilization == 0.6
        assert entry.max_model_len == 262144
        assert entry.extra_vllm_args == ["--load-format", "instanttensor"]
        assert entry.engine_image == "vllm/vllm-openai:v0.27.1"

    def test_an_entry_with_no_placement_lands_on_the_peer(self):
        """A solo instance names no nodes; without this the entry would be
        applied wherever the profile happens to run."""
        entry = self._capture({"instances": [self.PEER_SPEC]}).entries[0]
        assert entry.node_ids == ["n3"]

    def test_a_declared_placement_is_kept(self):
        spec = dict(self.PEER_SPEC, node_ids=["n3", "n2"], strategy="tensor")
        entry = self._capture({"instances": [spec]}).entries[0]
        assert entry.node_ids == ["n3", "n2"] and entry.strategy == "tensor"

    def test_an_unreachable_peer_falls_back_to_the_announcement(self):
        """An older build has no such route and a node that is down answers
        nothing. The model still belongs in the profile."""
        app = _head([_Node([{"model": "org/solo", "status": "serving"}])])
        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            profile = capture_profile(app, "def")
        entry = next(e for e in profile.entries if e.model == "org/solo")
        assert entry.node_ids == ["n3"]
        assert entry.gpu_memory_utilization is None

    def test_a_peer_that_answers_is_not_also_taken_from_the_announcement(self):
        profile = self._capture(
            {"instances": [self.PEER_SPEC]},
            announced=[{"model": "unsloth/Qwen3.8-27B-NVFP4", "status": "serving"}])
        rows = [e for e in profile.entries
                if e.model == "unsloth/Qwen3.8-27B-NVFP4"]
        assert len(rows) == 1
        assert rows[0].gpu_memory_utilization == 0.6   # the asked-for one won

    def test_a_peer_sending_rubbish_does_not_break_the_save(self):
        app = _head([_Node([{"model": "org/solo", "status": "serving"}])])
        with mock.patch("urllib.request.urlopen",
                        return_value=_answer({"instances": ["nonsense", {"kind": "llm"}]})):
            profile = capture_profile(app, "def")
        # The unusable rows are dropped; the announcement still contributes.
        assert [e.model for e in profile.entries] == ["org/solo"]

    def test_a_peer_without_a_fabric_ip_is_not_called(self):
        node = _Node([{"model": "org/solo", "status": "serving"}])
        node.fabric_ip = ""
        with mock.patch("urllib.request.urlopen") as urlopen:
            capture_profile(_head([node]), "def")
        urlopen.assert_not_called()


@pytest_asyncio.fixture
async def client():
    app = create_app(config=NodeConfig(node_id="n3", node_name="spark-659b"),
                     engine=None)
    async with TestClient(TestServer(app)) as c:
        yield c


class TestTheEndpoint:
    @pytest.mark.asyncio
    async def test_it_answers_with_this_nodes_instances(self, client, monkeypatch):
        config = _config(gpu_memory_utilization=0.7,
                         extra_vllm_args=["--enforce-eager"])
        client.app["instances"] = _Manager([_Instance(_Record(GLM), config)])
        payload = await (await client.get("/api/instances/launch-config")).json()
        assert payload["node_id"] == "n3"
        assert payload["instances"][0]["model"] == GLM
        assert payload["instances"][0]["extra_vllm_args"] == ["--enforce-eager"]

    @pytest.mark.asyncio
    async def test_a_node_running_nothing_answers_an_empty_list(self, client):
        payload = await (await client.get("/api/instances/launch-config")).json()
        assert payload["instances"] == []

    def test_both_callers_share_one_extraction(self):
        """A profile whose peer entries carried different fields from its
        local ones would restore two nodes differently from one cluster."""
        import inspect

        from ainode.api import server
        from ainode.profiles import apply

        assert "local_launch_specs" in inspect.getsource(server.handle_launch_config)
        assert "local_launch_specs(app)" in inspect.getsource(apply.capture_profile)


class TestADefaultIsNotAChoice:
    """Captured from the cluster's own profile, on a vision model:

        "kv_cache_dtype": "fp8",
        "extra_vllm_args": [... "--kv-cache-dtype", "auto" ...]

    The recipe says auto and carries the reason — "Vision models must NOT get
    fp8 KV on GB10 — it corrupts generation (proven 2026-07-06)". The node
    default is fp8, and capturing that default turned it into an instruction:
    on restore, a stated kv_cache_dtype is marked EXPLICIT, which is exactly
    the flag that disables the safety downgrade. The restored model would
    have produced garbage rather than an error.
    """

    def _spec(self, explicit):
        config = _config(kv_cache_dtype="fp8", kv_cache_dtype_explicit=explicit)
        app = _app_with_local([_Instance(_Record(GLM, []), config)])
        return local_launch_specs(app)[0]

    def test_the_node_default_is_not_recorded(self):
        assert self._spec(False)["kv_cache_dtype"] == ""

    def test_an_explicit_choice_is(self):
        assert self._spec(True)["kv_cache_dtype"] == "fp8"

    def test_the_safety_rule_is_what_this_protects(self):
        # An empty entry lets effective_kv_cache_dtype run at restore time,
        # which is where the model directory can be read.
        from ainode.engine.serve_args import effective_kv_cache_dtype

        source = __import__("inspect").getsource(effective_kv_cache_dtype)
        assert "kv_cache_dtype_explicit" in source
        assert "is_multimodal_model" in source
