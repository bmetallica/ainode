"""Applying a profile converges the node onto it.

"Apply" means the node ends up serving what the profile describes — not
"additionally serve these". A profile that leaves the previous set running
would hold memory for models nobody asked for, and applying the same profile to
two nodes would produce two different nodes.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web

import ainode.engine.sharding_routes as sharding_routes
import ainode.models.api_routes as api_routes
from ainode.core.config import NodeConfig
from ainode.profiles.apply import apply_profile, capture_profile, startup_restore
from ainode.profiles.store import Profile, ProfileStore


class _Record:
    def __init__(self, model, port, peers=()):
        self.instance_id = f"head:{model}"
        self.model = model
        self.api_port = port
        self.peer_ips = list(peers)


class _Backend:
    def __init__(self, config):
        self.config = config
        self.stopped = False

    def stop(self):
        self.stopped = True


class _Instance:
    def __init__(self, model, port, peers=(), **cfg):
        self.record = _Record(model, port, peers)
        self.backend = _Backend(NodeConfig(node_id="head", model=model, **cfg))


class _Manager:
    def __init__(self, instances=()):
        self._by_id = {i.record.instance_id: i for i in instances}

    def instances(self):
        return list(self._by_id.values())

    def remove(self, instance_id):
        return self._by_id.pop(instance_id, None)


class _Embeddings:
    def __init__(self, loaded=()):
        self.loaded = list(loaded)
        self.saved = 0

    def list_loaded(self):
        return [{"id": m} for m in self.loaded]

    def is_loaded(self, model_id):
        return model_id in self.loaded

    def unload(self, model_id):
        self.loaded.remove(model_id)
        return True

    async def aload(self, model_id, force=False):
        self.loaded.append(model_id)
        return {"id": model_id}

    def save_manifest(self):
        self.saved += 1


@pytest.fixture()
def app():
    return {"config": NodeConfig(node_id="head"), "instances": _Manager(),
            "engine": None, "cluster_state": None, "embedding_manager": None}


@pytest.fixture(autouse=True)
def _no_side_effects(monkeypatch):
    """Neither the manifest nor the readiness poll belongs in a unit test."""
    monkeypatch.setattr(api_routes, "save_instance_manifest", lambda app: None)

    async def _ready(port, timeout=0):
        return True

    monkeypatch.setattr(api_routes, "_wait_port_ready", _ready)


@pytest.fixture()
def launches(monkeypatch):
    """Record what would have been launched, solo and distributed."""
    calls = {"solo": [], "distributed": []}

    def _solo(app, model, gmu=None, *, overrides=None, persist=True):
        calls["solo"].append({"model": model, "gmu": gmu,
                              "overrides": dict(overrides or {})})
        return {"ok": True, "model": model, "api_port": 8000 + len(calls["solo"])}

    async def _distributed(request):
        body = await request.json()
        calls["distributed"].append(body)
        return web.json_response({"api_port": 8000, "parallel_plan": {"label": "PP=3"}})

    monkeypatch.setattr(api_routes, "append_solo_instance", _solo)
    monkeypatch.setattr(sharding_routes, "handle_sharding_launch", _distributed)
    return calls


def _apply(app, profile):
    return asyncio.run(apply_profile(app, profile))


class TestConvergence:
    def test_starts_what_is_missing(self, app, launches):
        report = _apply(app, Profile(name="p", entries=[{"model": "a/b"}]))
        assert report["ok"] is True
        assert [c["model"] for c in launches["solo"]] == ["a/b"]

    def test_stops_what_the_profile_does_not_mention(self, app, launches):
        stale = _Instance("old/model", 8000)
        app["instances"] = _Manager([stale])
        report = _apply(app, Profile(name="p", entries=[{"model": "a/b"}]))
        assert stale.backend.stopped is True
        assert report["stopped"] == ["old/model"]
        assert app["instances"].instances() == []

    def test_leaves_a_matching_instance_alone(self, app, launches):
        running = _Instance("a/b", 8000, gpu_memory_utilization=0.4)
        app["instances"] = _Manager([running])
        report = _apply(app, Profile(name="p", entries=[
            {"model": "a/b", "gpu_memory_utilization": 0.4}]))
        assert running.backend.stopped is False
        assert launches["solo"] == []
        assert report["results"][0]["action"] == "already_running"

    def test_relaunches_an_instance_whose_config_drifted(self, app, launches):
        running = _Instance("a/b", 8000, gpu_memory_utilization=0.4)
        app["instances"] = _Manager([running])
        _apply(app, Profile(name="p", entries=[
            {"model": "a/b", "gpu_memory_utilization": 0.8}]))
        assert [c["model"] for c in launches["solo"]] == ["a/b"]

    def test_an_entry_that_says_nothing_accepts_what_is_running(self, app, launches):
        # Otherwise a minimal entry would restart a healthy model on every apply.
        running = _Instance("a/b", 8000, kv_cache_dtype="fp8", max_model_len=4096)
        app["instances"] = _Manager([running])
        _apply(app, Profile(name="p", entries=[{"model": "a/b"}]))
        assert launches["solo"] == []

    def test_node_count_change_forces_a_relaunch(self, app, launches):
        # Running on one node, profile wants three: that is a different
        # deployment even though the model id matches.
        running = _Instance("a/b", 8000)
        app["instances"] = _Manager([running])
        _apply(app, Profile(name="p", entries=[
            {"model": "a/b", "node_ids": ["head", "n2", "n3"]}]))
        assert launches["distributed"] and launches["solo"] == []


class TestEntryRouting:
    def test_several_nodes_go_to_the_distributed_route(self, app, launches):
        _apply(app, Profile(name="p", entries=[
            {"model": "a/b", "node_ids": ["head", "n2", "n3"], "strategy": "pipeline"}]))
        assert launches["solo"] == []
        body = launches["distributed"][0]
        assert body["node_ids"] == ["head", "n2", "n3"]
        assert body["strategy"] == "pipeline"

    def test_one_node_goes_to_the_solo_route(self, app, launches):
        _apply(app, Profile(name="p", entries=[
            {"model": "a/b", "node_ids": ["head"]}]))
        assert launches["distributed"] == []
        assert launches["solo"][0]["model"] == "a/b"

    def test_per_load_knobs_reach_the_solo_launch(self, app, launches):
        _apply(app, Profile(name="p", entries=[{
            "model": "a/b", "gpu_memory_utilization": 0.35, "max_model_len": 8192,
            "kv_cache_dtype": "fp8", "extra_vllm_args": ["--enable-prefix-caching"],
        }]))
        call = launches["solo"][0]
        assert call["gmu"] == 0.35
        assert call["overrides"]["max_model_len"] == 8192
        assert call["overrides"]["kv_cache_dtype"] == "fp8"
        assert call["overrides"]["extra_vllm_args"] == ["--enable-prefix-caching"]


class TestFailureHandling:
    def test_one_failing_entry_does_not_stop_the_others(self, app, monkeypatch):
        attempted = []

        def _solo(app, model, gmu=None, *, overrides=None, persist=True):
            attempted.append(model)
            if model == "bad/model":
                return {"ok": False, "error": "no such model"}
            return {"ok": True, "api_port": 8000}

        monkeypatch.setattr(api_routes, "append_solo_instance", _solo)
        report = _apply(app, Profile(name="p", entries=[
            {"model": "bad/model"}, {"model": "good/model"}]))
        assert attempted == ["bad/model", "good/model"]
        assert report["ok"] is False
        assert report["results"][0]["error"] == "no such model"
        assert report["results"][1]["ok"] is True

    def test_a_distributed_rejection_is_reported_with_its_reason(self, app, monkeypatch):
        async def _reject(request):
            return web.json_response({"error": "Selected node(s) not available"},
                                     status=422)

        monkeypatch.setattr(sharding_routes, "handle_sharding_launch", _reject)
        report = _apply(app, Profile(name="p", entries=[
            {"model": "a/b", "node_ids": ["head", "ghost"]}]))
        assert report["ok"] is False
        assert "not available" in report["results"][0]["error"]


class TestEmbeddingEntries:
    def test_an_embedding_entry_uses_the_embedding_manager(self, app, launches):
        app["embedding_manager"] = _Embeddings()
        _apply(app, Profile(name="p", entries=[
            {"model": "nomic-ai/nomic-embed-text-v1.5", "kind": "embedding"}]))
        assert app["embedding_manager"].loaded == ["nomic-ai/nomic-embed-text-v1.5"]
        assert launches["solo"] == []

    def test_an_embedding_model_outside_the_profile_is_unloaded(self, app, launches):
        app["embedding_manager"] = _Embeddings(["old/embedder"])
        report = _apply(app, Profile(name="p", entries=[{"model": "a/b"}]))
        assert app["embedding_manager"].loaded == []
        assert "old/embedder" in report["stopped"]

    def test_an_already_loaded_embedding_model_is_left_alone(self, app, launches):
        app["embedding_manager"] = _Embeddings(["keep/embedder"])
        report = _apply(app, Profile(name="p", entries=[
            {"model": "keep/embedder", "kind": "embedding"}]))
        assert report["results"][0]["action"] == "already_running"


class TestCapture:
    def test_captures_what_is_running(self, app):
        app["instances"] = _Manager([
            _Instance("a/b", 8000, gpu_memory_utilization=0.4, max_model_len=8192),
        ])
        app["embedding_manager"] = _Embeddings(["nomic-ai/nomic-embed-text-v1.5"])
        profile = capture_profile(app, "Endausbau", "wie es laufen soll")
        models = {e.model: e for e in profile.entries}
        assert models["a/b"].gpu_memory_utilization == 0.4
        assert models["a/b"].max_model_len == 8192
        assert models["nomic-ai/nomic-embed-text-v1.5"].kind == "embedding"

    def test_a_captured_profile_can_be_applied_back(self, app, launches):
        app["instances"] = _Manager([_Instance("a/b", 8000, gpu_memory_utilization=0.4)])
        profile = capture_profile(app, "snapshot")
        # Nothing running any more — applying it has to bring the model back.
        app["instances"] = _Manager()
        _apply(app, profile)
        assert launches["solo"][0]["gmu"] == 0.4


class TestStartupRestore:
    def test_without_a_default_profile_nothing_happens(self, app, tmp_path):
        app["profiles"] = ProfileStore(tmp_path / "profiles.json")
        assert asyncio.run(startup_restore(app)) is False

    def test_a_default_profile_is_applied(self, app, launches, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.profiles.apply.SOLO_GRACE", 0)
        store = ProfileStore(tmp_path / "profiles.json")
        store.put(Profile(name="p", entries=[{"model": "a/b"}]))
        store.set_default("p")
        app["profiles"] = store
        assert asyncio.run(startup_restore(app)) is True
        assert [c["model"] for c in launches["solo"]] == ["a/b"]

    def test_a_failing_default_profile_still_counts_as_handled(
            self, app, tmp_path, monkeypatch):
        # Returning False would send the caller into the manifest replay and
        # start a second, contradictory set of models.
        monkeypatch.setattr("ainode.profiles.apply.SOLO_GRACE", 0)

        async def _boom(app, profile, wait=True):
            raise RuntimeError("nope")

        monkeypatch.setattr("ainode.profiles.apply.apply_profile", _boom)
        store = ProfileStore(tmp_path / "profiles.json")
        store.put(Profile(name="p", entries=[{"model": "a/b"}]))
        store.set_default("p")
        app["profiles"] = store
        assert asyncio.run(startup_restore(app)) is True
