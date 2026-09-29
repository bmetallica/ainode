# ruff: noqa: F811  (the fixtures app/launches are imported and then requested by name)
"""A profile converges every node it uses, and may hold replicas.

wizzard.md E1 and E4. Applying a profile used to converge only the node it ran
on: a solo model on Spark2 that the profile did not mention went on running and
held the memory the wizard had planned for something else, and a peer's own
entries were relaunched on every apply because nothing compared them.
"""

from __future__ import annotations

import asyncio

import pytest

import ainode.models.api_routes as api_routes
from ainode.profiles import apply as apply_module
from ainode.profiles.store import Profile, ProfileError
from tests.test_profile_apply import _Embeddings, _Instance, _Manager  # noqa: F401
from tests.test_profile_apply import app, launches  # noqa: F401


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(api_routes, "save_instance_manifest", lambda app: None)

    async def _ready(port, timeout=0):
        return True

    monkeypatch.setattr(api_routes, "_wait_port_ready", _ready)


@pytest.fixture
def peers(monkeypatch):
    """Record every converge call a head makes, and answer as a peer would."""
    calls = []

    async def _converge(app, node_id, phase, entries, timeout=120.0):
        calls.append((node_id, phase, [e.model for e in entries]))
        if node_id == "old":
            return None           # a build without the route
        if phase == "stop":
            return {"stopped": [f"stale-on-{node_id}"]}
        return {"results": [{"model": e.model, "action": "launched", "ok": True,
                             "error": ""} for e in entries]}

    monkeypatch.setattr(apply_module, "_peer_converge", _converge)
    return calls


def _apply(app, profile):
    return asyncio.run(apply_module.apply_profile(app, profile))


class TestEveryNodeTheProfileUses:
    def test_peers_are_stopped_before_anything_starts(self, app, launches, peers):
        profile = Profile(name="p", entries=[
            {"model": "big/model", "node_ids": ["head", "s2"], "strategy": "tensor"},
            {"model": "coder", "node_ids": ["s3"]},
        ])
        report = _apply(app, profile)
        phases = [(n, p) for n, p, _ in peers]
        # Both peers stop first — s2 only hosts a rank, and still has to be
        # cleared — then s3 starts its own entry.
        assert phases[:2] == [("s2", "stop"), ("s3", "stop")]
        assert ("s3", "start") in phases and ("s2", "start") not in phases
        assert "stale-on-s2 (s2)" in report["stopped"]
        assert launches["distributed"][0]["model"] == "big/model"
        assert report["ok"] and set(report["nodes"]) == {"head", "s2", "s3"}

    def test_a_node_the_profile_does_not_name_is_left_alone(self, app, launches, peers):
        _apply(app, Profile(name="p", entries=[{"model": "coder", "node_ids": ["s3"]}]))
        assert {n for n, _, _ in peers} == {"s3"}

    def test_the_peer_gets_only_its_own_entries(self, app, launches, peers):
        _apply(app, Profile(name="p", entries=[
            {"model": "coder", "node_ids": ["s3"]},
            {"model": "rag", "kind": "embedding", "node_ids": ["s3"]},
            {"model": "chat", "node_ids": ["head"]},
        ]))
        # One start per entry, in order, so each shows its own progress and a
        # cancel takes effect between them.
        starts = [m for n, p, m in peers if p == "start"]
        assert starts == [["coder"], ["rag"]]
        assert [c["model"] for c in launches["solo"]] == ["chat"]

    def test_an_older_peer_falls_back_to_the_load_routes(self, app, launches,
                                                         peers, monkeypatch):
        started = []

        async def _old_start(app, entry):
            started.append(entry.model)
            return apply_module.ApplyResult(entry.model, "launched", True)

        monkeypatch.setattr(apply_module, "_start_llm_entry", _old_start)
        report = _apply(app, Profile(name="p", entries=[
            {"model": "coder", "node_ids": ["old"]}]))
        assert started == ["coder"]
        assert report["unreachable"] == ["old"]


class TestConvergingHere:
    def test_stop_keeps_what_matches_and_stops_the_rest(self, app, launches):
        keep = _Instance("coder", 8001)
        stale = _Instance("stale", 8002)
        app["instances"] = _Manager([keep, stale])
        app["config"].node_id = "s3"
        from ainode.profiles.store import ProfileEntry

        result = asyncio.run(apply_module.converge_here(
            app, "stop", [ProfileEntry(model="coder", node_ids=["s3"])]))
        assert result["stopped"] == ["stale"]
        assert stale.backend.stopped and not keep.backend.stopped

    def test_start_does_not_relaunch_a_match(self, app, launches):
        app["instances"] = _Manager([_Instance("coder", 8001)])
        app["config"].node_id = "s3"
        from ainode.profiles.store import ProfileEntry

        result = asyncio.run(apply_module.converge_here(
            app, "start", [ProfileEntry(model="coder", node_ids=["s3"])]))
        assert result["results"][0]["action"] == "already_running"
        assert launches["solo"] == []

    def test_unload_stops_exactly_the_named(self, app):
        a, b = _Instance("a", 8001), _Instance("b", 8002)
        app["instances"] = _Manager([a, b])
        from ainode.profiles.store import ProfileEntry

        result = asyncio.run(apply_module.converge_here(
            app, "unload", [ProfileEntry(model="a")]))
        assert result["stopped"] == ["a"] and not b.backend.stopped

    @pytest.mark.asyncio
    async def test_the_route_wants_the_cluster_key(self, tmp_path, monkeypatch):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from ainode.auth import cluster_key as ck
        from ainode.profiles.api_routes import register_profile_routes

        monkeypatch.setattr(ck, "key_path", lambda: tmp_path / "cluster.key")
        ck._CACHE.update(mtime=None, path=None, key="")
        web_app = web.Application()
        web_app["config"] = __import__("ainode.core.config", fromlist=["x"]).NodeConfig()
        web_app["profiles"] = __import__("ainode.profiles.store",
                                         fromlist=["x"]).ProfileStore(tmp_path / "p.json")
        register_profile_routes(web_app)
        async with TestClient(TestServer(web_app)) as client:
            resp = await client.post("/api/profiles/converge",
                                     json={"phase": "stop", "entries": []})
            assert resp.status == 403
            key = ck.cluster_key(create=True)
            resp = await client.post("/api/profiles/converge",
                                     json={"phase": "nonsense", "entries": []},
                                     headers={ck.CLUSTER_HEADER: key})
            assert resp.status == 400
        ck._CACHE.update(mtime=None, path=None, key="")


class TestReplicas:
    def test_the_same_model_on_two_nodes_is_allowed(self):
        Profile(name="p", entries=[{"model": "coder", "node_ids": ["s2"]},
                                   {"model": "coder", "node_ids": ["s3"]}])

    @pytest.mark.parametrize("second", [["s2"], [], ["head", "s2"]])
    def test_not_twice_on_one_node(self, second):
        with pytest.raises(ProfileError):
            Profile(name="p", entries=[{"model": "coder", "node_ids": ["s2"]},
                                       {"model": "coder", "node_ids": second}])

    def test_the_wizard_draft_round_trips(self):
        draft = {"version": 1, "models": [{"id": "m1", "model": "coder"}]}
        profile = Profile.from_dict({"name": "p", "entries": [], "wizard": draft})
        assert Profile.from_dict(profile.to_dict()).wizard == draft
