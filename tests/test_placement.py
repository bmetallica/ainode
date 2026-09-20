"""Where a model runs, decided once and remembered.

Covers the store, the API surface, and the two launch paths that consult it:
the distributed one (multi-node pins) and the single-node dispatch (one-node
pins). The rule under all of it is the same — an explicit choice in the
request always wins, and a pin can never be the reason a launch fails.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ainode.placement.api_routes import (
    get_placement_store,
    handle_delete,
    handle_list,
    handle_put,
)
from ainode.placement.store import Placement, PlacementError, PlacementStore


@pytest.fixture()
def store(tmp_path):
    return PlacementStore(tmp_path / "placement.json")


class _Node:
    def __init__(self, node_id):
        self.node_id = node_id


class _Cluster:
    def __init__(self, ids):
        self._nodes = [_Node(i) for i in ids]

    def get_nodes(self):
        return list(self._nodes)


class _Config:
    def __init__(self, node_id="head"):
        self.node_id = node_id


class _Req:
    def __init__(self, app, body=None, **match):
        self.app = app
        self._body = body
        self.match_info = match

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _app(store, node_ids=("head", "n2", "n3")):
    return {
        "placement_store": store,
        "config": _Config(),
        "cluster_state": _Cluster(node_ids),
    }


def _call(handler, app, body=None, **match):
    return asyncio.run(handler(_Req(app, body, **match)))


def _json(resp):
    return json.loads(resp.body)


class TestPlacement:
    def test_it_needs_a_model(self):
        with pytest.raises(PlacementError):
            Placement(model="  ")

    def test_a_node_named_twice_counts_once(self):
        # Two entries would be read as two ranks, halving the memory the
        # planner believes each rank has.
        assert Placement(model="a/b", node_ids=["n2", "n2", "head"]).node_ids == \
            ["n2", "head"]

    def test_order_is_kept(self):
        # The first node is the head of a distributed launch, so the order is
        # not decoration.
        assert Placement(model="a/b", node_ids=["n3", "head"]).node_ids == \
            ["n3", "head"]

    def test_blank_nodes_are_dropped(self):
        assert Placement(model="a/b", node_ids=["", "  ", "n2"]).node_ids == ["n2"]

    def test_strategy_is_normalised(self):
        assert Placement(model="a/b", strategy=" Tensor ").strategy == "tensor"


class TestStore:
    def test_nothing_stored_reads_as_empty(self, store):
        assert store.load() == {}
        assert store.get("a/b") is None

    def test_a_placement_survives_a_round_trip(self, store):
        store.put(Placement(model="a/b", node_ids=["head", "n2"], strategy="tensor"))
        again = PlacementStore(store.path).get("a/b")
        assert again.node_ids == ["head", "n2"]
        assert again.strategy == "tensor"

    def test_removing_reports_whether_anything_went(self, store):
        store.put(Placement(model="a/b", node_ids=["head"]))
        assert store.remove("a/b") is True
        assert store.remove("a/b") is False

    def test_models_do_not_collide(self, store):
        store.put(Placement(model="a/b", node_ids=["head"]))
        store.put(Placement(model="c/d", node_ids=["n3"]))
        assert sorted(store.load()) == ["a/b", "c/d"]

    def test_a_broken_file_costs_a_click_not_a_boot(self, store):
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text("{ this is not json")
        assert store.load() == {}

    def test_the_write_is_atomic(self, store):
        store.put(Placement(model="a/b", node_ids=["head"]))
        # No .tmp left behind: a reader that caught one would see a file that
        # parses as "no placements".
        assert not list(store.path.parent.glob("*.tmp"))


class TestRoutes:
    def test_put_then_list(self, store):
        app = _app(store)
        resp = _call(handle_put, app,
                     {"model": "a/b", "node_ids": ["head", "n2"],
                      "strategy": "tensor"})
        assert resp.status == 200
        listed = _json(_call(handle_list, app))["placements"]
        assert listed == [{"model": "a/b", "node_ids": ["head", "n2"],
                           "strategy": "tensor"}]

    def test_an_unknown_node_is_refused(self, store):
        # A typo here only surfaces at the next launch, where it reads like
        # the cluster lost a node rather than like a setting that was wrong.
        resp = _call(handle_put, _app(store),
                     {"model": "a/b", "node_ids": ["head", "typo"]})
        assert resp.status == 400
        assert "typo" in _json(resp)["error"]

    def test_a_model_is_required(self, store):
        assert _call(handle_put, _app(store), {"node_ids": ["head"]}).status == 400

    def test_an_empty_node_set_removes_the_pin(self, store):
        app = _app(store)
        _call(handle_put, app, {"model": "a/b", "node_ids": ["head"]})
        _call(handle_put, app, {"model": "a/b", "node_ids": []})
        assert store.get("a/b") is None

    def test_delete(self, store):
        app = _app(store)
        _call(handle_put, app, {"model": "a/b", "node_ids": ["head"]})
        assert _json(_call(handle_delete, app, model="a/b"))["removed"] is True
        assert _json(_call(handle_delete, app, model="a/b"))["removed"] is False

    def test_validation_is_skipped_before_discovery_has_run(self, store):
        # An empty cluster means "we do not know yet", not "no node exists" —
        # refusing everything would make the setting unusable on a cold head.
        app = {"placement_store": store, "config": _Config(""), "cluster_state": None}
        assert _call(handle_put, app,
                     {"model": "a/b", "node_ids": ["n9"]}).status == 200

    def test_the_store_is_created_once_per_app(self, tmp_path):
        app = {}
        assert get_placement_store(app) is get_placement_store(app)


class TestTheLaunchPathReadsIt:
    def test_a_pin_supplies_the_nodes(self, store):
        from ainode.engine import sharding_routes

        store.put(Placement(model="a/b", node_ids=["head", "n2"]))
        assert sharding_routes._remembered_placement(
            {"placement_store": store}, "a/b").node_ids == ["head", "n2"]

    def test_an_unreadable_store_is_not_a_failed_launch(self, monkeypatch):
        from ainode.engine import sharding_routes

        class _Boom(PlacementStore):
            def get(self, model):
                raise OSError("disk went away")

        app = {"placement_store": _Boom()}
        assert sharding_routes._remembered_placement(app, "a/b") is None

    def test_only_a_single_node_pin_reaches_the_solo_dispatch(self, store):
        # A two-node pin honoured here would load a sharded model whole.
        from ainode.api import server

        store.put(Placement(model="a/b", node_ids=["n3"]))
        store.put(Placement(model="c/d", node_ids=["head", "n2"]))
        app = {"placement_store": store}
        assert server._placed_node(app, "a/b") == "n3"
        assert server._placed_node(app, "c/d") == ""
        assert server._placed_node(app, "") == ""


class TestTheRoutesAreReachable:
    def test_they_exist(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="head"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/api/placement" in paths
        assert "/api/placement/{model}" in paths


WEB = __import__("pathlib").Path(__file__).resolve().parent.parent / "ainode" / "web"
APP_JS = (WEB / "static" / "js" / "app.js").read_text()


class TestTheUI:
    def test_the_checkbox_exists(self):
        assert 'id="launch-pin"' in (WEB / "templates" / "index.html").read_text()

    def test_a_pin_outranks_the_free_memory_recommendation(self):
        # Otherwise a pinned model moves to whichever nodes are idle right now.
        assert "if (!self.applyPlacement(opt.value))" in APP_JS

    def test_changing_the_selection_keeps_the_pin_current(self):
        assert "repinIfPinned" in APP_JS
