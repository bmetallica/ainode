"""An image model runs on whichever node has room — node 3 is a preference,
not a property.

Two defects the question turned up. The admission gate asked "is there room
in the cluster" where it meant "is there room HERE", so a load onto a full
node passed because a different one had space. And nothing stopped a
multi-node selection, which would have formed a Ray cluster for vLLM and
failed minutes later about a checkpoint vLLM cannot read — true, and no help
in working out that the mistake was the node count.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ainode.models.registry import ModelManager
from ainode.planner.compute import NodeBudget, plan_for_image
from ainode.safety.admission import check_admission

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
          "static" / "js" / "app.js").read_text()


def _node(node_id, free_gb):
    return type("N", (), {
        "node_id": node_id, "node_name": node_id.upper(), "status": "online",
        "gpu_memory_gb": 122.0, "gpu_memory_total_mb": 124928.0,
        "gpu_memory_used_mb": 124928.0 - free_gb * 1024})()


class _Cluster:
    def __init__(self, nodes):
        self._nodes = nodes

    def members(self):
        return list(self._nodes)


def _pipeline_on_disk():
    directory = Path(tempfile.mkdtemp())
    repo = directory / "someone--img"
    repo.mkdir()
    (repo / "model_index.json").write_text("{}")
    (repo / "w.safetensors").write_bytes(b"x" * 2048)
    return ModelManager(models_dir=str(directory))


class _Config:
    node_id = "n2"
    api_port = 8000
    max_image_size = 1024


def _app(nodes):
    return {"config": _Config(), "cluster_state": _Cluster(nodes),
            "model_manager": _pipeline_on_disk()}


class TestTheGateAsksAboutTheRightNode:
    def _weights(self, monkeypatch, gb=33.0):
        import ainode.planner.api_routes as planner_routes

        monkeypatch.setattr(planner_routes, "_image_weights_gb",
                            lambda manager, model: gb)

    def test_a_full_node_is_refused_even_when_another_has_room(self, monkeypatch):
        # The defect: the roomiest node in the cluster answered for a load
        # that was going somewhere else entirely.
        self._weights(monkeypatch)
        app = _app([_node("n1", 100), _node("n2", 10)])
        refusal = check_admission(app, "someone/img", node_ids=["n2"])
        assert refusal and "40 GB needed" in refusal

    def test_a_node_with_room_is_allowed(self, monkeypatch):
        self._weights(monkeypatch)
        app = _app([_node("n1", 100), _node("n2", 10)])
        assert check_admission(app, "someone/img", node_ids=["n1"]) == ""

    def test_every_node_is_a_candidate_none_is_special(self, monkeypatch):
        # Whichever one has the memory. There is no preferred node in the
        # code — that is a property of the deployment, not of AINode.
        self._weights(monkeypatch)
        for name in ("n1", "n2", "n3"):
            app = _app([_node(n, 100 if n == name else 5)
                        for n in ("n1", "n2", "n3")])
            assert check_admission(app, "someone/img", node_ids=[name]) == ""

    def test_the_load_path_scopes_to_the_node_it_runs_on(self):
        # handle_model_load runs where the instance will live — the cluster
        # dispatch has already forwarded it — so "is there room" means here.
        import inspect

        from ainode.models import api_routes

        source = inspect.getsource(api_routes.handle_model_load)
        assert "wanted_nodes = str_list_field(body, \"node_ids\") or None" in source
        assert "wanted_nodes = [own]" in source

    def test_a_measurement_is_also_read_against_that_node(self):
        import inspect

        from ainode.safety import admission

        source = inspect.getsource(admission._planner_says)
        assert "_measured_says(app, model, measured, max_model_len, node_ids)" \
            in source
        assert "_image_says(app, manager, model, node_ids)" in source


class TestThePlannerPicksWhicheverFits:
    def test_it_chooses_the_roomiest_node_by_name(self):
        nodes = [NodeBudget("n1", "N1", 122.0, 20.0),
                 NodeBudget("n2", "N2", 122.0, 80.0),
                 NodeBudget("n3", "N3", 122.0, 40.0)]
        plan = plan_for_image(18.0, nodes, max_image_size=1024)
        assert plan.node_ids == ["n2"]

    def test_the_answer_moves_with_the_memory(self):
        nodes = [NodeBudget("n1", "N1", 122.0, 90.0),
                 NodeBudget("n2", "N2", 122.0, 20.0)]
        assert plan_for_image(18.0, nodes).node_ids == ["n1"]

    def test_it_can_be_asked_about_one_node_only(self):
        # Which is what the launch form does once a node is picked.
        nodes = [NodeBudget("n3", "N3", 122.0, 20.0)]
        assert plan_for_image(33.0, nodes, max_image_size=2048).fits is False


class TestAnImageModelIsNeverSplit:
    def test_the_distributed_route_refuses_it_by_name(self):
        import asyncio

        from ainode.engine import sharding_routes

        class _Req:
            def __init__(self, app, body):
                self.app = app
                self._body = body

            async def json(self):
                return self._body

        app = _app([_node("n1", 100), _node("n2", 100)])
        resp = asyncio.run(sharding_routes.handle_sharding_launch(
            _Req(app, {"model": "someone/img", "node_ids": ["n1", "n2"]})))
        assert resp.status == 422
        import json

        error = json.loads(resp.body)["error"]
        assert "one process on one node" in error
        # And it says the choice of node is free, since that is the next
        # question anyone reading it will have.
        assert "any of them can serve it" in error.lower()

    def test_an_llm_is_untouched_by_that_check(self):
        import inspect

        from ainode.engine import sharding_routes

        source = inspect.getsource(sharding_routes._refuse_distributed_image)
        assert 'return ""' in source          # the not-an-image path

    def test_the_form_lights_one_dot_for_it(self):
        block = APP_JS.split("toggleImageFields(model) {")[1].split("\n  },")[0]
        assert "this._selectNodeIds([lit[0]])" in block

    def test_the_hint_says_why_rather_than_just_doing_it(self):
        assert "one process on one node. Any node can serve it" in APP_JS
