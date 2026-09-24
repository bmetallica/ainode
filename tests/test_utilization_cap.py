"""An engine may not ask for memory the machine does not have.

Reported from the cluster, after a two-node launch:

    wenn ich z.b. minimax3 auf node 1 und node2 starte läuft der speicher
    sofort auf 122gb voll und die sparks schmieren komplett ab

122 GB of 128 is what gpu_memory_utilization=0.97 means on unified memory —
and the model card for that checkpoint recommends exactly that, because it
was written for a machine whose GPU memory is not also the operating
system's. The memory guard cannot save this: the allocation crosses its
reserve faster than any poll interval, so the launch has to be the thing that
does not ask.
"""

from __future__ import annotations

from ainode.planner.compute import NodeBudget
from ainode.safety.utilization import (
    MIN_UTILIZATION,
    cap_utilization,
    utilization_ceiling,
)


class _App(dict):
    pass


def _app(monkeypatch, budgets):
    monkeypatch.setattr("ainode.safety.admission._budgets_with_reserve",
                        lambda app, node_ids=None: list(budgets))
    return _App()


def _spark(free_gb, node="spark-1", total=128.0):
    return NodeBudget(node_id=node, name=node, total_gb=total, free_gb=free_gb)


class TestTheCeiling:
    def test_an_idle_spark_lands_near_the_proven_value(self, monkeypatch):
        # 0.85 is the figure this cluster has been serving MoE at; the
        # ceiling arriving in that region is the arithmetic agreeing with
        # what the hardware already told us.
        ceiling, where = utilization_ceiling(_app(monkeypatch, [_spark(120.0)]))
        assert 0.80 <= ceiling <= 0.92
        assert where == "spark-1"

    def test_a_busy_node_gives_less(self, monkeypatch):
        ceiling, _ = utilization_ceiling(_app(monkeypatch, [_spark(40.0)]))
        assert 0.25 <= ceiling <= 0.32

    def test_the_tightest_node_decides(self, monkeypatch):
        # Every rank of a tensor-parallel launch gets the same fraction.
        app = _app(monkeypatch, [_spark(120.0, "spark-1"), _spark(30.0, "spark-2")])
        ceiling, where = utilization_ceiling(app)
        assert where == "spark-2"
        assert ceiling < 0.3

    def test_no_budgets_is_silence(self, monkeypatch):
        assert utilization_ceiling(_app(monkeypatch, [])) == (0.0, "")


class TestTheCap:
    def test_the_card_recommended_value_is_lowered(self, monkeypatch):
        app = _app(monkeypatch, [_spark(120.0)])
        value, note = cap_utilization(app, 0.97)
        assert value < 0.97
        assert "takes the node down" in note
        assert "force" in note

    def test_something_that_fits_is_left_alone(self, monkeypatch):
        app = _app(monkeypatch, [_spark(120.0)])
        assert cap_utilization(app, 0.5) == (0.5, "")

    def test_force_is_obeyed(self, monkeypatch):
        # An operator who knows what a checkpoint needs is allowed to be right.
        app = _app(monkeypatch, [_spark(20.0)])
        assert cap_utilization(app, 0.97, force=True) == (0.97, "")

    def test_it_refuses_rather_than_capping_into_uselessness(self, monkeypatch):
        # Below the floor the engine gets less than its own weights need, so
        # capping there produces a launch that is certain to fail ninety
        # seconds later with a message about a staging buffer. Refusing says
        # the true thing now.
        app = _app(monkeypatch, [_spark(1.0)])
        value, note = cap_utilization(app, 0.9)
        assert value is None
        assert "cannot hold its own weights" in note
        assert "force" in note

    def test_the_floor_is_still_a_floor_when_there_is_room_above_it(self, monkeypatch):
        # A node with a little room caps to something launchable rather than
        # being refused.
        app = _app(monkeypatch, [_spark(28.0)])
        value, note = cap_utilization(app, 0.9)
        assert value is not None and value >= MIN_UTILIZATION
        assert "lowered from" in note

    def test_nothing_requested_is_nothing_changed(self, monkeypatch):
        assert cap_utilization(_app(monkeypatch, [_spark(120.0)]), None) == (None, "")

    def test_an_unreadable_cluster_does_not_block_a_launch(self, monkeypatch):
        assert cap_utilization(_App(), 0.9) == (0.9, "")


class TestBothLaunchPathsApplyIt:
    def test_the_solo_path_does(self):
        import inspect

        from ainode.models.api_routes import append_solo_instance

        source = inspect.getsource(append_solo_instance)
        assert "cap_utilization" in source
        # After the existing instance is stopped: a reload frees what it held.
        assert source.index("existing.backend.stop()") < source.index("cap_utilization")

    def test_the_distributed_path_does(self):
        import inspect

        from ainode.engine import sharding_routes

        source = inspect.getsource(sharding_routes.handle_sharding_launch)
        assert "cap_utilization" in source
        # Across every participating node, head included.
        assert "[config.node_id] + [n.node_id for n in chosen]" in source

    def test_the_operator_is_told(self):
        import inspect

        from ainode.engine import sharding_routes

        source = inspect.getsource(sharding_routes.handle_sharding_launch)
        # The note reaches the response the operator reads, not only the log.
        assert "plan_note" in source.split("cap_note")[4]

    def test_a_node_with_no_room_is_refused_not_capped(self):
        import inspect

        from ainode.engine import sharding_routes
        from ainode.models import api_routes

        assert "capped is None" in inspect.getsource(
            sharding_routes.handle_sharding_launch)
        assert "if gmu is None:" in inspect.getsource(
            api_routes.append_solo_instance)


class _Node:
    def __init__(self, node_id, total_mb=128000.0, used_mb=8000.0):
        self.node_id = node_id
        self.node_name = node_id
        self.status = "online"
        self.gpu_memory_total_mb = total_mb
        self.gpu_memory_used_mb = used_mb
        self.gpu_memory_gb = total_mb / 1024
        self.instances = []


class _Cluster:
    def __init__(self, nodes):
        self._nodes = nodes

    def members(self):
        return self._nodes


class TestItCapsTheNodeYouAreLaunchingOn:
    """Reported from the cluster, launching Qwen on node 3 while DeepSeek
    held nodes 1 and 2:

        gpu-memory-utilization lowered from 0.60 to 0.15: that fraction of
        spark-1432's total memory is what is free there

    spark-1432 is node 2. It had nothing to do with the launch — but the cap
    asked every node and took the tightest, which is right for a launch that
    spans them and wrong for one that does not.
    """

    def _app(self):
        return {
            "config": type("C", (), {"node_id": "node3"})(),
            "cluster_state": _Cluster([
                _Node("node1", used_mb=110000.0),   # busy: DeepSeek
                _Node("node2", used_mb=118000.0),   # busier
                _Node("node3", used_mb=8000.0),     # empty, and the target
            ]),
        }

    def test_a_busy_neighbour_does_not_cap_this_launch(self):
        from ainode.safety.utilization import cap_utilization

        value, note = cap_utilization(self._app(), 0.60, node_ids=["node3"])
        assert value == 0.60
        assert note == ""

    def test_without_a_scope_the_tightest_still_decides(self):
        # Right for a launch that spans the nodes — which is why the scope
        # has to be passed rather than the behaviour changed. node2 has
        # nothing free at all, so the unscoped answer is a refusal naming it.
        from ainode.safety.utilization import cap_utilization

        value, note = cap_utilization(self._app(), 0.60)
        assert value is None
        assert "node2" in note

    def test_the_solo_path_passes_its_own_node(self):
        import inspect

        from ainode.models.api_routes import append_solo_instance

        source = inspect.getsource(append_solo_instance)
        assert "node_ids=[config.node_id]" in source

    def test_the_distributed_path_passes_every_participant(self):
        import inspect

        from ainode.engine import sharding_routes

        assert "[config.node_id] + [n.node_id for n in chosen]" in \
            inspect.getsource(sharding_routes.handle_sharding_launch)


class TestThisNodesOwnFigureIsRead_Live:
    """A node's own entry in cluster state is refreshed by the broadcast it
    SENDS, not the one it receives — so for itself it can be the figure it
    had at startup, before it loaded anything."""

    class _Collector:
        def __init__(self, used_mb):
            self._used = used_mb

        def get_gpu_metrics(self):
            return {"memory_total_mb": 128000.0, "memory_used_mb": self._used}

    def _app(self, stale_used, live_used):
        return {
            "config": type("C", (), {"node_id": "node1"})(),
            "cluster_state": _Cluster([_Node("node1", used_mb=stale_used)]),
            "metrics_collector": self._Collector(live_used),
        }

    def test_the_collector_wins_for_this_node(self):
        from ainode.planner.api_routes import node_budgets

        # Stale says 8 GB used; it is really 110.
        budget = node_budgets(self._app(8000.0, 110000.0))[0]
        assert budget.free_gb < 20

    def test_a_peer_is_left_to_its_broadcast(self):
        from ainode.planner.api_routes import node_budgets

        app = self._app(8000.0, 110000.0)
        app["cluster_state"] = _Cluster([_Node("node2", used_mb=8000.0)])
        assert node_budgets(app)[0].free_gb > 100

    def test_an_unreadable_collector_falls_back(self):
        from ainode.planner.api_routes import node_budgets

        class _Broken:
            def get_gpu_metrics(self):
                raise RuntimeError("nvml is unhappy")

        app = self._app(8000.0, 110000.0)
        app["metrics_collector"] = _Broken()
        assert node_budgets(app)[0].free_gb > 100

    def test_an_error_payload_falls_back_too(self):
        from ainode.planner.api_routes import node_budgets

        class _Erroring:
            def get_gpu_metrics(self):
                return {"error": "no gpu"}

        app = self._app(8000.0, 110000.0)
        app["metrics_collector"] = _Erroring()
        assert node_budgets(app)[0].free_gb > 100


class TestForceIsInTheForm:
    """Every refusal in the product ends with `"force": true`, and it was an
    escape hatch you could only reach over the API — which is no use to the
    person reading the refusal in a browser.
    """

    HTML = (__import__("pathlib").Path(__file__).resolve().parent.parent /
            "ainode" / "web" / "templates" / "index.html").read_text()
    APP_JS = (__import__("pathlib").Path(__file__).resolve().parent.parent /
              "ainode" / "web" / "static" / "js" / "app.js").read_text()

    def test_the_form_has_it(self):
        assert 'id="launch-force"' in self.HTML

    def test_it_says_what_it_skips_and_why_that_matters(self):
        block = self.HTML.split('id="launch-force"')[1][:600]
        assert "admission checks" in block
        assert "taking the node" in " ".join(block.split())

    def test_it_reaches_the_launch_body(self):
        assert "advanced.force = true" in self.APP_JS

    def test_it_does_not_survive_the_launch(self):
        # A checkbox left ticked turns every later launch into an unchecked
        # one, which is the opposite of deliberate.
        assert "usedForce.checked = false" in self.APP_JS

    def test_both_launch_paths_read_it(self):
        import inspect

        from ainode.engine import sharding_routes
        from ainode.models import api_routes

        for source in (inspect.getsource(sharding_routes.handle_sharding_launch),
                       inspect.getsource(api_routes.handle_model_load)):
            assert 'body.get("force")' in source

    def test_force_skips_the_cap(self, monkeypatch):
        from ainode.safety.utilization import cap_utilization

        app = _app(monkeypatch, [_spark(1.0)])
        assert cap_utilization(app, 0.9, force=True) == (0.9, "")


class TestAPlanThatDoesNotFitExplainsTheMissingButton:
    """Reported: "ich kann nichtmehr use this plan auswählen, warum?"

    Because the button is only drawn for a plan that fits, and the plan had
    stopped fitting. The blocker was on screen; the connection between the
    two was not.
    """

    APP_JS = (__import__("pathlib").Path(__file__).resolve().parent.parent /
              "ainode" / "web" / "static" / "js" / "app.js").read_text()

    def test_the_absence_is_explained(self):
        block = self.APP_JS.split("if (!plan.fits) {")[1][:900]
        assert "no plan to apply" in block
        assert "not offered" in block

    def test_it_says_what_would_change_the_answer(self):
        block = self.APP_JS.split("if (!plan.fits) {")[1][:900]
        assert "Free memory on a node" in block
        assert "Launch anyway" in block

    def test_the_button_is_still_only_for_a_plan_that_fits(self):
        # The explanation is not a way to apply a plan that does not exist.
        block = self.APP_JS.split("if (!plan.fits) {")[1][:900]
        assert "plan-apply" not in block
