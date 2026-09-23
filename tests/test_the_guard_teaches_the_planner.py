"""What the guard had to kill, the planner has to know.

From the cluster, after the guard stopped a worker on a two-node launch:

    ich denke wir müssen die ram grenze auch beim modell planen
    berücksichtigen, der worker ist hier jetzt zum wiederholten mal vom
    memoryguard (diesmal wirklich) beendet worden

Two halves to that. The planner must plan against the line the guard
actually enforces — it held back its own 4 GB while the guard refused to go
below 8, so the launch dialog was more optimistic than the gate and both
were more optimistic than the hardware. And a kill is evidence: the next
launch of the same model, asking for the same thing, should be refused with
it rather than discovered the same way.
"""

from __future__ import annotations

import time
from pathlib import Path

from ainode.measure.store import MeasurementStore


class _Guard:
    def __init__(self, warn_mb):
        self.warn_mb = warn_mb

    def read(self):
        return type("R", (), {"warn_mb": self.warn_mb})()


class _Cluster:
    def __init__(self, nodes):
        self._nodes = nodes

    def members(self):
        return self._nodes


def _node(node_id, total_mb=128000, used_mb=8000):
    return type("N", (), {
        "node_id": node_id, "node_name": node_id, "status": "online",
        "gpu_memory_total_mb": total_mb, "gpu_memory_used_mb": used_mb,
        "gpu_memory_gb": total_mb / 1024, "instances": [],
    })()


class TestOneLineForBoth:
    def test_the_guards_reserve_is_held_back(self):
        from ainode.planner.api_routes import budgets_with_guard_reserve

        app = {"cluster_state": _Cluster([_node("n1")]), "memory_guard": _Guard(8192)}
        plain = budgets_with_guard_reserve({"cluster_state": app["cluster_state"]})
        guarded = budgets_with_guard_reserve(app)
        # 8 GB warning line against the planner's own 4 GB reserve: 4 GB less
        # to plan into, which is exactly the difference between the two.
        assert round(plain[0].free_gb - guarded[0].free_gb) == 4

    def test_a_raised_reserve_moves_it_further(self):
        from ainode.planner.api_routes import budgets_with_guard_reserve

        app = {"cluster_state": _Cluster([_node("n1")]), "memory_guard": _Guard(20480)}
        assert budgets_with_guard_reserve(app)[0].free_gb < 105

    def test_without_a_guard_only_the_headroom_is_held_back(self):
        from ainode.planner.api_routes import budgets_with_guard_reserve
        from ainode.planner.compute import plan_headroom_gb

        app = {"cluster_state": _Cluster([_node("n1")])}
        budget = budgets_with_guard_reserve(app)[0]
        assert round(budget.free_gb) == round(117.2 - plan_headroom_gb(125.0))

    def test_the_headroom_is_not_the_guards_line(self):
        # Two different things: the guard's line is where it starts acting,
        # the headroom is the distance a plan keeps from it.
        from ainode.planner.compute import plan_headroom_gb

        assert 7.0 <= plan_headroom_gb(128.0) <= 8.0

    def test_it_is_a_share_of_the_machine_not_a_constant(self):
        # 8 GB is right on a Spark and absurd on a 16 GB CI box.
        from ainode.planner.compute import plan_headroom_gb

        assert plan_headroom_gb(16.0) <= 1.5
        assert plan_headroom_gb(512.0) <= 8.0
        assert plan_headroom_gb(0.0) >= 1.0

    def test_the_dialog_and_the_gate_ask_the_same_function(self):
        import inspect

        from ainode.planner import api_routes
        from ainode.safety import admission

        assert "budgets_with_guard_reserve" in inspect.getsource(
            api_routes.handle_plan)
        assert "budgets_with_guard_reserve" in inspect.getsource(
            admission._budgets_with_reserve)


class TestAKillIsEvidence:
    MODEL = "sparkarena/Minimax-M3-v0-NVFP4-REAP50"

    def _app(self, tmp_path, **stop):
        store = MeasurementStore(tmp_path / "measurements.json")
        if stop:
            store.record_guard_stop(self.MODEL, **stop)
        return {"measurement_store": store, "model_manager": None}

    def test_the_stop_is_written_down(self, tmp_path):
        store = MeasurementStore(tmp_path / "m.json")
        store.record_guard_stop(self.MODEL, free_gb=0.9, node_id="n1",
                                gpu_memory_utilization=0.85,
                                max_model_len=65536, nodes=2)
        entry = store.get(self.MODEL)
        assert entry.guard_stops == 1
        assert entry.guard_stop_gmu == 0.85
        assert entry.guard_stop_nodes == 2

    def test_the_next_identical_launch_is_refused(self, tmp_path):
        from ainode.safety.admission import check_admission

        app = self._app(tmp_path, gpu_memory_utilization=0.85,
                        max_model_len=65536, nodes=2)
        refusal = check_admission(app, self.MODEL, node_ids=["n1", "n2"],
                                  gpu_memory_utilization=0.85,
                                  max_model_len=65536)
        assert "already had to stop" in refusal
        assert "not an estimate" in refusal

    def test_it_says_what_the_killed_launch_asked_for(self, tmp_path):
        from ainode.safety.admission import check_admission

        app = self._app(tmp_path, gpu_memory_utilization=0.85,
                        max_model_len=65536, nodes=2)
        refusal = check_admission(app, self.MODEL, node_ids=["n1", "n2"],
                                  gpu_memory_utilization=0.85)
        assert "0.85" in refusal and "65,536" in refusal

    def test_asking_for_less_memory_is_a_different_question(self, tmp_path):
        from ainode.safety.admission import check_admission

        app = self._app(tmp_path, gpu_memory_utilization=0.85, nodes=2)
        assert check_admission(app, self.MODEL, node_ids=["n1", "n2"],
                               gpu_memory_utilization=0.6) == ""

    def test_asking_for_less_context_is_too(self, tmp_path):
        from ainode.safety.admission import check_admission

        app = self._app(tmp_path, max_model_len=65536, nodes=2)
        assert check_admission(app, self.MODEL, node_ids=["n1", "n2"],
                               max_model_len=8192) == ""

    def test_spreading_it_wider_is_too(self, tmp_path):
        from ainode.safety.admission import check_admission

        app = self._app(tmp_path, nodes=2, gpu_memory_utilization=0.85)
        assert check_admission(app, self.MODEL, node_ids=["n1", "n2", "n3"],
                               gpu_memory_utilization=0.85) == ""

    def test_force_overrides_it(self, tmp_path):
        from ainode.safety.admission import check_admission

        app = self._app(tmp_path, nodes=2, gpu_memory_utilization=0.85)
        assert check_admission(app, self.MODEL, node_ids=["n1", "n2"],
                               gpu_memory_utilization=0.85, force=True) == ""

    def test_a_model_that_has_since_run_is_forgiven(self, tmp_path):
        # Whatever was in the way is evidently no longer there.
        from ainode.safety.admission import check_admission

        store = MeasurementStore(tmp_path / "measurements.json")
        store.record_guard_stop(self.MODEL, nodes=2, gpu_memory_utilization=0.85)
        store.record_launch(self.MODEL, ok=True, memory_gb=40.0)
        app = {"measurement_store": store, "model_manager": None}
        assert check_admission(app, self.MODEL, node_ids=["n1", "n2"],
                               gpu_memory_utilization=0.85) == ""

    def test_a_model_never_stopped_is_never_refused(self, tmp_path):
        from ainode.safety.admission import check_admission

        app = self._app(tmp_path)
        assert check_admission(app, "org/innocent", node_ids=["n1"]) == ""

    def test_it_counts_repeats(self, tmp_path):
        from ainode.safety.admission import check_admission

        store = MeasurementStore(tmp_path / "measurements.json")
        for _ in range(3):
            store.record_guard_stop(self.MODEL, nodes=2, gpu_memory_utilization=0.85)
        app = {"measurement_store": store, "model_manager": None}
        refusal = check_admission(app, self.MODEL, node_ids=["n1", "n2"],
                                  gpu_memory_utilization=0.85)
        assert "3 times" in refusal

    def test_it_is_asked_before_the_memory_arithmetic(self):
        # Evidence beats estimate: there is nothing for the planner to add to
        # "this was killed here".
        import inspect

        from ainode.safety import admission

        source = inspect.getsource(admission.check_admission)
        assert source.index("_guard_history_says") < source.index("_planner_says")


class TestTheGuardWritesIt:
    def test_act_records_against_the_model(self, tmp_path):
        from ainode.safety.memory_guard import MemoryGuard

        class _Config:
            gpu_memory_utilization = 0.85
            max_model_len = 65536

        class _Backend:
            config = _Config()

            def kill(self):
                pass

        class _Record:
            model = "org/big"
            peer_ips = ["10.0.0.2"]
            load_error = ""
            load_phase = ""
            status = "serving"

        class _Instance:
            record = _Record()
            backend = _Backend()

        class _Manager:
            def instances(self):
                return [_Instance()]

        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal: 131072000 kB\nMemAvailable: 900000 kB\n")
        store = MeasurementStore(tmp_path / "measurements.json")
        guard = MemoryGuard({"instances": _Manager(), "measurement_store": store},
                            meminfo=meminfo, critical_gb=4, warn_gb=8)
        guard.act(guard.read())

        entry = store.get("org/big")
        assert entry.guard_stops == 1
        assert entry.guard_stop_gmu == 0.85
        assert entry.guard_stop_max_model_len == 65536
        assert entry.guard_stop_nodes == 2
        assert time.time() - entry.last_guard_stop < 10


class TestThePlanDoesNotAimAtTheCliff:
    """The whole chain, because every link of it was sound and the result
    still killed a node.

    The launch form is pre-filled with the planner's gpu_memory_utilization,
    and the planner derived it from a claim that consumed every gigabyte down
    to the guard's warning line. So a launch that went exactly to plan landed
    the node ON the line where the guard starts refusing work — and the next
    ordinary fluctuation took it under the line where the guard starts
    killing.

    Two idle 128 GB nodes, a 129 GB checkpoint at TP=2: 65 GB of weights per
    node, which fits with room to spare. They filled up anyway.
    """

    def _budgets(self, free_gb=120.0, total_mb=128000, warn_mb=8192):
        from ainode.planner.api_routes import budgets_with_guard_reserve

        used = total_mb - free_gb * 1024
        app = {"cluster_state": _Cluster([_node("n1", total_mb, used),
                                          _node("n2", total_mb, used)]),
               "memory_guard": _Guard(warn_mb)}
        return budgets_with_guard_reserve(app)

    def _plan(self, budgets):
        from ainode.planner.compute import plan_for
        from ainode.planner.facts import facts_from_config

        facts = facts_from_config(
            {"num_hidden_layers": 62, "num_attention_heads": 64,
             "num_key_value_heads": 8, "hidden_size": 6144,
             "max_position_embeddings": 196608, "torch_dtype": "bfloat16"},
            "sparkarena/Minimax-M3", int(128.9e9))
        return plan_for(facts, budgets, kv_cache_dtype="fp8")

    def test_the_weights_do_fit_on_two_nodes(self):
        # Saying so plainly: 129 GB across two nodes is 65 GB each, on nodes
        # with 120 free. Nothing about this launch was too big.
        plan = self._plan(self._budgets())
        assert plan.fits
        assert plan.tensor_parallel_size == 2
        assert 60 <= plan.weights_per_node_gb <= 72

    def test_what_it_plans_to_claim_leaves_the_guard_alone(self):
        free_gb, warn_gb = 120.0, 8.0
        budgets = self._budgets(free_gb=free_gb)
        plan = self._plan(budgets)
        total_gb = 128000 / 1024
        claimed = plan.gpu_memory_utilization * total_gb
        left = free_gb - claimed
        # The number that matters: what is still free once the engine has
        # taken its share. Before the headroom existed this was the warning
        # line exactly — nothing between a plan going right and the guard
        # acting.
        assert left > warn_gb + 3, (
            f"plan claims {claimed:.0f} GB of {free_gb:.0f} free, leaving "
            f"{left:.0f} GB against a {warn_gb:.0f} GB guard line")

    def test_a_tighter_guard_line_moves_the_plan_not_the_margin(self):
        # Raising the reserve in Settings has to make the plan smaller, not
        # make the margin disappear.
        for warn_mb in (8192, 16384, 24576):
            budgets = self._budgets(warn_mb=warn_mb)
            plan = self._plan(budgets)
            if not plan.fits:
                continue
            claimed = plan.gpu_memory_utilization * (128000 / 1024)
            assert 120.0 - claimed > warn_mb / 1024


class TestTheRecordCanBeDropped:
    """The refusal is the right default and the wrong permanent state.

        ih wollte jetzt nochmal versuchen das minimax3 zu laden bekomme aber
        ständig diese bzw. ähnliche meldungen … kann ich das irgendwie
        zurücksetzen?

    What made a launch impossible is usually fixed by something the record
    cannot see — a flag added, an image rebuilt, a model unloaded elsewhere.
    """

    MODEL = "sparkarena/Minimax-M3-v0-NVFP4-REAP50"

    def _store(self, tmp_path):
        store = MeasurementStore(tmp_path / "measurements.json")
        store.record_launch(self.MODEL, ok=True, memory_gb=61.0, load_seconds=210.0)
        store.record_guard_stop(self.MODEL, gpu_memory_utilization=0.82,
                                max_model_len=3072, nodes=2)
        return store

    def test_clearing_lifts_the_refusal(self, tmp_path):
        from ainode.safety.admission import check_admission

        store = self._store(tmp_path)
        app = {"measurement_store": store, "model_manager": None}
        assert "already had to stop" in check_admission(
            app, self.MODEL, node_ids=["n1", "n2"], gpu_memory_utilization=0.82)

        assert store.forget_guard_stops(self.MODEL) is True
        assert check_admission(app, self.MODEL, node_ids=["n1", "n2"],
                               gpu_memory_utilization=0.82) == ""

    def test_the_measurements_survive_it(self, tmp_path):
        # They are still true: the model did load, and it did cost that.
        store = self._store(tmp_path)
        store.forget_guard_stops(self.MODEL)
        entry = store.get(self.MODEL)
        assert entry.memory_gb == 61.0
        assert entry.load_seconds == 210.0
        assert entry.guard_stops == 0

    def test_the_kills_leave_the_history_too(self, tmp_path):
        store = self._store(tmp_path)
        store.forget_guard_stops(self.MODEL)
        assert not any(h.get("guard_stop")
                       for h in store.get(self.MODEL).history)

    def test_clearing_nothing_says_so(self, tmp_path):
        store = MeasurementStore(tmp_path / "m.json")
        assert store.forget_guard_stops("org/never-stopped") is False

    def test_the_refusal_names_the_way_out(self, tmp_path):
        from ainode.safety.admission import check_admission

        app = {"measurement_store": self._store(tmp_path), "model_manager": None}
        refusal = check_admission(app, self.MODEL, node_ids=["n1", "n2"],
                                  gpu_memory_utilization=0.82)
        assert "forget-stops" in refusal
        assert self.MODEL in refusal

    def test_the_refusal_is_marked_clearable(self, tmp_path):
        from ainode.safety.admission import check_admission

        app = {"measurement_store": self._store(tmp_path), "model_manager": None}
        refusal = check_admission(app, self.MODEL, node_ids=["n1", "n2"],
                                  gpu_memory_utilization=0.82)
        assert getattr(refusal, "clearable", "") == self.MODEL

    def test_another_refusal_is_not(self, tmp_path):
        # Only a record the operator can drop gets the button.
        from ainode.safety.admission import AdmissionRefusal

        assert getattr(AdmissionRefusal("no room"), "clearable", "") == ""

    def test_both_routes_pass_it_on(self):
        import inspect

        from ainode.engine import sharding_routes
        from ainode.models import api_routes

        for source in (inspect.getsource(sharding_routes.handle_sharding_launch),
                       inspect.getsource(api_routes.handle_model_load)):
            assert '"clearable"' in source

    def test_the_ui_offers_it(self):
        app_js = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
                  "static" / "js" / "app.js").read_text()
        assert "offerToClearTheRecord" in app_js
        assert "/api/measurements/forget-stops" in app_js

    def test_the_endpoint_is_registered(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="n1"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/api/measurements/forget-stops" in paths


class TestUnlockingFromTheUI:
    """A command in a paragraph is not a feature.

        bau das bitte ins UI ein um modelle zu entsperren
    """

    APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
              "static" / "js" / "app.js").read_text()

    def test_the_guard_panel_lists_what_it_stopped(self):
        assert "Stopped by the guard" in self.APP_JS
        assert "_blockedModelsCard" in self.APP_JS

    def test_each_one_has_an_unlock_button(self):
        assert "data-unlock-model" in self.APP_JS
        assert "unlockModel(" in self.APP_JS

    def test_it_reads_the_fleet_not_just_this_node(self):
        # The node that ran out is rarely the one you are looking at.
        card = self.APP_JS.split("async renderConfigMemory()")[1][:900]
        assert "/api/cluster/measurements" in card

    def test_an_empty_list_explains_itself(self):
        # Rather than an empty box that could equally mean "not loaded".
        assert "Nothing is blocked" in self.APP_JS

    def test_it_says_what_unlocking_does_not_do(self):
        # Dropping the record changes nothing about what the launch asks for.
        assert "it will run out again" in self.APP_JS

    def test_the_model_card_carries_a_badge(self):
        assert "blockedBadge" in self.APP_JS
        assert "Blocked</span>" in self.APP_JS

    def test_the_badge_points_at_where_to_lift_it(self):
        badge = self.APP_JS.split("var blockedBadge")[1][:500]
        assert "Settings → Memory Guard" in badge

    def test_the_cluster_route_exists(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="n1"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/api/cluster/measurements/forget-stops" in paths

    def test_the_dispatcher_knows_the_local_handler(self):
        # Without this the head's own record could only be cleared by the
        # head talking to itself over the fabric.
        import inspect

        from ainode.api import server

        source = inspect.getsource(server._cluster_dispatch)
        assert "forget-stops" in source
        assert "handle_forget_stops" in source

    def test_the_shape_matches_what_the_endpoint_returns(self):
        # /api/cluster/measurements answers {models: {name: [per-node]}} —
        # reading it as {nodes: [...]} would list nothing, silently.
        import inspect

        from ainode.measure import api_routes

        assert '{"models": by_model}' in inspect.getsource(api_routes.handle_cluster)
        assert "cluster.models" in self.APP_JS
