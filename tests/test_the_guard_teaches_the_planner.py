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

    def test_no_guard_changes_nothing(self):
        from ainode.planner.api_routes import budgets_with_guard_reserve

        app = {"cluster_state": _Cluster([_node("n1")])}
        # (128000 - 8000) MB, in GB: nothing held back beyond the planner's own.
        assert round(budgets_with_guard_reserve(app)[0].free_gb) == 117

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
