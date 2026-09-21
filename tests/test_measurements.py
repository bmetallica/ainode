"""What a model actually did here, written down by the node that ran it.

The catalog carries estimates — a size from the Hub, a memory figure someone
worked out with a calculator, a `verified` flag meaning "a person ran this and
said it worked". Every number that would replace them already existed at the
moment a model came up: the load timings, the host memory reading, the request
counters. Nobody was writing them down, so they were read off a card by a
person and typed back into a source file.

What is tested here is that the machine does that for itself, and that it
does not overreach while doing it: a measurement taken one way must not be
quoted about a launch made another way.
"""

from __future__ import annotations

import json

import pytest

from ainode.measure.store import HISTORY, MeasurementStore


@pytest.fixture()
def store(tmp_path):
    return MeasurementStore(tmp_path / "measurements.json")


class TestTheStore:
    def test_nothing_measured_reads_as_empty(self, store):
        assert store.load() == {}
        assert store.get("org/m") is None

    def test_a_successful_launch_is_written_down(self, store):
        store.record_launch("org/m", ok=True, load_seconds=292.0,
                            memory_gb=68.2, max_model_len=65536,
                            node_id="n3")
        entry = MeasurementStore(store.path).get("org/m")
        assert entry.measured is True
        assert entry.memory_gb == 68.2
        assert entry.load_seconds == 292.0
        assert entry.max_model_len == 65536
        assert entry.launches == 1

    def test_a_failed_one_is_counted_but_not_measured(self, store):
        # The time a failed load took is the time it took to fail, which is
        # not how long this model needs.
        store.record_launch("org/m", ok=False, load_seconds=40.0,
                            memory_gb=12.0)
        entry = store.get("org/m")
        assert entry.failures == 1
        assert entry.launches == 0
        assert entry.load_seconds == 0.0
        assert entry.measured is False

    def test_launches_accumulate(self, store):
        for _ in range(3):
            store.record_launch("org/m", ok=True, load_seconds=100.0,
                                memory_gb=40.0)
        assert store.get("org/m").launches == 3

    def test_the_newest_figures_win(self, store):
        store.record_launch("org/m", ok=True, memory_gb=40.0, load_seconds=100)
        store.record_launch("org/m", ok=True, memory_gb=44.0, load_seconds=90)
        entry = store.get("org/m")
        assert entry.memory_gb == 44.0 and entry.load_seconds == 90.0

    def test_the_history_is_bounded(self, store):
        for i in range(HISTORY + 4):
            store.record_launch("org/m", ok=True, memory_gb=float(i))
        assert len(store.get("org/m").history) == HISTORY

    def test_speed_is_recorded_separately(self, store):
        store.record_launch("org/m", ok=True, memory_gb=40.0)
        store.record_speed("org/m", tokens_per_second=98.5)
        assert store.get("org/m").tokens_per_second == 98.5

    def test_speed_for_an_unmeasured_model_is_ignored(self, store):
        # It would create an entry claiming a speed for a model that never
        # loaded here.
        store.record_speed("org/never", tokens_per_second=98.5)
        assert store.get("org/never") is None

    def test_one_can_be_forgotten(self, store):
        # A measurement outlives the thing it measured: a new engine image, a
        # different quantisation, a node with more memory. When it stops
        # describing reality, deleting it is the honest move.
        store.record_launch("org/m", ok=True, memory_gb=40.0)
        assert store.forget("org/m") is True
        assert store.forget("org/m") is False

    def test_a_broken_file_costs_an_estimate_not_a_launch(self, store):
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text("{ not json")
        assert store.load() == {}

    def test_an_unwritable_store_does_not_raise(self, tmp_path):
        # Losing a measurement must never become losing a launch.
        blocked = MeasurementStore(tmp_path / "nope" / "x" / "m.json")
        blocked.path.parent.parent.mkdir(parents=True)
        blocked.path.parent.write_text("not a directory")
        blocked.record_launch("org/m", ok=True, memory_gb=1.0)

    def test_the_write_is_atomic(self, store):
        store.record_launch("org/m", ok=True, memory_gb=1.0)
        assert not list(store.path.parent.glob("*.tmp"))


class _Backend:
    def __init__(self, phase, timeline=None, config=None):
        self.load_phase = phase
        self.load_timeline = timeline or []
        self.load_seconds = 0
        self.config = config


class _Record:
    def __init__(self, model, kind="llm"):
        self.model = model
        self.kind = kind


class _Instance:
    def __init__(self, model, backend, kind="llm"):
        self.record = _Record(model, kind)
        self.backend = backend


class _Manager:
    def __init__(self, instances):
        self._instances = instances

    def instances(self):
        return list(self._instances)


class _Guard:
    def __init__(self, available_gb):
        self.available_gb = available_gb

    def read(self):
        from ainode.safety.memory_guard import MemoryReading

        return MemoryReading(available_mb=self.available_gb * 1024,
                             total_mb=122 * 1024, readable=True)


class _Config:
    node_id = "n3"
    max_model_len = 65536
    gpu_memory_utilization = 0.87
    engine_backend = ""


class TestTheRecorder:
    def _app(self, store, backend, guard_gb=100.0, kind="llm"):
        return {"config": _Config(), "measurement_store": store,
                "memory_guard": _Guard(guard_gb),
                "instances": _Manager([_Instance("org/m", backend, kind)])}

    def test_the_cost_of_a_load_is_a_subtraction(self, store):
        # The only moment the baseline can be taken is while it is still
        # loading; afterwards the memory is already gone.
        from ainode.measure.recorder import Recorder

        backend = _Backend("loading_weights", config=_Config())
        app = self._app(store, backend, guard_gb=100.0)
        recorder = Recorder(app)
        recorder.poll()                       # baseline: 100 GB free
        backend.load_phase = "ready"
        app["memory_guard"] = _Guard(32.0)    # 68 GB gone
        recorder.poll()
        assert store.get("org/m").memory_gb == 68.0

    def test_the_timeline_is_kept_with_it(self, store):
        from ainode.measure.recorder import Recorder

        backend = _Backend("starting", config=_Config())
        app = self._app(store, backend)
        recorder = Recorder(app)
        recorder.poll()
        backend.load_phase = "ready"
        backend.load_timeline = [{"phase": "starting", "seconds": 38.0},
                                 {"phase": "profiling", "seconds": 183.0}]
        recorder.poll()
        entry = store.get("org/m")
        assert entry.load_seconds == 221.0
        assert entry.load_timeline[0]["phase"] == "starting"

    def test_the_launch_settings_are_kept_as_context(self, store):
        # A memory figure at 64k context says nothing about 256k, so the
        # figure is worthless without what it was measured at.
        from ainode.measure.recorder import Recorder

        backend = _Backend("starting", config=_Config())
        app = self._app(store, backend)
        recorder = Recorder(app)
        recorder.poll()
        backend.load_phase = "ready"
        recorder.poll()
        entry = store.get("org/m")
        assert entry.max_model_len == 65536
        assert entry.gpu_memory_utilization == 0.87

    def test_a_failure_is_recorded_too(self, store):
        from ainode.measure.recorder import Recorder

        backend = _Backend("loading_weights", config=_Config())
        app = self._app(store, backend)
        recorder = Recorder(app)
        recorder.poll()
        backend.load_phase = "failed"
        recorder.poll()
        entry = store.get("org/m")
        assert entry.failures == 1 and entry.launches == 0

    def test_it_records_once(self, store):
        from ainode.measure.recorder import Recorder

        backend = _Backend("starting", config=_Config())
        app = self._app(store, backend)
        recorder = Recorder(app)
        recorder.poll()
        backend.load_phase = "ready"
        recorder.poll()
        recorder.poll()
        recorder.poll()
        assert store.get("org/m").launches == 1

    def test_an_instance_already_up_when_first_seen_is_not_a_launch(self, store):
        # A restart, or the recorder starting later. It did not happen now.
        from ainode.measure.recorder import Recorder

        app = self._app(store, _Backend("ready", config=_Config()))
        Recorder(app).poll()
        assert store.get("org/m") is None

    def test_a_speed_from_too_few_requests_is_not_written_down(self, store):
        # Three requests is noise, and writing it down makes it look like a
        # fact.
        from ainode.measure.recorder import Recorder

        store.record_launch("org/m", ok=True, memory_gb=40.0)

        class _Collector:
            def model_stats(self):
                return {"org/m": {"requests": 3, "avg_tokens_per_second": 999.0}}

        app = self._app(store, _Backend("ready", config=_Config()))
        app["metrics_collector"] = _Collector()
        Recorder(app).poll()
        assert store.get("org/m").tokens_per_second == 0.0

    def test_a_speed_from_enough_is(self, store):
        from ainode.measure.recorder import Recorder

        store.record_launch("org/m", ok=True, memory_gb=40.0)

        class _Collector:
            def model_stats(self):
                return {"org/m": {"requests": 40, "avg_tokens_per_second": 98.5}}

        app = self._app(store, _Backend("ready", config=_Config()))
        app["metrics_collector"] = _Collector()
        Recorder(app).poll()
        assert store.get("org/m").tokens_per_second == 98.5

    def test_a_broken_instance_list_is_silence(self, store):
        from ainode.measure.recorder import Recorder

        class _Boom:
            def instances(self):
                raise RuntimeError("no")

        Recorder({"instances": _Boom(), "measurement_store": store}).poll()

    def test_it_runs_on_every_node_not_only_with_telemetry(self):
        # A measurement that only existed when MQTT was configured would be
        # missing from exactly the deployments that most need it.
        import inspect

        from ainode.api import server

        assert "recorder.poll()" in inspect.getsource(server._cluster_sync_loop)


class _Req:
    def __init__(self, app, **match):
        self.app = app
        self.match_info = match


class TestItIsVisible:
    def test_the_local_route_lists_them(self, store):
        import asyncio

        from ainode.measure.api_routes import handle_local

        store.record_launch("org/m", ok=True, memory_gb=68.2, node_id="n3")
        app = {"config": _Config(), "measurement_store": store}
        body = json.loads(asyncio.run(handle_local(_Req(app))).body)
        assert body["measurements"][0]["memory_gb"] == 68.2
        assert body["measurements"][0]["measured"] is True

    def test_one_can_be_thrown_away_over_the_api(self, store):
        import asyncio

        from ainode.measure.api_routes import handle_forget

        store.record_launch("org/m", ok=True, memory_gb=1.0)
        app = {"measurement_store": store}
        body = json.loads(asyncio.run(
            handle_forget(_Req(app, model="org/m"))).body)
        assert body["forgotten"] is True

    def test_the_routes_exist(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="head"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert {"/api/measurements", "/api/cluster/measurements"} <= paths


class TestThePlannerPrefersIt:
    def test_the_plan_carries_the_measurement_beside_its_estimate(self, store):
        # Not instead of. The difference between the two is the interesting
        # part — a plan that predicted 68 GB for something that cost 74 is a
        # planner worth correcting.
        from ainode.planner.api_routes import _attach_measurement

        store.record_launch("org/m", ok=True, memory_gb=74.0,
                            load_seconds=292.0, max_model_len=65536)
        payload = {"weights_gb": 68.0}
        _attach_measurement({"measurement_store": store}, "org/m", payload)
        assert payload["measured"]["memory_gb"] == 74.0
        assert payload["measured"]["vs_plan_gb"] == 6.0

    def test_an_unmeasured_model_adds_nothing(self, store):
        from ainode.planner.api_routes import _attach_measurement

        payload = {"weights_gb": 68.0}
        _attach_measurement({"measurement_store": store}, "org/never", payload)
        assert "measured" not in payload

    def test_the_gate_uses_what_it_cost_last_time(self, store):
        from ainode.safety.admission import _measured_says

        store.record_launch("org/m", ok=True, memory_gb=90.0,
                            max_model_len=65536)
        app = {"measurement_store": store, "config": _Config(),
               "cluster_state": None}
        measured = {"memory_gb": 90.0, "max_model_len": 65536}
        # No nodes known: falls through rather than refusing on nothing.
        assert _measured_says(app, "org/m", measured, 65536) is None

    def test_a_different_context_length_is_not_quoted_at(self, store):
        # A model measured at 64k and launched at 256k needs a different
        # amount; pretending otherwise is the overconfidence the estimate is
        # criticised for.
        from ainode.safety.admission import _measured_says

        measured = {"memory_gb": 90.0, "max_model_len": 65536}
        assert _measured_says({}, "org/m", measured, 262144) is None


class TestTheUIShowsIt:
    def test_the_hint_puts_it_beside_the_estimate(self):
        from pathlib import Path

        app_js = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
                  "static" / "js" / "app.js").read_text()
        assert "Measured here:" in app_js
        assert "vs the plan" in app_js
        assert "line + measured + warn" in app_js
