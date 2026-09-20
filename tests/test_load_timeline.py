"""Where a load's minutes went.

A launch on this hardware takes minutes, and "minutes" answers neither of the
two questions actually asked about it: "is it stuck?" and "why is this slower
than the same model somewhere else?" Both need to know WHICH minutes. The
phases were already being detected from the engine's own output; the only
thing missing was a clock on them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ainode.engine.load_phase import LoadPhaseTracker


@pytest.fixture()
def clock(monkeypatch):
    """A monotonic clock this test drives by hand."""
    now = {"t": 1000.0}

    import ainode.engine.load_phase as module

    monkeypatch.setattr(module.time, "monotonic", lambda: now["t"])
    return now


def _tracker(clock):
    tracker = LoadPhaseTracker()
    tracker.reset()
    return tracker


class TestTheClock:
    def test_each_phase_is_measured_as_it_is_left(self, clock):
        tracker = _tracker(clock)
        clock["t"] += 40                      # container start + engine import
        tracker.advance("loading_weights")
        clock["t"] += 70                      # reading the weights
        tracker.advance("profiling")
        clock["t"] += 180                     # compiling and sizing the cache
        tracker.observe("INFO: Application startup complete.")

        assert tracker.timeline() == [
            {"phase": "starting", "seconds": 40.0},
            {"phase": "loading_weights", "seconds": 70.0},
            {"phase": "profiling", "seconds": 180.0},
        ]

    def test_the_running_phase_counts_up_during_the_load(self, clock):
        # The answer has to be useful DURING a slow load, which is when it is
        # asked — not only once it is over.
        tracker = _tracker(clock)
        clock["t"] += 25
        assert tracker.timeline() == [{"phase": "starting", "seconds": 25.0}]
        clock["t"] += 10
        assert tracker.timeline()[-1]["seconds"] == 35.0

    def test_readiness_from_the_api_poll_stops_the_clock(self, clock):
        # wait_ready() routinely beats the log line that says the same thing.
        tracker = _tracker(clock)
        clock["t"] += 30
        tracker.advance("loading_weights")
        clock["t"] += 50
        tracker.mark_ready()
        clock["t"] += 600                     # the model has been serving since
        assert tracker.timeline() == [
            {"phase": "starting", "seconds": 30.0},
            {"phase": "loading_weights", "seconds": 50.0},
        ]
        assert tracker.elapsed == 680.0

    def test_marking_ready_twice_does_not_add_a_phase(self, clock):
        tracker = _tracker(clock)
        clock["t"] += 10
        tracker.mark_ready()
        clock["t"] += 10
        tracker.mark_ready()
        assert len(tracker.timeline()) == 1

    def test_a_relaunch_starts_a_new_timeline(self, clock):
        tracker = _tracker(clock)
        clock["t"] += 90
        tracker.mark_ready()
        tracker.reset()
        clock["t"] += 5
        assert tracker.timeline() == [{"phase": "starting", "seconds": 5.0}]

    def test_a_failed_launch_keeps_what_it_measured(self, clock):
        tracker = _tracker(clock)
        clock["t"] += 44
        tracker.advance("loading_weights")
        clock["t"] += 12
        tracker.fail("the launcher exited (code 1)")
        assert tracker.timeline() == [{"phase": "starting", "seconds": 44.0}]

    def test_idle_is_not_a_step_of_a_launch(self, clock):
        tracker = LoadPhaseTracker()
        clock["t"] += 3600
        tracker.advance("starting")
        assert tracker.timeline() == [{"phase": "starting", "seconds": 0.0}]

    def test_the_phases_are_read_from_the_engines_own_output(self, clock):
        # No new parsing: the markers that already drive the progress bar are
        # what the clock is attached to.
        tracker = _tracker(clock)
        clock["t"] += 35
        tracker.observe("INFO 09-20 10:02:11 [gpu_model_runner.py:1] "
                        "Starting to load model Qwen/Qwen3.8...")
        clock["t"] += 62
        tracker.observe("INFO 09-20 10:03:13 [gpu_worker.py:2] "
                        "Memory profiling takes 20.00 seconds")
        clock["t"] += 190
        tracker.observe("INFO: Application startup complete.")
        assert [e["phase"] for e in tracker.timeline()] == [
            "starting", "loading_weights", "profiling"]
        assert tracker.timeline()[-1]["seconds"] == 190.0


class TestItReachesTheHead:
    def test_the_record_carries_it(self):
        from ainode.discovery.instance import InstanceRecord

        record = InstanceRecord(instance_id="i", model="m", head_node_id="n")
        record.load_timeline = [{"phase": "starting", "seconds": 40.0}]
        record.load_seconds = 40.0
        assert InstanceRecord.from_dict(record.to_dict()).load_timeline == \
            [{"phase": "starting", "seconds": 40.0}]

    def test_a_record_from_an_older_build_still_parses(self):
        from ainode.discovery.instance import InstanceRecord

        record = InstanceRecord.from_dict(
            {"instance_id": "i", "model": "m", "head_node_id": "n"})
        assert record.load_timeline == []

    def test_the_backend_exposes_it(self):
        from unittest import mock

        from ainode.core.config import NodeConfig
        from ainode.engine.backends import eugr

        backend = eugr.EugrBackend(NodeConfig(node_id="n", model="org/m"))
        with mock.patch.object(eugr, "detect_gpu", return_value=None):
            assert backend.load_timeline == []
            assert backend.load_seconds >= 0

    def test_it_is_stamped_onto_the_advertised_record(self):
        from ainode.api.server import _stamp_load_state

        class _Backend:
            load_error = ""
            load_phase = "profiling"
            load_detail = ""
            load_timeline = [{"phase": "starting", "seconds": 40.0}]
            load_seconds = 40.0

        class _Record:
            load_error = ""
            load_phase = ""
            load_detail = ""
            load_timeline = []
            load_seconds = 0.0

        class _Instance:
            backend = _Backend()
            record = _Record()

        instance = _Instance()
        _stamp_load_state(instance)
        assert instance.record.load_timeline[0]["seconds"] == 40.0
        assert instance.record.load_seconds == 40.0

    def test_a_backend_without_a_timeline_is_not_an_error(self):
        from ainode.api.server import _stamp_load_state

        class _Backend:
            load_error = ""
            load_phase = "ready"
            load_detail = ""

        class _Record:
            pass

        instance = type("I", (), {"backend": _Backend(), "record": _Record()})()
        _stamp_load_state(instance)
        assert instance.record.load_timeline == []


APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
          "static" / "js" / "app.js").read_text()


class TestTheCard:
    def test_it_is_drawn(self):
        assert "loadTimelineBlock" in APP_JS
        assert "load-timeline" in APP_JS

    def test_the_phases_are_named_in_the_operators_terms(self):
        # "profiling" is vLLM's word for compiling kernels and sizing the KV
        # cache, and it tells an operator nothing about what is slow.
        assert "compiling + sizing the cache" in APP_JS
        assert "container + engine start" in APP_JS

    def test_it_survives_readiness(self):
        # The question is usually asked once the model is already up.
        assert "loaded in " in APP_JS
