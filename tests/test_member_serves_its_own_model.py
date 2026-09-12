"""A member node serving its own model must be visible from the head.

The target deployment is one model per node: a chat model on node 1, a coding
model on node 2, embeddings on node 3, each serving on its own. Loading the
second one from the head's UI worked — and then nothing appeared anywhere,
because a node in "member" mode had its announced model blanked
unconditionally ("members serve via the head's sharded engine, not their own
model"). True of a member inside a distributed launch; false of this.

The second half is the same blindness during loading: an instance was
advertised only once its engine answered, so a model that takes twenty
minutes to load did not exist for those twenty minutes.
"""

from __future__ import annotations

import asyncio


from ainode.api.server import _instance_is_starting, _live_instance_records


class _Backend:
    def __init__(self, running=True, phase="loading_weights"):
        self._running = running
        self.load_phase = phase

    def is_running(self):
        return self._running


class _Record:
    def __init__(self, model, status="starting"):
        self.model = model
        self.status = status
        self.instance_id = f"head:{model}"

    def to_dict(self):
        return {"model": self.model, "status": self.status}


class _Instance:
    def __init__(self, model, running=True, phase="loading_weights", status="starting"):
        self.record = _Record(model, status)
        self.backend = _Backend(running, phase)


class _Manager:
    def __init__(self, instances):
        self._i = list(instances)

    def instances(self):
        return self._i

    def is_empty(self):
        return not self._i


def _records(manager, serving):
    """Run the advertiser with a stubbed liveness probe."""
    import ainode.api.server as server

    async def _fake_serving(backend, loop):
        return serving

    original = server._engine_serving
    server._engine_serving = _fake_serving
    try:
        return asyncio.run(_live_instance_records(manager, None))
    finally:
        server._engine_serving = original


class TestStartingInstancesAreAdvertised:
    def test_a_loading_instance_is_advertised_as_starting(self):
        manager = _Manager([_Instance("a/b")])
        records = _records(manager, serving=False)
        assert [r.model for r in records] == ["a/b"]
        assert records[0].status == "starting"

    def test_a_serving_instance_is_advertised_as_serving(self):
        manager = _Manager([_Instance("a/b")])
        records = _records(manager, serving=True)
        assert records[0].status == "serving"

    def test_a_dead_process_drops_out(self):
        # The phantom-READY protection this filter existed for.
        manager = _Manager([_Instance("a/b", running=False)])
        assert _records(manager, serving=False) == []

    def test_a_failed_launch_drops_out(self):
        manager = _Manager([_Instance("a/b", phase="failed")])
        assert _records(manager, serving=False) == []

    def test_a_crashed_instance_is_marked_failed(self):
        inst = _Instance("a/b", running=False, status="serving")
        _records(_Manager([inst]), serving=False)
        assert inst.record.status == "failed"


class TestStartingDetection:
    def test_running_and_not_failed_counts(self):
        assert _instance_is_starting(_Instance("a/b")) is True

    def test_a_failed_phase_does_not(self):
        assert _instance_is_starting(_Instance("a/b", phase="failed")) is False

    def test_a_backend_that_raises_does_not(self):
        class _Boom:
            load_phase = "starting"

            def is_running(self):
                raise RuntimeError("docker is gone")

        inst = _Instance("a/b")
        inst.backend = _Boom()
        assert _instance_is_starting(inst) is False


class TestMemberAnnouncement:
    """The rule itself, read off the source — it lives inside a long loop."""

    def test_the_model_is_gated_on_liveness_not_on_role(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "ainode" / "api" /
               "server.py").read_text()
        assert 'updates["model"] = "" if not engine_serving' in src
        assert 'dmode == "member" or not engine_serving' not in src

    def test_a_serving_member_reports_serving(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "ainode" / "api" /
               "server.py").read_text()
        assert '"serving" if engine_serving else "member-ready"' in src
