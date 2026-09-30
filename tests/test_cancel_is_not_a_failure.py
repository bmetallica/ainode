"""A launch the operator stops is not a failed launch, and the primary keeps
its kind across a restart.

From the export of 2026-09-30: three cancelled Qwen3.8-Flash-Next launches in
its history as "killed with SIGKILL … almost always the kernel's OOM killer",
and Qwen-Image-2.1 on the head recorded as "llm" after AINode restarted.
"""

from __future__ import annotations

from ainode.core.config import NodeConfig
from ainode.engine.load_phase import PHASE_FAILED, LoadPhaseTracker


class TestTheTracker:
    def test_an_exit_after_a_stop_is_not_a_failure(self):
        t = LoadPhaseTracker()
        t.arm()
        t.reset()
        t.halt()
        t.reset()                      # stop() resets; the log thread ends after it
        t.fail_exit(-9)
        assert t.phase != PHASE_FAILED and t.error == ""

    def test_the_next_launch_fails_normally_again(self):
        t = LoadPhaseTracker()
        t.halt()
        t.arm()
        t.reset()
        t.fail_exit(-9)
        assert t.phase == PHASE_FAILED and "SIGKILL" in t.error

    def test_a_launch_nobody_stopped_still_fails(self):
        t = LoadPhaseTracker()
        t.reset()
        t.fail_exit(1)
        assert t.phase == PHASE_FAILED


class TestTheBackendsSayWhoStoppedThem:
    def test_eugr_halts_before_it_kills_and_arms_on_start(self):
        import inspect

        from ainode.engine.backends.eugr import EugrBackend

        stop = inspect.getsource(EugrBackend.stop)
        assert stop.index("self._phase.halt()") < stop.index("send_signal")
        assert "self._phase.arm()" in inspect.getsource(EugrBackend.start_solo)
        assert "self._phase.arm()" in inspect.getsource(EugrBackend.start_distributed)

    def test_diffusers_too(self):
        import inspect

        from ainode.engine.backends.diffusers import DiffusersBackend

        assert "self._phase.halt()" in inspect.getsource(DiffusersBackend.stop)
        assert "self._phase.arm()" in inspect.getsource(DiffusersBackend.start)


class TestThePrimaryKeepsItsKind:
    def test_an_image_primary_is_an_image(self):
        from ainode.api.server import _head_instances, _primary_kind

        assert _primary_kind(NodeConfig(engine_backend="diffusers")) == "image"
        assert _primary_kind(NodeConfig(engine_backend="eugr")) == "llm"
        records = _head_instances(NodeConfig(model="Qwen/Qwen-Image-2.1", node_id="h",
                                             engine_backend="diffusers",
                                             distributed_mode="head", peer_ips=["10.0.0.2"]))
        assert records[0]["kind"] == "image"
