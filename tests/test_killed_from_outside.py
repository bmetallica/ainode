"""A launch that ends on a signal did not crash — something ended it.

Reported from the cluster, on a two-node launch of
``sparkarena/Minimax-M3-v0-NVFP4-REAP50``:

    FAILED — reading the weights from disk
    the launcher exited (code -9) — (Worker_TP0 pid=261) INFO ... MiniMax M3
    sparse attention selected Triton ... Using FlashInfer CUTLASS Unquantized
    MoE backend ...

Every line of evidence is an INFO line from a healthy startup, because the
process never got to say anything about its own death: SIGKILL cannot be
caught. The exit code was the whole message, and "code -9" is the least
actionable string in the product.

Two things are tested here. The translation — what a signal means on this
hardware — and the attribution: when AINode's own memory guard is what
killed it, that has to survive the launcher exiting a moment later.
"""

from __future__ import annotations

from ainode.engine.load_phase import LoadPhaseTracker


def _failed(rc):
    phase = LoadPhaseTracker()
    phase.reset()
    phase.observe("INFO 09-23 09:20:01 [unquantized.py:316] Using FlashInfer\n")
    phase.fail_exit(rc)
    return phase.failure_reason()


class TestTranslatingTheSignal:
    def test_sigkill_says_something_else_stopped_it(self):
        reason = _failed(-9)
        assert "SIGKILL" in reason
        assert "outside the engine" in reason

    def test_it_names_the_likely_culprit_on_this_hardware(self):
        reason = _failed(-9)
        assert "OOM killer" in reason
        assert "one pool" in reason

    def test_it_says_how_to_confirm(self):
        # So the next person does not have to guess between the kernel, the
        # guard and a stray docker stop.
        assert "dmesg" in _failed(-9)

    def test_it_distinguishes_itself_from_the_guard(self):
        # The guard records its own stops. If this message is all there is,
        # the kernel did it.
        assert "memory guard records its own stops" in _failed(-9)

    def test_a_shells_128_plus_n_is_the_same_death(self):
        # The launcher is a shell script, and a shell reports a killed child
        # as 137 rather than -9.
        assert "SIGKILL" in _failed(137)

    def test_sigterm_is_not_an_oom(self):
        reason = _failed(-15)
        assert "SIGTERM" in reason and "asked it to stop" in reason

    def test_a_segfault_says_where_to_look(self):
        assert "native code" in _failed(-11)

    def test_an_unknown_signal_still_says_it_was_one(self):
        assert "signal 7" in _failed(-7)

    def test_an_ordinary_exit_code_is_unchanged(self):
        reason = _failed(1)
        assert "code 1" in reason
        assert "signal" not in reason

    def test_no_code_at_all_still_reads(self):
        assert "stopped producing output" in _failed(None)


class TestTheFirstExplanationWins:
    """The guard kills, the launcher exits, and the exit used to overwrite
    the only message that said who did it."""

    def test_a_later_exit_does_not_replace_the_reason(self):
        phase = LoadPhaseTracker()
        phase.reset()
        phase.fail("Stopped by the host memory guard: only 900 MB were left")
        phase.fail_exit(-9)
        assert "memory guard" in phase.failure_reason()
        assert "code -9" not in phase.failure_reason()

    def test_a_first_exit_is_still_reported(self):
        phase = LoadPhaseTracker()
        phase.reset()
        phase.fail_exit(-9)
        assert "code -9" in phase.failure_reason()

    def test_a_serving_engine_is_never_retracted(self):
        # A detached launcher exiting after a successful start is normal.
        phase = LoadPhaseTracker()
        phase.reset()
        phase.mark_ready()
        phase.fail_exit(-9)
        assert phase.failure_reason() == ""


class _Phase:
    def __init__(self):
        self.reason = ""

    def fail(self, reason):
        self.reason = reason


class TestTheGuardTellsTheBackend:
    def test_both_backends_can_be_told(self):
        from ainode.engine.backends.diffusers import DiffusersBackend
        from ainode.engine.backends.eugr import EugrBackend

        for backend in (EugrBackend, DiffusersBackend):
            assert callable(getattr(backend, "note_external_stop", None))

    def test_the_guard_tells_it_before_it_kills(self):
        import inspect

        from ainode.safety.memory_guard import MemoryGuard

        source = inspect.getsource(MemoryGuard.act)
        assert "note_external_stop" in source
        # Before, while the engine is still running: afterwards the log
        # stream has already ended and the exit code has already been read.
        assert source.index("note_external_stop") < source.index('for method in ("kill", "stop")')

    def test_the_backend_passes_it_to_the_phase(self):
        class _Backend(_Phase):
            pass

        from ainode.engine.backends.eugr import EugrBackend

        backend = EugrBackend.__new__(EugrBackend)
        backend._phase = _Phase()
        backend.note_external_stop("Stopped by the host memory guard")
        assert backend._phase.reason == "Stopped by the host memory guard"
