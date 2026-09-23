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


class TestTheBudgetHintNamesTheUtilization:
    """Reported from the cluster, on an empty node:

        RuntimeError: buffer_size (5086090240 B) exceeds device memory
        budget (1168039936 B)

    A gigabyte of budget on a 128 GB node is not another model in the way —
    it is gpu_memory_utilization x total, minus the weights, and the
    utilization was 0.15 because a since-fixed bug capped this launch against
    a different node. The hint sent the reader to the loader, which is the
    one thing that was working as intended.
    """

    def _hint(self):
        from ainode.engine.load_phase import _INSTANTTENSOR_BUDGET_HINT

        return _INSTANTTENSOR_BUDGET_HINT

    def test_it_says_where_the_budget_comes_from(self):
        assert "gpu-memory-utilization x the node's total memory" in self._hint()

    def test_it_does_the_arithmetic_for_the_reader(self):
        # A number in the message is worth a paragraph of explanation.
        assert "0.15 there leaves 19 GB" in self._hint()

    def test_it_still_offers_the_loader_knobs(self):
        hint = self._hint()
        assert "drop:--load-format" in hint
        assert "INSTANTTENSOR_BUFFER_SIZE=67108864" in hint

    def test_it_still_mentions_a_second_model(self):
        assert "already loaded on that node" in self._hint()

    def test_the_pattern_still_matches_the_engine_line(self):
        from ainode.engine.load_phase import LoadPhaseTracker

        phase = LoadPhaseTracker()
        phase.reset()
        phase.observe("RuntimeError: buffer_size (5086090240 B) exceeds "
                      "device memory budget (1168039936 B)\n")
        phase.fail_exit(1)
        assert "InstantTensor" in phase.failure_reason()


class TestItSaysWhatTheNodeHadFree:
    """The number every memory-shaped failure is really about, and the one
    nobody has.

    Three attempts on the same node reported budgets of 1.09 GiB, 1.49 GiB
    and 449 MiB. Against what? By the time anyone looks, the engine that was
    holding the memory is gone — so the reading has to be taken at the moment
    of failure, not when the message is read.
    """

    def _phase(self, free_mb, line):
        from ainode.engine.load_phase import LoadPhaseTracker

        phase = LoadPhaseTracker()
        phase.reset()
        phase.observe(line)
        phase.fail("the launcher exited (code 1)")
        phase.free_mb_at_failure = free_mb
        return phase

    BUDGET = ("RuntimeError: buffer_size (5086090240 B) exceeds device "
              "memory budget (471216128 B)\n")

    def test_a_memory_failure_carries_the_reading(self):
        reason = self._phase(1200.0, self.BUDGET).failure_reason()
        assert "1.2 GB free" in reason

    def test_an_unrelated_failure_does_not(self):
        # A number that turns out to be irrelevant teaches people to ignore
        # the ones that are not.
        reason = self._phase(1200.0,
                             "ValueError: unrecognized arguments: --nope\n"
                             ).failure_reason()
        assert "GB free" not in reason

    def test_an_unreadable_meminfo_is_silence(self):
        assert "GB free" not in self._phase(None, self.BUDGET).failure_reason()

    def test_it_is_read_at_the_moment_of_failure(self):
        import inspect

        from ainode.engine.load_phase import LoadPhaseTracker

        assert "self.free_mb_at_failure = _host_free_mb()" in \
            inspect.getsource(LoadPhaseTracker.fail)

    def test_the_evidence_still_comes_first(self):
        # Evidence, then explanation, then the reading — the rule this
        # message has followed since the hints were added.
        reason = self._phase(1200.0, self.BUDGET).failure_reason()
        assert reason.index("buffer_size") < reason.index("InstantTensor")
        assert reason.index("InstantTensor") < reason.index("GB free")

    def test_a_sigkill_counts_as_memory_shaped(self):
        from ainode.engine.load_phase import LoadPhaseTracker

        phase = LoadPhaseTracker()
        phase.reset()
        phase.observe("INFO loading weights\n")
        phase.fail_exit(-9)
        phase.free_mb_at_failure = 300.0
        assert "0.3 GB free" in phase.failure_reason()
