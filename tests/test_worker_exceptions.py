"""The root cause of a distributed failure is in a worker, not in the parent.

Reported:

    the launcher exited (code 1) — File ".../multiproc_executor.py", line 808,
    in wait_for_ready | raise e from None | Exception: WorkerProc
    initialization failed due to an exception in a background process. See
    stack trace for root cause.

"See stack trace for root cause" is the one line in a traceback that says
nothing, and it was the only one AINode showed. Two reasons, both fixed here:

  * the exception matcher allowed a single "(...)" prefix. A worker's line
    carries the whole relay chain plus a level, a timestamp and a source
    location — so worker exceptions matched nothing, and a distributed launch
    fails in a worker by definition.
  * the first match won outright. The parent's wrapper is raised after the
    inner exception, so even once workers were visible the wrapper could
    still take the slot.
"""

from __future__ import annotations

from ainode.engine.load_phase import LoadPhaseTracker

WORKER = ("(EngineCore pid=676) (RayWorkerProc pid=918) (Worker_TP0 pid=918) "
          "ERROR 09-14 07:08:06 [multiproc_executor.py:991] "
          "torch.AcceleratorError: CUDA error: an illegal memory access was "
          "encountered")
WRAPPER = ("Exception: WorkerProc initialization failed due to an exception in "
           "a background process. See stack trace for root cause.")


def _reason(*lines: str) -> str:
    tracker = LoadPhaseTracker()
    tracker.reset()
    for line in lines:
        tracker.observe(line)
    tracker.fail("the launcher exited (code 1)")
    return tracker.failure_reason()


class TestAWorkerLineIsSeen:
    def test_the_relay_chain_does_not_hide_it(self):
        assert "torch.AcceleratorError" in _reason(WORKER)

    def test_a_plain_exception_still_matches(self):
        assert "ValueError" in _reason("ValueError: something is wrong")

    def test_one_prefix_still_matches(self):
        assert "RuntimeError" in _reason("(APIServer pid=552) RuntimeError: boom")

    def test_a_level_and_timestamp_are_stepped_over(self):
        assert "KeyError" in _reason(
            "(EngineCore pid=1) ERROR 09-17 22:17:36 [core.py:1378] KeyError: 'x'")


class TestTheWrapperYields:
    def test_a_real_exception_after_it_wins(self):
        """The parent raises last, so without this the wrapper always won."""
        reason = _reason(WRAPPER, WORKER)
        assert "illegal memory access" in reason
        assert "See stack trace" not in reason

    def test_a_real_exception_before_it_is_kept(self):
        reason = _reason(WORKER, WRAPPER)
        assert "illegal memory access" in reason

    def test_one_wrapper_does_not_overwrite_another(self):
        reason = _reason(WORKER, WRAPPER, WRAPPER)
        assert "illegal memory access" in reason

    def test_a_wrapper_alone_is_still_reported(self):
        """Better than nothing: it at least says the launch died."""
        assert "WorkerProc initialization failed" in _reason(WRAPPER)

    def test_the_first_real_exception_wins_over_later_ones(self):
        """Teardown throws its own; the first one is the cause."""
        reason = _reason(WORKER, "RuntimeError: Engine core has died")
        assert "illegal memory access" in reason


class TestItStillDoesTheOldJob:
    def test_a_pydantic_continuation_is_still_attached(self):
        reason = _reason(
            "ValidationError: 1 validation error for ModelConfig",
            "quantization",
            "  Input should be 'awq', 'gptq' [input_value='modelopt_mixed']")
        assert "modelopt_mixed" in reason

    def test_a_healthy_launch_reports_nothing(self):
        tracker = LoadPhaseTracker()
        tracker.reset()
        tracker.observe("GPU KV cache size: 1,072,101 tokens")
        assert tracker.failure_reason() == ""
