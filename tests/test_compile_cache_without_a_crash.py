"""Clearing the compiled-kernel cache must not require a crash first.

Reported, live, mid-session: GLM emitted an endless run of "!" — token id 0,
which is what argmax returns over NaN logits. The engine log had

    RuntimeError: Worker failed with error 'CUDA error: an illegal memory
    access was encountered'
    (Worker_TP0) torch.AcceleratorError: cudaErrorIllegalAddress

The advice was "press Clear compile cache in the UI". The operator could not
find it, and was right: that button exists only inside a failure note, gated
on the error text matching /compile cache|illegal instruction|compiled
kernel/. "illegal memory access" is not "illegal instruction", so the regex
declined to draw it — and the instance was not even in a failed state.

Two fixes: the regex now covers the fault this cluster actually produced, and
the cache is clearable from every instance card, because clearing it is
ordinary maintenance rather than a crash remedy.
"""

from __future__ import annotations

import re
from pathlib import Path

from ainode.engine.load_phase import LoadPhaseTracker

APP_JS = Path(__file__).resolve().parent.parent / "ainode" / "web" / "static" / "js" / "app.js"

FAULT = ("(EngineCore pid=676) ERROR 09-14 07:08:06 [core.py:1485] RuntimeError: "
         "Worker failed with error 'CUDA error: an illegal memory access was "
         "encountered")


class TestTheButtonIsAlwaysThere:
    def _source(self) -> str:
        return APP_JS.read_text()

    def test_every_instance_card_carries_it(self):
        assert "instance-cache-clear" in self._source()

    def test_it_targets_every_node_the_instance_runs_on(self):
        # Kernels are compiled per node; clearing one of two leaves the fault
        # in place on the other, which reads as "clearing did not help".
        source = self._source()
        assert 'data-nodes="' in source
        assert "for (var i = 0; i < (labels.length || 1); i++)" in source

    def test_it_strips_the_port_from_a_node_label(self):
        # The card renders "spark-1432:8001" for a stacked instance, and
        # _nodeIdForLabel only knows the name.
        assert "l.split(':')[0]" in self._source()

    def test_it_says_to_unload_first(self):
        # The cache is read at launch; clearing it under a running engine
        # achieves nothing and can land mid-compile.
        assert "Unload the model first" in self._source()


class TestTheFailureNoteRecognisesThisFault:
    def test_illegal_memory_access_now_matches(self):
        source = APP_JS.read_text()
        pattern = re.search(r"/(compile cache[^/]*)/i\.test\(error\)", source)
        assert pattern, "the failure-note trigger moved"
        regex = re.compile(pattern.group(1), re.I)
        assert regex.search("CUDA error: an illegal memory access was encountered")
        assert regex.search("torch.AcceleratorError: cudaErrorIllegalAddress")

    def test_the_old_faults_still_match(self):
        source = APP_JS.read_text()
        regex = re.compile(
            re.search(r"/(compile cache[^/]*)/i\.test\(error\)", source).group(1),
            re.I)
        assert regex.search("an illegal instruction was encountered")
        assert regex.search("stale compile cache")


class TestTheHint:
    def _reason(self) -> str:
        tracker = LoadPhaseTracker()
        tracker.reset()
        tracker.observe(FAULT)
        tracker.fail("the launcher exited (code 1)")
        return tracker.failure_reason()

    def test_the_evidence_survives(self):
        # A hint never replaces the line it explains.
        assert "illegal memory access" in self._reason()

    def test_it_warns_that_earlier_output_is_suspect(self):
        """The distinguishing property of this fault: the engine does not
        necessarily stop. It keeps answering, with garbage."""
        assert "cannot be trusted" in self._reason()

    def test_it_names_the_container_removal(self):
        # A stopped container is reused by name, so a plain relaunch can hand
        # back the poisoned process.
        assert "docker rm -f" in self._reason()

    def test_it_offers_the_backend_swap_last(self):
        reason = self._reason()
        assert "--attention-backend FLASHINFER" in reason
        assert reason.index("compile cache") < reason.index("--attention-backend")

    def test_it_is_not_confused_with_the_instruction_fault(self):
        tracker = LoadPhaseTracker()
        tracker.reset()
        tracker.observe("RuntimeError: CUDA driver error: an illegal instruction was encountered")
        tracker.fail("the launcher exited (code 1)")
        assert "cannot be trusted" not in tracker.failure_reason()

    def test_a_healthy_launch_gets_no_hint(self):
        tracker = LoadPhaseTracker()
        tracker.reset()
        for line in ("Loading model from scratch...",
                     "Using FLASHINFER attention backend",
                     "GPU KV cache size: 1,072,101 tokens"):
            tracker.observe(line)
        assert "cannot be trusted" not in tracker.failure_reason()
