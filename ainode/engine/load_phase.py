"""Coarse engine load phase, derived from the engine's own log stream.

The UI's launching card turns this into a percentage. Without it a launch shows
a flat 8% — the fallback for "no phase reported" — for the whole of a load that
can take minutes on a frontier model, which is indistinguishable from a hang.
That was the state of the eugr backend, the default one.

The markers are deliberately substring matches on lower-cased lines rather than
anything structured: vLLM's log format is not a contract, and a phase that
occasionally fails to advance is a cosmetic problem, while a parser that throws
on an unexpected line is not.
"""

from __future__ import annotations

__all__ = ["LOAD_PHASE_MARKERS", "LOAD_PHASE_ORDER", "PHASE_FAILED",
           "LoadPhaseTracker"]

# How much of the tail to keep for a failure message.
_TAIL_LINES = 12

# Monotonic: a phase only ever moves forward within one launch.
LOAD_PHASE_ORDER = [
    "idle", "starting", "distributing", "loading_weights",
    "distributed_init", "profiling", "ready",
]

# Terminal, and outside the ordering: a launch that died did not reach a later
# phase, it stopped. Without this a dead launcher is indistinguishable from a
# slow load — the card sits at "starting" forever and the only way to find out
# is to read a log file.
PHASE_FAILED = "failed"

LOAD_PHASE_MARKERS = [
    ("loading_weights", ("loading model weights", "loading weights",
                         "loading safetensors")),
    # launch-cluster.sh's own output counts: it copies the image check, starts
    # Ray head and workers and waits for the cluster before vLLM says anything.
    # Without these the bar sits at "starting" through the slowest part of a
    # distributed launch.
    ("distributed_init", ("nccl info", "init_process_group", "rayworkerwrapper",
                          "ray worker", "starting ray", "ray status",
                          "waiting for cluster", "cluster head is responsive",
                          "starting container")),
    ("profiling", ("memory profiling", "available kv cache", "gpu kv cache",
                   "warming up", "autotuning", "capturing cuda graph")),
]

# Lines that mean the API is up. Both appear; whichever lands first wins.
READY_MARKERS = ("uvicorn running on", "application startup complete")


class LoadPhaseTracker:
    """Tracks how far a launch has got, from log lines.

    Shared by both backends so the UI reports the same thing regardless of
    which one is serving. Readiness is a latch the backend may also set from
    its API-poll path — ``wait_ready()`` can win the race against the log
    stream, which would otherwise leave the phase stuck on a model that is
    already serving.
    """

    def __init__(self) -> None:
        self.phase = "idle"
        self.ready = False
        self.error = ""
        #: Last lines seen, so a failure can quote the cause instead of
        #: pointing at a log file the operator then has to go and find.
        self.tail: list = []

    def reset(self) -> None:
        """A fresh log stream means a fresh launch — start the clock over."""
        self.phase = "starting"
        self.ready = False
        self.error = ""
        self.tail = []

    def fail(self, reason: str) -> None:
        """Mark the launch dead. Ignored once the engine is serving — the
        launcher exiting after a successful start is normal for a detached
        engine, and must not retract a working model."""
        if self.ready:
            return
        self.phase = PHASE_FAILED
        self.error = reason.strip()

    def advance(self, phase: str) -> None:
        """Move to ``phase`` only if it is later than the current one."""
        try:
            if LOAD_PHASE_ORDER.index(phase) > LOAD_PHASE_ORDER.index(self.phase):
                self.phase = phase
        except ValueError:
            pass

    def observe(self, line: str) -> bool:
        """Feed one log line. Returns True the first time readiness is seen."""
        if self.ready:
            return False
        stripped = line.rstrip()
        if stripped:
            self.tail.append(stripped)
            del self.tail[:-_TAIL_LINES]
        low = line.lower()
        for phase, markers in LOAD_PHASE_MARKERS:
            if any(m in low for m in markers):
                self.advance(phase)
                break
        if any(m in low for m in READY_MARKERS):
            self.ready = True
            self.phase = "ready"
            return True
        return False

    def current(self, ready_latch: bool = False) -> str:
        """The phase to report. ``ready_latch`` is the backend's own flag."""
        if self.ready or ready_latch:
            return "ready"
        return self.phase

    def failure_reason(self) -> str:
        """One line an operator can act on, or ""."""
        if self.phase != PHASE_FAILED:
            return ""
        detail = " | ".join(self.tail[-3:])
        return f"{self.error}{(' — ' + detail) if detail else ''}"
