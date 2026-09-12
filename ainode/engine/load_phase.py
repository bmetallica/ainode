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

import re

__all__ = ["LOAD_PHASE_MARKERS", "LOAD_PHASE_ORDER", "PHASE_FAILED",
           "LoadPhaseTracker"]

# How much of the tail to keep for a failure message.
_TAIL_LINES = 24

# Lines the launcher prints while tearing down after a failure. They are the
# LAST thing in the log and describe the cleanup, not the cause — quoting them
# produced "Stopping cluster... | Stopping head node... | Cluster stopped." as
# the explanation for a launch that died on an unrecognised vLLM argument.
_TEARDOWN_MARKERS = (
    "stopping cluster", "stopping head node", "stopping worker",
    "cluster stopped", "cleanup", "removing container",
)


def _is_teardown(line: str) -> bool:
    low = line.lower()
    return any(m in low for m in _TEARDOWN_MARKERS)

# An exception line, with vLLM's process prefix tolerated:
#   (EngineCore pid=143) AttributeError: 'NoneType' object has no attribute ...
# The FIRST match in a launch is the root cause. What follows it is usually a
# second traceback from the supervising process ending in vLLM's own
# "Engine core initialization failed. See root cause above" — the tail of the
# log, and the least useful line in it.
_EXCEPTION_RE = re.compile(
    r"^\s*(?:\([^)]*\)\s*)?(?:\[[^\]]*\]\s*)?"
    r"([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Exit)): (.+)$"
)

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

# Failures worth naming before the engine reaches its own traceback, because
# the traceback describes the symptom and not the mistake.
#
# A repository whose architecture resolves to *DraftModel holds the draft half
# of a speculative-decoding pair. It is not servable on its own: vLLM loads it,
# reaches for the speculative_config that would name its base model, finds None
# and dies with "AttributeError: 'NoneType' object has no attribute
# 'draft_model_config'" — which says nothing about what the operator actually
# did wrong. The draft belongs in --speculative-config alongside a base model.
_DRAFTER_HINT = (
    "this repository is a speculative-decoding DRAFT model, not a servable "
    "model. Load the base model it belongs to, and pass this one in "
    "--speculative-config if you want speculative decoding."
)

# Architecture names that mean "drafter". Enumerated because they do not share
# a suffix: two real cases here were DFlashDraftModel and Qwen3DSparkModel, and
# the next vendor will spell it differently again — hence the second,
# name-independent pattern below.
_DRAFTER_ARCHS = ("draftmodel", "dsparkmodel", "dflash", "eagle", "mtpmodel")

# vLLM exits 2 from argparse on an unknown flag, and prints the offending one.
# Worth naming because the usual cause is a recipe written for a different
# engine build than the one actually running.
_ARGPARSE_HINT = (
    "the engine rejected a command-line flag. A catalog recipe is written for a "
    "specific engine build (its engine_image); running it against a different "
    "one fails exactly like this."
)

_FATAL_PATTERNS = [
    ("resolved architecture:", arch, _DRAFTER_HINT) for arch in _DRAFTER_ARCHS
] + [
    # The net for a drafter nobody has enumerated: whatever its architecture is
    # called, it dies reaching through a speculative_config that is None —
    #   AttributeError: 'NoneType' object has no attribute 'draft_model_config'
    ("draft_model_config", "nonetype", _DRAFTER_HINT),
    ("unrecognized arguments", "unrecognizedarguments", _ARGPARSE_HINT),
    ("error: argument", "error:argument", _ARGPARSE_HINT),
]


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
        #: First exception line of this launch — the root cause. Preferred over
        #: the tail, which is usually a supervising process's own traceback.
        self.root_cause = ""
        #: A known mistake recognised from the log, explained in the operator's
        #: terms rather than the engine's. Outranks the root cause.
        self.fatal_hint = ''

    def reset(self) -> None:
        """A fresh log stream means a fresh launch — start the clock over."""
        self.phase = "starting"
        self.ready = False
        self.error = ""
        self.tail = []
        self.root_cause = ""
        self.fatal_hint = ""

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
            if not self.root_cause:
                match = _EXCEPTION_RE.match(stripped)
                if match:
                    self.root_cause = f"{match.group(1)}: {match.group(2)}".strip()
        low = line.lower()
        for lead, needle, message in _FATAL_PATTERNS:
            if lead in low and needle in low.replace(" ", ""):
                self.fatal_hint = message
                break
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
        """One line an operator can act on, or "".

        The root cause wins over the tail. vLLM reports an engine crash twice:
        the real exception in the worker, then "Engine core initialization
        failed. See root cause above" from the supervisor — and that second one
        is what the tail of the log actually contains.
        """
        if self.phase != PHASE_FAILED:
            return ""
        interesting = [ln for ln in self.tail if not _is_teardown(ln)]
        detail = (self.fatal_hint or self.root_cause
                  or " | ".join(interesting[-3:] or self.tail[-3:]))
        return f"{self.error}{(' — ' + detail) if detail else ''}"
