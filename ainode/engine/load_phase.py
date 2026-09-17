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

# Exceptions whose own line says nothing actionable — the detail follows.
# pydantic is the case that matters here: "1 validation error for ModelConfig"
# is the same sentence whichever argument was wrong.
_CONTINUES_RE = re.compile(r"validation error|ValidationError", re.I)
_CONTINUATION_LINES = 4

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
    # The phrasing is vLLM's and it changes between builds. "Loading model
    # from scratch..." is what the current one prints, and it matched nothing
    # here — so a launch that was reading 23 GB off disk showed "starting ·
    # 12%" for its entire duration, which is indistinguishable from a hang and
    # was reported as one.
    ("loading_weights", ("loading model weights", "loading weights",
                         "loading safetensors", "loading model from scratch",
                         "starting to load model", "loading weights took",
                         "model loading took", "instanttensor")),
    # launch-cluster.sh's own output counts: it copies the image check, starts
    # Ray head and workers and waits for the cluster before vLLM says anything.
    # Without these the bar sits at "starting" through the slowest part of a
    # distributed launch.
    ("distributed_init", ("nccl info", "init_process_group", "rayworkerwrapper",
                          "ray worker", "starting ray", "ray status",
                          "waiting for cluster", "cluster head is responsive",
                          "starting container")),
    ("profiling", ("memory profiling", "available kv cache", "gpu kv cache",
                   "warming up", "autotuning", "capturing cuda graph",
                   "init engine", "torch.compile", "compiling a graph",
                   "graph capturing finished")),
]

# Lines that say what is happening in words, for the phases whose slow part is
# invisible from the phase alone. "Loading weights" covers both reading them
# off a local disk in two minutes and pulling them from Hugging Face in forty,
# and the operator very much wants to know which one they are watching.
DETAIL_MARKERS = [
    (("unauthenticated requests to the hf hub", "downloading from",
      "resolve/main", "fetching "),
     "downloading the model from Hugging Face — this is the slow one"),
    ((".safetensors:", "model-0000"),
     "downloading the model from Hugging Face — this is the slow one"),
    (("loading model from scratch", "loading safetensors",
      "loading weights"),
     "reading the weights from disk"),
    (("torch.compile", "compiling a graph", "inductor"),
     "compiling kernels — a first launch pays this once"),
    (("capturing cuda graph",),
     "capturing CUDA graphs"),
    (("memory profiling", "available kv cache"),
     "sizing the KV cache"),
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

# There is no pattern here for the resolved ARCHITECTURE name, and that is
# the point. One used to match "Resolved architecture: *DraftModel",
# "*MTPModel", "*Eagle*" and friends — and a model with BUILT-IN
# multi-token prediction resolves exactly such an architecture as part of a
# perfectly healthy launch. GLM 5.3 Flash logs
#
#   INFO [model.py:686] Resolved architecture: Glm5NextMTPModel
#
# while doing what its recipe asks, and the hint hijacked an unrelated failure
# to tell the operator to go and load a different model.
#
# The failure that really does mean "a drafter was served alone" is below, and
# it is a symptom rather than a name: vLLM reaches for a speculative_config
# that is None. The other guard is better still and does not involve the log
# at all — models.api_routes.drafter_base_model refuses such a load up front,
# from the catalog, before anything starts.

# vLLM exits 2 from argparse on an unknown flag, and prints the offending one.
# Worth naming because the usual cause is a recipe written for a different
# engine build than the one actually running.
_ARGPARSE_HINT = (
    "the engine rejected a command-line flag. A catalog recipe is written for a "
    "specific engine build (its engine_image); running it against a different "
    "one fails exactly like this."
)

# An illegal instruction means the GPU was handed machine code it will not
# execute. On this hardware the traceback from a real launch put it precisely:
#
#   cudagraph_utils.py:385 in capture: forward_fn(CUDAGraphMode.NONE)
#   ... /root/.cache/vllm/torch_compile_cache/torch_aot_compile/<hash>/
#       inductor_cache/x7/cx7....py:1067 in call
#       triton_red_fused__to_copy_abs_clamp_cutlass_scaled_mm_...run(...)
#   RuntimeError: CUDA driver error: an illegal instruction was encountered
#
# — a Triton kernel that Inductor generated and cached on disk. That cache
# lives on the HOST (/root/.cache/vllm, mounted into every engine container),
# so it outlives an image swap: a kernel compiled by one toolchain can be
# loaded by another. Which is the first thing to rule out, and the cheapest.
#
# This was previously attributed to FlashInfer's prefill kernel under graph
# capture, a different known GB10/sm120 failure. The remedy for that one
# (--enforce-eager) does not necessarily help here: the crash happens in the
# warmup call, CUDAGraphMode.NONE, before any graph is captured.
_ILLEGAL_INSTRUCTION_HINT = (
    "the GPU refused to run a compiled kernel. Most often a stale compile "
    "cache: it keys on the model and its settings but not on the toolchain "
    "that built the kernels, so changing the engine image can leave one the "
    "GPU will not execute. Use Clear compile cache below, then relaunch. If it "
    "comes back, add --enforce-eager to the model's extra vLLM args; if it "
    "still comes back, add --compilation-config '{\"mode\":0}' to stop "
    "Inductor generating kernels at all."
)

# An out-of-bounds access inside a worker kernel. Seen on this cluster mid
# session, on GLM 5.3 Flash served by the experimental B12X stack:
#
#   RuntimeError: Worker failed with error 'CUDA error: an illegal memory
#   access was encountered'
#   (Worker_TP0) torch.AcceleratorError: cudaErrorIllegalAddress
#
# Distinct from cudaErrorIllegalInstruction, and worse in one specific way:
# the model does not necessarily stop. The CUDA context is poisoned, so
# subsequent kernels return whatever is in memory, and the reported symptom
# was an endless run of "!" — token id 0, which is what argmax returns over
# NaN logits. An operator watching output rather than logs sees a model that
# has gone mad, not a crash.
_ILLEGAL_ADDRESS_HINT = (
    "a kernel read or wrote outside its memory. The engine's CUDA context is "
    "unusable afterwards, so anything it produced since — including output "
    "that looked fine — cannot be trusted; a run of \"!!!!\" is the usual "
    "tell. Remove the engine container on EVERY node before relaunching "
    "(docker rm -f), because a stopped one is reused by name. Then clear the "
    "compile cache. If it returns, the attention backend is the next suspect: "
    "try --attention-backend FLASHINFER --block-size 16 to swap the "
    "experimental kernels for vLLM's standard ones."
)

# The B12X fast loader asks the platform for a capability it does not always
# have. Seen on a three-node GB10 cluster:
#
#   RuntimeError: the initial b12x loader requires GPU host page tables
#
# Everything after it in that log was fallout — Ray workers dying, actor
# handles from a dead session — so naming this line is the difference between
# one sentence and two hundred.
_B12X_LOADER_HINT = (
    "the B12X fast loader needs a platform capability this engine container "
    "does not have. The rest of the B12X stack does not depend on it: try "
    "--load-format auto in the model's extra vLLM args, which keeps the b12x "
    "attention, MoE and linear backends and loads the weights the ordinary "
    "way."
)

# A mixed-precision AutoRound / GPTQ checkpoint. Seen on
# aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx:
#
#   Value error, Unsupported weight_bits: 16, currently only support
#   {8, 2, 3, 4}
#
# The 16 is not a mistake in the checkpoint. Its quantization_config carries
# bits=16 as the GLOBAL default and then 22,249 per-layer overrides naming the
# real widths — 2, 3, 4 or float per module. Stock vLLM reads the global value,
# finds 16, and stops; reading the overrides is what the vendor's own plugin
# exists to do. No combination of serve flags changes that.
_MIXED_BITS_HINT = (
    "this checkpoint quantises each layer to a different width and records "
    "the widths per module, with an unquantized default. vLLM's quantization "
    "parser reads only the default, so it sees 16 bits and stops. It is not a "
    "flag this launch is missing: reading a mixed-bit checkpoint needs the "
    "quantization plugin the model's own card names, in the engine image. "
    "Check the model card for the serving stack it was built against."
)

# eugr's launcher copies its launch script to /workspace in the engine
# container and execs it there. Its own images build everything under that
# path, so the assumption is invisible until someone points a DISTRIBUTED
# launch at an image that does not have it:
#
#   Copying launch script to head node...
#   Error response from daemon: Could not find the file /workspace in
#   container vllm_node
#   Error: docker cp to head node failed
#
# vllm/vllm-openai uses /vllm-workspace and has no /workspace. It serves a
# SOLO launch happily — which is what makes this confusing: the same image
# works until the launch becomes distributed.
_WORKSPACE_HINT = (
    "the cluster launcher copied its start script to /workspace and the "
    "engine image has no such directory — the stock vllm/vllm-openai images "
    "use /vllm-workspace. A current launcher creates it first; one from "
    "before that fix does not, so check whether this node's ainode image is "
    "up to date before blaming the engine image. Failing that, clear the "
    "model's engine image so the launch uses the node default, which has "
    "/workspace already."
)

# A model whose kernels need DeepGEMM, on an image where the module is
# present but does not load. Seen launching DeepSeek-V4-Flash:
#
#   RuntimeError: Sparse Attention Indexer CUDA op requires DeepGEMM support
#   in the current vLLM environment.
#
# The cause is usually visible much earlier in the same log, as a warning
# nothing else reacts to:
#
#   Module vllm.third_party.deep_gemm was found but failed to import
#   ImportError: .../deep_gemm/_C...so: undefined symbol:
#   _ZN3c1010ValueErrorC1ENS_14SourceLocation...
#
# That symbol is c10::ValueError from libtorch: the extension was compiled
# against a different torch than the image ships. Models that do not reach
# for DeepGEMM start anyway, so the image looks healthy until one does.
_DEEPGEMM_HINT = (
    "this model needs DeepGEMM kernels and the engine image cannot load them. "
    "Search the same log for \"deep_gemm\" — an ImportError about an "
    "undefined c10 symbol there means the module was built against a "
    "different torch than the image ships, which is a fault in the image "
    "rather than in the model or the launch. Models that do not use DeepGEMM "
    "start on it regardless, which is why it goes unnoticed."
)

# vLLM implements pipeline parallelism per architecture, and a model that does
# not gets minutes into a launch before saying so. The planner refuses this for
# curated models; the log net covers everything else.
_NO_PIPELINE_HINT = (
    "this model cannot be split along the pipeline axis — vLLM implements that "
    "per architecture. Tensor-parallel is the only axis left, and it needs a "
    "power-of-two rank count, so select 1, 2, 4 or 8 nodes rather than the "
    "number you picked."
)

_FATAL_PATTERNS = [
    # Whatever a drafter's architecture is called, serving one alone dies
    # reaching through a speculative_config that is None —
    #   AttributeError: 'NoneType' object has no attribute 'draft_model_config'
    # A symptom, and unlike a name it does not also occur in healthy launches.
    ("draft_model_config", "nonetype", _DRAFTER_HINT),
    ("pipeline parallelism is not supported", "pipelineparallelismisnotsupported",
     _NO_PIPELINE_HINT),
    ("b12x loader requires", "b12xloaderrequires", _B12X_LOADER_HINT),
    ("unsupported weight_bits", "unsupportedweight_bits", _MIXED_BITS_HINT),
    ("requires deepgemm", "requiresdeepgemm", _DEEPGEMM_HINT),
    ("could not find the file /workspace", "couldnotfindthefile/workspace",
     _WORKSPACE_HINT),
    ("cudaerrorillegalinstruction", "cudaerrorillegalinstruction", _ILLEGAL_INSTRUCTION_HINT),
    ("illegal instruction", "illegalinstruction", _ILLEGAL_INSTRUCTION_HINT),
    ("cudaerrorillegaladdress", "cudaerrorillegaladdress", _ILLEGAL_ADDRESS_HINT),
    ("illegal memory access", "illegalmemoryaccess", _ILLEGAL_ADDRESS_HINT),
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
        #: How many further lines still belong to that root cause. Some
        #: exceptions put nothing useful on their own line:
        #:
        #:   ValidationError: 1 validation error for ModelConfig
        #:   quantization
        #:     Input should be 'awq', 'gptq', ... [input_value='modelopt_mixed']
        #:
        #: The first line names the class and the count; the field and the
        #: rejected value — the only parts anyone can act on — come after it.
        self._root_cause_continues = 0
        #: A known mistake recognised from the log, explained in the operator's
        #: terms rather than the engine's. Accompanies the evidence; it does
        #: not replace it.
        self.fatal_hint = ''
        #: The log line that triggered the hint — the one naming the rejected
        #: flag or the unservable architecture. This is the part an operator
        #: can act on.
        self.offending_line = ''
        #: What is happening right now, in the operator's words — set by the
        #: backend for work that produces no log lines of its own. A 20 GB
        #: engine-image pull and a multi-hundred-GB weight copy both happen
        #: before the launcher writes its first line, so without this the UI
        #: shows "starting" and the log file is empty, which is exactly what a
        #: hang looks like.
        self.detail = ''

    def reset(self) -> None:
        """A fresh log stream means a fresh launch — start the clock over."""
        self.phase = "starting"
        self.ready = False
        self.error = ""
        self.tail = []
        self.root_cause = ""
        self._root_cause_continues = 0
        self.fatal_hint = ""
        self.offending_line = ""
        self.detail = ""

    def note(self, detail: str) -> None:
        """Say what is happening now. ``""`` clears it."""
        self.detail = (detail or "").strip()

    def fail(self, reason: str) -> None:
        """Mark the launch dead. Ignored once the engine is serving — the
        launcher exiting after a successful start is normal for a detached
        engine, and must not retract a working model."""
        if self.ready:
            return
        self.phase = PHASE_FAILED
        self.error = reason.strip()

    def advance(self, phase: str) -> None:
        """Move to ``phase`` only if it is later than the current one.

        Moving on clears the detail: it described the phase being left, and a
        stale "copying weights to spark-2" under "loading weights" is worse
        than no detail at all.
        """
        try:
            if LOAD_PHASE_ORDER.index(phase) > LOAD_PHASE_ORDER.index(self.phase):
                self.phase = phase
                self.detail = ""
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
                    if _CONTINUES_RE.search(self.root_cause):
                        self._root_cause_continues = _CONTINUATION_LINES
            elif self._root_cause_continues and not _is_teardown(stripped):
                self._root_cause_continues -= 1
                if len(self.root_cause) < 600:
                    self.root_cause = f"{self.root_cause} | {stripped[:200]}"
        low = line.lower()
        for lead, needle, message in _FATAL_PATTERNS:
            if lead in low and needle in low.replace(" ", ""):
                self.fatal_hint = message
                if not self.offending_line:
                    self.offending_line = stripped[:300]
                break
        for phase, markers in LOAD_PHASE_MARKERS:
            if any(m in low for m in markers):
                self.advance(phase)

        # After advance(), which clears a detail belonging to the phase just
        # left — so this sets the detail of the phase now current.
        for markers, text in DETAIL_MARKERS:
            if any(m in low for m in markers):
                self.detail = text
                break
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

        Evidence first, explanation second. The hint used to REPLACE the log
        line it was explaining, which threw away the only part that identified
        the failure: "the engine rejected a command-line flag" is true of
        every exit-2 and names none of them, while "unrecognized arguments:
        --speculative_config" says exactly what to change. Observed on
        hardware, where a launch failed and the message could not distinguish
        which flag had been refused.

        The root cause wins over the tail. vLLM reports an engine crash twice:
        the real exception in the worker, then "Engine core initialization
        failed. See root cause above" from the supervisor — and that second one
        is what the tail of the log actually contains.
        """
        if self.phase != PHASE_FAILED:
            return ""
        interesting = [ln for ln in self.tail if not _is_teardown(ln)]
        evidence = (self.offending_line or self.root_cause
                    or " | ".join(interesting[-3:] or self.tail[-3:]))
        parts = [p for p in (evidence, self.fatal_hint) if p]
        detail = " — ".join(parts)
        return f"{self.error}{(' — ' + detail) if detail else ''}"
