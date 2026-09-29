"""What the engine says about its own memory, read back out of its log.

The measurement store records one number for a launch: the drop in
``MemAvailable`` on the node between before the load and after. That is the
whole footprint — weights, engine, and the KV cache the engine sized to fill
its pool — and it was being compared against the planner's *weights*, which is
a different quantity. From the cluster, Smaug-Flash across two nodes:

    ▣ Measured here: 83.1 GB ... (-0.6 GB vs the plan)
      Weights: 159.4 GB on disk, split 2 ways = 83.7 GB per node

Those two agreeing to within a gigabyte looked like a planner that was right.
It was two errors cancelling. The engine's own figures for that launch:

    Model loading took 67.7 GiB memory and ... seconds
    GPU KV cache size: 1,109,643 tokens

67.7 of weights plus 13.7 of cache plus the engine is the 83.1 — and the
planner's 83.7 was an estimate of the weights ALONE, which are really 67.7.
Nearly twenty gigabytes per node of over-estimate, hidden behind a coincidence.

vLLM prints all of it. So rather than infer a split from one host-level number,
this reads the three lines the engine writes about itself, and the store keeps
weights and cache apart. Which also makes the ratio between the checkpoint on
disk and the weights in memory a measured quantity instead of TP_REPLICATION,
a constant put there by someone guessing which direction to round.

All three are GiB in the log and decimal GB here, because everything the
planner compares is decimal (ainode/core/units.py).
"""

from __future__ import annotations

import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)

__all__ = ["parse_engine_report"]

#: "Model loading took 67.7038 GiB memory and 421.882 seconds" — per rank.
_WEIGHTS = re.compile(r"Model loading took\s+([\d.]+)\s*GiB", re.I)

#: "Available KV cache memory: 13.72 GiB" — per rank, after profiling.
_KV_MEMORY = re.compile(r"Available KV cache memory:\s+([\d.]+)\s*GiB", re.I)

#: "GPU KV cache size: 1,109,643 tokens, Maximum concurrency for 131,072
#:  tokens per request: 8.47x" — across the whole cache, not per rank.
_KV_TOKENS = re.compile(r"KV cache size:\s*([\d,]+)\s*tokens", re.I)
_KV_CONCURRENCY = re.compile(
    r"Maximum concurrency for\s+([\d,]+)\s+tokens per request:\s*([\d.]+)x",
    re.I)


#: "vLLM API server version 0.11.1rc2.dev104+g1a2b3c" — once per start.
_VERSION = re.compile(r"vLLM API server version\s+([0-9][\w.+\-]*)", re.I)


def _last(pattern, text: str, group: int = 1):
    """The last match, because a log is appended to across relaunches."""
    found = pattern.findall(text)
    if not found:
        return None
    value = found[-1]
    if isinstance(value, tuple):
        value = value[group - 1]
    return value


def _number(text) -> float:
    digits = str(text or "").replace(",", "").strip()
    try:
        return float(digits)
    except ValueError:
        return 0.0


def parse_engine_report(text: str) -> Dict[str, object]:
    """The engine's own memory figures, or {} for the ones it did not print.

    Never raises. A log that says nothing about memory produces an empty dict,
    and the caller keeps the host-level measurement it already had — a partial
    answer must not overwrite a whole one.
    """
    if not text:
        return {}
    from ainode.core.units import gb_from_gib

    out: Dict[str, object] = {}
    try:
        weights = _number(_last(_WEIGHTS, text))
        if weights > 0:
            out["weights_gb"] = round(gb_from_gib(weights), 1)

        kv_memory = _number(_last(_KV_MEMORY, text))
        if kv_memory > 0:
            out["kv_cache_gb"] = round(gb_from_gib(kv_memory), 1)

        tokens = _number(_last(_KV_TOKENS, text))
        if tokens > 0:
            out["kv_tokens"] = int(tokens)

        version = _last(_VERSION, text)
        if version:
            out["engine_version"] = str(version)

        concurrency = _KV_CONCURRENCY.findall(text)
        if concurrency:
            window, times = concurrency[-1]
            if _number(window) > 0:
                out["kv_at_max_model_len"] = int(_number(window))
            if _number(times) > 0:
                out["kv_concurrency"] = round(_number(times), 2)
    except Exception:
        logger.debug("could not parse the engine's memory report",
                     exc_info=True)
        return {}
    return out
