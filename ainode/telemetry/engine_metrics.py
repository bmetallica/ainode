"""What the engine itself reports, distilled.

Everything AINode published about a model until now was measured at the
proxy: how many requests went through it, how long they took, how many tokens
came back. That describes the traffic, not the engine. The questions an
operator actually has during a busy hour are inside vLLM and were invisible:

  * how full is the KV cache — the thing that decides how many people can use
    the model at once, and the thing that filled up on GLM here;
  * is anything being PREEMPTED — requests pushed out of the cache and
    recomputed later. A rising preemption count is the early warning before
    "the model suddenly got slow"; it was in this cluster's logs and nowhere
    else;
  * how deep is the queue — running versus waiting, which says whether
    --max-num-seqs is set anywhere near right;
  * how long until the first token, separately from the decode rate, because
    a prefill-bound load and a decode-bound one look identical in tokens per
    second and need opposite fixes.

The image engine publishes the same way, with its own names — how many
pictures, how long each took, how many denoising steps a second. Same topic,
same shape, so a dashboard does not have to know which engine is behind a
model to read how busy it is.

vLLM exposes all of it in Prometheus text format on each instance's own port.
This reads that and republishes a distilled subset: a handful of named fields
rather than several hundred bucket lines, because the point is a dashboard
that can be built without first learning vLLM's metric catalogue.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["parse_prometheus", "distil", "with_slots", "WANTED"]

#: name{labels} value
_LINE_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
                      r"(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[^\s]+)\s*$")

#: The gauges and counters worth a field of their own, and what to call them.
#: Names change between vLLM builds, so each entry lists the spellings seen.
WANTED = (
    # The image engine's own counters. Same topic, same shape: a dashboard
    # should not need to know which engine is behind a model to read how busy
    # it is.
    ("images_generated_total", ("ainode:images_generated_total",), 1.0),
    ("image_seconds_total", ("ainode:image_seconds_total",), 1.0),
    ("seconds_per_image", ("ainode:seconds_per_image",), 1.0),
    ("steps_per_second", ("ainode:steps_per_second",), 1.0),
    ("kv_cache_percent", ("vllm:gpu_cache_usage_perc",
                          "vllm:kv_cache_usage_perc"), 100.0),
    ("requests_running", ("vllm:num_requests_running",
                          "ainode:requests_running"), 1.0),
    ("requests_waiting", ("vllm:num_requests_waiting",), 1.0),
    ("requests_swapped", ("vllm:num_requests_swapped",), 1.0),
    ("preemptions_total", ("vllm:num_preemptions_total",
                           "vllm:num_preemptions",), 1.0),
    ("prompt_tokens_total", ("vllm:prompt_tokens_total",), 1.0),
    ("generation_tokens_total", ("vllm:generation_tokens_total",), 1.0),
    # Prompt tokens looked up in the prefix cache, and how many were there.
    # vLLM has no "cached_tokens" metric; that name exists only in a
    # response's usage, and only with --enable-prompt-tokens-details.
    ("prefix_cache_queries_total", ("vllm:prefix_cache_queries_total",
                                    "vllm:gpu_prefix_cache_queries_total"), 1.0),
    ("prefix_cache_hits_total", ("vllm:prefix_cache_hits_total",
                                 "vllm:gpu_prefix_cache_hits_total"), 1.0),
    ("spec_accepted_tokens_total", ("vllm:spec_decode_num_accepted_tokens_total",), 1.0),
    ("spec_draft_tokens_total", ("vllm:spec_decode_num_draft_tokens_total",), 1.0),
)

#: Fields that are rates or averages rather than counts. Rounding one of
#: these to a whole number turns 0.48 steps a second into 0, which reads as a
#: stalled engine — and 41.3 seconds an image into 41, which is merely wrong.
_FLOAT_FIELDS = frozenset({"seconds_per_image", "steps_per_second"})

#: Histograms worth an average. A full bucket set is dozens of lines per
#: histogram and nothing a dashboard can use without a Prometheus behind it;
#: sum/count is the average, which is what a gauge panel wants.
_AVERAGES = (
    ("time_to_first_token_s", "vllm:time_to_first_token_seconds"),
    ("time_per_output_token_s", "vllm:time_per_output_token_seconds"),
    ("e2e_latency_s", "vllm:e2e_request_latency_seconds"),
    ("queue_time_s", "vllm:request_queue_time_seconds"),
)


def parse_prometheus(text: str) -> Dict[str, float]:
    """Metric name → summed value, labels collapsed.

    Labels are dropped deliberately: an instance serves one model, so vLLM's
    model_name label carries no information here, and a finished request's
    metrics are split across label sets that only make sense added up.
    """
    out: Dict[str, float] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE_RE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        name = match.group("name")
        out[name] = out.get(name, 0.0) + value
    return out


def _first(metrics: Dict[str, float], names) -> Optional[float]:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def distil(text: str) -> Dict[str, Any]:
    """The subset worth publishing, from one instance's /metrics output."""
    metrics = parse_prometheus(text)
    if not metrics:
        return {}
    out: Dict[str, Any] = {}

    for field, names, scale in WANTED:
        value = _first(metrics, names)
        if value is None:
            continue
        if field in ("seconds_per_image", "steps_per_second") and not value:
            # The server only emits these once it has made something. A zero
            # would read as "infinitely fast" on a gauge.
            continue
        # Counters are integers; a gauge scaled to a percentage is not, and
        # neither is a rate.
        if scale != 1.0:
            out[field] = round(value * scale, 2)
        elif field in _FLOAT_FIELDS:
            out[field] = round(value, 3)
        else:
            out[field] = round(value)

    for field, base in _AVERAGES:
        total = metrics.get(f"{base}_sum")
        count = metrics.get(f"{base}_count")
        if total is not None and count:
            out[field] = round(total / count, 4)

    running = out.get("requests_running")
    waiting = out.get("requests_waiting")
    if running is not None and waiting is not None:
        # One number for "is this model over capacity right now". Waiting
        # without running means the queue is not moving at all.
        out["requests_total_in_flight"] = running + waiting

    hits = out.get("prefix_cache_hits_total")
    queries = out.get("prefix_cache_queries_total")
    if hits is not None and queries:
        # Share of prompt tokens that did not have to be prefilled again —
        # since the engine started, not over the last interval.
        out["prefix_cache_hit_rate"] = round(hits / queries, 3)

    accepted = out.get("spec_accepted_tokens_total")
    drafted = out.get("spec_draft_tokens_total")
    if accepted is not None and drafted:
        # The only figure that says whether speculative decoding is earning
        # its keep. Below about 0.5 the drafter is costing more than it saves.
        out["spec_acceptance_rate"] = round(accepted / drafted, 3)
    return out


def with_slots(metrics: Dict[str, Any], slots: int) -> Dict[str, Any]:
    """Add the parallel slots: ``--max-num-seqs``, how many are taken, how many free.

    vLLM does not export its --max-num-seqs, so the number comes from the
    launch arguments. Without it (or without a running count) nothing is
    added — a free-slot count guessed from vLLM's default would be a number
    about a different engine.
    """
    running = metrics.get("requests_running")
    if not slots or running is None or "images_generated_total" in metrics:
        return metrics
    out = dict(metrics)
    out["slots_total"] = int(slots)
    out["slots_used"] = int(running)
    out["slots_free"] = max(0, int(slots) - int(running))
    return out


def _slots_for(app, model: str, args) -> int:
    """--max-num-seqs of this instance: its launch args, then the catalog recipe."""
    from ainode.clients.opencode import _concurrency

    class _Launch:
        extra_vllm_args = list(args or [])

    info = None
    manager = app.get("model_manager")
    if manager is not None:
        try:
            info = manager._find_catalog_by_hf_repo(model)
        except Exception:
            info = None
    try:
        return _concurrency(_Launch, info)
    except Exception:
        return 0


async def collect(app) -> Dict[str, Dict[str, Any]]:
    """instance name → distilled metrics, for every instance on this node."""
    import aiohttp

    from ainode.telemetry.logs import _slug

    session = app.get("client_session")
    if session is None:
        return {}
    targets: List[tuple] = []
    manager = app.get("instances")
    try:
        instances = list(manager.instances()) if manager is not None else []
    except Exception:
        instances = []
    for instance in instances:
        record = getattr(instance, "record", None)
        model = str(getattr(record, "model", "") or "")
        port = getattr(record, "api_port", 0)
        if model and port:
            cfg = getattr(getattr(instance, "backend", None), "config", None)
            targets.append((_slug(model), port, _slots_for(
                app, model, getattr(cfg, "extra_vllm_args", None))))
    if not targets:
        config = app.get("config")
        model = str(getattr(config, "model", "") or "")
        port = getattr(config, "api_port", 0)
        if model and port:
            targets.append((_slug(model), port, _slots_for(
                app, model, getattr(config, "extra_vllm_args", None))))

    async def _one(name: str, port: int, slots: int = 0):
        try:
            async with session.get(
                    f"http://localhost:{port}/metrics",
                    timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status != 200:
                    return name, {}
                return name, with_slots(distil(await resp.text()), slots)
        except Exception:
            # An engine that is still loading has no /metrics yet, and one
            # that has died has none either. Both are reported elsewhere;
            # neither is worth an error here.
            logger.debug("no /metrics from %s on %s", name, port, exc_info=True)
            return name, {}

    # All at once: a stacked node has several, and a two-second timeout each
    # in sequence would hold the publish loop for longer than its interval.
    import asyncio

    results = await asyncio.gather(*[_one(*t) for t in targets],
                                   return_exceptions=True)
    out: Dict[str, Dict[str, Any]] = {}
    for result in results:
        if isinstance(result, BaseException):
            continue
        name, distilled = result
        if distilled:
            out[name] = distilled
    return out
