"""The facts that travel with a failure.

Everything here is assembled on the head and handed to a model that has never
seen this cluster: what was being launched and with which settings, what the
nodes look like right now, and the part of the engine log that carries the
failure. The operator's own error text is never replaced by any of it — it is
shown as it always was, and this is what goes AROUND it.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = ["log_excerpt", "collect_context", "render_context"]

# Lines that say nothing about a failure and would otherwise eat the whole
# budget: progress bars redraw hundreds of times, and a weight loader prints
# one line per shard.
_NOISE_RE = re.compile(
    r"(\d+%\|)|(\bit/s\])|(\bs/it\])|"
    r"(Loading safetensors checkpoint shards)|"
    r"(^\s*$)"
)

# Lines worth keeping from further back than the tail.
_SIGNAL_RE = re.compile(
    r"(Traceback \(most recent call last\))|(^\s*File \")|"
    r"(\bERROR\b)|(\bCRITICAL\b)|(Error:)|(Exception)|(raise\s)|"
    r"(CUDA)|(out of memory)|(assert)|(Unsupported)|(not supported)|"
    r"(ValueError)|(RuntimeError)|(KeyError)|(OSError)",
    re.IGNORECASE,
)


def log_excerpt(text: str, max_lines: int = 90, max_chars: int = 7000) -> str:
    """The part of an engine log that is about the failure.

    A vLLM log is mostly progress. Handing a model the last 500 lines of it
    spends the whole context on shard counters and leaves the traceback — 300
    lines further up, behind the noise — out of the prompt entirely.
    """
    lines = [ln.rstrip() for ln in str(text or "").splitlines()]
    kept = [ln for ln in lines if not _NOISE_RE.search(ln)]
    if not kept:
        return ""
    tail = kept[-max_lines:]
    earlier = kept[:-max_lines] if len(kept) > max_lines else []
    signal = [ln for ln in earlier if _SIGNAL_RE.search(ln)][-25:]
    out = signal + (["... (log truncated) ..."] if signal else []) + tail
    text = "\n".join(out)
    if len(text) > max_chars:
        # Trimmed from the front: the end of a log is where the failure is.
        text = "... (truncated) ...\n" + text[-max_chars:]
    return text


def _nodes_summary(app) -> List[dict]:
    cluster = app.get("cluster_state")
    out: List[dict] = []
    for node in (cluster.members() if cluster is not None else []):
        status = node.status.value if hasattr(node.status, "value") else str(node.status)
        total = float(getattr(node, "gpu_memory_gb", 0) or 0)
        used_mb = float(getattr(node, "gpu_memory_used_mb", 0) or 0)
        total_mb = float(getattr(node, "gpu_memory_total_mb", 0) or 0)
        used_pct = (used_mb / total_mb * 100) if total_mb else 0.0
        models = [str(i.get("model")) for i in (getattr(node, "instances", []) or [])
                  if isinstance(i, dict) and i.get("model")]
        if getattr(node, "model", "") and node.model not in models:
            models.insert(0, str(node.model))
        out.append({
            "node_id": str(getattr(node, "node_id", "") or ""),
            "node_name": str(getattr(node, "node_name", "") or ""),
            "gpu": str(getattr(node, "gpu_name", "") or ""),
            "memory_gb": round(total, 1),
            "memory_used_pct": round(used_pct),
            "status": status,
            "models": models,
            "embedding_models": [str(e) for e in
                                 (getattr(node, "embedding_models", []) or []) if e],
        })
    return out


def collect_context(app, model: str, node_id: str, error: str,
                    launch: Optional[dict] = None,
                    log_text: str = "") -> dict:
    """Assemble everything known about one failure."""
    nodes = _nodes_summary(app)
    return {
        "model": str(model or ""),
        "node_id": str(node_id or ""),
        "error": str(error or ""),
        "launch": launch or {},
        "nodes": nodes,
        "gpu_names": [n["gpu"] for n in nodes if n["gpu"]],
        "log": log_excerpt(log_text),
    }


def _render_launch(launch: dict) -> str:
    if not launch:
        return "  (the launch parameters for this instance could not be read)"
    interesting = [
        ("nodes", launch.get("node_ids")),
        ("parallelism", launch.get("strategy")),
        ("gpu_memory_utilization", launch.get("gpu_memory_utilization")),
        ("max_model_len", launch.get("max_model_len")),
        ("max_num_seqs", launch.get("max_num_seqs")),
        ("kv_cache_dtype", launch.get("kv_cache_dtype")),
        ("quantization", launch.get("quantization")),
        ("trust_remote_code", launch.get("trust_remote_code")),
        ("engine_image", launch.get("engine_image")),
        ("extra vLLM flags", " ".join(launch.get("extra_vllm_args") or [])),
        ("engine environment", " ".join(
            f"{k}={v}" for k, v in (launch.get("extra_env") or {}).items())),
    ]
    rows = [f"  {name}: {value}" for name, value in interesting
            if value not in (None, "", [], {}, False)]
    # An unset field is a real answer — it means the default applied — so say
    # so once rather than letting the model assume a value was chosen.
    return "\n".join(rows) if rows else "  (all settings left at their defaults)"


def render_context(context: dict) -> str:
    """The context as the model reads it."""
    parts = [f"FAILING MODEL\n  {context.get('model') or '(unknown)'}"]
    node = context.get("node_id")
    if node:
        parts[0] += f"\n  launched from node: {node}"

    parts.append("LAUNCH SETTINGS\n" + _render_launch(context.get("launch") or {}))

    rows = []
    for n in context.get("nodes") or []:
        serving = ", ".join(n["models"] + n["embedding_models"]) or "nothing"
        rows.append(
            f"  {n['node_name'] or n['node_id']} ({n['node_id']}): {n['gpu']}, "
            f"{n['memory_gb']:g} GB, {n['memory_used_pct']}% in use, {n['status']}, "
            f"serving {serving}")
    parts.append("CLUSTER RIGHT NOW\n" + ("\n".join(rows) or "  (no nodes reported)"))

    parts.append("THE ERROR AINODE SHOWED\n" +
                 (context.get("error") or "(no error text was captured)"))

    log = context.get("log")
    if log:
        parts.append("ENGINE LOG (filtered, most recent last)\n" + log)
    else:
        parts.append("ENGINE LOG\n  (no log could be read from that node)")
    return "\n\n".join(parts)
