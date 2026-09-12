"""What AINode publishes about itself, as plain JSON dicts.

Separate from the transport on purpose: the payloads are what an operator
builds dashboards and alerts on, so they are worth testing directly, and a
second transport later (a webhook, a push gateway) should not have to
reimplement them.

Three groups, because they answer three different questions and change at
different rates:

``system``   is this node healthy — CPU, memory, disk, network, temperature.
``gpu``      is the accelerator busy, and how hot.
``models``   what is loaded here, how much it is used, how fast it answers.

Plus ``cluster`` from the head only: the fleet view, which no member can
assemble because only the head sees every node's announcements.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["build_payloads", "node_identity"]


def node_identity(config) -> Dict[str, Any]:
    """The fields every payload carries, so a message stands on its own."""
    from ainode import __version__

    return {
        "node_id": getattr(config, "node_id", "") or "",
        "node_name": getattr(config, "node_name", "") or "",
        "version": __version__,
        "timestamp": round(time.time(), 3),
    }


def _instances(app) -> List[Dict[str, Any]]:
    manager = app.get("instances")
    if manager is None:
        return []
    out = []
    for instance in manager.instances():
        record = instance.record
        config = getattr(instance.backend, "config", None)
        entry: Dict[str, Any] = {
            "model": record.model,
            "api_port": record.api_port,
            "status": getattr(record, "status", ""),
            "nodes": 1 + len(list(getattr(record, "peer_ips", []) or [])),
        }
        if config is not None:
            gmu = getattr(config, "gpu_memory_utilization", None)
            if gmu is not None:
                entry["gpu_memory_utilization"] = gmu
            if getattr(config, "max_model_len", None):
                entry["max_model_len"] = config.max_model_len
        phase = getattr(instance.backend, "load_phase", "")
        if phase:
            entry["load_phase"] = phase
        out.append(entry)
    return out


def _embeddings(app) -> List[str]:
    manager = app.get("embedding_manager")
    if manager is None:
        return []
    try:
        return [m.get("id", "") for m in manager.list_loaded() if m.get("id")]
    except Exception:
        logger.debug("embedding list unavailable", exc_info=True)
        return []


def _cluster(app) -> Optional[Dict[str, Any]]:
    """The fleet view — head only.

    A member publishing this would publish its own partial picture under the
    same topic as the head's complete one, and whichever arrived last would
    win. One publisher per fact.
    """
    config = app.get("config")
    cluster = app.get("cluster_state")
    if cluster is None or config is None:
        return None
    if (getattr(config, "distributed_mode", "solo") or "solo") == "member":
        return None

    nodes = []
    total_vram = 0.0
    online = 0
    try:
        members = list(cluster.members())
    except Exception:
        logger.debug("cluster members unavailable", exc_info=True)
        return None

    for node in members:
        status = node.status.value if hasattr(node.status, "value") else str(node.status)
        vram = float(getattr(node, "gpu_memory_gb", 0) or 0)
        total_vram += vram
        if status in ("online", "serving", "member-ready"):
            online += 1
        nodes.append({
            "node_id": node.node_id,
            "node_name": getattr(node, "node_name", ""),
            "status": status,
            "model": getattr(node, "model", "") or "",
            "gpu_memory_gb": round(vram, 1),
            "gpu_memory_used_percent": round(
                float(getattr(node, "gpu_memory_used_pct", 0) or 0), 1),
        })

    return {
        "nodes_total": len(nodes),
        "nodes_online": online,
        "vram_total_gb": round(total_vram, 1),
        "nodes": nodes,
    }


def build_payloads(app, sampler) -> Dict[str, Dict[str, Any]]:
    """Every payload this node has to publish, keyed by topic suffix.

    Never raises: a metric that cannot be read is left out. Telemetry that
    takes the node down with it is worse than no telemetry.
    """
    config = app.get("config")
    identity = node_identity(config)
    collector = app.get("metrics_collector")

    payloads: Dict[str, Dict[str, Any]] = {}

    try:
        payloads["system"] = {**identity, **sampler.sample()}
    except Exception:
        logger.exception("system metrics failed")

    if collector is not None:
        try:
            gpu = collector.get_gpu_metrics() or {}
            if not gpu.get("error"):
                payloads["gpu"] = {**identity, **gpu}
        except Exception:
            logger.exception("gpu metrics failed")

    try:
        models: Dict[str, Any] = {
            **identity,
            "loaded": _instances(app),
            "embeddings": _embeddings(app),
        }
        if collector is not None:
            stats = collector.get_snapshot()
            models["requests_total"] = stats.get("requests", {}).get("total", 0)
            models["errors_total"] = stats.get("requests", {}).get("errors", 0)
            models["uptime_seconds"] = stats.get("uptime_seconds", 0)
            models["per_model"] = collector.model_stats()
        payloads["models"] = models
    except Exception:
        logger.exception("model metrics failed")

    try:
        cluster = _cluster(app)
        if cluster is not None:
            payloads["cluster"] = {**identity, **cluster}
    except Exception:
        logger.exception("cluster metrics failed")

    return payloads
