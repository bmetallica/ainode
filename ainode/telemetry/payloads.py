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
``fabric``   what the RDMA links are carrying, which ``system.network``
             cannot see: RDMA bypasses the kernel stack, so a saturated ring
             reads as zero bytes there.
``safety``   what the host memory guard sees, and what it has had to do.
``transfers`` downloads and mirror runs in flight — hours of work that was
             visible only to whoever had the browser open.

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


def _as_float(value, default: float = 0.0) -> float:
    """A number from whatever a peer sent, or the default.

    Values here arrive from other nodes' announcements, which are written by
    whatever build that node runs. One unparseable field used to take the
    whole cluster payload with it — the topic simply stopped being published,
    which is the least debuggable way for telemetry to fail.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _safety(app) -> Optional[Dict[str, Any]]:
    """What the host memory guard sees, and what it has had to do.

    Published because the guard can kill a running engine, and until now the
    only trace of that was a line in a log file. A dashboard that shows
    memory but not the thing acting on it explains half of what happened.
    """
    guard = app.get("memory_guard")
    if guard is None:
        return None
    try:
        reading = guard.read()
    except Exception:
        logger.debug("memory guard unreadable", exc_info=True)
        return None
    payload: Dict[str, Any] = {
        "memory_guard_enabled": bool(getattr(guard, "enabled", False)),
        "host_available_mb": round(reading.available_mb),
        "host_total_mb": round(reading.total_mb),
        # The ENFORCED lines, not the configured ones: the reserve is capped
        # against the size of the machine, and an alert built on a number the
        # guard is not actually using would fire at the wrong moment.
        "warn_mb": round(reading.warn_mb),
        "critical_mb": round(reading.critical_mb),
        "blocking_launches": bool(reading.blocking),
        "below_critical": bool(reading.critical),
    }
    if not reading.readable:
        payload["host_memory_readable"] = False
    payload["stops_total"] = int(getattr(guard, "stops", 0) or 0)
    actions = list(reading.actions or [])
    if actions:
        payload["last_stop"] = actions[-1]
    return payload


def _transfers(app) -> Optional[Dict[str, Any]]:
    """Downloads and mirror runs in flight.

    These take hours on a 200 GB checkpoint and were visible only to whoever
    had the browser open. A transfer that stalls at 40% overnight is exactly
    the thing telemetry is for.
    """
    out: Dict[str, Any] = {}

    downloads = []
    for job in (app.get("download_jobs") or {}).values():
        if not isinstance(job, dict):
            continue
        if job.get("status") not in ("downloading", "starting", "queued"):
            continue
        entry = {"model": job.get("hf_repo") or job.get("model") or "",
                 "status": job.get("status"),
                 "percent": round(float(job.get("progress") or 0), 1)}
        for field in ("downloaded_bytes", "total_bytes"):
            if job.get(field):
                entry[field] = int(job[field])
        downloads.append(entry)
    if downloads:
        out["downloads"] = downloads

    mirror = app.get("mirror_jobs") or {}
    if mirror.get("running"):
        out["mirror"] = {
            "running": True,
            "model": mirror.get("current") or mirror.get("model") or "",
            "done": int(mirror.get("done") or 0),
            "total": int(mirror.get("total") or 0),
        }
    return out or None


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
        vram = _as_float(getattr(node, "gpu_memory_gb", 0))
        total_vram += vram
        if status in ("online", "serving", "member-ready"):
            online += 1
        # The node object carries megabytes; the percentage is computed for
        # the HTTP response and does not exist here. Reading a field that is
        # not there returned 0.0 for every node, on a cluster with two models
        # loaded — a metric that is always zero is worse than no metric, since
        # a dashboard built on it looks healthy while the node is full.
        used_mb = _as_float(getattr(node, "gpu_memory_used_mb", 0))
        total_mb = _as_float(getattr(node, "gpu_memory_total_mb", 0))
        entry = {
            "node_id": node.node_id,
            "node_name": getattr(node, "node_name", ""),
            "status": status,
            "model": getattr(node, "model", "") or "",
            "gpu_memory_gb": round(vram, 1),
        }
        if total_mb:
            entry["gpu_memory_used_mb"] = round(used_mb)
            entry["gpu_memory_total_mb"] = round(total_mb)
            entry["gpu_memory_used_percent"] = round(used_mb / total_mb * 100, 1)
        utilization = getattr(node, "gpu_utilization", None)
        if isinstance(utilization, (int, float)):
            entry["gpu_utilization_percent"] = round(float(utilization), 1)
        version = str(getattr(node, "version", "") or "")
        if version:
            entry["version"] = version
        # A node on another build can send anything here; iterating a value
        # that is not a list would drop the whole cluster payload.
        raw_instances = getattr(node, "instances", None)
        instances = [
            {"model": i.get("model"), "api_port": i.get("api_port"),
             "status": i.get("status")}
            for i in (raw_instances if isinstance(raw_instances, list) else [])
            if isinstance(i, dict) and i.get("model")
        ]
        if instances:
            entry["instances"] = instances
        nodes.append(entry)

    payload = {
        "nodes_total": len(nodes),
        "nodes_online": online,
        "vram_total_gb": round(total_vram, 1),
        "nodes": nodes,
    }
    # A fleet that does not agree with itself about the build it is running
    # is a distributed launch waiting to fail: the launcher compares engine
    # images across nodes and aborts when they differ, minutes in. One
    # boolean makes that an alert instead of a failed start.
    versions = sorted({n["version"] for n in nodes if n.get("version")})
    if len(versions) > 1:
        payload["versions_agree"] = False
        payload["versions"] = versions
    elif versions:
        payload["versions_agree"] = True
    return payload


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
            # The collector has computed these all along and nothing carried
            # them out. An average latency hides the tail, and the tail is
            # what a user notices.
            latency = stats.get("requests", {}).get("latency_ms")
            if isinstance(latency, dict) and latency:
                models["latency_ms"] = latency
            models["uptime_seconds"] = stats.get("uptime_seconds", 0)
            models["per_model"] = collector.model_stats()
        payloads["models"] = models
    except Exception:
        logger.exception("model metrics failed")

    try:
        transfers = _transfers(app)
        if transfers is not None:
            payloads["transfers"] = {**identity, **transfers}
    except Exception:
        logger.exception("transfer metrics failed")

    try:
        safety = _safety(app)
        if safety is not None:
            payloads["safety"] = {**identity, **safety}
    except Exception:
        logger.exception("safety metrics failed")

    try:
        fabric = app.get("_fabric_sampler")
        if fabric is None:
            from ainode.metrics.fabric import FabricSampler

            fabric = FabricSampler()
            app["_fabric_sampler"] = fabric
        ports = fabric.sample()
        if ports:
            payloads["fabric"] = {**identity, "ports": ports}
    except Exception:
        logger.exception("fabric metrics failed")

    try:
        cluster = _cluster(app)
        if cluster is not None:
            payloads["cluster"] = {**identity, **cluster}
    except Exception:
        logger.exception("cluster metrics failed")

    return payloads
