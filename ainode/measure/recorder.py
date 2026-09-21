"""Writing down what a launch did, while it is happening.

Everything here is derived from state the node already keeps: the load-phase
tracker's timings, the host memory reading the guard takes every two seconds,
and the request counters the metrics collector has been keeping all along.
Nothing new is measured — it is only that nobody was writing it down.

Driven from the cluster sync loop, which runs every five seconds on every
node whether or not telemetry is configured. A measurement that only existed
when MQTT was set up would be missing from exactly the deployments that most
need it.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["Recorder"]

_TERMINAL = ("ready", "failed")


class Recorder:
    """Turns instance state changes into measurements."""

    def __init__(self, app):
        self._app = app
        self._phase: Dict[str, str] = {}
        #: Host memory free when each model's load was first seen, so the
        #: cost of the load is a subtraction rather than a guess.
        self._baseline: Dict[str, float] = {}
        self._store = None

    @property
    def store(self):
        if self._store is None:
            from ainode.measure.store import MeasurementStore

            self._store = self._app.get("measurement_store") or MeasurementStore()
            self._app["measurement_store"] = self._store
        return self._store

    # -- the loop calls this ------------------------------------------------

    def poll(self) -> None:
        """Never raises: a bookkeeping failure must not disturb a node."""
        try:
            self._poll()
        except Exception:
            logger.debug("could not record measurements", exc_info=True)

    def _poll(self) -> None:
        available = self._host_available_gb()
        seen = set()
        for model, instance in self._instances():
            seen.add(model)
            backend = instance.backend
            phase = str(getattr(backend, "load_phase", "") or "")
            previous = self._phase.get(model)
            self._phase[model] = phase

            if previous is None:
                # First sight. If it is still loading, remember what the node
                # had free before it finished — that is the only moment the
                # baseline can be taken.
                if phase not in _TERMINAL and available:
                    self._baseline.setdefault(model, available)
                continue
            if phase == previous or phase not in _TERMINAL:
                continue
            self._record(model, instance, phase, available)

        for model in [m for m in self._phase if m not in seen]:
            del self._phase[model]
            self._baseline.pop(model, None)

        self._record_speeds()

    # -- writing ------------------------------------------------------------

    def _record(self, model: str, instance, phase: str,
                available: float) -> None:
        backend = instance.backend
        record = getattr(instance, "record", None)
        config = getattr(backend, "config", None)
        baseline = self._baseline.pop(model, 0.0)
        cost = 0.0
        if phase == "ready" and baseline and available:
            cost = max(0.0, baseline - available)

        timeline = list(getattr(backend, "load_timeline", None) or [])
        seconds = sum(e.get("seconds", 0) for e in timeline) or float(
            getattr(backend, "load_seconds", 0) or 0)

        self.store.record_launch(
            model,
            ok=(phase == "ready"),
            kind=str(getattr(record, "kind", "") or "llm") or "llm",
            node_id=str(getattr(self._app.get("config"), "node_id", "") or ""),
            engine_backend=str(getattr(config, "engine_backend", "") or ""),
            load_seconds=seconds,
            load_timeline=timeline,
            memory_gb=cost,
            max_model_len=int(getattr(config, "max_model_len", 0) or 0),
            gpu_memory_utilization=float(
                getattr(config, "gpu_memory_utilization", 0) or 0),
            max_image_size=int(getattr(config, "max_image_size", 0) or 0)
            if str(getattr(config, "engine_backend", "")) == "diffusers" else 0,
        )

    def _record_speeds(self) -> None:
        """How fast each model answers, from what it has actually served.

        Only for models that have done enough to have an average worth
        keeping — a speed from three requests is noise, and writing it down
        would make it look like a fact.

        An image model has no tokens, so its speed is the average latency:
        one request, one picture. The proxy times those exactly as it times a
        completion, which is why nothing extra has to be measured for it.
        """
        collector = self._app.get("metrics_collector")
        if collector is None:
            return
        try:
            stats = collector.model_stats() or {}
        except Exception:
            return
        kinds = self._kinds()
        for model, entry in stats.items():
            if not isinstance(entry, dict):
                continue
            if (entry.get("requests") or 0) < 5:
                continue
            try:
                if kinds.get(model) == "image":
                    latency = entry.get("avg_latency_ms") or 0
                    if latency:
                        self.store.record_speed(
                            model, seconds_per_image=float(latency) / 1000.0)
                    continue
                speed = entry.get("avg_tokens_per_second")
                if speed:
                    self.store.record_speed(model, tokens_per_second=float(speed))
            except Exception:
                logger.debug("could not record speed for %s", model,
                             exc_info=True)

    def _kinds(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for model, instance in self._instances():
            record = getattr(instance, "record", None)
            out[model] = str(getattr(record, "kind", "") or "llm")
        return out

    # -- reading ------------------------------------------------------------

    def _instances(self):
        manager = self._app.get("instances")
        out = []
        for instance in (manager.instances() if manager is not None else []):
            record = getattr(instance, "record", None)
            model = str(getattr(record, "model", "") or "")
            if model and getattr(instance, "backend", None) is not None:
                out.append((model, instance))
        return out

    def _host_available_gb(self) -> float:
        """Free host memory now, from the guard if it is running.

        The guard reads /proc/meminfo every two seconds anyway; asking it
        costs nothing and keeps one definition of "free" in the process.
        """
        guard = self._app.get("memory_guard")
        if guard is not None:
            try:
                reading = guard.read()
                if reading.readable and reading.available_mb:
                    return reading.available_mb / 1024
            except Exception:
                logger.debug("could not read host memory", exc_info=True)
        from ainode.safety.memory_guard import host_available_mb

        value = host_available_mb()
        return (value / 1024) if value else 0.0


def measured_for(app, model: str) -> Optional[dict]:
    """The measurement for one model on this node, or None."""
    try:
        from ainode.measure.store import MeasurementStore

        store = app.get("measurement_store") or MeasurementStore()
        entry = store.get(model)
    except Exception:
        return None
    return entry.to_dict() if entry is not None and entry.measured else None
