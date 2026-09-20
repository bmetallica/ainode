"""Things that happen once, published when they happen.

The metric topics answer "what is true now". Some facts are not states but
events — a load finished, a load died — and a dashboard that only samples
every thirty seconds sees a five-minute launch as a phase field that changed
at some point and gives no total, no breakdown, and nothing to alert on.

Watched rather than hooked into the launch paths, for one reason: a launch
can start from the API, from a profile being applied, or from the instance
replay after a restart, and a watcher catches all three without three
separate call sites that each have to remember to fire. The cost is that an
event arrives with the next publish rather than the instant it happens, which
for a launch that took minutes is a rounding error. (The memory guard is the
opposite case and gets the opposite treatment: there the timing IS the
message, so it publishes out of band.)
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

__all__ = ["LaunchWatcher"]

#: The phases that end a launch.
_TERMINAL = ("ready", "failed")


class LaunchWatcher:
    """Turns per-instance phase changes into launch events."""

    def __init__(self, app):
        self._app = app
        self._phase: Dict[str, str] = {}

    def poll(self) -> List[dict]:
        """Events since the last call. Never raises."""
        events: List[dict] = []
        try:
            instances = self._instances()
        except Exception:
            logger.debug("could not read instances for launch events",
                         exc_info=True)
            return events

        seen = set()
        for model, backend in instances:
            seen.add(model)
            phase = str(getattr(backend, "load_phase", "") or "")
            previous = self._phase.get(model)
            self._phase[model] = phase
            if phase == previous or phase not in _TERMINAL:
                continue
            if previous is None:
                # First sight of an instance that is already finished — a
                # restart, or telemetry switched on later. Not an event: it
                # did not happen now, and dating it now would be a lie.
                continue
            events.append(self._event(model, backend, phase))

        for model in [m for m in self._phase if m not in seen]:
            del self._phase[model]
        return events

    def _instances(self):
        out = []
        manager = self._app.get("instances")
        for instance in (manager.instances() if manager is not None else []):
            record = getattr(instance, "record", None)
            model = str(getattr(record, "model", "") or "")
            backend = getattr(instance, "backend", None)
            if model and backend is not None:
                out.append((model, backend))
        if not out:
            engine = self._app.get("engine")
            model = str(getattr(self._app.get("config"), "model", "") or "")
            if engine is not None and model:
                out.append((model, engine))
        return out

    def _event(self, model: str, backend, phase: str) -> dict:
        event = {"model": model,
                 "outcome": "ready" if phase == "ready" else "failed"}
        timeline = list(getattr(backend, "load_timeline", None) or [])
        if timeline:
            event["timeline"] = timeline
            # The total as its own field: a dashboard should not have to sum
            # a list to draw the one number everybody asks for.
            event["seconds"] = round(sum(e.get("seconds", 0) for e in timeline), 1)
        elif getattr(backend, "load_seconds", 0):
            event["seconds"] = round(float(backend.load_seconds), 1)
        if phase == "failed":
            reason = str(getattr(backend, "load_error", "") or "")
            if reason:
                # Truncated: a vLLM traceback can be kilobytes, and an event
                # is not a log. The log topic carries the rest.
                event["error"] = reason[:1000]
        return event
