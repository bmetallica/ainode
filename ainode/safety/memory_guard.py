"""Stop an engine before it takes the machine with it.

Two nodes were lost launching a model whose memory did not fit. Nothing
noticed: the only admission check on the way in compares ratios of
``gpu_memory_utilization``, applies to stacked loads only, and is absent
entirely from the distributed path — and once a launch is running, nothing
watches at all. vLLM allocates its KV cache after the weights are in, so the
moment the machine dies is minutes after the point where anything checked.

On GB10 this is not an ordinary out-of-memory. The GPU's memory IS the host's
memory: an engine that over-allocates does not get a CUDA error, it starves
the kernel, and the node has to be power-cycled.

Two design decisions worth stating, because both are load-bearing:

* **Its own thread, not an asyncio task.** The launch path blocks the event
  loop for minutes (docker stop, an rsync of the weights, the launcher). A
  guard living in that loop would be deaf during exactly the window it exists
  to cover.
* **/proc/meminfo, not psutil.** Read from inside a container, ``MemAvailable``
  in /proc is the HOST's figure, which is what has to be protected. psutil
  reports the same file but through an abstraction that a future cgroup-aware
  version could quietly reinterpret as the container's own limit, and that
  number would be the wrong one.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["MemoryGuard", "MemoryReading", "host_available_mb", "PRESETS"]

MEMINFO = Path("/proc/meminfo")

#: How often to look. Two seconds is fast enough to catch a cache allocation
#: going wrong and slow enough to cost nothing.
POLL_SECONDS = 2.0

#: A single dip is not a verdict. The KV cache is allocated in one go, and the
#: kernel reclaims page cache right behind it, so a momentary reading below the
#: line is normal. Two in a row is a trend.
BREACHES_BEFORE_ACTING = 2

#: Named starting points. The Spark numbers are for a 128 GB unified-memory
#: node where the GPU allocation and the operating system share one pool; a
#: machine with discrete VRAM needs far less headroom, because an engine that
#: over-allocates there fails with a CUDA error instead of starving the kernel.
PRESETS = {
    "dgx-spark": {"warn_gb": 8.0, "critical_gb": 4.0},
    "generic": {"warn_gb": 2.0, "critical_gb": 1.0},
}

#: Ceiling on the reserve, as a share of the machine's total memory. The
#: defaults are sized for a 128 GB node; applied unchanged to a small machine
#: — a laptop, a CI runner, a container with 8 GB — an 8 GB reserve would
#: refuse every launch there and the guard would read as a broken product
#: rather than as a setting that does not fit. Held back on a Spark this is
#: 19 GB, well above the 8 GB default, so it changes nothing where it matters.
MAX_RESERVE_SHARE = 0.15


@dataclass
class MemoryReading:
    available_mb: float = 0.0
    total_mb: float = 0.0
    warn_mb: float = 0.0
    critical_mb: float = 0.0
    #: True while a new load would be refused.
    blocking: bool = False
    critical: bool = False
    readable: bool = True
    #: What the guard has stopped, newest first.
    actions: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "available_mb": round(self.available_mb),
            "total_mb": round(self.total_mb),
            "warn_mb": round(self.warn_mb),
            "critical_mb": round(self.critical_mb),
            "blocking": self.blocking,
            "critical": self.critical,
            "readable": self.readable,
            "actions": list(self.actions),
        }


def host_available_mb(path: Path = MEMINFO) -> Optional[float]:
    """The host's reclaimable-plus-free memory in MB, or None.

    ``MemAvailable`` rather than ``MemFree``: the kernel counts the page cache
    it can hand back, and on a node that has just read forty gigabytes of
    weights off disk, MemFree is near zero while the machine is perfectly
    healthy. Refusing a launch on that reading would refuse every launch.
    """
    try:
        for line in path.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        logger.debug("could not read %s", path, exc_info=True)
    return None


def host_total_mb(path: Path = MEMINFO) -> Optional[float]:
    try:
        for line in path.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


class MemoryGuard:
    """Watches host memory and stops the newest engine before the node dies."""

    def __init__(self, app, *, warn_gb: float = 8.0, critical_gb: float = 4.0,
                 enabled: bool = True, poll_seconds: float = POLL_SECONDS,
                 meminfo: Path = MEMINFO,
                 clock: Callable[[], float] = time.monotonic):
        self._app = app
        self.warn_mb = max(0.0, float(warn_gb) * 1024)
        self.critical_mb = max(0.0, float(critical_gb) * 1024)
        self.enabled = bool(enabled)
        self._poll = float(poll_seconds)
        self._meminfo = Path(meminfo)
        self._clock = clock
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._breaches = 0
        self._actions: List[dict] = []
        self._last = MemoryReading(warn_mb=self.warn_mb,
                                   critical_mb=self.critical_mb)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        if not self.enabled or self._thread is not None:
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="memory-guard",
                                        daemon=True)
        self._thread.start()
        logger.info("host memory guard: warn below %.0f MB, stop the newest "
                    "engine below %.0f MB", self.warn_mb, self.critical_mb)
        return True

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5)

    def configure(self, *, warn_gb: Optional[float] = None,
                  critical_gb: Optional[float] = None,
                  enabled: Optional[bool] = None) -> None:
        with self._lock:
            if warn_gb is not None:
                self.warn_mb = max(0.0, float(warn_gb) * 1024)
            if critical_gb is not None:
                self.critical_mb = max(0.0, float(critical_gb) * 1024)
            # A critical line above the warning line would stop engines without
            # ever having refused a load, which is the wrong order of defence.
            if self.critical_mb > self.warn_mb:
                self.warn_mb = self.critical_mb
            if enabled is not None:
                self.enabled = bool(enabled)
        if self.enabled and self._thread is None:
            self.start()

    # -- reading -----------------------------------------------------------

    def read(self) -> MemoryReading:
        available = host_available_mb(self._meminfo)
        reading = MemoryReading(
            warn_mb=self.warn_mb, critical_mb=self.critical_mb,
            actions=list(self._actions[-5:]),
        )
        if available is None:
            # Unreadable means unknown, and unknown must not block a launch:
            # this guard is a safety net, not a gate that fails closed on a
            # platform whose /proc looks different.
            reading.readable = False
            return reading
        reading.available_mb = available
        reading.total_mb = host_total_mb(self._meminfo) or 0.0
        warn_mb, critical_mb = self._effective(reading.total_mb)
        reading.warn_mb, reading.critical_mb = warn_mb, critical_mb
        if self.enabled:
            reading.blocking = available < warn_mb
            reading.critical = available < critical_mb
        self._last = reading
        return reading

    def _effective(self, total_mb: float) -> tuple:
        """The reserve actually applied, capped against the machine's size."""
        if total_mb <= 0:
            return self.warn_mb, self.critical_mb
        ceiling = total_mb * MAX_RESERVE_SHARE
        warn = min(self.warn_mb, ceiling)
        # The critical line keeps its distance from the warning line when both
        # are squeezed, or a small machine would refuse loads and kill engines
        # at the same reading.
        critical = min(self.critical_mb, warn / 2 if warn else 0.0)
        return warn, critical

    @property
    def last(self) -> MemoryReading:
        return self._last

    def accepting_loads(self) -> Optional[str]:
        """"" when a load may proceed, else why it may not."""
        reading = self.read()
        if not reading.blocking:
            return ""
        return (
            f"Refusing to start: only {reading.available_mb:.0f} MB of host "
            f"memory are free, below the {reading.warn_mb:.0f} MB this node "
            f"keeps in reserve. On this hardware the GPU allocation and the operating "
            f"system share one pool, so a launch here does not fail — it takes "
            f"the node down. Unload a model, or lower the reserve in Settings "
            f"if you know what this launch needs."
        )

    # -- acting ------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(self._poll):
            try:
                self._tick()
            except Exception:  # pragma: no cover - the guard must not die
                logger.exception("memory guard tick failed")

    def _tick(self) -> None:
        reading = self.read()
        if not reading.readable or not reading.critical:
            self._breaches = 0
            return
        self._breaches += 1
        if self._breaches < BREACHES_BEFORE_ACTING:
            logger.warning("host memory at %.0f MB, below the %.0f MB line "
                           "(%d/%d)", reading.available_mb, self.critical_mb,
                           self._breaches, BREACHES_BEFORE_ACTING)
            return
        self._breaches = 0
        self.act(reading)

    def act(self, reading: MemoryReading) -> Optional[str]:
        """Stop the newest engine. Returns the model stopped, or None."""
        instance = self._newest_instance()
        if instance is None:
            logger.error("host memory at %.0f MB with no instance to stop",
                         reading.available_mb)
            return None
        model = str(getattr(getattr(instance, "record", None), "model", "") or "")
        reason = (
            f"Stopped by the host memory guard: only "
            f"{reading.available_mb:.0f} MB of host memory were left "
            f"(limit {reading.critical_mb:.0f} MB). This was the most recently "
            f"started engine on this node. Nothing else was touched."
        )
        logger.error("%s — stopping %s", reason, model or "the newest instance")
        backend = getattr(instance, "backend", None)
        # kill, not stop: a graceful shutdown takes ten seconds or more, and a
        # node this close to the edge does not have ten seconds.
        for method in ("kill", "stop"):
            fn = getattr(backend, method, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception:
                    logger.exception("%s() failed on the newest instance", method)
        record = getattr(instance, "record", None)
        if record is not None:
            try:
                record.load_error = reason
                record.load_phase = "failed"
                record.status = "failed"
            except Exception:
                logger.debug("could not mark the record", exc_info=True)
        self._actions.append({
            "at": time.time(), "model": model, "reason": reason,
            "available_mb": round(reading.available_mb),
        })
        del self._actions[:-20]
        return model or None

    def _newest_instance(self):
        manager = self._app.get("instances") if self._app is not None else None
        if manager is None:
            return None
        try:
            instances = list(manager.instances())
        except Exception:
            return None
        return instances[-1] if instances else None
