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

#: Engine container names, as the backends create them. A member node has no
#: instance record to stop, so these are what it has to work with. Kept in
#: step with EugrBackend.CONTAINER_BASENAME and DiffusersBackend's; a test
#: asserts they still match.
ENGINE_CONTAINERS = ("vllm_node", "ainode_image")

#: How often to look. One second, because the thing being watched moves at
#: memory bandwidth: a 100 GB KV allocation on a 273 GB/s machine crosses the
#: whole reserve in well under a second, and two seconds of grace was two
#: seconds spent after the node was already unrecoverable.
POLL_SECONDS = 1.0

#: A single dip is not a verdict. The KV cache is allocated in one go, and the
#: kernel reclaims page cache right behind it, so a momentary reading below the
#: WARNING line is normal. Below the CRITICAL line it is not: acting on the
#: second sample meant acting a second late, on a node that had four
#: gigabytes left when the first one was taken.
BREACHES_BEFORE_ACTING = 1

#: Falling this fast, in MB per second, while already inside the warning band
#: is treated as a breach in its own right. Waiting for the critical line
#: there is waiting for a number that will be passed between two samples: an
#: engine sizing its cache goes from "plenty" to "none" without ever being
#: observed in between, which is exactly how two nodes were lost.
FAST_DROP_MB_PER_SECOND = 4096.0

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
                 clock: Callable[[], float] = time.monotonic,
                 on_action: Optional[Callable[[dict], None]] = None):
        self._app = app
        #: Called right after an engine is stopped, with the action record.
        #: The guard samples every two seconds and telemetry publishes every
        #: thirty, so without this the one event worth knowing about arrives
        #: up to half a minute late — or not at all, if the node goes down in
        #: between.
        self.on_action = on_action
        #: Every stop since this process started. reading.actions keeps only
        #: the last few, so it cannot answer "has this happened before".
        self.stops = 0
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
        #: (monotonic, available_mb) of the previous sample, for the slope.
        self._previous: Optional[tuple] = None
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
        if not reading.readable:
            self._breaches = 0
            self._previous = None
            return

        falling = self._falling_fast(reading)
        if not reading.critical and not falling:
            self._breaches = 0
            return
        if falling and not reading.critical:
            logger.error("host memory at %.0f MB and falling faster than "
                         "%.0f MB/s — acting inside the warning band rather "
                         "than waiting for %.0f MB, which would be crossed "
                         "between two samples", reading.available_mb,
                         FAST_DROP_MB_PER_SECOND, self.critical_mb)
        self._breaches += 1
        if self._breaches < BREACHES_BEFORE_ACTING:
            logger.warning("host memory at %.0f MB, below the %.0f MB line "
                           "(%d/%d)", reading.available_mb, self.critical_mb,
                           self._breaches, BREACHES_BEFORE_ACTING)
            return
        self._breaches = 0
        self.act(reading)

    def _falling_fast(self, reading: MemoryReading) -> bool:
        """Inside the warning band and dropping faster than the reserve lasts."""
        now = self._clock()
        previous, self._previous = self._previous, (now, reading.available_mb)
        if previous is None or not reading.blocking:
            return False
        elapsed = now - previous[0]
        if elapsed <= 0:
            return False
        return (previous[1] - reading.available_mb) / elapsed > FAST_DROP_MB_PER_SECOND

    def act(self, reading: MemoryReading) -> Optional[str]:
        """Stop the newest engine. Returns the model stopped, or None."""
        instance = self._newest_instance()
        if instance is None:
            # A MEMBER node has no instance record for a distributed launch:
            # its engine container is started over SSH by the head's launcher,
            # and AINode here never created one. That is the node this guard
            # could do nothing for — and in a two-node launch it is half the
            # cluster. The container is right there on the socket.
            stopped = self._kill_engine_containers()
            if stopped:
                self._record(reading, ", ".join(stopped), (
                    f"Stopped by the host memory guard: only "
                    f"{reading.available_mb:.0f} MB of host memory were left "
                    f"(limit {reading.critical_mb:.0f} MB). No instance is "
                    f"recorded here — this node is a member of a launch the "
                    f"head started — so the engine container(s) were killed "
                    f"directly: {', '.join(stopped)}."))
                return ", ".join(stopped)
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
        # Tell the backend first, while it is still running. Its log stream
        # ends a moment after the kill and the launcher's exit code is the
        # only thing left to explain the death — "the launcher exited (code
        # -9)", which is true, uninformative, and used to overwrite this
        # message on the next sync. The first explanation wins now, and this
        # is it.
        note = getattr(backend, "note_external_stop", None)
        if callable(note):
            try:
                note(reason)
            except Exception:
                logger.debug("could not tell the backend why", exc_info=True)
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
        self._record(reading, model, reason)
        self._remember(model, instance, reading)
        return model or None

    def _remember(self, model: str, instance, reading: MemoryReading) -> None:
        """Write the stop down against the model, not only in the log.

        The next launch of the same model, asking for the same thing, should
        not have to be discovered the same way. A refusal that says "this was
        killed here, at this utilization, on this date" is evidence rather
        than an estimate — and it is the only kind of evidence available for
        a model that has never successfully served.
        """
        if not model:
            return
        config = getattr(getattr(instance, "backend", None), "config", None)
        record = getattr(instance, "record", None)
        try:
            from ainode.measure.store import MeasurementStore

            store = (self._app.get("measurement_store")
                     if self._app is not None else None) or MeasurementStore()
            store.record_guard_stop(
                model,
                free_gb=round(reading.available_mb / 1024, 1),
                node_id=str(getattr(self._app.get("config"), "node_id", "") or ""),
                gpu_memory_utilization=float(
                    getattr(config, "gpu_memory_utilization", 0) or 0),
                max_model_len=int(getattr(config, "max_model_len", 0) or 0),
                nodes=1 + len(list(getattr(record, "peer_ips", []) or [])),
                extra_args=list(getattr(config, "extra_vllm_args", None) or []),
            )
        except Exception:
            logger.debug("could not record the stop against %s", model,
                         exc_info=True)

    def _record(self, reading: MemoryReading, model: str, reason: str) -> None:
        action = {
            "at": time.time(), "model": model, "reason": reason,
            "available_mb": round(reading.available_mb),
            "critical_mb": round(reading.critical_mb),
        }
        self._actions.append(action)
        del self._actions[:-20]
        self.stops += 1
        if self.on_action is not None:
            try:
                self.on_action(action)
            except Exception:
                logger.debug("the memory guard's action callback failed",
                             exc_info=True)

    def _kill_engine_containers(self) -> List[str]:
        """``docker kill`` every engine container this node is running.

        Names, not images: the launcher names them, and an engine image can be
        anything an operator points the node at. Nothing else on the socket is
        touched — AINode's own container is not in this list, and killing it
        would take the guard with it.
        """
        import subprocess

        names: List[str] = []
        try:
            listing = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10)
        except Exception:
            logger.exception("could not list containers")
            return []
        for name in (listing.stdout or "").split():
            if any(name == base or name.startswith(f"{base}-")
                   for base in ENGINE_CONTAINERS):
                names.append(name)
        killed = []
        for name in names:
            try:
                # kill, not stop: this node has no time for a graceful exit.
                subprocess.run(["docker", "kill", name], capture_output=True,
                               text=True, timeout=15)
                killed.append(name)
                logger.error("host memory guard killed container %s", name)
            except Exception:
                logger.exception("could not kill %s", name)
        return killed

    def _newest_instance(self):
        manager = self._app.get("instances") if self._app is not None else None
        if manager is None:
            return None
        try:
            instances = list(manager.instances())
        except Exception:
            return None
        return instances[-1] if instances else None
