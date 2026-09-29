"""What a node uses with nothing loaded.

A profile describes a node as it will be once applied, so the wizard plans
against the node's TOTAL memory rather than what happens to be free right now.
That needs one more figure: what the node itself takes before any model does —
Linux, docker, the AINode container, the registry caches. It is not small, and
it is not the same everywhere: the head measured about 27 GB less usable than
its peers (R3 in upgrade-fixes.md).

So each node writes down its memory in use whenever it is genuinely idle — no
instance, no embedding model, no engine container of any kind (a peer can host
a rank of the head's distributed model without knowing it as an instance) —
and announces the latest figure. The planner uses it; a node never seen idle
falls back to DEFAULT_BASELINE_GB and says it is an estimate.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["IdleBaseline", "DEFAULT_BASELINE_GB", "ENGINE_CONTAINER_PREFIXES"]

#: Assumed when a node has never been seen idle: roughly Linux, docker and the
#: AINode container on a Spark with nothing else on it.
DEFAULT_BASELINE_GB = 8.0

#: Containers that hold model memory. Anything else (the orchestrator itself,
#: the registry caches) is part of the baseline.
ENGINE_CONTAINER_PREFIXES = ("vllm_node", "ainode-vllm-node-solo", "ainode_image")

#: How often to look. Idle is the rare state on a working cluster; checking
#: once a minute catches it without a docker call every five seconds.
CHECK_SECONDS = 60.0


def _engine_containers_running() -> Optional[bool]:
    """True/False, or None when docker cannot be asked."""
    try:
        done = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                              capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    if done.returncode != 0:
        return None
    return any(name.startswith(ENGINE_CONTAINER_PREFIXES)
               for name in (done.stdout or "").split())


class IdleBaseline:
    """Records memory in use at idle; ``poll`` is blocking (docker), so call
    it from an executor."""

    def __init__(self, app, path: Optional[Path] = None):
        self._app = app
        self._path = path
        self._checked = 0.0
        self._value: Optional[float] = None

    @property
    def path(self) -> Path:
        if self._path is not None:
            return self._path
        from ainode.core.config import AINODE_HOME

        return Path(AINODE_HOME) / "baseline.json"

    def used_mb(self) -> float:
        """The last idle reading in MiB, 0.0 when there has not been one."""
        if self._value is None:
            try:
                self._value = float(json.loads(self.path.read_text())
                                    .get("used_mb") or 0)
            except (OSError, ValueError, AttributeError):
                self._value = 0.0
        return self._value

    def _idle_here(self) -> bool:
        manager = self._app.get("instances")
        if manager is not None:
            try:
                if list(manager.instances()):
                    return False
            except Exception:
                return False
        embeddings = self._app.get("embedding_manager")
        if embeddings is not None:
            try:
                if list(embeddings.list_loaded()):
                    return False
            except Exception:
                return False
        return _engine_containers_running() is False

    def poll(self, now: Optional[float] = None) -> None:
        """Take a reading if the node is idle and it is time. Never raises."""
        now = time.time() if now is None else now
        if now - self._checked < CHECK_SECONDS:
            return
        self._checked = now
        try:
            if not self._idle_here():
                return
            collector = self._app.get("metrics_collector")
            metrics = collector.get_gpu_metrics() if collector is not None else {}
            used = float((metrics or {}).get("memory_used_mb") or 0)
            if used <= 0:
                return
            self._value = used
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"used_mb": used, "at": now}))
        except Exception:
            logger.debug("could not record the idle baseline", exc_info=True)
