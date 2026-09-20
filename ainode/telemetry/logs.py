"""Log lines over MQTT: this node's own, and each engine instance's.

Two sources, published on the same loop as the metrics:

  <prefix>/<node>/logs/ainode           what the orchestrator logged
  <prefix>/<node>/logs/vllm/<instance>  what each engine wrote

MQTT is a poor transport for a firehose, and a vLLM log IS a firehose: it
redraws a progress bar several times a second and prints a line per weight
shard. So nothing here forwards a log — it forwards what has been ADDED since
the last publish, with the redraws dropped and a hard cap per message, and it
says how many lines it dropped rather than pretending it sent everything.

Off unless asked for. A broker that suddenly receives four megabytes a minute
from three nodes because someone ticked telemetry on is not a feature.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["LogBuffer", "LogTail", "LogPublisher", "install_log_buffer"]

#: Progress bars and per-shard chatter. The same rule the error assistant
#: applies to a log it is about to hand a model — for the same reason, that
#: the interesting lines are otherwise buried under redraws.
_NOISE_RE = re.compile(
    r"(\d+%\|)|(\bit/s\])|(\bs/it\])|"
    r"(Loading safetensors checkpoint shards)"
)

#: Never put more than this in one message, however far behind we are.
DEFAULT_MAX_LINES = 100

#: And never more than this many bytes, because one vLLM traceback line can
#: be several kilobytes on its own.
MAX_BYTES = 64 * 1024


def _interesting(line: str) -> bool:
    return bool(line.strip()) and not _NOISE_RE.search(line)


class LogBuffer(logging.Handler):
    """AINode's own log, kept in memory for the publisher to drain.

    A handler rather than a second file: the orchestrator's log goes to
    journald, and asking a container to read the journal of the host that
    started it is a worse dependency than keeping the last few hundred lines
    here.
    """

    def __init__(self, capacity: int = 2000):
        super().__init__()
        self._lines: deque = deque(maxlen=capacity)
        self.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._lines.append((record.levelno, self.format(record)))
        except Exception:  # pragma: no cover - a logging handler must not raise
            pass

    def drain(self, min_level: int = logging.INFO,
              max_lines: int = DEFAULT_MAX_LINES) -> tuple:
        """Everything buffered, newest last. Returns (lines, dropped)."""
        collected: List[str] = []
        while self._lines:
            level, text = self._lines.popleft()
            if level >= min_level:
                collected.append(text)
        return _cap(collected, max_lines)


def install_log_buffer(capacity: int = 2000) -> LogBuffer:
    """Attach the buffer to AINode's own logger tree, once."""
    root = logging.getLogger("ainode")
    for handler in root.handlers:
        if isinstance(handler, LogBuffer):
            return handler
    buffer = LogBuffer(capacity=capacity)
    buffer.setLevel(logging.DEBUG)
    root.addHandler(buffer)
    return buffer


class LogTail:
    """New lines appended to one file since the last read.

    Survives the file being replaced or truncated — a relaunch writes a fresh
    log, and a tail that kept seeking past the old end would go silent for the
    rest of the node's uptime.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._offset = 0
        self._inode: Optional[int] = None
        #: Bytes skipped because the file had grown further than one message
        #: may carry. Reported, because a gap the subscriber cannot see is
        #: worse than a gap.
        self.skipped_bytes = 0

    def read(self, max_lines: int = DEFAULT_MAX_LINES) -> tuple:
        try:
            stat = self.path.stat()
        except OSError:
            return [], 0
        if stat.st_ino != self._inode or stat.st_size < self._offset:
            # A new file (a relaunch writes one) or one truncated in place:
            # read it from the start, because its beginning is the launch the
            # subscriber wants. Bounded, so a log that was already large when
            # we noticed does not arrive in one message.
            self._inode = stat.st_ino
            start = max(0, stat.st_size - MAX_BYTES)
            self.skipped_bytes += start
            self._offset = start
        if stat.st_size <= self._offset:
            return [], 0
        try:
            with self.path.open("rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read(stat.st_size - self._offset)
                self._offset = handle.tell()
        except OSError:
            return [], 0
        text = chunk.decode("utf-8", errors="replace")
        # A partial last line is kept out rather than published broken; the
        # next read starts before it.
        if not text.endswith("\n"):
            cut = text.rfind("\n")
            if cut >= 0:
                self._offset -= len(text[cut + 1:].encode("utf-8"))
                text = text[:cut + 1]
        lines = [ln.rstrip("\r") for ln in text.splitlines() if _interesting(ln)]
        return _cap(lines, max_lines)

    def start_at_end(self) -> None:
        try:
            stat = self.path.stat()
        except OSError:
            return
        self._inode = stat.st_ino
        self._offset = stat.st_size


def _cap(lines: List[str], max_lines: int) -> tuple:
    """(lines, dropped) — the newest that fit, and how many did not."""
    dropped = 0
    if max_lines > 0 and len(lines) > max_lines:
        dropped = len(lines) - max_lines
        lines = lines[-max_lines:]
    size = 0
    kept: List[str] = []
    for line in reversed(lines):
        size += len(line.encode("utf-8")) + 1
        if size > MAX_BYTES:
            dropped += 1
            continue
        kept.append(line)
    kept.reverse()
    return kept, dropped


class LogPublisher:
    """Builds the log payloads for one publish cycle."""

    def __init__(self, app):
        self._app = app
        self._tails: Dict[str, LogTail] = {}
        self._buffer: Optional[LogBuffer] = None

    def enabled(self) -> bool:
        return bool(getattr(self._app.get("config"), "mqtt_logs", False))

    def payloads(self) -> Dict[str, dict]:
        """topic suffix → payload. Empty when there is nothing to say."""
        if not self.enabled():
            return {}
        config = self._app.get("config")
        max_lines = _max_lines(config)
        out: Dict[str, dict] = {}

        lines, dropped = self._own_lines(config, max_lines)
        if lines:
            out["logs/ainode"] = _payload(config, "ainode", "", lines, dropped)

        for instance, path in self._engine_logs().items():
            tail = self._tails.get(instance)
            if tail is None or tail.path != path:
                tail = LogTail(path)
                # A tail created now starts at the end of the file, or the
                # first publish after a restart ships the whole of the last
                # load.
                tail.start_at_end()
                self._tails[instance] = tail
                continue
            tail.skipped_bytes = 0
            lines, dropped = tail.read(max_lines)
            if lines:
                payload = _payload(config, "vllm", instance, lines, dropped)
                if tail.skipped_bytes:
                    payload["skipped_bytes"] = tail.skipped_bytes
                out[f"logs/vllm/{instance}"] = payload
        return out

    def _own_lines(self, config, max_lines: int) -> tuple:
        if self._buffer is None:
            self._buffer = install_log_buffer()
        level = getattr(logging, str(
            getattr(config, "mqtt_log_level", "INFO") or "INFO").upper(),
            logging.INFO)
        return self._buffer.drain(min_level=level, max_lines=max_lines)

    def _engine_logs(self) -> Dict[str, Path]:
        """instance name → its log file, for every instance on this node."""
        found: Dict[str, Path] = {}
        manager = self._app.get("instances")
        instances = []
        try:
            instances = list(manager.instances()) if manager is not None else []
        except Exception:
            logger.debug("could not list instances for log publishing",
                         exc_info=True)
        for instance in instances:
            backend = getattr(instance, "backend", None)
            path = getattr(backend, "log_path", None)
            record = getattr(instance, "record", None)
            model = str(getattr(record, "model", "") or "")
            if path is None or not model:
                continue
            found[_slug(model)] = Path(path)
        engine = self._app.get("engine")
        if not found and engine is not None:
            path = getattr(engine, "log_path", None)
            model = str(getattr(self._app.get("config"), "model", "") or "")
            if path is not None and model:
                found[_slug(model)] = Path(path)
        return found


def _slug(model: str) -> str:
    """A model id as an MQTT topic level: no slashes, no wildcards."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "instance"


def _max_lines(config) -> int:
    try:
        return max(1, min(1000, int(getattr(config, "mqtt_log_lines", 0)
                                    or DEFAULT_MAX_LINES)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_LINES


def _payload(config, source: str, instance: str, lines: List[str],
             dropped: int) -> dict:
    import time

    payload = {
        "node_id": getattr(config, "node_id", "") or "",
        "node_name": getattr(config, "node_name", "") or "",
        "source": source,
        "lines": lines,
        "count": len(lines),
        "timestamp": time.time(),
    }
    if instance:
        payload["instance"] = instance
    if dropped:
        # Said out loud. A gap the subscriber cannot see is worse than a gap.
        payload["dropped"] = dropped
    return payload
