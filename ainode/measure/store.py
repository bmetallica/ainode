"""What a model actually did on this cluster, written down by the cluster.

The catalog carries estimates: a size from the Hub, a memory figure someone
worked out with a calculator, a `verified` flag meaning "a person ran this
and said it worked". All three age badly and none of them are measurements.

Meanwhile every number that would replace them already exists at the moment a
model comes up and again when it goes down — how long the load took and where
the time went, how much host memory it actually cost, how fast it answers.
Until now those were read off a card by a person and typed back into a source
file, which is the sort of work a machine should be doing for itself.

So: the node writes them down. The planner prefers a measurement to an
estimate, the card says "measured here" rather than "verified by someone",
and nothing has to be typed anywhere.

One file per node in ``~/.ainode/measurements.json``, because a measurement
is a property of the model ON THIS HARDWARE and a node that has never run
something has nothing to say about it. The head gathers them the same way it
gathers everything else about its peers.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["Measurement", "MeasurementStore"]

#: How many launches to remember per model. Enough to see that a figure is
#: stable, few enough that the file stays a file.
HISTORY = 5


@dataclass
class Measurement:
    """One model, as this node has actually run it."""

    model: str
    #: "llm" or "image" — what the figures below mean depends on it.
    kind: str = "llm"
    node_id: str = ""
    engine_backend: str = ""
    #: Unix time of the most recent successful load.
    last_ok: float = 0.0
    #: How many times it has come up here, and how many times it has failed.
    launches: int = 0
    failures: int = 0
    #: Seconds from launch to serving, and where they went.
    load_seconds: float = 0.0
    load_timeline: List[dict] = field(default_factory=list)
    #: Host memory the load actually cost, in GB. The figure the planner
    #: wants and the one nobody can work out from a parameter count.
    memory_gb: float = 0.0
    #: What it was launched with, so a figure can be read in context: a
    #: memory cost at 64k context says nothing about the same model at 256k.
    max_model_len: int = 0
    gpu_memory_utilization: float = 0.0
    max_image_size: int = 0
    #: Serving speed, once anything has been asked of it.
    tokens_per_second: float = 0.0
    seconds_per_image: float = 0.0
    #: The last few loads, newest last: [{at, seconds, memory_gb, ok}].
    history: List[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model = str(self.model or "").strip()
        if not self.model:
            raise ValueError("a measurement needs a model")

    @property
    def measured(self) -> bool:
        """True when this node has actually served the model."""
        return self.launches > 0 and self.last_ok > 0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["measured"] = self.measured
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Measurement":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


class MeasurementStore:
    """The measurements this node holds, as one JSON file."""

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else None

    @property
    def path(self) -> Path:
        if self._path is not None:
            return self._path
        from ainode.core.config import AINODE_HOME

        return AINODE_HOME / "measurements.json"

    def load(self) -> Dict[str, Measurement]:
        """Everything measured here. A broken file reads as empty — losing a
        measurement costs an estimate, not a launch."""
        try:
            raw = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            logger.exception("could not read %s; continuing without measurements",
                             self.path)
            return {}
        out: Dict[str, Measurement] = {}
        for model, entry in (raw.get("measurements") or {}).items():
            try:
                measurement = Measurement.from_dict({**(entry or {}),
                                                     "model": model})
            except Exception:
                logger.warning("dropping unusable measurement for %s", model)
                continue
            out[measurement.model] = measurement
        return out

    def get(self, model: str) -> Optional[Measurement]:
        return self.load().get(str(model or "").strip())

    def record_launch(self, model: str, *, ok: bool, kind: str = "llm",
                      node_id: str = "", engine_backend: str = "",
                      load_seconds: float = 0.0, load_timeline=None,
                      memory_gb: float = 0.0, max_model_len: int = 0,
                      gpu_memory_utilization: float = 0.0,
                      max_image_size: int = 0) -> Measurement:
        """Write down what a launch did. Never raises."""
        current = self.load()
        entry = current.get(model) or Measurement(model=model)
        entry.kind = kind or entry.kind
        entry.node_id = node_id or entry.node_id
        entry.engine_backend = engine_backend or entry.engine_backend
        if ok:
            entry.launches += 1
            entry.last_ok = time.time()
            # Only from a successful load: the time a failed one took is the
            # time it took to fail, which is not how long this model needs.
            if load_seconds:
                entry.load_seconds = round(float(load_seconds), 1)
            if load_timeline:
                entry.load_timeline = list(load_timeline)
            if memory_gb > 0:
                entry.memory_gb = round(float(memory_gb), 1)
            if max_model_len:
                entry.max_model_len = int(max_model_len)
            if gpu_memory_utilization:
                entry.gpu_memory_utilization = float(gpu_memory_utilization)
            if max_image_size:
                entry.max_image_size = int(max_image_size)
        else:
            entry.failures += 1
        entry.history.append({
            "at": round(time.time(), 1), "ok": bool(ok),
            "seconds": round(float(load_seconds or 0), 1),
            "memory_gb": round(float(memory_gb or 0), 1),
        })
        del entry.history[:-HISTORY]
        current[entry.model] = entry
        self._write(current)
        return entry

    def record_speed(self, model: str, *, tokens_per_second: float = 0.0,
                     seconds_per_image: float = 0.0) -> None:
        """How fast it answers, measured while it served."""
        current = self.load()
        entry = current.get(model)
        if entry is None:
            return
        if tokens_per_second > 0:
            entry.tokens_per_second = round(float(tokens_per_second), 1)
        if seconds_per_image > 0:
            entry.seconds_per_image = round(float(seconds_per_image), 2)
        self._write(current)

    def forget(self, model: str) -> bool:
        current = self.load()
        if model not in current:
            return False
        del current[model]
        self._write(current)
        return True

    def _write(self, measurements: Dict[str, Measurement]) -> None:
        payload = {"measurements": {}}
        for model, entry in sorted(measurements.items()):
            data = entry.to_dict()
            data.pop("model", None)
            data.pop("measured", None)
            payload["measurements"][model] = data
        path = self.path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
            tmp.replace(path)
        except OSError:
            # A measurement that cannot be written is a lost estimate, not a
            # lost launch. Never let it become one.
            logger.exception("could not write %s", path)
