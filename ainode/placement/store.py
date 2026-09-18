"""Where a model runs, decided once and remembered.

A launch takes a node set every time. That is right for an experiment and
wrong for a deployment: a cluster settles into an arrangement — the coder
across two nodes, the chat model and the embeddings on the third — and then
every restart, every relaunch after a config change, every retry after a
failed load asks the same question again. Answering it from memory is the
difference between running a cluster and re-configuring one.

Profiles already record a whole arrangement, which is the right shape for
"bring everything back". This is the smaller companion: one model, one
placement, applied whenever that model is launched without an explicit
choice. The two do not compete — a profile entry names its nodes outright,
and never consults this.

Stored in ``~/.ainode/placement.json``, beside config.json and
profiles.json, so a backup of that directory carries it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["Placement", "PlacementStore", "PlacementError"]


class PlacementError(ValueError):
    """A placement is not usable as written."""


@dataclass
class Placement:
    """The nodes one model should run on, and how it is split across them."""

    model: str
    node_ids: List[str] = field(default_factory=list)
    #: "" lets the planner decide from the node count, which is what the
    #: launch path already does. An explicit value is for the cases where the
    #: obvious choice is wrong — a model that cannot do pipeline, say.
    strategy: str = ""

    def __post_init__(self) -> None:
        self.model = str(self.model or "").strip()
        if not self.model:
            raise PlacementError("A placement needs a model.")
        seen: List[str] = []
        for node in self.node_ids or []:
            node = str(node).strip()
            # Deduplicated in order: the same node twice would be read as two
            # ranks and silently halve the memory the planner thinks it has.
            if node and node not in seen:
                seen.append(node)
        self.node_ids = seen
        self.strategy = str(self.strategy or "").strip().lower()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Placement":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


class PlacementStore:
    """The placements on this node, persisted as one JSON file."""

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else None

    @property
    def path(self) -> Path:
        if self._path is not None:
            return self._path
        from ainode.core.config import AINODE_HOME

        return AINODE_HOME / "placement.json"

    def load(self) -> Dict[str, Placement]:
        """Everything remembered. A broken file reads as empty rather than
        taking the node down — a placement is a convenience, and losing it
        must cost a click, not a boot."""
        try:
            raw = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            logger.exception("could not read %s; continuing without placements",
                             self.path)
            return {}
        out: Dict[str, Placement] = {}
        for model, entry in (raw.get("placements") or {}).items():
            try:
                placement = Placement.from_dict({**(entry or {}), "model": model})
            except PlacementError:
                logger.warning("dropping unusable placement for %s", model)
                continue
            out[placement.model] = placement
        return out

    def get(self, model: str) -> Optional[Placement]:
        return self.load().get(str(model or "").strip())

    def put(self, placement: Placement) -> None:
        current = self.load()
        current[placement.model] = placement
        self._write(current)

    def remove(self, model: str) -> bool:
        current = self.load()
        if str(model) not in current:
            return False
        del current[str(model)]
        self._write(current)
        return True

    def _write(self, placements: Dict[str, Placement]) -> None:
        payload = {"placements": {m: {"node_ids": p.node_ids,
                                      "strategy": p.strategy}
                                  for m, p in sorted(placements.items())}}
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside and moved into place: a half-written file here would
        # be read as "no placements" on the next boot, silently putting models
        # back where they were not wanted.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(path)
