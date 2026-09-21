"""Profiles — a named description of what this node should be serving.

The operator's real deployment is not one model, it is a set: a chat model for
the team, a coding model with a large KV cache, an embedding model for RAG, all
of them up at the same time and all of them back after a restart. Loading that
by hand means three launches in the right order with the right memory fractions
typed in each time, and a reboot restores only the solo ones.

A profile records the whole set once. ``ProfileEntry`` carries exactly the
fields a launch accepts — the per-load overrides plus where it runs — so an
entry is a serialised launch, not a second dialect that has to be kept in sync
with the launch routes.

Stored in ``~/.ainode/profiles.json``, next to ``config.json`` and
``instances.json``, so a backup of that directory carries the deployment with
it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["Profile", "ProfileEntry", "ProfileStore", "ProfileError"]

#: What a profile name may contain. A name reaches the filesystem only as a
#: JSON key, but it also reaches URLs and the UI, so keep it boring.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$")

#: Entry kinds. "llm" is a vLLM instance (solo or distributed, decided by
#: node_ids); "embedding" goes through the embedding manager, which is a
#: different runtime entirely — in-process sentence-transformers, no container.
KIND_LLM = "llm"
KIND_EMBEDDING = "embedding"
#: Image generation. A separate kind rather than an LLM with a flag, because
#: restoring one means starting a different engine with different settings —
#: a profile that got that wrong would bring a node back serving nothing.
KIND_IMAGE = "image"
KINDS = (KIND_LLM, KIND_EMBEDDING, KIND_IMAGE)


class ProfileError(ValueError):
    """A profile or entry is not usable as written."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ProfileEntry:
    """One model in a profile, with where and how it runs.

    ``node_ids`` is the placement: empty or a single id means a solo load on
    this node, several ids mean a distributed launch across exactly those
    nodes. Ids rather than addresses, because an address changes with the
    fabric and an id does not; :mod:`ainode.profiles.apply` reports a node that
    has gone missing instead of quietly starting the model somewhere else.

    Every other field is optional and omitted-means-default, so a minimal entry
    is ``{"model": "..."}`` — the catalog recipe fills in the rest at launch
    time, the same way a bare click in the dashboard does.
    """

    model: str
    kind: str = KIND_LLM
    node_ids: List[str] = field(default_factory=list)
    strategy: str = ""
    gpu_memory_utilization: Optional[float] = None
    max_model_len: Optional[int] = None
    kv_cache_dtype: str = ""
    quantization: str = ""
    served_model_name: List[str] = field(default_factory=list)
    trust_remote_code: Optional[bool] = None
    extra_vllm_args: List[str] = field(default_factory=list)
    extra_env: Dict[str, str] = field(default_factory=dict)
    engine_image: str = ""
    # Image generation. engine_backend decides WHICH engine restores this
    # entry, so it is not optional decoration: an image instance restored as
    # an LLM starts vLLM on a diffusers pipeline and fails.
    engine_backend: str = ""
    max_image_size: Optional[int] = None
    image_steps: Optional[int] = None
    image_size: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        self.model = str(self.model or "").strip()
        if not self.model:
            raise ProfileError("Every profile entry needs a model.")
        self.kind = str(self.kind or KIND_LLM).strip().lower()
        if self.kind not in KINDS:
            raise ProfileError(
                f"Unknown entry kind {self.kind!r}; use one of {', '.join(KINDS)}."
            )
        self.node_ids = [str(n).strip() for n in (self.node_ids or []) if str(n).strip()]
        self.served_model_name = [
            str(s).strip() for s in (self.served_model_name or []) if str(s).strip()
        ]
        self.extra_vllm_args = [str(a) for a in (self.extra_vllm_args or [])]
        self.extra_env = {
            str(k): str(v) for k, v in (self.extra_env or {}).items() if str(k)
        }
        if self.gpu_memory_utilization is not None:
            try:
                self.gpu_memory_utilization = float(self.gpu_memory_utilization)
            except (TypeError, ValueError):
                raise ProfileError(
                    f"gpu_memory_utilization for {self.model} is not a number."
                ) from None
            if not 0.0 < self.gpu_memory_utilization <= 1.0:
                raise ProfileError(
                    f"gpu_memory_utilization for {self.model} must be between "
                    f"0 and 1, got {self.gpu_memory_utilization}."
                )
        if self.max_model_len is not None:
            try:
                self.max_model_len = int(self.max_model_len)
            except (TypeError, ValueError):
                raise ProfileError(
                    f"max_model_len for {self.model} is not a number."
                ) from None

    @property
    def is_distributed(self) -> bool:
        return len(self.node_ids) > 1

    def launch_body(self) -> dict:
        """The request body this entry stands for.

        Only fields the operator actually set are included: an absent field
        must stay absent so the launch path applies the catalog recipe and the
        config defaults, rather than pinning everything to whatever the profile
        was written with.
        """
        body: dict = {"model": self.model}
        if self.node_ids:
            body["node_ids"] = list(self.node_ids)
        if self.strategy:
            body["strategy"] = self.strategy
        if self.gpu_memory_utilization is not None:
            body["gpu_memory_utilization"] = self.gpu_memory_utilization
        if self.max_model_len is not None:
            body["max_model_len"] = self.max_model_len
        if self.kv_cache_dtype:
            body["kv_cache_dtype"] = self.kv_cache_dtype
        if self.quantization:
            body["quantization"] = self.quantization
        if self.served_model_name:
            body["served_model_name"] = list(self.served_model_name)
        if self.trust_remote_code is not None:
            body["trust_remote_code"] = bool(self.trust_remote_code)
        if self.extra_vllm_args:
            body["extra_vllm_args"] = list(self.extra_vllm_args)
        if self.extra_env:
            body["extra_env"] = dict(self.extra_env)
        if self.engine_image:
            body["engine_image"] = self.engine_image
        if self.engine_backend:
            body["engine_backend"] = self.engine_backend
        if self.max_image_size is not None:
            body["max_image_size"] = self.max_image_size
        if self.image_steps is not None:
            body["image_steps"] = self.image_steps
        if self.image_size:
            body["image_size"] = self.image_size
        return body

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProfileEntry":
        if not isinstance(data, dict):
            raise ProfileError("A profile entry must be an object.")
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Profile:
    """A named set of entries, applied as a whole."""

    name: str
    description: str = ""
    entries: List[ProfileEntry] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        self.name = str(self.name or "").strip()
        if not _NAME_RE.match(self.name):
            raise ProfileError(
                "A profile name must be 1-64 characters of letters, digits, "
                "space, dot, dash or underscore, and start with a letter or "
                "digit."
            )
        self.description = str(self.description or "").strip()
        self.entries = [
            e if isinstance(e, ProfileEntry) else ProfileEntry.from_dict(e)
            for e in (self.entries or [])
        ]
        seen = set()
        for e in self.entries:
            if e.model in seen:
                raise ProfileError(
                    f"{e.model} appears twice in profile {self.name!r}; a model "
                    f"can only be served once per node set."
                )
            seen.add(e.model)
        self.created_at = self.created_at or _now()
        self.updated_at = self.updated_at or self.created_at

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "entries": [e.to_dict() for e in self.entries],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        if not isinstance(data, dict):
            raise ProfileError("A profile must be an object.")
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            entries=data.get("entries") or [],
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


class ProfileStore:
    """The profiles on this node, plus which one is the default.

    Reads are tolerant: a corrupt or hand-edited file yields an empty store and
    a log line rather than an exception at boot, because a broken profile file
    must not stop the node from serving what it already has.
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else self._default_path()
        self._profiles: Dict[str, Profile] = {}
        self._default: str = ""
        self.load()

    @staticmethod
    def _default_path() -> Path:
        from ainode.core.config import AINODE_HOME
        return Path(AINODE_HOME) / "profiles.json"

    @property
    def path(self) -> Path:
        return self._path

    # -- persistence ---------------------------------------------------------

    def load(self) -> None:
        self._profiles = {}
        self._default = ""
        try:
            if not self._path.exists():
                return
            raw = json.loads(self._path.read_text())
        except Exception:
            logger.exception("Could not read %s; starting with no profiles",
                             self._path)
            return
        for item in (raw.get("profiles") or []) if isinstance(raw, dict) else []:
            try:
                profile = Profile.from_dict(item)
            except ProfileError:
                logger.exception("Skipping unusable profile in %s", self._path)
                continue
            self._profiles[profile.name] = profile
        default = str(raw.get("default") or "") if isinstance(raw, dict) else ""
        self._default = default if default in self._profiles else ""

    def save(self) -> None:
        payload = {
            "profiles": [p.to_dict() for p in self._profiles.values()],
            "default": self._default,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a half-written profiles.json read at the next boot
        # would lose every profile, and the default one would not come up.
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self._path)

    # -- CRUD ----------------------------------------------------------------

    def names(self) -> List[str]:
        return sorted(self._profiles)

    def all(self) -> List[Profile]:
        return [self._profiles[n] for n in self.names()]

    def get(self, name: str) -> Optional[Profile]:
        return self._profiles.get(str(name or "").strip())

    def put(self, profile: Profile) -> Profile:
        existing = self._profiles.get(profile.name)
        if existing is not None:
            profile.created_at = existing.created_at
        profile.updated_at = _now()
        self._profiles[profile.name] = profile
        self.save()
        return profile

    def delete(self, name: str) -> bool:
        name = str(name or "").strip()
        if name not in self._profiles:
            return False
        del self._profiles[name]
        if self._default == name:
            self._default = ""
        self.save()
        return True

    # -- default -------------------------------------------------------------

    @property
    def default_name(self) -> str:
        return self._default

    def set_default(self, name: str) -> None:
        """Set (or, with ``""``, clear) the profile applied at startup."""
        name = str(name or "").strip()
        if name and name not in self._profiles:
            raise ProfileError(f"No profile named {name!r}.")
        self._default = name
        self.save()

    def default_profile(self) -> Optional[Profile]:
        return self._profiles.get(self._default) if self._default else None
