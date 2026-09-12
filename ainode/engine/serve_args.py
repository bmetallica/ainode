"""Serve-argument decisions shared by the engine backends.

Both backends end up writing a ``vllm serve`` command line, and the rules for
what belongs on it are properties of the model and the hardware, not of the
backend: fp8 KV corrupts vision-model generation on GB10 whichever backend
started the container, and a flag the caller supplied explicitly must suppress
the built-in one either way.

These lived as private methods on the NVIDIA backend, so the eugr backend —
the default one — quietly served without them: a KV dtype chosen in the UI, an
API alias, ``trust_remote_code`` all went nowhere.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

__all__ = [
    "effective_kv_cache_dtype",
    "is_multimodal_model",
    "local_model_dir",
    "supplied_flags",
]

_VISION_ARCH_RE = re.compile(r"VL|Vision|vision")


def local_model_dir(model: str, models_dir: str) -> Optional[str]:
    """This model's on-disk weight directory, or None if it is not there.

    The flat ``org--name`` layout our downloader writes.
    """
    if not model or not models_dir:
        return None
    path = Path(models_dir) / model.replace("/", "--")
    try:
        if path.is_dir() and any(path.iterdir()):
            return str(path)
    except OSError:
        pass
    return None


def is_multimodal_model(model_dir: Optional[str]) -> bool:
    """True when the local ``config.json`` describes a vision model.

    Only the local directory is inspected. A repo-id that has not been
    downloaded yet reads as False — deliberately: fetching a remote config
    from inside a command-line builder would put the network in the path of
    every launch.
    """
    if not model_dir:
        return False
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
    except Exception:
        return False
    if "vision_config" in config:
        return True
    archs = config.get("architectures") or []
    if isinstance(archs, str):
        archs = [archs]
    return any(_VISION_ARCH_RE.search(str(a)) for a in archs)


def effective_kv_cache_dtype(config, model_dir: Optional[str]) -> str:
    """The KV-cache dtype to serve with, after the vision-model safety rule.

    fp8 KV corrupts vision-model generation on GB10 (verified: Qwen2.5-VL emits
    garbage on fp8, clean output on auto; text models are unaffected), so the
    fp8 *default* is downgraded to auto for a multimodal model. An explicit
    value always wins, including an explicit fp8 — that is the operator's way
    back in for a model they know handles it.
    """
    dtype = getattr(config, "kv_cache_dtype", "") or ""
    explicit = bool(getattr(config, "kv_cache_dtype_explicit", False))
    if dtype == "fp8" and not explicit and is_multimodal_model(model_dir):
        return "auto"
    return dtype


def supplied_flags(extra_args: Optional[Iterable]) -> Set[str]:
    """Flag names present in ``extra_vllm_args``, in both ``--f v`` and ``--f=v``.

    A built-in flag is skipped when the caller supplied the same one: vLLM
    rejects duplicates, and the explicit value is the intent.
    """
    args: List[str] = [str(a) for a in (extra_args or [])]
    return {a.split("=", 1)[0] for a in args if a.startswith("--")}
