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
    "DROP_PREFIX",
    "dropped_flags",
    "effective_kv_cache_dtype",
    "is_multimodal_model",
    "local_model_dir",
    "merge_vllm_args",
    "split_vllm_args",
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


def split_vllm_args(args: Optional[Iterable]) -> List[List[str]]:
    """Group a flat argument list into ``[flag, value...]`` runs.

    ``["--moe-backend", "marlin", "--enable-prefix-caching"]`` becomes
    ``[["--moe-backend", "marlin"], ["--enable-prefix-caching"]]``. Tokens
    before the first flag are kept as their own leading group so nothing is
    lost.
    """
    groups: List[List[str]] = []
    for token in [str(a) for a in (args or [])]:
        if token.startswith("--") or not groups:
            groups.append([token])
        else:
            groups[-1].append(token)
    return groups


#: Written in the extra-args field to REMOVE a flag the recipe supplies,
#: e.g. ``drop:--speculative-config``. Overriding a flag was possible from the
#: start; removing one was not, and there is no value that means "not set" for
#: flags like --quantization or --speculative-config. Diagnosing a model whose
#: recipe carries a flag its engine build cannot use needed exactly this.
DROP_PREFIX = "drop:"


def dropped_flags(caller_args: Optional[Iterable]) -> Set[str]:
    """Flags the caller asked to remove from the recipe."""
    dropped = set()
    for arg in [str(a) for a in (caller_args or [])]:
        if arg.startswith(DROP_PREFIX):
            flag = arg[len(DROP_PREFIX):].strip().split("=", 1)[0]
            if flag:
                dropped.add(flag if flag.startswith("-") else f"--{flag}")
    return dropped


def merge_vllm_args(recipe_args: Optional[Iterable],
                    caller_args: Optional[Iterable]) -> List[str]:
    """Caller arguments, plus the recipe flags the caller did not mention.

    Setting one thing in the UI must not throw away everything else the model
    needs. Typing a ``--max-num-seqs`` for Qwen3.8 used to replace its whole
    recipe — reasoning parser, tool-call parser and speculative config gone —
    because the caller's list simply took the place of the recipe's. Now the
    caller wins per flag, and the rest of the recipe survives.

    A ``drop:--flag`` entry removes a recipe flag instead of replacing it, and
    is itself never passed on.
    """
    caller = [str(a) for a in (caller_args or [])]
    dropped = dropped_flags(caller)
    caller = [a for a in caller if not a.startswith(DROP_PREFIX)]
    if not recipe_args:
        return caller
    supplied = supplied_flags(caller)
    merged = list(caller)
    for group in split_vllm_args(recipe_args):
        flag = group[0].split("=", 1)[0]
        if flag not in supplied and flag not in dropped:
            merged.extend(group)
    return merged
