"""Which ``VLLM_*`` variables does this engine image actually read?

From the cluster's own launch log, on a model that was being tuned:

    WARNING [interface.py:1274] Unknown vLLM environment variable detected:
    VLLM_BASE_DIR

That is the whole feedback. It appears once, in the middle of a launch that
then proceeds and fails for its own reasons, and it means the knob the
operator set did nothing at all. AINode passes ``extra_env`` through blind: a
catalog recipe and an Advanced field both land in ``docker -e`` and neither is
checked against the image they are going to.

The image can be asked. vLLM keeps its registry at
``vllm.envs.environment_variables``, a dict whose keys are every name it will
read; a ``VLLM_*`` name that is not in it is ignored, with that one warning.
So the check is a question to the image rather than a list maintained here —
which matters on a rolling engine build, where a list would be wrong within
the week.

Only ``VLLM_*`` names are judged. ``NCCL_*``, ``HF_*``, ``TRITON_*``,
``INSTANTTENSOR_*`` and the rest are read by other libraries that keep no such
registry, and calling those unknown would be a false alarm on every launch
that does anything interesting.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

__all__ = ["probe_image_env", "known_env_names", "unregistered_names",
           "env_warning"]

#: How long a probe result stays good. The engine image is rebuilt, not
#: mutated, so this only bounds how long a stale answer can survive a rebuild
#: that reused the tag.
CACHE_SECONDS = 24 * 3600

_PROBE = ("import json, vllm.envs as e; "
          "print(json.dumps(sorted(e.environment_variables)))")

#: In-process memo, so a launch does not start a container to ask twice.
_MEMO: Dict[str, Set[str]] = {}


def _cache_path(image: str) -> Path:
    from ainode.core.config import AINODE_HOME

    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in image)
    return Path(AINODE_HOME) / "engine-env" / f"{safe}.json"


def probe_image_env(image: str, timeout: int = 120) -> Set[str]:
    """Every name ``image``'s vLLM will read, or an empty set.

    Empty means "could not ask", never "reads nothing" — the caller must treat
    those the same way, which is to say by keeping quiet. An image that cannot
    be probed is not evidence that an operator's variable is wrong.
    """
    if not image:
        return set()
    try:
        done = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "python3", image,
             "-c", _PROBE],
            capture_output=True, text=True, timeout=timeout)
    except Exception:
        logger.debug("could not probe %s for its env registry", image,
                     exc_info=True)
        return set()
    if done.returncode != 0:
        logger.debug("env probe of %s exited %d: %s", image, done.returncode,
                     (done.stderr or "")[-400:])
        return set()
    # The image prints INFO lines on import (Triton, platform detection), so
    # the JSON is the last line that parses rather than the whole of stdout.
    for line in reversed((done.stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            return {str(n) for n in json.loads(line)}
        except ValueError:
            continue
    return set()


def known_env_names(image: str, *, refresh: bool = False) -> Set[str]:
    """``probe_image_env`` behind a memo and a file cache."""
    if not image:
        return set()
    if not refresh and image in _MEMO:
        return _MEMO[image]
    path = _cache_path(image)
    if not refresh:
        try:
            blob = json.loads(path.read_text())
            if time.time() - float(blob.get("at") or 0) < CACHE_SECONDS:
                names = {str(n) for n in blob.get("names") or []}
                if names:
                    _MEMO[image] = names
                    return names
        except (OSError, ValueError, TypeError):
            pass
    names = probe_image_env(image)
    if names:
        _MEMO[image] = names
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"at": time.time(),
                                        "names": sorted(names)}))
        except OSError:
            logger.debug("could not cache the env registry for %s", image,
                         exc_info=True)
    return names


def unregistered_names(env: Optional[Iterable], known: Set[str]) -> List[str]:
    """The ``VLLM_*`` names in ``env`` that ``known`` does not contain.

    An empty ``known`` answers "nothing to report": see probe_image_env.
    """
    if not known:
        return []
    names = list(env.keys()) if isinstance(env, dict) else list(env or [])
    return sorted(n for n in (str(k) for k in names)
                  if n.startswith("VLLM_") and n not in known)


def env_warning(env: Optional[Iterable], image: str) -> str:
    """One sentence for the operator, or "".

    Said before the launch rather than found afterwards in a log line that
    scrolls past between a Triton notice and a NCCL banner.
    """
    missing = unregistered_names(env, known_env_names(image))
    if not missing:
        return ""
    return (f"{', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not read by this engine "
            f"image — vLLM registers every variable it honours, and "
            f"{'this name is' if len(missing) == 1 else 'these names are'} "
            f"not in that registry. The launch will proceed and the "
            f"{'setting' if len(missing) == 1 else 'settings'} will do "
            f"nothing.")
