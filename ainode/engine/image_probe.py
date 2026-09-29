"""What the engine image will accept, asked of the engine image.

Two questions a launch keeps getting wrong by guessing, both answered by the
image itself — the same move as engine_env.py makes for ``VLLM_*`` variables:

* **Which reasoning parsers exist** (F1 in upgrade-fixes.md). Smaug, DeepSeek
  and Qwen3.8 think by default, and none of the recipes here set a
  ``--reasoning-parser``, because a name the image does not know kills the
  launch at second three. Without a parser the thinking arrives as ordinary
  content, and a client told to expect it separately reads an empty answer.
  Asked of the image, the name is either there or it is not.

* **Which ``vllm serve`` flags exist, and what they accept** (F2). ``vllm serve
  --help`` prints nothing useful in the ``vllm-node`` image, but the parser can
  be built in Python — ``make_arg_parser(FlexibleArgumentParser())`` — and its
  actions listed. A recipe or Advanced flag the image does not have
  (``--quantization modelopt_fp4``, a parser renamed upstream) is then refused
  before the weights are mirrored and Ray is formed, instead of minutes later
  in a log.

Keyed by image ID rather than by tag: ``vllm-node:latest`` is rebuilt in place
from a rolling wheel release, and an answer about last week's build is an answer
about another program. An ID never changes its contents, so a cached answer for
one is good forever.

The probe tolerates every vLLM layout it knows of and reports what it could
not read, rather than failing whole: an image that cannot be asked is never
evidence that a flag is wrong, and every caller treats an empty answer as
"keep quiet".
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["probe", "image_id", "PROBE_MARK"]

PROBE_MARK = "AINODE_PROBE "

#: Run inside the image. Every import is guarded: the modules moved between
#: vLLM releases (vllm.utils.FlexibleArgumentParser became
#: vllm.utils.argparse_utils.FlexibleArgumentParser; tool parsers moved out of
#: vllm.entrypoints.openai), and a probe that dies on one missing module would
#: lose the answers it could have given.
_SCRIPT = r'''
import json
out = {}

def names_of(manager):
    found = set()
    for attr in ("reasoning_parsers", "tool_parsers", "_reasoning_parsers",
                 "_tool_parsers", "lazy_parsers", "_lazy_parsers"):
        value = getattr(manager, attr, None)
        if isinstance(value, dict):
            found.update(str(k) for k in value)
    for attr in ("list_registered", "get_registered", "registered_names"):
        fn = getattr(manager, attr, None)
        if callable(fn):
            try:
                found.update(str(k) for k in fn())
            except Exception:
                pass
    return sorted(found)

try:
    from vllm.reasoning import ReasoningParserManager
    out["reasoning_parsers"] = names_of(ReasoningParserManager)
except Exception as exc:
    out["reasoning_error"] = repr(exc)[:300]

for module in ("vllm.tool_parsers", "vllm.entrypoints.openai.tool_parsers"):
    try:
        mod = __import__(module, fromlist=["ToolParserManager"])
        out["tool_parsers"] = names_of(mod.ToolParserManager)
        break
    except Exception as exc:
        out["tool_error"] = repr(exc)[:300]

try:
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except Exception:
        from vllm.utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    parser = make_arg_parser(FlexibleArgumentParser())
    options = {}
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        try:
            listed = sorted(str(c) for c in choices) if choices else None
        except Exception:
            listed = None
        for flag in action.option_strings:
            options[flag] = listed
    out["options"] = options
except Exception as exc:
    out["options_error"] = repr(exc)[:300]

try:
    import vllm
    out["vllm_version"] = str(getattr(vllm, "__version__", ""))
except Exception:
    pass

print("AINODE_PROBE " + json.dumps(out))
'''

#: In-process memo, by image id.
_MEMO: Dict[str, dict] = {}


def _cache_path(identity: str) -> Path:
    from ainode.core.config import AINODE_HOME

    safe = "".join(c if c.isalnum() else "_" for c in identity)[-80:]
    return Path(AINODE_HOME) / "engine-probe" / f"{safe}.json"


def image_id(image: str) -> str:
    """The image's content ID, or "" when docker cannot say."""
    if not image:
        return ""
    try:
        done = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True, text=True, timeout=20)
    except Exception:
        return ""
    return (done.stdout or "").strip() if done.returncode == 0 else ""


def _run(image: str, timeout: int) -> dict:
    try:
        done = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "python3", image,
             "-c", _SCRIPT],
            capture_output=True, text=True, timeout=timeout)
    except Exception:
        logger.debug("could not probe %s", image, exc_info=True)
        return {}
    for line in reversed((done.stdout or "").splitlines()):
        if line.startswith(PROBE_MARK):
            try:
                answer = json.loads(line[len(PROBE_MARK):])
                return answer if isinstance(answer, dict) else {}
            except ValueError:
                return {}
    logger.debug("probe of %s printed no answer (exit %s): %s", image,
                 done.returncode, (done.stderr or "")[-400:])
    return {}


def probe(image: str, timeout: int = 180) -> dict:
    """What ``image`` accepts: ``{"reasoning_parsers": [...], "tool_parsers":
    [...], "options": {flag: choices-or-None}, "vllm_version": ...}``, with any
    part it could not read left out. ``{}`` when the image cannot be asked."""
    identity = image_id(image)
    if not identity:
        return {}
    if identity in _MEMO:
        return _MEMO[identity]
    path = _cache_path(identity)
    try:
        cached = json.loads(path.read_text())
        if isinstance(cached, dict) and cached:
            _MEMO[identity] = cached
            return cached
    except (OSError, ValueError):
        pass
    answer = _run(image, timeout)
    if answer.get("options") or answer.get("reasoning_parsers"):
        _MEMO[identity] = answer
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(answer))
        except OSError:
            logger.debug("could not cache the probe of %s", image,
                         exc_info=True)
    return answer


def cached_probe(image: str) -> Optional[dict]:
    """The probe of ``image`` if one has been taken, without taking one."""
    identity = image_id(image)
    if not identity:
        return None
    if identity in _MEMO:
        return _MEMO[identity]
    try:
        return json.loads(_cache_path(identity).read_text())
    except (OSError, ValueError):
        return None
