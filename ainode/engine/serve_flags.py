"""Check a launch's extra flags against the engine image, before the launch.

F2 in upgrade-fixes.md. Every flag a recipe or the Advanced field carries goes
to ``vllm serve`` unchecked, and vLLM's argparse is the first to read it — on a
distributed launch that is after the engine image has been compared across the
nodes, the weights mirrored and the Ray cluster formed. A flag the image does
not have (``--quantization modelopt_fp4`` on a build that calls it
``modelopt_mixed``, a parser renamed upstream) costs all of that and then reads
as ``unrecognized arguments`` in a log.

The image's own parser, listed by engine/image_probe.py, answers the same
question in advance. Only what can be decided without running anything is
checked: that each ``--flag`` exists, and that a flag with a fixed set of
choices was given one of them. Values are otherwise vLLM's business.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

__all__ = ["flag_problems", "normalise"]

#: A probe with fewer options than this did not see vLLM's real parser —
#: a partial import, a stub — and is not a basis for refusing anything.
MIN_OPTIONS = 100

#: Flags whose valid values can be extended at launch by a plugin file, so the
#: image's registry is not the whole list when the plugin flag is present.
_PLUGGABLE = {
    "--reasoning-parser": "--reasoning-parser-plugin",
    "--tool-call-parser": "--tool-parser-plugin",
}


def normalise(flag: str) -> str:
    """``--max_model_len`` and ``--max-model-len`` are one flag to vLLM, and
    ``--compilation-config.mode`` is a key of ``--compilation-config``."""
    head = flag.split("=", 1)[0]
    if head.startswith("--"):
        head = "--" + head[2:].split(".", 1)[0].replace("_", "-")
    return head


def flag_problems(args: Optional[Iterable], options: Optional[Dict[str, Optional[List[str]]]]) -> List[str]:
    """One sentence per flag the image will reject; [] when all is well or
    when ``options`` is too thin to judge by."""
    if not options or len(options) < MIN_OPTIONS:
        return []
    known = {normalise(k): v for k, v in options.items()}
    tokens = [str(a) for a in (args or [])]
    present = {normalise(t) for t in tokens if t.startswith("--")}
    problems: List[str] = []
    for index, token in enumerate(tokens):
        if not token.startswith("--"):
            continue
        flag = normalise(token)
        if flag not in known:
            # BooleanOptionalAction registers --no-<flag> itself, so a
            # negation that is not listed really is not there.
            problems.append(f"{token.split('=', 1)[0]} is not a flag this "
                            f"engine image's vLLM accepts")
            continue
        choices = known[flag]
        if not choices:
            continue
        plugin = _PLUGGABLE.get(flag)
        if plugin and plugin in present:
            continue
        if "=" in token:
            value = token.split("=", 1)[1]
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            value = tokens[index + 1]
        else:
            continue
        if value not in choices:
            shown = ", ".join(choices[:12]) + (" …" if len(choices) > 12 else "")
            problems.append(f"{flag} {value}: this image accepts {shown}")
    return problems
