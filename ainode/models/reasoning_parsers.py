"""Which reasoning parser a model needs, set only where the engine has it.

A model that thinks before it answers writes ``<think>…</think>`` (or its
family's equivalent) into its output. vLLM moves that into
``reasoning_content`` only when started with ``--reasoning-parser <name>``;
without it the thinking arrives as ordinary content, and a client configured
for a reasoning model — opencode with ``"reasoning": true`` — sees an answer
that is all thinking, or none. Diagnosed twice on this cluster.

Nobody set it because a wrong name is fatal: vLLM's argparse refuses a parser
it does not know, and the launch dies at second three. So the name is chosen
here and then CHECKED against the engine image (engine/image_probe.py) — never
added blind. An image that cannot be asked gets nothing, exactly as before.

The family → parser table is read off the ``--reasoning-parser`` lines of the
recipes in eugr/spark-vllm-docker (MIT, ``recipes/*.yaml``): deepseek_v4 for
DeepSeek-V4-Flash, qwen3 for Qwen3.5/3.6/3.8, glm45 for GLM-4.7/5.3,
minimax_m2 for MiniMax-M2.x, gemma4, openai_gptoss for gpt-oss, inkling, and
nemotron_v3 for Nemotron-3 Super/Lightning. Nemotron-3 Nano is left out: its
parser is a plugin file the recipe ships, not a name the image has.

Deliberately NOT matched: Qwen3-Coder and the Qwen3 "Instruct-2507" line. They
do not think, and the qwen3 parser applied to output with no end-of-thinking
marker can file the whole answer as reasoning.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

__all__ = ["reasoning_parser_for", "reasoning_args"]

# By model id, first match wins. (pattern, parser, pattern that vetoes it)
_BY_NAME: List[Tuple[str, str, str]] = [
    (r"deepseek-?v4|dspark", "deepseek_v4", ""),
    (r"qwen-?3", "qwen3", r"coder|instruct-?2507"),
    (r"glm-?(?:4\.[5-9]|5)", "glm45", ""),
    (r"minimax-?m[23]", "minimax_m2", ""),
    (r"gemma-?4|diffusiongemma", "gemma4", ""),
    (r"gpt-?oss", "openai_gptoss", ""),
    (r"inkling", "inkling", ""),
    (r"nemotron-?3", "nemotron_v3", r"nano"),
]

# By the checkpoint's architecture, for a finetune whose name says nothing —
# abacusai/Smaug-Flash is DeepSeek-V4-Flash under another name. Only
# architectures where every model thinks; the Qwen3 ones do not qualify, since
# Qwen3-Coder shares them.
_BY_ARCHITECTURE: List[Tuple[str, str]] = [
    (r"deepseekv4", "deepseek_v4"),
    (r"glm4moe", "glm45"),
    (r"minimaxm2", "minimax_m2"),
    (r"gptoss", "openai_gptoss"),
]


def reasoning_parser_for(model: str, architecture: str = "") -> str:
    """The parser this model's family uses, or "" when it is not known to
    think (or is known not to)."""
    name = (model or "").lower()
    for pattern, parser, veto in _BY_NAME:
        if re.search(pattern, name):
            if veto and re.search(veto, name):
                return ""
            return parser
    arch = re.sub(r"[^a-z0-9]", "", (architecture or "").lower())
    for pattern, parser in _BY_ARCHITECTURE:
        if re.search(pattern, arch):
            return parser
    return ""


def reasoning_args(model: str, existing: Optional[Iterable[str]],
                   available: Optional[Iterable[str]],
                   architecture: str = "") -> List[str]:
    """``["--reasoning-parser", name]`` to add, or ``[]``.

    Nothing when the launch already names one (a recipe or an operator is
    not second-guessed), when the family is unknown, or when ``available`` —
    the image's own registry — does not contain the name. ``available`` of
    None or empty means the image could not be asked, and then nothing is
    added: the whole point is never to add a name that might kill the launch.
    """
    supplied = {str(a).split("=", 1)[0].replace("_", "-")
                for a in (existing or []) if str(a).startswith("--")}
    if "--reasoning-parser" in supplied or "--reasoning-parser-plugin" in supplied:
        return []
    names = set(available or [])
    if not names:
        return []
    parser = reasoning_parser_for(model, architecture)
    if not parser or parser not in names:
        return []
    return ["--reasoning-parser", parser]
