"""Which tool-call parser a model needs, so nobody has to know.

vLLM will not answer an OpenAI ``tool_choice: "auto"`` request unless it was
started with ``--enable-auto-tool-choice`` *and* a ``--tool-call-parser`` that
matches how the model emits calls. Open WebUI sends ``tool_choice: "auto"`` by
default, so a model launched without those two flags rejects every chat that
has a tool attached — with an error about server configuration that is
accurate and tells the user nothing they can act on. llama.cpp never needed
this because it parses tool calls itself; vLLM makes it the operator's
problem.

That problem does not belong to the operator. The parser follows from the
model family, and the mapping below is read off the proven recipes in
eugr/spark-vllm-docker (MIT) — ``recipes/*.yaml``, the ``--tool-call-parser``
line of each — plus the parsers vLLM documents for the older Llama and
Mistral families.

Deliberately conservative: an unrecognised model gets nothing, exactly as
before. Guessing a parser for a model whose chat template has no tool support
would turn a working launch into a broken one, and "it used to start" is worth
more than "it might have had tool calling".
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

__all__ = ["AUTO", "OFF", "KNOWN_PARSERS", "parser_for_model", "tool_call_args"]

#: Sentinel values the API and UI use for the tool-calling choice.
AUTO = "auto"
OFF = "off"

# Ordered: the first pattern that matches a model id wins, so specific rules
# (Qwen3-Coder) come before general ones (any Qwen).
#
# Sources, per entry: the recipe in eugr/spark-vllm-docker that serves that
# family (MIT), except llama3_json / mistral / hermes, which are vLLM's own
# documented parsers for families the recipes do not cover.
_RULES: List[Tuple[str, str]] = [
    # Qwen3.6 and newer, and the Nemotron 3.x line, emit XML-tagged calls.
    (r"qwen-?3\.(?:[6-9]|\d{2})", "qwen3_xml"),
    (r"nemotron-?3", "qwen3_xml"),
    # Qwen3-Coder and the Qwen3.5 generation.
    (r"qwen3-?coder", "qwen3_coder"),
    (r"qwen-?3\.5", "qwen3_coder"),
    (r"qwen-?3", "qwen3_coder"),
    (r"gemma-?4|diffusiongemma", "gemma4"),
    (r"glm-?[45]", "glm47"),
    (r"minimax", "minimax_m2"),
    (r"deepseek-?v4", "deepseek_v4"),
    (r"step-?3", "step3p5"),
    (r"inkling", "inkling"),
    (r"gpt-?oss", "openai"),
    (r"hermes", "hermes"),
    (r"mistral|mixtral|magistral", "mistral"),
    (r"llama-?3", "llama3_json"),
]

#: Offered in the UI. "Automatic" and "off" are not parsers; they are choices.
KNOWN_PARSERS = (
    "qwen3_xml", "qwen3_coder", "gemma4", "glm47", "minimax_m2",
    "deepseek_v4", "step3p5", "inkling", "openai", "hermes", "mistral",
    "llama3_json", "llama4_json", "granite", "internlm", "jamba", "phi4_mini_json",
)


def parser_for_model(model: str) -> str:
    """The tool-call parser for this model id, or "" when unrecognised."""
    name = (model or "").lower()
    if not name:
        return ""
    for pattern, parser in _RULES:
        if re.search(pattern, name):
            return parser
    return ""


def tool_call_args(model: str, existing: Optional[List[str]],
                   choice: str = AUTO) -> List[str]:
    """The tool-calling flags to add to ``existing``, possibly none.

    ``choice`` is ``AUTO`` (derive from the model), ``OFF`` (add nothing), or
    a parser name to force. Anything already present wins: a recipe that names
    its own parser, or an operator who typed one, is not second-guessed.
    """
    if choice == OFF:
        return []
    supplied = {a.split("=", 1)[0] for a in (existing or []) if str(a).startswith("--")}
    if "--tool-call-parser" in supplied:
        # The parser is set; make sure the switch that activates it is too,
        # since one without the other is the failure this module exists for.
        return [] if "--enable-auto-tool-choice" in supplied else ["--enable-auto-tool-choice"]

    parser = parser_for_model(model) if choice == AUTO else choice.strip()
    if not parser:
        return []
    args = ["--tool-call-parser", parser]
    if "--enable-auto-tool-choice" not in supplied:
        args.append("--enable-auto-tool-choice")
    return args
