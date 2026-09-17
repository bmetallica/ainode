"""Generate an OpenCode provider config for whatever this cluster is serving.

Writing one by hand needs four facts per model, and three of them are easy to
get wrong in ways that surface hours later:

  * whether it reasons. A client told otherwise ignores the reasoning field,
    sees an empty ``content`` and treats the turn as finished — the model
    "stops by itself". Diagnosed twice on this cluster, on two models, before
    the pattern was recognised.
  * whether it takes images. Offering them to a text-only model produces
    errors from the engine rather than from the client.
  * the context limit. It has to fit under the ``max_model_len`` the instance
    was LAUNCHED with, which is often far below what the model advertises —
    DeepSeek V4 claims 1M and serves 131072 here — and it has to leave room,
    because a client's token accounting is an estimate. 96000 + 32768 against
    131072 leaves 2304 tokens of margin, and a request that overshoots is
    refused mid-session.

So the numbers come from the running instances, not from the catalog's
aspirations: the same per-node launch configuration a profile captures.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["build_opencode_config", "limits_for"]

#: Cap on generated tokens. Reasoning models spend most of their budget
#: thinking — 979 tokens measured here for "count from 1 to 30" — so this is
#: deliberately generous, and still bounded so one request cannot claim the
#: whole window.
MAX_OUTPUT_TOKENS = 32768

#: Reserve below max_model_len, as a fraction, floored. A client counts
#: tokens by estimate: system prompt, tool schemas and attachments are
#: routinely undercounted, and the engine refuses the whole request when the
#: sum overshoots.
MARGIN_FRACTION = 16
MIN_MARGIN_TOKENS = 4096


def limits_for(max_model_len: int) -> Dict[str, int]:
    """``{"context": …, "output": …}`` that fit inside ``max_model_len``.

    The two together stay under the window with room to spare, because the
    client adds them up and the engine rejects the total.
    """
    window = max(1024, int(max_model_len))
    output = min(MAX_OUTPUT_TOKENS, max(256, window // 4))
    margin = max(MIN_MARGIN_TOKENS, window // MARGIN_FRACTION)
    context = window - output - margin
    if context < window // 4:
        # A small window cannot afford a generous answer; shrink the answer
        # rather than leaving nothing to read from.
        output = max(256, window // 4)
        margin = max(256, window // MARGIN_FRACTION)
        context = window - output - margin
    return {"context": max(512, context), "output": output}


def _launched_context(entry, info) -> Optional[int]:
    """The ``max_model_len`` this instance actually runs with.

    Three places, in order of authority: the launch override, the flag inside
    the launch arguments, and the recipe's own flag. The model's advertised
    context length is deliberately NOT one of them — that is the number that
    made a 1M-context model produce a config nothing could use.
    """
    direct = getattr(entry, "max_model_len", None)
    if direct:
        return int(direct)
    for source in (getattr(entry, "extra_vllm_args", None) or [],
                   getattr(info, "extra_vllm_args", None) or []):
        args = [str(a) for a in source]
        for index, arg in enumerate(args):
            if arg == "--max-model-len" and index + 1 < len(args):
                try:
                    return int(args[index + 1])
                except ValueError:
                    continue
            if arg.startswith("--max-model-len="):
                try:
                    return int(arg.split("=", 1)[1])
                except ValueError:
                    continue
    return None


def _capabilities(info) -> Tuple[bool, bool, bool]:
    """(tool_call, reasoning, attachment) from the catalog entry."""
    caps = {str(c).lower() for c in (getattr(info, "capabilities", None) or [])}
    return ("tool_use" in caps,
            "reasoning" in caps,
            bool(caps & {"vision", "image", "multimodal"}))


def _served_llms(app) -> List[Tuple[str, object]]:
    """(model id, profile entry) for every LLM running anywhere in the cluster.

    Through the profile capture, which already asks each node how it started
    its models — one place that knows, rather than a second one that drifts.
    """
    from ainode.profiles.apply import capture_profile
    from ainode.profiles.store import KIND_LLM

    try:
        profile = capture_profile(app, "opencode-export")
    except Exception:
        logger.exception("could not read what the cluster is serving")
        return []
    return [(e.model, e) for e in profile.entries if e.kind == KIND_LLM and e.model]


def build_opencode_config(app, base_url: str) -> dict:
    """The whole opencode.json, ready to paste."""
    manager = app.get("model_manager")
    models: Dict[str, dict] = {}
    notes: List[str] = []

    for model_id, entry in _served_llms(app):
        info = None
        if manager is not None:
            try:
                info = manager._find_catalog_by_hf_repo(model_id)
            except Exception:
                logger.debug("no catalog entry for %s", model_id, exc_info=True)

        tool_call, reasoning, attachment = _capabilities(info)
        window = _launched_context(entry, info)
        if window is None:
            # No launch flag and no recipe flag: the engine is running the
            # model's own maximum, whatever that is. Say so rather than
            # inventing a window — a wrong limit fails mid-session, which is
            # the hardest kind of failure to attribute.
            window = int(getattr(info, "context_length", 0) or 32768)
            notes.append(
                f"{model_id}: no --max-model-len was set, so this uses the "
                f"model's own {window}. Check it against the engine log if "
                f"requests are refused.")

        models[model_id] = {
            "name": str(getattr(info, "name", "") or model_id.split("/")[-1]),
            "tool_call": tool_call,
            "reasoning": reasoning,
            "attachment": attachment,
            "limit": limits_for(window),
        }

    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "vllm": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "AINode cluster",
                "options": {"baseURL": base_url.rstrip("/") + "/v1",
                            "apiKey": "sk-no-auth"},
                "models": models,
            }
        },
    }
    if models:
        # Biggest window first: the model with the most room is the one worth
        # pointing a coding session at. Smallest as the helper, so titles and
        # summaries do not occupy the expensive one.
        by_window = sorted(models, key=lambda m: models[m]["limit"]["context"])
        config["model"] = f"vllm/{by_window[-1]}"
        if len(by_window) > 1:
            config["small_model"] = f"vllm/{by_window[0]}"
    return {"config": config, "notes": notes}
