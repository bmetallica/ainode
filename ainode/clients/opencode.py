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
    refused mid-session. And it has to fit the KV cache once per concurrent
    session, or the engine preempts one of them mid-answer.

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


def _concurrency(entry, info) -> int:
    """``--max-num-seqs`` of this instance, launch before recipe; 0 unknown."""
    for source in (getattr(entry, "extra_vllm_args", None) or [],
                   getattr(info, "extra_vllm_args", None) or []):
        args = [str(a) for a in source]
        for index, arg in enumerate(args):
            value = None
            if arg == "--max-num-seqs" and index + 1 < len(args):
                value = args[index + 1]
            elif arg.startswith("--max-num-seqs="):
                value = arg.split("=", 1)[1]
            if value is not None:
                try:
                    return max(0, int(value))
                except ValueError:
                    continue
    return 0


def _cache_tokens(app, model_id: str, window: int) -> int:
    """How many tokens of KV cache the engine reported for this launch.

    vLLM's own figure (``GPU KV cache size: N tokens``), which #219 records
    per model on the node that served it. Only when it belongs to a launch
    with this window — a figure from a load at another context length is a
    figure about a different cache. 0 when there is none.
    """
    try:
        from ainode.measure.recorder import measured_for

        measured = measured_for(app, model_id) or {}
    except Exception:
        return 0
    tokens = int(measured.get("kv_tokens") or 0)
    launched = int(measured.get("max_model_len") or 0)
    if not tokens or (launched and launched != int(window)):
        return 0
    return tokens


def _launch_args(entry, info) -> List[str]:
    """Every vLLM flag this instance was started with, launch before recipe."""
    out: List[str] = []
    for source in (getattr(entry, "extra_vllm_args", None) or [],
                   getattr(info, "extra_vllm_args", None) or []):
        out.extend(str(a) for a in source)
    return out


def _capabilities(info, entry=None, model_id: str = "") -> Tuple[bool, bool, bool]:
    """(tool_call, reasoning, attachment), from what the engine was told.

    Two fields with different rules, deliberately. ``tool_call`` may come from
    the catalog, because the launcher adds a parser for a family it recognises
    even when nothing asked. ``reasoning`` may not: nothing adds a reasoning
    parser, so a catalog claim that the model thinks is not evidence that the
    thinking will arrive where the client looks for it.

    The catalog was the only source, which meant a model nobody curated got
    ``"tool_call": false`` — and on a coding agent that is not a cosmetic
    field, it is the difference between an assistant that edits files and one
    that describes how it would.

    Reported from the cluster, on Qwen3-Coder-Next served from the Hub:

        "tool_call": false, "reasoning": false

    while AINode had itself chosen ``--tool-call-parser qwen3_coder`` for that
    launch (models/tool_parsers.py matches ``qwen3-?coder``). Two parts of the
    same program disagreeing about the same model, and the wrong one is the
    one written into the client config.

    So the launch arguments come first: a flag the engine is running is not an
    opinion. The catalog stays as the source for what flags cannot show —
    vision, and a curated model's own declaration.
    """
    caps = {str(c).lower() for c in (getattr(info, "capabilities", None) or [])}
    tool_call = "tool_use" in caps
    attachment = bool(caps & {"vision", "image", "multimodal"})

    args = _launch_args(entry, info)
    joined = " ".join(args)
    if "--tool-call-parser" in args or "--enable-auto-tool-choice" in args \
            or "--tool-call-parser=" in joined:
        tool_call = True

    # reasoning is NOT taken from the catalog, and that is the difference
    # between it and tool_call. The catalog capability says the model thinks;
    # the client field says the reasoning arrives in its own place on the wire,
    # and only --reasoning-parser makes vLLM put it there. Without the flag the
    # <think> block comes back as ordinary content, and a client told to expect
    # it separated gets a first response it cannot read.
    #
    # Reported from the cluster, on Smaug-Flash — catalog capabilities carry
    # "reasoning", the recipe carries --tool-call-parser and no reasoning
    # parser, and the generated config said "reasoning": true:
    #
    #     also mit "limit": {"context": 90112, ...} bricht er quasi sofort ab
    #
    # Same shape as the tool_call bug one field over: two parts of one program
    # disagreeing, with the wrong one written into the client config.
    reasoning = ("--reasoning-parser" in args or "--reasoning-config" in args
                 or "--reasoning-parser=" in joined)

    # Nothing in the flags and nothing in the catalog: ask the same table the
    # launcher asks. It answers for a family rather than a repo, which is why
    # it is the last resort and not the first.
    if not tool_call and model_id:
        try:
            from ainode.models.tool_parsers import parser_for_model

            tool_call = bool(parser_for_model(model_id))
        except Exception:
            logger.debug("could not ask for %s's tool parser", model_id,
                         exc_info=True)
    return tool_call, reasoning, attachment


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


def build_opencode_config(app, base_url: str, served=None,
                          cache_tokens: Optional[Dict[str, int]] = None) -> dict:
    """The whole opencode.json, ready to paste.

    ``served`` is [(model id, entry)] — what runs now when omitted, a profile's
    entries for build_opencode_config_for_profile. ``cache_tokens`` overrides
    the measured cache size per model (a profile's plan knows it before
    anything has run).
    """
    manager = app.get("model_manager")
    models: Dict[str, dict] = {}
    notes: List[str] = []

    for model_id, entry in (_served_llms(app) if served is None else served):
        info = None
        if manager is not None:
            try:
                info = manager._find_catalog_by_hf_repo(model_id)
            except Exception:
                logger.debug("no catalog entry for %s", model_id, exc_info=True)

        tool_call, reasoning, attachment = _capabilities(
            info, entry, model_id)
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

        # The window is what one request may be. Whether that many tokens
        # fit for every session at once is a question for the cache, and the
        # generator never asked it: 2 sessions x 131072 against a cache of
        # 180000 tokens means the engine preempts one mid-answer — the agent
        # "stops working" with nothing in the log that says why. So the
        # limit is each session's share of the cache when that is smaller.
        seqs = _concurrency(entry, info)
        cache = (cache_tokens or {}).get(model_id) or _cache_tokens(app, model_id, window)
        if seqs > 1 and cache and cache // seqs < window:
            share = cache // seqs
            notes.append(
                f"{model_id}: {seqs} concurrent sessions share a cache of "
                f"{cache:,} tokens, so each gets {share:,} — not the "
                f"{window:,} the window allows. The limit is sized for "
                f"that; relaunch with fewer concurrent requests for more.")
            window = share

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
    # What the copied config says, in one short string: the dashboard keeps
    # the one it handed out and compares it with this, so a config that went
    # stale when a model was reloaded with another window says so (F3) — the
    # 608,512-against-131,072 case, which surfaced as an agent that stopped.
    import hashlib
    import json as _json

    digest = hashlib.sha1(_json.dumps(models, sort_keys=True).encode()).hexdigest()
    return {"config": config, "notes": notes, "fingerprint": digest[:12]}


class _ProfileLaunch:
    """A profile entry as the config builder reads it, with the flags the
    launch will add on its own (a reasoning parser the engine image has, F1)."""

    def __init__(self, entry, extra_args):
        self.model = entry.model
        self.max_model_len = entry.max_model_len
        self.extra_vllm_args = list(extra_args)


def build_opencode_config_for_profile(app, profile, base_url: str) -> dict:
    """An opencode.json for what ``profile`` will serve — before applying it.

    The windows and session counts are the profile's own; the cache each model
    gets comes from planning the profile (planner/household.py) where the
    wizard made it. A model that runs as replicas is one model to the client:
    the router spreads it, so it gets the smaller of the replicas' limits.
    """
    available = None
    try:
        from ainode.engine.image_probe import cached_probe

        found = cached_probe("vllm-node:latest") or {}
        available = found.get("reasoning_parsers")
    except Exception:
        logger.debug("no engine probe for the reasoning parsers", exc_info=True)

    cache: Dict[str, int] = {}
    if getattr(profile, "wizard", None):
        try:
            from ainode.planner.household import plan_household

            plan = plan_household(app, profile.wizard)
            for planned in plan.get("models") or []:
                tokens = int(planned.get("kv_tokens") or 0)
                if tokens:
                    model = planned["model"]
                    cache[model] = min(cache.get(model, tokens), tokens)
        except Exception:
            logger.exception("could not plan %s for its client config", profile.name)

    served = []
    seen: Dict[str, int] = {}
    for entry in profile.entries:
        if entry.kind not in ("", "llm"):
            continue
        args = list(entry.extra_vllm_args or [])
        if getattr(app.get("config"), "auto_reasoning_parser", True):
            from ainode.models.reasoning_parsers import reasoning_args

            args += reasoning_args(entry.model, args, available)
        launch = _ProfileLaunch(entry, args)
        if entry.model in seen:
            # A replica: keep the smaller window of the two.
            kept = served[seen[entry.model]][1]
            if entry.max_model_len and (not kept.max_model_len
                                        or entry.max_model_len < kept.max_model_len):
                served[seen[entry.model]] = (entry.model, launch)
            continue
        seen[entry.model] = len(served)
        served.append((entry.model, launch))
    result = build_opencode_config(app, base_url, served=served, cache_tokens=cache)
    result["profile"] = profile.name
    return result

