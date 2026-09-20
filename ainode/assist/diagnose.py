"""Ask a model that is already running what a failure means.

Deliberately not a model of its own. AINode does not bundle a small helper
model and does not download one: if something is loaded and answering, it can
read a traceback, and if nothing is loaded there is no assistant — the button
is not drawn. A cluster whose models have all failed to start is exactly the
case where a bundled helper would also fail to start.

The failing model is never used, even when it is somehow still answering: it
is the subject of the question.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from ainode.assist.briefing import SYSTEM_BRIEFING, hardware_notes
from ainode.assist.context import render_context

logger = logging.getLogger(__name__)

__all__ = ["helper_candidates", "build_messages", "ask_helper", "AssistError"]

#: The whole rendered context. A helper with a short window is the norm, not
#: the exception, and a prompt that overflows it fails with a 400 that says
#: nothing useful to the operator.
MAX_CONTEXT_CHARS = 12000

#: Enough for the three-part answer the briefing asks for, with room for a
#: reasoning model to think first.
MAX_ANSWER_TOKENS = 1200


class AssistError(RuntimeError):
    """The assistant could not answer. Carries an HTTP status for the route."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def helper_candidates(app, exclude: str = "") -> List[Tuple[str, str, int]]:
    """Every model that could answer, as (model, host, port).

    Local first — it is the cheapest hop and the one most likely to be
    reachable — then remote, each group in a stable order so two calls a
    second apart do not pick two different models.
    """
    from ainode.api.server import _routing_table

    config = app.get("config")
    cluster = app.get("cluster_state")
    if cluster is None or config is None:
        return []
    try:
        table = _routing_table(cluster, config.node_id, config.api_port)
    except Exception:
        logger.debug("could not build the routing table", exc_info=True)
        return []
    skip = str(exclude or "").strip()
    local, remote = [], []
    for model in sorted(table):
        if model == skip:
            continue
        host, port = table[model]
        (local if host == "localhost" else remote).append((model, host, int(port)))
    return local + remote


def build_messages(context: dict) -> List[dict]:
    system = SYSTEM_BRIEFING + hardware_notes(context.get("gpu_names") or [])
    body = render_context(context)
    if len(body) > MAX_CONTEXT_CHARS:
        # From the front: the launch settings and the error are at the top and
        # the log tail at the bottom, and both ends matter more than the
        # middle of a log.
        head, tail = body[:MAX_CONTEXT_CHARS // 3], body[-(MAX_CONTEXT_CHARS // 3 * 2):]
        body = head + "\n\n... (context truncated) ...\n\n" + tail
    return [
        {"role": "system", "content": system},
        {"role": "user", "content":
            body + "\n\nDiagnose this failure for the operator."},
    ]


async def ask_helper(session, model: str, host: str, port: int,
                     messages: List[dict], timeout: float = 240.0) -> dict:
    """One non-streaming chat completion. Raises AssistError on anything else."""
    import aiohttp

    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        # Low but not zero: a diagnosis is a judgement, and greedy decoding on
        # a long technical prompt tends to loop.
        "temperature": 0.2,
        "max_tokens": MAX_ANSWER_TOKENS,
        "stream": False,
    }
    try:
        async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            raw = await resp.text()
            if resp.status != 200:
                raise AssistError(
                    f"{model} answered {resp.status}: {raw[:400]}", status=502)
            data = await resp.json(content_type=None)
    except AssistError:
        raise
    except Exception as exc:
        raise AssistError(f"could not reach {model} at {host}:{port}: {exc}",
                          status=502) from exc

    answer, thought = _extract(data)
    if not answer and thought:
        # A reasoning model can spend the whole budget thinking and return an
        # empty `content` — the same failure mode that makes a chat client
        # believe the turn ended. The thinking is still an answer of sorts, so
        # it is shown rather than reported as nothing.
        answer = thought
        thought = ""
    if not answer:
        raise AssistError(f"{model} returned an empty answer", status=502)
    usage = data.get("usage") or {}
    return {
        "answer": answer.strip(),
        "reasoning": (thought or "").strip(),
        "helper_model": model,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def _extract(data: dict) -> Tuple[str, str]:
    choices = (data or {}).get("choices") or []
    if not choices:
        return "", ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        # Some servers answer with content parts rather than a string.
        content = "".join(p.get("text", "") for p in content
                          if isinstance(p, dict))
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if not isinstance(reasoning, str):
        reasoning = ""
    return str(content or ""), reasoning


def choose_helper(app, exclude: str = "",
                  wanted: str = "") -> Optional[Tuple[str, str, int]]:
    """The helper to use: the requested one if it is serving, else the first."""
    candidates = helper_candidates(app, exclude=exclude)
    if wanted:
        for entry in candidates:
            if entry[0] == wanted:
                return entry
        return None
    return candidates[0] if candidates else None
