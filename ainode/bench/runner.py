"""Measure what a model on this cluster actually delivers.

Capacity planning on this hardware is not something to reason about from
parameter counts. Two measurements made the point on the same afternoon:

  * a 26B MoE served 70 tok/s single-stream and 544 tok/s across 16 streams,
    so the cluster carried far more people than the arithmetic suggested;
  * a 230B MoE across two nodes served 20.5 tok/s where its catalog entry
    claimed 42 — half, because two all-reduces per layer cost more than the
    pooled bandwidth gains.

Neither is knowable without running it. This module is the throwaway script
that produced those numbers, kept: one place, in the product, so the next
question about "how many users" is answered by the cluster rather than by an
estimate.

Deliberately over the OpenAI-compatible endpoint rather than in-process. That
is the path a real client takes — proxy, routing, tokenizer, sampling — so
what comes out is what a user would get, not what the engine could do in
isolation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["BenchError", "BenchSpec", "BenchResult", "build_prompt",
           "run_benchmark", "PROMPT_STYLES"]

#: Ceilings. A benchmark is a load generator pointed at a production cluster:
#: without limits a typo in the UI can hold every KV slot for an hour.
MAX_CONCURRENCY = 64
MAX_OUTPUT_TOKENS = 4096
MAX_PROMPT_TOKENS = 1_000_000
MAX_LEVELS = 8

#: Roughly four characters per token for English and German prose. Padding is
#: sized by this and then MEASURED — the response reports the token count the
#: engine actually saw, because an estimate presented as a context length is
#: how a capacity plan goes wrong.
CHARS_PER_TOKEN = 4

PROMPT_STYLES: Dict[str, str] = {
    "chat": (
        "Erkläre in wenigen Absätzen, worauf es bei der Kapazitätsplanung "
        "eines lokalen LLM-Clusters ankommt."
    ),
    "code": (
        "Schreibe eine Python-Funktion, die eine CSV-Datei einliest, nach "
        "einer Spalte gruppiert und den Mittelwert je Gruppe als Dictionary "
        "zurückgibt. Erkläre danach kurz, wie sie arbeitet."
    ),
    "rag": (
        "Beantworte die Frage ausschließlich anhand des obenstehenden "
        "Kontexts. Frage: Welche Aussagen des Kontexts widersprechen "
        "einander, und warum?"
    ),
}


class BenchError(RuntimeError):
    """The benchmark cannot be run as specified."""


@dataclass
class BenchSpec:
    """What to measure.

    ``prompt_tokens`` is the interesting one: throughput at 1K of context and
    throughput at 64K are different numbers, and the second is what a RAG or
    coding workload actually looks like. Padding the prompt is the only way to
    measure it, since the engine's cost is driven by what it has to attend
    over, not by what the operator hoped.
    """

    model: str
    concurrency: List[int] = field(default_factory=lambda: [1, 4, 8])
    max_tokens: int = 256
    prompt_tokens: int = 0
    style: str = "chat"
    temperature: float = 0.7
    base_url: str = ""

    def validate(self) -> "BenchSpec":
        if not self.model:
            raise BenchError("no model selected")
        levels = [int(n) for n in self.concurrency if int(n) > 0]
        if not levels:
            raise BenchError("no concurrency levels given")
        if len(levels) > MAX_LEVELS:
            raise BenchError(f"at most {MAX_LEVELS} concurrency levels")
        if max(levels) > MAX_CONCURRENCY:
            raise BenchError(
                f"concurrency above {MAX_CONCURRENCY} is refused: this points "
                f"a load generator at a production cluster")
        if not 1 <= int(self.max_tokens) <= MAX_OUTPUT_TOKENS:
            raise BenchError(f"max_tokens must be 1..{MAX_OUTPUT_TOKENS}")
        if not 0 <= int(self.prompt_tokens) <= MAX_PROMPT_TOKENS:
            raise BenchError(f"prompt_tokens must be 0..{MAX_PROMPT_TOKENS}")
        if self.style not in PROMPT_STYLES:
            raise BenchError(f"unknown prompt style {self.style!r}")
        self.concurrency = sorted(set(levels))
        return self


@dataclass
class BenchResult:
    """One concurrency level, measured."""

    concurrency: int
    ok: int = 0
    failed: int = 0
    wall_seconds: float = 0.0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens_per_second: float = 0.0
    per_stream_tokens_per_second: float = 0.0
    first_token_seconds: float = 0.0
    slowest_seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def build_prompt(style: str, prompt_tokens: int) -> str:
    """The question, padded to roughly ``prompt_tokens``.

    The padding is filler text the model is told to ignore, not repetition of
    a single token: a run of identical tokens is unrepresentative of anything
    and prefix caching would collapse it, reporting a context length that was
    never attended over.
    """
    question = PROMPT_STYLES.get(style) or PROMPT_STYLES["chat"]
    if prompt_tokens <= 0:
        return question

    wanted_chars = int(prompt_tokens) * CHARS_PER_TOKEN
    # Varied, so neither the tokenizer nor the prefix cache sees a pattern.
    filler_parts: List[str] = []
    produced = 0
    index = 0
    while produced < wanted_chars:
        index += 1
        sentence = (
            f"Abschnitt {index}: Der Messwert {index * 7 % 997} wurde am "
            f"Knoten {index % 5 + 1} erhoben und betrug "
            f"{(index * 13) % 89},{index % 100:02d} Einheiten. "
        )
        filler_parts.append(sentence)
        produced += len(sentence)
    filler = "".join(filler_parts)[:wanted_chars]
    return (
        "Der folgende Kontext ist Füllmaterial für eine Durchsatzmessung.\n\n"
        f"{filler}\n\n{question}"
    )


async def _one_request(session, url: str, headers: dict, body: dict,
                       timeout) -> tuple:
    """(seconds, completion_tokens, prompt_tokens, error)."""
    started = time.monotonic()
    try:
        async with session.post(url, json=body, headers=headers,
                                timeout=timeout) as response:
            payload = await response.json()
            if response.status != 200:
                message = payload.get("error") if isinstance(payload, dict) else None
                if isinstance(message, dict):
                    message = message.get("message")
                return 0.0, 0, 0, str(message or f"HTTP {response.status}")[:200]
    except asyncio.TimeoutError:
        return 0.0, 0, 0, "timed out"
    except Exception as exc:
        return 0.0, 0, 0, str(exc)[:200]

    elapsed = time.monotonic() - started
    usage = (payload.get("usage") or {}) if isinstance(payload, dict) else {}
    return (elapsed,
            int(usage.get("completion_tokens") or 0),
            int(usage.get("prompt_tokens") or 0),
            "")


async def run_benchmark(
    spec: BenchSpec,
    *,
    session=None,
    on_progress: Optional[Callable[[dict], None]] = None,
    request_timeout: float = 900.0,
) -> List[BenchResult]:
    """Run ``spec`` and return one result per concurrency level.

    Levels run in ascending order and sequentially, never overlapping: two
    levels in flight at once would measure each other rather than the model.
    """
    import aiohttp

    spec.validate()
    base = (spec.base_url or "http://127.0.0.1:3000").rstrip("/")
    url = f"{base}/v1/chat/completions"
    prompt = build_prompt(spec.style, spec.prompt_tokens)
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    headers = {"Content-Type": "application/json"}
    body = {
        "model": spec.model,
        "max_tokens": int(spec.max_tokens),
        "temperature": float(spec.temperature),
        "messages": [{"role": "user", "content": prompt}],
    }

    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    results: List[BenchResult] = []
    try:
        for level in spec.concurrency:
            if on_progress is not None:
                try:
                    on_progress({"state": "running", "concurrency": level})
                except Exception:  # pragma: no cover
                    logger.exception("bench progress callback failed")

            started = time.monotonic()
            gathered = await asyncio.gather(*[
                _one_request(session, url, headers, body, timeout)
                for _ in range(level)
            ])
            wall = time.monotonic() - started

            durations = [d for d, _, _, err in gathered if not err]
            tokens = sum(t for _, t, _, err in gathered if not err)
            prompts = [p for _, _, p, err in gathered if not err]
            errors = [err for _, _, _, err in gathered if err]

            result = BenchResult(concurrency=level)
            result.ok = len(durations)
            result.failed = len(errors)
            result.wall_seconds = round(wall, 2)
            result.completion_tokens = tokens
            result.prompt_tokens = prompts[0] if prompts else 0
            if wall > 0:
                result.total_tokens_per_second = round(tokens / wall, 1)
            if durations:
                per = [t / d for d, t, _, err in gathered
                       if not err and d > 0 and t]
                if per:
                    result.per_stream_tokens_per_second = round(sum(per) / len(per), 1)
                result.slowest_seconds = round(max(durations), 2)
            if errors:
                # The first one, in full. A count of failures says nothing
                # about whether the model was missing or the context too long.
                result.error = errors[0]
            results.append(result)

            if on_progress is not None:
                try:
                    on_progress({"state": "level_done", "result": result.to_dict()})
                except Exception:  # pragma: no cover
                    logger.exception("bench progress callback failed")

            # Every request failing is a configuration problem, not a data
            # point — running the heavier levels would take minutes to repeat
            # the same error.
            if result.ok == 0:
                break
    finally:
        if owns_session:
            await session.close()
    return results
