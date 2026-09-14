"""Run a throughput measurement from the UI, and read the result back.

A benchmark takes minutes, so it cannot be an HTTP request the browser holds
open. It is a job: POST starts it, GET polls it, and the last finished run
stays readable so a comparison does not have to be written down by hand.

One at a time, cluster-wide-ish: two benchmarks in flight measure each other.
The guard is per-node, which is the honest scope — a second operator on
another node's UI could still overlap, and the results would say so (a
throughput that halves for no reason). Locking across the cluster for this
would be more machinery than the problem deserves.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from aiohttp import web

from ainode.api.params import as_object, int_field, str_field
from ainode.bench.runner import (
    MAX_CONCURRENCY,
    MAX_LEVELS,
    MAX_OUTPUT_TOKENS,
    PROMPT_STYLES,
    BenchError,
    BenchSpec,
    run_benchmark,
)

logger = logging.getLogger(__name__)

__all__ = ["register_bench_routes"]

_STATE = "bench_state"


def register_bench_routes(app: web.Application) -> None:
    app.setdefault(_STATE, {"running": False})
    app.router.add_get("/api/bench/options", handle_options)
    app.router.add_post("/api/bench/run", handle_run)
    app.router.add_get("/api/bench/status", handle_status)
    app.router.add_post("/api/bench/cancel", handle_cancel)


async def handle_options(request: web.Request) -> web.Response:
    """What can be measured, and against what: the models this node can route
    to. Asking the operator to type a model id correctly is a way to produce
    404s that look like failed benchmarks."""
    models = []
    try:
        from ainode.api.server import handle_v1_models

        response = await handle_v1_models(request)
        import json as _json

        models = [entry.get("id") for entry
                  in (_json.loads(response.body).get("data") or [])
                  if entry.get("id")]
    except Exception:
        logger.exception("could not list models for the benchmark")
    return web.json_response({
        "models": models,
        "styles": sorted(PROMPT_STYLES),
        "limits": {"concurrency": MAX_CONCURRENCY, "levels": MAX_LEVELS,
                   "max_tokens": MAX_OUTPUT_TOKENS},
        "defaults": {"concurrency": [1, 4, 8], "max_tokens": 256,
                     "prompt_tokens": 0, "style": "chat"},
    })


async def handle_run(request: web.Request) -> web.Response:
    state: dict = request.app.setdefault(_STATE, {"running": False})
    if state.get("running"):
        # _public(): the state carries the asyncio.Task, which is not JSON.
        # Passing it straight out turned "another run is in progress" — a
        # perfectly ordinary answer — into a 500 with a traceback.
        return web.json_response(
            {"error": "a benchmark is already running", "status": _public(state)},
            status=409)

    body = as_object(await _json(request))
    levels = body.get("concurrency")
    if not isinstance(levels, list):
        levels = [1, 4, 8]
    spec = BenchSpec(
        model=str_field(body, "model"),
        concurrency=[int(n) for n in levels if isinstance(n, (int, float))],
        max_tokens=int_field(body, "max_tokens", default=256),
        prompt_tokens=int_field(body, "prompt_tokens", default=0),
        style=str_field(body, "style") or "chat",
        base_url=f"http://127.0.0.1:{request.app['config'].web_port}",
    )
    try:
        spec.validate()
    except BenchError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    state.clear()
    state.update({
        "running": True, "started_at": time.time(), "finished_at": None,
        "model": spec.model, "style": spec.style,
        "requested_prompt_tokens": spec.prompt_tokens,
        "max_tokens": spec.max_tokens, "concurrency": spec.concurrency,
        "current": None, "results": [], "error": "",
    })

    async def _run() -> None:
        def progress(event: dict) -> None:
            if event.get("state") == "running":
                state["current"] = event.get("concurrency")
            elif event.get("state") == "level_done":
                state["results"].append(event["result"])
                state["current"] = None

        try:
            await run_benchmark(spec, session=request.app.get("client_session"),
                                on_progress=progress)
        except asyncio.CancelledError:
            state["error"] = "cancelled"
            raise
        except BenchError as exc:
            state["error"] = str(exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("benchmark failed")
            state["error"] = str(exc)
        finally:
            state["running"] = False
            state["current"] = None
            state["finished_at"] = time.time()

    state["task"] = asyncio.get_event_loop().create_task(_run())
    return web.json_response({"started": True, "spec": {
        "model": spec.model, "concurrency": spec.concurrency,
        "max_tokens": spec.max_tokens, "prompt_tokens": spec.prompt_tokens,
        "style": spec.style}}, status=202)


async def handle_status(request: web.Request) -> web.Response:
    state: dict = request.app.get(_STATE) or {"running": False}
    return web.json_response(_public(state))


def _public(state: dict) -> dict:
    """The state without the task handle — everything else is JSON."""
    return {k: v for k, v in state.items() if k != "task"}


async def handle_cancel(request: web.Request) -> web.Response:
    """Stop it. A benchmark holds KV slots that a real user wants back, and
    the levels above the current one can take minutes each."""
    state: dict = request.app.get(_STATE) or {}
    task: Optional[asyncio.Task] = state.get("task")
    if not state.get("running") or task is None or task.done():
        return web.json_response({"cancelled": False, "reason": "not running"})
    task.cancel()
    return web.json_response({"cancelled": True})


async def _json(request: web.Request):
    try:
        return await request.json()
    except Exception:
        return {}
