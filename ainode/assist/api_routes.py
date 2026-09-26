"""The error assistant's HTTP surface.

Three routes, on every node:

* ``GET  /api/engine/log``      the tail of this node's engine log. The head
                                reads it from the node that failed, which may
                                not be itself.
* ``GET  /api/assist/status``   whether an assistant is available at all, and
                                which models could answer.
* ``POST /api/assist/diagnose`` the diagnosis itself.

The raw error is not touched by any of this. It stays where it was, rendered
by the instance card; a diagnosis is added beneath it, attributed to the model
that produced it.
"""

from __future__ import annotations

import logging

import aiohttp
from aiohttp import web

from ainode.api.params import as_object, int_field, str_field
from ainode.assist.context import collect_context
from ainode.assist.diagnose import (
    AssistError,
    ask_helper,
    build_messages,
    choose_helper,
    helper_candidates,
)

logger = logging.getLogger(__name__)

__all__ = ["register_assist_routes"]

#: How much log to read. Generous, because the filtering happens afterwards
#: and a distributed launch interleaves the output of several workers.
DEFAULT_LOG_LINES = 400
MAX_LOG_LINES = 2000


def register_assist_routes(app: web.Application) -> None:
    app.router.add_get("/api/engine/log", handle_engine_log)
    app.router.add_get("/api/engine/env", handle_engine_env)
    app.router.add_get("/api/assist/status", handle_assist_status)
    app.router.add_post("/api/assist/diagnose", handle_assist_diagnose)


async def handle_engine_log(request: web.Request) -> web.Response:
    """GET /api/engine/log?model=&lines= — this node's engine log tail."""
    lines = min(int(request.query.get("lines") or DEFAULT_LOG_LINES), MAX_LOG_LINES)
    model = request.query.get("model") or ""
    text = _local_log(request.app, model, lines)
    return web.json_response({
        "node_id": str(getattr(request.app.get("config"), "node_id", "") or ""),
        "model": model,
        "lines": lines,
        "text": text,
    })


async def handle_engine_env(request: web.Request) -> web.Response:
    """GET /api/engine/env?image=&names=A,B — what this image will read.

    ``names`` is optional; with it the answer is only the verdict on those,
    which is what a launch form needs. Without it the whole registry comes
    back, which is what someone writing a recipe needs.
    """
    from ainode.engine.engine_env import (env_warning, known_env_names,
                                          unregistered_names)

    image = request.query.get("image") or "vllm-node:latest"
    refresh = str(request.query.get("refresh") or "").lower() in ("1", "true")
    known = known_env_names(image, refresh=refresh)
    names = [n for n in (request.query.get("names") or "").split(",") if n]
    payload = {
        "image": image,
        "probed": bool(known),
        "count": len(known),
    }
    if names:
        payload["unregistered"] = unregistered_names(names, known)
        payload["warning"] = env_warning({n: "" for n in names}, image)
    else:
        payload["names"] = sorted(known)
    if not known:
        payload["note"] = (
            f"could not ask {image} for its environment registry — no verdict "
            f"is given rather than a wrong one")
    return web.json_response(payload)


async def handle_assist_status(request: web.Request) -> web.Response:
    """GET /api/assist/status?exclude=<model> — can anything answer right now?

    The UI draws the button from this: no model serving means no assistant,
    and saying so costs one field rather than a failed request after a click.
    """
    exclude = request.query.get("exclude") or ""
    candidates = helper_candidates(request.app, exclude=exclude)
    return web.json_response({
        "available": bool(candidates),
        "helpers": [c[0] for c in candidates],
        "helper_model": candidates[0][0] if candidates else "",
    })


async def handle_assist_diagnose(request: web.Request) -> web.Response:
    """POST /api/assist/diagnose {model, node_id, error, helper_model?}."""
    try:
        body = as_object(await request.json())
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    model = str_field(body, "model")
    node_id = str_field(body, "node_id")
    error = str_field(body, "error")
    if not error and not model:
        return web.json_response(
            {"error": "nothing to diagnose: send the error text or the model"},
            status=400)

    helper = choose_helper(request.app, exclude=model,
                           wanted=str_field(body, "helper_model"))
    if helper is None:
        # 409 rather than 503: nothing is broken, there is simply no model
        # loaded to ask. The UI hides the button in this state, so reaching
        # here means the last one was unloaded between drawing and clicking.
        return web.json_response(
            {"error": "no other model is loaded, so there is nothing to ask. "
                      "Load any chat model on any node and try again."},
            status=409)
    helper_model, host, port = helper

    lines = int_field(body, "log_lines", default=DEFAULT_LOG_LINES,
                      minimum=10) or DEFAULT_LOG_LINES
    log_text = await _log_for(request, node_id, model, min(lines, MAX_LOG_LINES))
    launch = await _launch_for(request, node_id, model)

    context = collect_context(request.app, model, node_id, error,
                              launch=launch, log_text=log_text)
    session = request.app.get("client_session")
    if session is None:  # pragma: no cover - the app always has one
        session = aiohttp.ClientSession()
    try:
        result = await ask_helper(session, helper_model, host, port,
                                  build_messages(context))
    except AssistError as exc:
        return web.json_response({"error": str(exc), "helper_model": helper_model},
                                 status=exc.status)
    result["ok"] = True
    # Returned so the operator can see exactly what the model was told. An
    # answer built on context they cannot inspect is not checkable.
    result["context_sent"] = {
        "log_lines": len((context.get("log") or "").splitlines()),
        "nodes": len(context.get("nodes") or []),
        "launch_known": bool(launch),
    }
    return web.json_response(result)


# ---------------------------------------------------------------------------
# Gathering, local and remote
# ---------------------------------------------------------------------------

def _local_log(app, model: str, lines: int) -> str:
    """The engine log on this node, preferring the instance that was asked for."""
    manager = app.get("instances")
    backends = []
    if manager is not None and model:
        try:
            instance = manager.by_model(model)
        except Exception:
            instance = None
        if instance is not None and getattr(instance, "backend", None) is not None:
            backends.append(instance.backend)
    engine = app.get("engine")
    if engine is not None:
        backends.append(engine)
    if manager is not None and not backends:
        try:
            backends.extend(i.backend for i in manager.instances()
                            if getattr(i, "backend", None) is not None)
        except Exception:
            logger.debug("could not list instances for the log", exc_info=True)
    for backend in backends:
        reader = getattr(backend, "logs", None)
        if not callable(reader):
            continue
        try:
            text = reader(lines)
        except Exception:
            logger.debug("could not read a backend log", exc_info=True)
            continue
        if text:
            return str(text)
    return ""


def _node_address(app, node_id: str):
    """(host, web_port) for a node that is not this one, or None."""
    config = app.get("config")
    if not node_id or node_id == getattr(config, "node_id", ""):
        return None
    cluster = app.get("cluster_state")
    node = cluster.get_node(node_id) if cluster is not None else None
    host = (getattr(node, "fabric_ip", "") or "") if node else ""
    if not host:
        return None
    return host, getattr(node, "web_port", 3000)


async def _log_for(request: web.Request, node_id: str, model: str,
                   lines: int) -> str:
    address = _node_address(request.app, node_id)
    if address is None:
        return _local_log(request.app, model, lines)
    host, port = address
    data = await _fetch(request, f"http://{host}:{port}/api/engine/log",
                        params={"model": model, "lines": str(lines)})
    return str((data or {}).get("text") or "")


async def _launch_for(request: web.Request, node_id: str, model: str) -> dict:
    """How this model was started, from the node that started it."""
    address = _node_address(request.app, node_id)
    if address is None:
        try:
            from ainode.profiles.apply import local_launch_specs

            specs = local_launch_specs(request.app)
        except Exception:
            logger.debug("could not read the local launch config", exc_info=True)
            specs = []
    else:
        host, port = address
        data = await _fetch(request,
                            f"http://{host}:{port}/api/instances/launch-config")
        specs = (data or {}).get("instances") or []
    for spec in specs:
        if str(spec.get("model") or "") == model:
            return spec
    # A launch that failed early may have left no instance at all. The rest of
    # the context still stands; the prompt says the settings are unknown
    # rather than implying defaults were used.
    return {}


async def _fetch(request: web.Request, url: str, params=None):
    session = request.app.get("client_session")
    if session is None:
        return None
    try:
        async with session.get(
                url, params=params,
                timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                return None
            return await resp.json(content_type=None)
    except Exception:
        # A peer that cannot be reached costs the diagnosis one section, not
        # the diagnosis.
        logger.debug("could not fetch %s", url, exc_info=True)
        return None
