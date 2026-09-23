"""What this node — and the fleet — has measured."""

from __future__ import annotations

import asyncio
import logging

import aiohttp
from aiohttp import web

from ainode.measure.store import MeasurementStore

logger = logging.getLogger(__name__)

__all__ = ["register_measurement_routes"]


def register_measurement_routes(app: web.Application) -> None:
    app.router.add_get("/api/measurements", handle_local)
    app.router.add_post("/api/measurements/forget-stops", handle_forget_stops)
    app.router.add_delete("/api/measurements/{model:.+}", handle_forget)
    app.router.add_get("/api/cluster/measurements", handle_cluster)


def _store(app) -> MeasurementStore:
    store = app.get("measurement_store")
    if store is None:
        store = MeasurementStore()
        app["measurement_store"] = store
    return store


async def handle_local(request: web.Request) -> web.Response:
    config = request.app.get("config")
    return web.json_response({
        "node_id": str(getattr(config, "node_id", "") or ""),
        "measurements": [m.to_dict()
                         for m in _store(request.app).load().values()],
    })


async def handle_forget(request: web.Request) -> web.Response:
    """Throw one away.

    Needed because a measurement outlives the thing it measured: a new engine
    image, a different quantisation of the same repo, a node with more memory
    free. When the figure stops describing reality the honest move is to
    delete it and let the next launch write a new one.
    """
    model = request.match_info.get("model", "")
    return web.json_response({
        "ok": True, "forgotten": _store(request.app).forget(model)})


async def handle_forget_stops(request: web.Request) -> web.Response:
    """POST /api/measurements/forget-stops {model} — clear the guard's record.

    The refusal built from a kill is the right default and the wrong
    permanent state. What made the launch impossible is usually fixed by
    something this store cannot see, so there has to be a way to say "that
    was then" without deleting the measurements, which are still true.

    A POST with the model in the body rather than in the path: a repo id
    contains a slash, and this route would otherwise have to fight the one
    below it for the same URL.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    model = str((body or {}).get("model") or "").strip()
    if not model:
        return web.json_response({"error": "model required"}, status=400)
    cleared = _store(request.app).forget_guard_stops(model)
    return web.json_response({"ok": True, "model": model, "cleared": cleared})


async def handle_cluster(request: web.Request) -> web.Response:
    """Every node's measurements, gathered.

    A model measured on node 3 tells you what it will cost on node 2 — the
    machines are identical — so the fleet's measurements are worth more than
    any one node's.
    """
    config = request.app.get("config")
    cluster = request.app.get("cluster_state")
    session = request.app.get("client_session")

    async def _peer(node) -> dict:
        host = (getattr(node, "fabric_ip", "") or "").strip()
        if not host or session is None:
            return {"node_id": node.node_id, "measurements": []}
        url = f"http://{host}:{getattr(node, 'web_port', 3000)}/api/measurements"
        try:
            async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return {"node_id": node.node_id, "measurements": []}
                return await resp.json(content_type=None)
        except Exception:
            logger.debug("could not ask %s for measurements", node.node_id,
                         exc_info=True)
            return {"node_id": node.node_id, "measurements": []}

    peers = []
    for node in (cluster.members() if cluster is not None else []):
        if node.node_id == getattr(config, "node_id", ""):
            continue
        status = node.status.value if hasattr(node.status, "value") else str(node.status)
        if status in ("online", "serving", "member-ready"):
            peers.append(node)
    results = await asyncio.gather(*[_peer(n) for n in peers],
                                   return_exceptions=True)

    import json as _json

    rows = [_json.loads((await handle_local(request)).body)]
    for result in results:
        if not isinstance(result, BaseException) and isinstance(result, dict):
            rows.append(result)

    by_model: dict = {}
    for row in rows:
        node_id = row.get("node_id") or ""
        for entry in row.get("measurements") or []:
            model = entry.get("model")
            if not model:
                continue
            bucket = by_model.setdefault(model, [])
            bucket.append({**entry, "node_id": entry.get("node_id") or node_id})
    return web.json_response({"models": by_model})
