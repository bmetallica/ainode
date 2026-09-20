"""Read and set where each model runs."""

from __future__ import annotations

import logging

from aiohttp import web

from ainode.api.params import as_object, str_field, str_list_field
from ainode.placement.store import Placement, PlacementError, PlacementStore

logger = logging.getLogger(__name__)

__all__ = ["register_placement_routes", "get_placement_store"]


def get_placement_store(app) -> PlacementStore:
    store = app.get("placement_store")
    if store is None:
        store = PlacementStore()
        app["placement_store"] = store
    return store


def register_placement_routes(app: web.Application) -> None:
    app.router.add_get("/api/placement", handle_list)
    app.router.add_put("/api/placement", handle_put)
    app.router.add_delete("/api/placement/{model:.+}", handle_delete)


async def handle_list(request: web.Request) -> web.Response:
    store = get_placement_store(request.app)
    return web.json_response({
        "placements": [p.to_dict() for p in store.load().values()],
    })


async def handle_put(request: web.Request) -> web.Response:
    try:
        body = as_object(await request.json())
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    model = str_field(body, "model")
    if not model:
        return web.json_response({"error": "model required"}, status=400)
    node_ids = str_list_field(body, "node_ids")

    # Known nodes only. A typo here is not caught until the next launch, and
    # by then it looks like the cluster lost a node rather than like a setting
    # that was never right.
    known = _known_node_ids(request.app)
    unknown = [n for n in node_ids if known and n not in known]
    if unknown:
        return web.json_response(
            {"error": f"unknown node(s): {', '.join(unknown)}",
             "known": sorted(known)}, status=400)

    try:
        placement = Placement(model=model, node_ids=node_ids,
                              strategy=str_field(body, "strategy"))
    except PlacementError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    store = get_placement_store(request.app)
    try:
        if placement.node_ids:
            store.put(placement)
        else:
            # An empty set means "wherever", which is the absence of a
            # placement rather than a placement onto nothing.
            store.remove(placement.model)
    except OSError as exc:
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response({"ok": True, "placement": placement.to_dict()})


async def handle_delete(request: web.Request) -> web.Response:
    model = request.match_info.get("model", "")
    store = get_placement_store(request.app)
    try:
        removed = store.remove(model)
    except OSError as exc:
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response({"ok": True, "removed": removed})


def _known_node_ids(app) -> set:
    """Every node this one can see, itself included. Empty when discovery has
    not run, and an empty set skips the check rather than refusing everything.
    """
    ids = set()
    config = app.get("config")
    own = str(getattr(config, "node_id", "") or "")
    if own:
        ids.add(own)
    cluster = app.get("cluster_state")
    if cluster is not None:
        try:
            ids.update(n.node_id for n in cluster.get_nodes() if n.node_id)
        except Exception:
            logger.debug("could not list nodes for placement validation",
                         exc_info=True)
    return ids
