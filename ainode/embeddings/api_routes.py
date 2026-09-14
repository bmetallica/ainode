"""API route handlers for embedding models.

Exposes:
- ``POST /v1/embeddings`` — OpenAI-compatible embeddings endpoint
- ``GET /api/embeddings/models`` — catalog of known embedding models
- ``POST /api/embeddings/models/{model_id}/load`` — eagerly load a model
- ``POST /api/embeddings/models/{model_id}/unload`` — drop from memory
"""

from __future__ import annotations

import logging
from typing import List

import aiohttp
from aiohttp import web

from ainode.embeddings.manager import (
    EmbeddingManager,
    KNOWN_EMBEDDING_MODELS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_embedding_routes(app: web.Application) -> None:
    """Attach embedding routes to the aiohttp app.

    The caller must have already placed an :class:`EmbeddingManager` at
    ``app["embedding_manager"]``.
    """
    app.router.add_post("/v1/embeddings", handle_v1_embeddings)
    app.router.add_get("/api/embeddings/models", handle_list_embedding_models)
    app.router.add_post(
        "/api/embeddings/models/{model_id:.+}/load", handle_load_embedding_model
    )
    app.router.add_post(
        "/api/embeddings/models/{model_id:.+}/unload", handle_unload_embedding_model
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    # Cheap whitespace approximation — matches OpenAI's "usage" stat well
    # enough for clients that just want a non-zero number.
    return max(1, len(text.split()))


def _error(message: str, *, code: str = "invalid_request_error", status: int = 400) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": code}}, status=status
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def peer_with_model(app, model_id: str):
    """(node_id, host, web_port) of a peer that has ``model_id`` loaded, or None.

    Embedding models are in-process, so each node can only answer for its own.
    A client pointed at the head — the normal arrangement, one address for the
    whole cluster — got "not loaded" for a model running perfectly well on
    node 3. Peers advertise what they have loaded, so the head can forward
    instead of refusing.
    """
    cluster = app.get("cluster_state")
    if cluster is None or not model_id:
        return None
    own = str(getattr(app.get("config"), "node_id", "") or "")
    try:
        nodes = cluster.get_nodes()
    except Exception:  # pragma: no cover - defensive
        return None
    for node in nodes:
        if node.node_id == own:
            continue
        if model_id not in (getattr(node, "embedding_models", []) or []):
            continue
        host = getattr(node, "fabric_ip", "") or ""
        if host:
            return node.node_id, host, node.web_port
    return None


async def _forward_embeddings(request: web.Request, body: dict, target) -> web.Response:
    node_id, host, port = target
    url = f"http://{host}:{port}/v1/embeddings"
    session = request.app.get("client_session")
    if session is None:  # pragma: no cover - the app always has one
        return _error(f"cannot reach node '{node_id}'", code="server_error", status=502)
    try:
        async with session.post(
                url, json=body, timeout=aiohttp.ClientTimeout(total=120)) as upstream:
            data = await upstream.read()
            ctype = upstream.headers.get(
                "Content-Type", "application/json").split(";")[0].strip()
            return web.Response(status=upstream.status, body=data, content_type=ctype)
    except aiohttp.ClientError as exc:
        return _error(f"failed to reach node '{node_id}' at {url}: {exc}",
                      code="server_error", status=502)


async def handle_v1_embeddings(request: web.Request) -> web.Response:
    """OpenAI-compatible embeddings endpoint."""
    manager: EmbeddingManager = request.app["embedding_manager"]

    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    if not isinstance(body, dict):
        return _error("Body must be a JSON object")

    model_id = body.get("model")
    if not model_id or not isinstance(model_id, str):
        return _error("'model' is required and must be a string")

    raw_input = body.get("input")
    if raw_input is None:
        return _error("'input' is required (string or array of strings)")

    if isinstance(raw_input, str):
        texts: List[str] = [raw_input]
    elif isinstance(raw_input, list):
        if not all(isinstance(x, str) for x in raw_input):
            return _error("'input' array must contain only strings")
        texts = raw_input
    else:
        return _error("'input' must be a string or array of strings")

    # Tag the request so the server-view log shows the embedding model.
    try:
        request["_log_model"] = model_id
    except Exception:
        pass

    # Not here? Ask the node that has it, rather than refusing on behalf of
    # the whole cluster.
    if not manager.is_loaded(model_id):
        target = peer_with_model(request.app, model_id)
        if target is not None:
            return await _forward_embeddings(request, body, target)

    try:
        vectors = await manager.aembed(model_id, texts)
    except RuntimeError as exc:
        return _error(str(exc), code="dependency_missing", status=503)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("embedding failure for %s", model_id)
        return _error(f"embedding failed: {exc}", code="server_error", status=500)

    total_tokens = sum(_approx_tokens(t) for t in texts)
    data = [
        {"object": "embedding", "embedding": vec, "index": idx}
        for idx, vec in enumerate(vectors)
    ]
    return web.json_response(
        {
            "object": "list",
            "data": data,
            "model": model_id,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }
    )


async def handle_list_embedding_models(request: web.Request) -> web.Response:
    manager: EmbeddingManager = request.app["embedding_manager"]
    loaded = {m["id"]: m for m in manager.list_loaded()}
    models = []
    for entry in manager.list_known():
        info = dict(entry)
        info["loaded"] = info["id"] in loaded
        models.append(info)
    # A model loaded from outside the curated list — which the UI can now ask
    # for by repo id — appeared nowhere: this listing walked the catalog and
    # marked it, so anything not in the catalog was invisible however loaded
    # it was, including in the tab that had just loaded it.
    known_ids = {entry["id"] for entry in models}
    for model_id, meta in sorted(loaded.items()):
        if model_id in known_ids:
            continue
        info = dict(meta)
        info["loaded"] = True
        info.setdefault("hf_repo", model_id)
        info.setdefault("description", "Loaded from Hugging Face")
        models.append(info)

    # Name the node for everything loaded, and merge in what peers report.
    # Without this the tab says LOADED for a model on this node and nothing at
    # all for one on another — so an operator who placed the RAG model on node
    # 3 sees an empty tab and reasonably concludes the load did not take.
    own_name = _own_node_name(request.app)
    for info in models:
        if info.get("loaded"):
            info.setdefault("node_name", own_name)
    by_id = {info["id"]: info for info in models}
    for node_id, node_name, model_id in _peer_embedding_models(request.app):
        existing = by_id.get(model_id)
        if existing is not None:
            if not existing.get("loaded"):
                existing["loaded"] = True
                existing["node_id"] = node_id
                existing["node_name"] = node_name
            continue
        entry = {"id": model_id, "hf_repo": model_id, "loaded": True,
                 "node_id": node_id, "node_name": node_name,
                 "description": "Loaded on " + node_name}
        by_id[model_id] = entry
        models.append(entry)

    return web.json_response({"models": models, "count": len(models)})


def _own_node_name(app) -> str:
    config = app.get("config")
    return str(getattr(config, "node_name", "") or
               getattr(config, "node_id", "") or "this node")


def _peer_embedding_models(app):
    """(node_id, node_name, model_id) for every peer's loaded model."""
    cluster = app.get("cluster_state")
    if cluster is None:
        return []
    own = str(getattr(app.get("config"), "node_id", "") or "")
    found = []
    try:
        nodes = cluster.get_nodes()
    except Exception:  # pragma: no cover - defensive
        return []
    for node in nodes:
        if node.node_id == own:
            continue
        name = node.node_name or node.node_id
        for model_id in (getattr(node, "embedding_models", []) or []):
            if model_id:
                found.append((node.node_id, name, model_id))
    return found


async def handle_load_embedding_model(request: web.Request) -> web.Response:
    manager: EmbeddingManager = request.app["embedding_manager"]
    model_id = request.match_info.get("model_id", "")
    if not model_id:
        return _error("model_id required")

    if manager.is_loaded(model_id):
        meta = next(
            (m for m in manager.list_loaded() if m["id"] == model_id), None
        )
        return web.json_response(
            {"ok": True, "model_id": model_id, "status": "loaded", "model": meta}
        )

    try:
        meta = await manager.aload(model_id)
    except RuntimeError as exc:
        return _error(str(exc), code="dependency_missing", status=503)
    except Exception as exc:
        logger.exception("failed to load embedding model %s", model_id)
        return _error(f"failed to load: {exc}", code="server_error", status=500)

    # Record it so a restart brings it back. An embedding model backing a RAG
    # pipeline is expected to stay available the way a served LLM does, and LLM
    # instances are already replayed on boot.
    manager.save_manifest()
    return web.json_response(
        {"ok": True, "model_id": model_id, "status": "loaded", "model": meta}
    )


async def handle_unload_embedding_model(request: web.Request) -> web.Response:
    manager: EmbeddingManager = request.app["embedding_manager"]
    model_id = request.match_info.get("model_id", "")
    if not model_id:
        return _error("model_id required")
    unloaded = manager.unload(model_id)
    if unloaded:
        manager.save_manifest()   # an unload must not come back on the next boot
    return web.json_response(
        {
            "ok": unloaded,
            "model_id": model_id,
            "status": "unloaded" if unloaded else "not_loaded",
        }
    )


__all__ = [
    "register_embedding_routes",
    "KNOWN_EMBEDDING_MODELS",
]
