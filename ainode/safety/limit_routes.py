"""POST /api/nodes/{node_id}/memory-limit — set a node's memory limit.

Set from the profile wizard on the head, for any node (wizzard.md §5): the
head writes its own config directly and forwards to a peer's PATCH /api/config
with the cluster key, which the peer's sign-in accepts from another node.
The value is NodeConfig.memory_limit_gb, decimal GB of total use, 0 = none.
"""

from __future__ import annotations

import logging

from aiohttp import web

logger = logging.getLogger(__name__)

__all__ = ["register_limit_routes", "MIN_LIMIT_GB"]

#: Below this nothing fits; 0 means "no limit".
MIN_LIMIT_GB = 8.0


def register_limit_routes(app: web.Application) -> None:
    app.router.add_post("/api/nodes/{node_id}/memory-limit", handle_set_limit)


def _parse(body) -> float:
    value = float((body or {}).get("gb") or 0)
    if value < 0 or 0 < value < MIN_LIMIT_GB:
        raise ValueError(f"a limit is 0 (none) or at least {MIN_LIMIT_GB:.0f} GB")
    return round(value, 1)


async def handle_set_limit(request: web.Request) -> web.Response:
    node_id = request.match_info["node_id"]
    try:
        gb = _parse(await request.json())
    except (ValueError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    config = request.app["config"]
    if node_id in ("", str(getattr(config, "node_id", "") or "")):
        config.memory_limit_gb = gb
        try:
            config.save()
        except Exception:
            logger.exception("could not save the memory limit")
            return web.json_response({"error": "could not save"}, status=500)
        return web.json_response({"ok": True, "node_id": config.node_id,
                                  "memory_limit_gb": gb})

    cluster = request.app.get("cluster_state")
    node = None
    if cluster is not None:
        node = next((n for n in cluster.members() if n.node_id == node_id), None)
    if node is None:
        return web.json_response({"error": f"no node {node_id!r} in the cluster"},
                                 status=404)
    host = (getattr(node, "fabric_ip", "") or getattr(node, "peer_ip", "") or "").strip()
    session = request.app.get("client_session")
    if not host or session is None:
        return web.json_response({"error": f"{node_id} cannot be reached"},
                                 status=502)
    url = f"http://{host}:{getattr(node, 'web_port', 3000) or 3000}/api/config"
    try:
        # The cluster key is added by the session's trace config for cluster
        # hosts (auth/cluster_key.py).
        async with session.patch(url, json={"memory_limit_gb": gb},
                                 timeout=10) as resp:
            payload = await resp.json(content_type=None)
            if resp.status != 200 or "memory_limit_gb" in (payload.get("rejected") or []):
                return web.json_response(
                    {"error": f"{node_id} refused it: {payload}"}, status=502)
    except Exception as exc:
        return web.json_response(
            {"error": f"{node_id} unreachable: {exc.__class__.__name__}"}, status=502)
    return web.json_response({"ok": True, "node_id": node_id, "memory_limit_gb": gb})
