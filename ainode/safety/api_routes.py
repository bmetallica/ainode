"""Read and set the host-memory reserve."""

from __future__ import annotations

import logging

from aiohttp import web

from ainode.api.params import as_object
from ainode.safety.memory_guard import PRESETS

logger = logging.getLogger(__name__)

__all__ = ["register_safety_routes", "get_memory_guard"]


def get_memory_guard(app):
    return app.get("memory_guard")


def register_safety_routes(app: web.Application) -> None:
    app.router.add_get("/api/safety/memory", handle_get)
    app.router.add_put("/api/safety/memory", handle_put)


async def handle_get(request: web.Request) -> web.Response:
    guard = get_memory_guard(request.app)
    if guard is None:
        return web.json_response({"enabled": False, "available": False,
                                  "presets": PRESETS})
    reading = guard.read()
    payload = reading.to_dict()
    payload.update({
        "enabled": guard.enabled,
        "available": True,
        "warn_gb": round(guard.warn_mb / 1024, 1),
        "critical_gb": round(guard.critical_mb / 1024, 1),
        "presets": PRESETS,
    })
    return web.json_response(payload)


async def handle_put(request: web.Request) -> web.Response:
    guard = get_memory_guard(request.app)
    if guard is None:
        return web.json_response({"error": "no memory guard on this node"},
                                 status=503)
    try:
        body = as_object(await request.json())
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    preset = str(body.get("preset") or "").strip()
    if preset:
        if preset not in PRESETS:
            return web.json_response(
                {"error": f"unknown preset {preset!r}",
                 "known": sorted(PRESETS)}, status=400)
        body = {**PRESETS[preset], **body}

    def _number(name):
        value = body.get(name)
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number")

    try:
        warn_gb, critical_gb = _number("warn_gb"), _number("critical_gb")
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    enabled = body.get("enabled")
    guard.configure(warn_gb=warn_gb, critical_gb=critical_gb,
                    enabled=None if enabled is None else bool(enabled))

    # Persisted, or the reserve resets to the default on the next restart —
    # which is the moment after a node has just been rebooted for running out.
    config = request.app.get("config")
    if config is not None:
        config.host_memory_warn_gb = round(guard.warn_mb / 1024, 2)
        config.host_memory_critical_gb = round(guard.critical_mb / 1024, 2)
        config.host_memory_guard = guard.enabled
        try:
            config.save()
        except Exception:
            logger.exception("could not persist the memory reserve")
    return await handle_get(request)
