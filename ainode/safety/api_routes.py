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
        **_headroom(request.app, payload.get("total_mb")),
    })
    return web.json_response(payload)


def _headroom(app, total_mb) -> dict:
    """The planning headroom: configured (None = automatic) and what it is
    on this node."""
    from ainode.core.units import gb_from_mib
    from ainode.planner.api_routes import configured_headroom
    from ainode.planner.compute import plan_headroom_gb

    configured = configured_headroom(app)
    total = gb_from_mib(float(total_mb or 0)) if total_mb else 0.0
    return {"plan_headroom_gb": configured,
            "plan_headroom_effective_gb": plan_headroom_gb(total, configured)}


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

    # The planning headroom: null or "" for automatic, else 0-30 GB.
    headroom = ...
    if "plan_headroom_gb" in body:
        raw = body.get("plan_headroom_gb")
        if raw is None or str(raw).strip() == "":
            headroom = -1.0
        else:
            try:
                headroom = float(raw)
            except (TypeError, ValueError):
                return web.json_response(
                    {"error": "plan_headroom_gb must be a number or empty"}, status=400)
            if not 0 <= headroom <= 30:
                return web.json_response(
                    {"error": "plan_headroom_gb must be between 0 and 30"}, status=400)

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
        if headroom is not ...:
            config.plan_headroom_gb = headroom
        try:
            config.save()
        except Exception:
            logger.exception("could not persist the memory reserve")
    return await handle_get(request)
